/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#include "MeasurementSet.h"
#include <icrar/leap-accelerate/ms/utils.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <boost/numeric/conversion/cast.hpp>
#include <cstddef>

namespace icrar
{
    MeasurementSet::MeasurementSet(const std::string& filepath, boost::optional<int> overrideNStations, bool readAutocorrelations)
    : m_measurementSet(std::make_unique<casacore::MeasurementSet>(filepath))
    , m_msmc(std::make_unique<casacore::MSMainColumns>(*m_measurementSet))
    , m_msc(std::make_unique<casacore::MSColumns>(*m_measurementSet))
    , m_filepath(filepath)
    , m_readAutocorrelations(false)
    {
        std:tie(m_antennas, m_readAutocorrelations) = CalculateUniqueAntennas();
        LOG(warning) << "unique antennas loaded";

        if(overrideNStations.is_initialized())
        {
            m_stations = overrideNStations.get();
            LOG(warning) << "overriding number of stations will be removed in future releases";
        }
        else if(m_antennas.size() != m_measurementSet->antenna().nrow())
        {
            // The antenna column may have blank entries for flagged antennas
            LOG(warning) << "total antennas = " << m_measurementSet->antenna().nrow();
            LOG(warning) << "unique antennas = " << m_antennas.size();
            LOG(warning) << "using unique antennas";

            m_stations = boost::numeric_cast<int>(m_antennas.size());
        }
        else
        {
            m_stations = boost::numeric_cast<int>(m_measurementSet->antenna().nrow());
        }

        m_numBaselines = CalculateNumBaselines(m_readAutocorrelations);
        m_numRows = boost::numeric_cast<uint32_t>(m_msmc->uvw().nrow());
        m_numTimesteps = boost::numeric_cast<uint32_t>(m_numRows / m_numBaselines);
        assert(m_measurementSet->polarization().nrow() > 0);
        m_numPols = m_msc->polarization().numCorr().get(0);
        Validate();
    }

    void MeasurementSet::Validate() const
    {
        // Stations
        if(m_antennas.size() != GetNumStations())
        {
            LOG(error) << "unique antennas does not match number of stations";
            LOG(error) << "unique antennas: " << m_antennas.size();
            LOG(error) << "stations: " << GetNumStations();
        }

        // Baselines
        // Validate number of baselines in first epoch
        casacore::Vector<double> time = m_msmc->time().getColumn();
        auto epoch = time[0];
        auto epochRows = std::count(time.begin(), time.end(), epoch);

        if(epochRows != GetNumBaselines())
        {
            LOG(error) << "epoch rows does not match baselines";
            LOG(error) << "epoch rows: " << epochRows;
            LOG(error) << "baselines: " << GetNumBaselines();
            throw exception("visibilities at first epoch does not match number of baselines", __FILE__, __LINE__);
        }

        if(GetNumRows() < GetNumBaselines())
        {
            std::stringstream ss;
            ss << "invalid number of rows, expected >=" << GetNumBaselines() << ", got " << GetNumRows();
            throw icrar::file_exception(GetFilepath().get_value_or("unknown"), ss.str(), __FILE__, __LINE__);
        }

        if(GetNumBaselines() == 0 || (GetNumRows() % GetNumBaselines() != 0))
        {
            LOG(error) << "number of rows not an integer multiple of number of baselines";
            LOG(error) << "baselines: " << GetNumBaselines()
                         << " rows: " << GetNumRows()
                         << "total epochs ~= " << static_cast<double>(GetNumRows()) / GetNumBaselines();
            throw exception("number of rows not an integer multiple of baselines", __FILE__, __LINE__);
        }
    }

    uint32_t MeasurementSet::GetNumRows() const
    {
        return m_numRows;
    }

    uint32_t MeasurementSet::GetTotalAntennas() const
    {
        return boost::numeric_cast<uint32_t>(m_measurementSet->antenna().nrow());
    }

    uint32_t MeasurementSet::GetNumTimesteps() const
    {
        return m_numTimesteps;
    }

    std::vector<double> MeasurementSet::GetEpochs() const
    {
        casacore::Vector<double> time = m_msmc->time().getColumn();
        uint32_t timesteps = GetNumTimesteps();
        std::vector<double> result;
        for(uint32_t i = 0; i < timesteps; ++i)
        {
            result.push_back(time[i * GetNumBaselines()]);
        }
        return result;
    }

    uint32_t MeasurementSet::GetNumStations() const
    {
        return m_stations;
    }

    uint32_t MeasurementSet::GetNumPols() const
    {
        if(m_measurementSet->polarization().nrow() > 0)
        {
            return m_msc->polarization().numCorr().get(0);
        }
        else
        {
            throw icrar::not_implemented_exception(__FILE__, __LINE__);
        }
    }

    uint32_t MeasurementSet::GetNumBaselines() const
    {
        return m_numBaselines;
    }

    uint32_t MeasurementSet::CalculateNumBaselines(bool useAutocorrelations) const
    {
        //TODO(calgray): cache value
        if(useAutocorrelations)
        {
            const auto num_stations = GetNumStations();
            return num_stations * (num_stations + 1) / 2;
        }
        else
        {
            const auto num_stations = GetNumStations();
            return num_stations * (num_stations - 1) / 2;
        }
    }

    uint32_t MeasurementSet::GetNumChannels() const
    {
        if(m_msc->spectralWindow().nrow() > 0)
        {
            return m_msc->spectralWindow().numChan().get(0);
        }
        else
        {
            return 0;
        }
    }

    Eigen::VectorXb MeasurementSet::GetFlaggedBaselines() const
    {
        auto nBaselines = GetNumBaselines();

        // Selects the flags of the first epoch
        auto epochIndices = casacore::Slicer(casacore::Slice(0, nBaselines));
        
        // Selects only the flags of the first channel and polarization
        // TODO(calgray): may want to consider using logical OR over each channel and polarization
        if(!m_msmc->flag().isNull() && m_msmc->flag().nrow() > 0 && m_msmc->flag().isDefined(0))
        {
            auto flagSlice = casacore::Slicer(
                casacore::IPosition(2, 0, 0),
                casacore::IPosition(2, 1, 1),
                casacore::IPosition(2, 1, 1));
            return ToVector<bool>(m_msmc->flag().getColumnRange(epochIndices, flagSlice));
        }
        else
        {
            LOG(warning) << "baseline flags not found";
            return Eigen::VectorXb::Zero(nBaselines);
        }
    }

    uint32_t MeasurementSet::GetNumFlaggedBaselines() const
    {
        auto flaggedBaselines = GetFlaggedBaselines();
        return boost::numeric_cast<uint32_t>(std::count(flaggedBaselines.cbegin(), flaggedBaselines.cend(), true));
    }
	
	Eigen::VectorXb MeasurementSet::GetShortBaselines(double minimumBaselineThreshold) const
    {
        auto nBaselines = GetNumBaselines();
        Eigen::VectorXb baselineFlags = Eigen::VectorXb::Zero(nBaselines); 

        // Filter short baselines
        if(minimumBaselineThreshold > 0.0)
        {
            auto firstChannelSlicer = casacore::Slicer(casacore::Slice(0, 1));
            casacore::Matrix<double> uv = m_msmc->uvw().getColumn(firstChannelSlicer);

            // TODO(calgray): uv is of size baselines * timesteps, consider throwing a warning if
            // short baselines change in later timesteps
            for(uint32_t i = 0; i < nBaselines; i++)
            {
                if(std::sqrt(uv(i, 0) * uv(i, 0) + uv(i, 1) * uv(i, 1)) < minimumBaselineThreshold)
                {
                    baselineFlags(i) = true;
                }
            }
        }

        return baselineFlags;
    }

    uint32_t MeasurementSet::GetNumShortBaselines(double minimumBaselineThreshold) const
    {
        auto shortBaselines = GetShortBaselines(minimumBaselineThreshold);
        return boost::numeric_cast<uint32_t>(std::count(shortBaselines.cbegin(), shortBaselines.cend(), true));
    }

    Eigen::VectorXb MeasurementSet::GetFilteredBaselines(double minimumBaselineThreshold) const
    {
        Eigen::VectorXb result = GetFlaggedBaselines() || GetShortBaselines(minimumBaselineThreshold);
        return result;
    }

    uint32_t MeasurementSet::GetNumFilteredBaselines(double minimumBaselineThreshold) const
    {
        auto filteredBaselines = GetFilteredBaselines(minimumBaselineThreshold);
        return boost::numeric_cast<uint32_t>(std::count(filteredBaselines.cbegin(), filteredBaselines.cend(), true));
	}

    Eigen::MatrixX3d MeasurementSet::GetCoords() const
    {
        return GetCoords(0, 1);
    }

    Eigen::MatrixX3d MeasurementSet::GetCoords(uint32_t startTimestep, uint32_t intervalTimesteps) const
    {
        return icrar::ms_read_coords1<double>(*m_measurementSet, startTimestep * GetNumBaselines(), intervalTimesteps * GetNumBaselines());
    }

    Eigen::Tensor<std::complex<double>, 3> MeasurementSet::GetVis() const
    {
        return GetVis(0, 1);
    }

    Eigen::Tensor<std::complex<double>, 3> MeasurementSet::GetVis(
        uint32_t startTimestep,
        uint32_t intervalTimesteps,
        Slice polarizationSlice) const
    {
        uint32_t nPolarizations = GetNumPols();

        std::cout << "pol:" << polarizationSlice.GetStart() << ":" << polarizationSlice.GetInterval() << ":" << polarizationSlice.GetEnd(); 

        //normal mode
        Range polarizationRange = polarizationSlice.Evaluate(nPolarizations);

        std::cout << "pol:" << polarizationRange.GetStart() << ":" << polarizationRange.GetInterval() << ":" << polarizationRange.GetEnd(); 
        // XX + YY mode (first + last) or (first)
        //Range polarizationRange = Range(0, std::max(1u, nPolarizations-1), nPolarizations-1);

        return ReadVis(startTimestep, intervalTimesteps, polarizationRange, "DATA");
    }

    Eigen::Tensor<std::complex<double>, 3> MeasurementSet::ReadVis(uint32_t startTimestep, uint32_t intervalTimesteps, Range polarizationRange, const char* column) const
    {
        const uint32_t num_numBaselines = GetNumBaselines();
        const uint32_t num_channels = GetNumChannels();
        const unsigned int total_rows = GetNumRows();

        auto timestep_slice = Eigen::seq(startTimestep, startTimestep+intervalTimesteps, intervalTimesteps);
        const unsigned int start_row = startTimestep * num_numBaselines;
        const unsigned int rows = intervalTimesteps * num_numBaselines;
        
        auto pols_slice = polarizationRange.ToSeq();
        const unsigned int pol_length = boost::numeric_cast<unsigned int>(pols_slice.sizeObject());
        const unsigned int pol_stride = boost::numeric_cast<unsigned int>(pols_slice.incrObject());
        
        if(!m_measurementSet->tableDesc().isColumn(column))
        {
            throw icrar::exception("ms column not found", __FILE__, __LINE__);
        }

        if(strcmp(column, "DATA")
        && strcmp(column, "CORRECTED_DATA")
        && strcmp(column, "MODEL_DATA"))
        {
            throw icrar::exception("expected a data column", __FILE__, __LINE__);
        }

        if (start_row >= total_rows)
        {
            std::stringstream ss;
            ss << "ms out of range " << start_row << " >= " << total_rows; 
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        // clamp num_numBaselines
        if (start_row + rows > total_rows)
        {
            std::stringstream ss;
            ss << "row selection [" << start_row << "," << start_row + rows << "] exceeds total range [" << 0 << "," << total_rows << "]";
            throw icrar::exception(ss.str(), __FILE__, __LINE__);
        }

        // Create slicers for table DATA
        // Slicer for table rows: array[baselines,timesteps]
        casacore::IPosition start1(1, start_row);
        casacore::IPosition length1(1, rows);
        casacore::Slicer row_range(start1, length1);

        // Slicer for row entries: matrix[polarizations,channels]
        casacore::IPosition start2(2, 0, 0);
        casacore::IPosition length2(2, pol_length, num_channels);
        casacore::IPosition stride2(2, pol_stride, 1u);
        casacore::Slicer array_section(start2, length2, stride2);

        // Read the data.
        casacore::ArrayColumn<std::complex<float>> ac(*m_measurementSet, column);
        casacore::Array<std::complex<float>> column_range = ac.getColumnRange(row_range, array_section);

        Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 3>> view(column_range.data(), pol_length, num_channels, num_numBaselines * intervalTimesteps);

        //TODO: Converting ICD format from [pol, channels, baselines*timesteps] to [pol, baselines*timesteps, channels]
        const Eigen::array<Eigen::DenseIndex, 3> shuffle = { 0, 2, 1 };
        Eigen::Tensor<std::complex<double>, 3> output = view.shuffle(shuffle).cast<std::complex<double>>();
        return output;
    }

    std::set<int32_t> MeasurementSet::GetMissingAntennas() const
    {
        std::set<std::int32_t> antennas;
        for(int32_t i = 0; i < boost::numeric_cast<int32_t>(m_antennas.size()); i++)
        {
            if(m_antennas.find(i) == m_antennas.end())
            {
                antennas.insert(i);
            }
        }
        return antennas;
    }

    std::set<int32_t> MeasurementSet::GetFlaggedAntennas() const
    {
        auto fg = GetFilteredBaselines();        
        int32_t totalStations = GetTotalAntennas();

        // start with a set of all antennas flagged and unflag the ones 
        // that contain unflagged baseline data
        Eigen::VectorXi antennas = Eigen::VectorXi::Ones(totalStations);
        for(uint32_t n = 0; n < GetNumBaselines(); n++)
        {
            if(!fg(n))
            {
                antennas(m_msmc->antenna1()(n)) = 0;
                antennas(m_msmc->antenna2()(n)) = 0;
            }
        }
        // see https://libigl.github.io/matlab-to-eigen.html
        std::set<int32_t> indexes;
        for(int32_t i = 0; i < boost::numeric_cast<int32_t>(antennas.size()); ++i)
        {
            if(antennas(i))
            {
                indexes.insert(i);
            }
        }
        return indexes;
    }

    // template<typename InputInterator1, typename InputIterator2>
    // bool HasMatches(InputInterator1 first1, InputInterator1 last1, InputInterator2 first2)
    // {
    // }

    std::tuple<std::set<int32_t>, bool> MeasurementSet::CalculateUniqueAntennas() const
    {
        casacore::Vector<casacore::Int> a1 = m_msmc->antenna1().getColumn();
        casacore::Vector<casacore::Int> a2 = m_msmc->antenna2().getColumn();
        std::set<std::int32_t> antennas;
        std::set_union(a1.cbegin(), a1.cend(), a2.cbegin(), a2.cend(), std::inserter(antennas, antennas.begin()));
        bool autoCorrelations = std::mismatch(a1.cbegin(), a1.cend(), a2.cbegin(), [](int a, int b) { return a != b; }).first != a1.cend();
        return std::make_tuple(antennas, autoCorrelations);
    }
} // namespace icrar
