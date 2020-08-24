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

namespace icrar
{
    MeasurementSet::MeasurementSet(std::string filepath)
    {
        m_measurementSet = std::make_unique<casacore::MeasurementSet>(filepath);
        m_msmc = std::make_unique<casacore::MSMainColumns>(*m_measurementSet);
        m_msc = std::make_unique<casacore::MSColumns>(*m_measurementSet);
    }

    MeasurementSet::MeasurementSet(const casacore::MeasurementSet& ms)
    {
        m_measurementSet = std::make_unique<casacore::MeasurementSet>(ms);
        m_msmc = std::make_unique<casacore::MSMainColumns>(*m_measurementSet);
        m_msc = std::make_unique<casacore::MSColumns>(*m_measurementSet);
    }

    MeasurementSet::MeasurementSet(std::istream& stream)
    {
        // don't skip the whitespace while reading
        std::cin >> std::noskipws;

        // use stream iterators to copy the stream to a string
        std::istream_iterator<char> it(std::cin);
        std::istream_iterator<char> end;
        std::string results = std::string(it, end);
    }

    unsigned int MeasurementSet::GetNumStations() const
    {
        return m_measurementSet->antenna().nrow();
    }

    unsigned int MeasurementSet::GetNumPols() const
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

    unsigned int MeasurementSet::GetNumBaselines() const
    {
        const size_t num_stations = (size_t)GetNumStations();
        return num_stations * (num_stations + 1) / 2; //TODO: +/- 1???
    }

    unsigned int MeasurementSet::GetNumChannels() const
    {
        return m_msc->spectralWindow().numChan().get(0);
    }

    Eigen::MatrixX3d MeasurementSet::GetCoords() const
    {
        GetCoords(0, GetNumBaselines());
    }

    Eigen::MatrixX3d MeasurementSet::GetCoords(unsigned int start_row, unsigned int nBaselines) const
    {
        Eigen::MatrixX3d matrix = Eigen::MatrixX3d::Zero(nBaselines, 3);
        icrar::ms_read_coords(
            *m_measurementSet,
            start_row,
            nBaselines,
            matrix(Eigen::all, 0).data(),
            matrix(Eigen::all, 1).data(),
            matrix(Eigen::all, 2).data());
        return matrix;
    }

    Eigen::Tensor<std::complex<double>, 3> MeasurementSet::GetVis() const
    {
        auto num_channels = GetNumChannels();
        auto num_baselines = GetNumBaselines();
        auto num_pols = GetNumPols();
        return GetVis(num_channels, num_baselines, num_pols);
    }

    Eigen::Tensor<std::complex<double>, 3> MeasurementSet::GetVis(std::uint32_t nChannels, std::uint32_t nBaselines, std::uint32_t nPolarizations) const
    {
        auto visibilities = Eigen::Tensor<std::complex<double>, 3>(nChannels, nBaselines, nPolarizations);
        int start_baseline = 0;
        int start_channel = 0;
        icrar::ms_read_vis(*m_measurementSet, start_baseline, start_channel, nChannels, nBaselines, nPolarizations, "DATA", (double*)visibilities.data());
        return visibilities;
    }
}