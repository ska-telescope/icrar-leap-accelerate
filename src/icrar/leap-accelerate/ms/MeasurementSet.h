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

#pragma once

#include <icrar/leap-accelerate/exception/exception.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays.h>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <boost/optional.hpp>

#include <iterator>
#include <string>
#include <exception>
#include <memory>
#include <tuple>
#include <vector>

namespace icrar
{
    class MeasurementSet
    {
        std::unique_ptr<casacore::MeasurementSet> m_measurementSet;
        std::unique_ptr<casacore::MSColumns> m_msc;
        std::unique_ptr<casacore::MSMainColumns> m_msmc;

        int m_stations;
        bool m_readAutocorrelations;

        boost::optional<std::string> m_filepath;

    public:
        MeasurementSet(std::string filepath, boost::optional<int> overrideNStations, bool readAutocorrelations);
        MeasurementSet(const casacore::MeasurementSet& ms, boost::optional<int> overrideNStations, bool readAutocorrelations);
        MeasurementSet(std::istream& stream, boost::optional<int> overrideNStations);

        boost::optional<std::string> GetFilepath() const { return m_filepath; }
        
        /**
         * @brief Gets a non-null pointer to a casacore::MeasurementSet
         * 
         * @return const casacore::MeasurementSet* 
         */
        const casacore::MeasurementSet* GetMS() const { return m_measurementSet.get(); }

        /**
         * @brief Gets a non-null pointer to a casacore::MSMainColumns
         * 
         * @return const casacore::MSMainColumns* 
         */
        const casacore::MSMainColumns* GetMSMainColumns() const { return m_msmc.get(); }
        
        /**
         * @brief Gets a non-null pointer to a casacore::MSColumns
         * 
         * @return const casacore::MSColumns* 
         */
        const casacore::MSColumns* GetMSColumns() const { return m_msc.get(); }

        /**
         * @brief Gets the number of stations excluding flagged stations. Overridable at construction.
         * 
         * @return unsigned int 
         */
        unsigned int GetNumStations() const;

        /**
         * @brief Get the number of baselines in the measurement set including autocorrelations (e.g. (0,0), (1,1), (2,2))
         * and including stations not recording rows.
         * @note TODO: baselines is always n*(n-1) / 2 and without autocorrelations
         * @return unsigned int 
         */
        unsigned int GetNumBaselines() const;

        /**
         * @brief Get the number of polarizations in the measurement set
         * 
         * @return unsigned int 
         */
        unsigned int GetNumPols() const;

        unsigned int GetNumChannels() const;

        unsigned int GetNumRows() const;

        //std::vector<casacore::MVuvw> MeasurementSet::GetCoordsCasa(unsigned int start_row) const;
        Eigen::MatrixX3d GetCoords() const;
        Eigen::MatrixX3d GetCoords(unsigned int start_row, unsigned int nBaselines) const;

        Eigen::Tensor<std::complex<double>, 3> GetVis(
            std::uint32_t startBaseline,
            std::uint32_t startChannel,
            std::uint32_t nChannels,
            std::uint32_t nBaselines,
            std::uint32_t nPolarizations) const;
        Eigen::Tensor<std::complex<double>, 3> GetVis() const;

    private:
        /**
         * @brief Get the number of baselines in the measurement set (e.g. (0,0), (1,1), (2,2))
         * 
         * @return unsigned int 
         */
        unsigned int GetNumBaselines(bool useAutocorrelations) const;
    };
}
