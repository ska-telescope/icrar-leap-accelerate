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

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/model/cpu/MVuvw.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>

#include <casacore/casa/Quanta/MVuvw.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <boost/optional.hpp>

#include <vector>
#include <array>
#include <complex>

namespace icrar
{
namespace cuda
{
    class DeviceIntegration;
}
}

namespace icrar
{
namespace cpu
{
    class MeasurementSet;

    /**
     * @brief A container for storing a visibilities tensor for accumulation during phase rotating.
     * 
     */
    class Integration
    {
    protected:
        int m_integrationNumber;
        int64_t m_rows;

        std::vector<MVuvw> m_UVW; //uvw is an array uvw[3][nbl] //Eigen::MatrixX3d
        Eigen::Tensor<std::complex<double>, 4> m_visibilities;

    public:
        Integration(
        int integrationNumber,
        const icrar::MeasurementSet& ms,
        uint32_t startTimestep,
        uint32_t intervalTimesteps,
        Slice polarizationSlice = Slice(0, boost::none, 1));

        bool operator==(const Integration& rhs) const;

        int GetIntegrationNumber() const { return m_integrationNumber; }

        size_t GetRows() const { return m_rows; }

        size_t GetNumPolarizations() const { return m_visibilities.dimension(0); }
        size_t GetNumChannels() const { return m_visibilities.dimension(1); }
        size_t GetNumBaselines() const { return m_visibilities.dimension(2); }
        size_t GetNumTimesteps() const { return m_visibilities.dimension(3); }

        /**
         * @brief Gets the UVW list
         * 
         * @return const std::vector<icrar::MVuvw>& 
         */
        const std::vector<icrar::MVuvw>& GetUVW() const { return m_UVW; }
        std::vector<icrar::MVuvw>& GetUVW() { return m_UVW; }

        /**
         * @brief Get the Visibilities object of size (polarizations, channels, baselines, timesteps)
         * 
         * @return Eigen::Tensor<std::complex<double>, 4>& 
         */
        const Eigen::Tensor<std::complex<double>, 4>& GetVis() const { return m_visibilities; }

        /**
         * @brief Get the Visibilities object of size size (polarizations, channels, baselines, timesteps)
         * 
         * @return Eigen::Tensor<std::complex<double>, 4>& 
         */
        Eigen::Tensor<std::complex<double>, 4>& GetVis() { return m_visibilities; }

        friend class icrar::cuda::DeviceIntegration;
    };
}
}
