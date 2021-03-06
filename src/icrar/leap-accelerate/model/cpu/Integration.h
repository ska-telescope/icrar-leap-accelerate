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

#include <icrar/leap-accelerate/common/MVuvw.h>
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

        int64_t index; // row index
        int64_t x; // number of rows
        int64_t channels; // channels
        int64_t baselines; // baselines

        std::vector<MVuvw> m_UVW; //uvw is an array uvw[3][nbl] //Eigen::MatrixX3d
        Eigen::Tensor<std::complex<double>, 3> m_visibilities; //[npol][nbl][nch]

    public:
        Integration(
            int integrationNumber,
            const icrar::MeasurementSet& ms,
            uint32_t index,
            uint32_t channels,
            uint32_t baselines,
            uint32_t polarizations);

        bool operator==(const Integration& rhs) const;

        int GetIntegrationNumber() const { return m_integrationNumber; }

        /**
         * @brief Gets the number of baselines
         * 
         * @return int 
         */
        size_t GetBaselines() const { return baselines; }

        /**
         * @brief Gets the UVW list
         * 
         * @return const std::vector<icrar::MVuvw>& 
         */
        const std::vector<icrar::MVuvw>& GetUVW() const { return m_UVW; }
        std::vector<icrar::MVuvw>& GetUVW() { return m_UVW; }

        /**
         * @brief Get the Visibilities object of size (polarizations, baselines, channels)
         * 
         * @return Eigen::Tensor<std::complex<double>, 3>& 
         */
        const Eigen::Tensor<std::complex<double>, 3>& GetVis() const { return m_visibilities; }

        /**
         * @brief Get the Visibilities object of size (polarizations, baselines, channels)
         * 
         * @return Eigen::Tensor<std::complex<double>, 3>& 
         */
        Eigen::Tensor<std::complex<double>, 3>& GetVis() { return m_visibilities; }

        friend class icrar::cuda::DeviceIntegration;
    };
}
}
