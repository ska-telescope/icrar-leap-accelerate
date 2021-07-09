/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#pragma once

#ifdef CUDA_ENABLED

#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>

#include <icrar/leap-accelerate/model/cpu/MVuvw.h>
#include <icrar/leap-accelerate/common/constants.h>

#include <icrar/leap-accelerate/cuda/device_tensor.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <boost/optional.hpp>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

#include <cuComplex.h>

namespace icrar
{
namespace cpu
{
    class Integration;
}
}

namespace icrar
{
namespace cuda
{
    /**
     * @brief A Cuda memory buffer instance of visibility data for integration
     * 
     */
    class DeviceIntegration
    {
        int m_integrationNumber;
        device_tensor3<std::complex<double>> m_visibilities; //[polarizations][baselines][channels]
        int64_t m_rows;
        
    public:
        /**
         * @brief Construct a new Device Integration object where visibilities is a zero tensor of @shape 
         * 
         * @param shape 
         */
        DeviceIntegration(int integrationNumber, Eigen::DSizes<Eigen::DenseIndex, 3> shape);

        /**
         * @brief Construct a new Device Integration object with a data syncronous copy
         * 
         * @param integration 
         */
        DeviceIntegration(const icrar::cpu::Integration& integration);

        /**
         * @brief Set the Data object
         * 
         * @param integration 
         */
        __host__ void Set(const icrar::cpu::Integration& integration);

        /**
         * @brief Set the Data object
         * 
         * @param integration 
         */
        __host__ void Set(const icrar::cuda::DeviceIntegration& integration);

        int GetIntegrationNumber() const { return m_integrationNumber; }

        int64_t GetRows() const { return m_rows; }
        uint64_t GetChannels() const { return m_visibilities.GetDimensionSize(2); }
        
        const device_tensor3<std::complex<double>>& GetVis() const { return m_visibilities; }
        device_tensor3<std::complex<double>>& GetVis() { return m_visibilities; }

        /**
         * @brief Copies device data to a host object
         * 
         * @param host object with data on cpu memory
         */
        __host__ void ToHost(cpu::Integration& host) const;
    };
} // namespace cuda
} // namepace icrar

#endif // CUDA_ENABLED
