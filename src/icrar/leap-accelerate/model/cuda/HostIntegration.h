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

#ifdef CUDA_ENABLED

#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <cuda_runtime.h>

namespace icrar
{
namespace cuda
{
    /**
     * @brief A cuda decorator for cpu::Integration. 
     * This class stores data on the host using pinned memory to allow for asyncronous read and write with cuda.
     */
    class HostIntegration : public cpu::Integration
    {
    public:
        HostIntegration(
            int integrationNumber,
            const icrar::MeasurementSet& ms,
            unsigned int index,
            unsigned int channels,
            unsigned int baselines,
            unsigned int polarizations)
        : Integration(
            integrationNumber,
            ms,
            index,
            channels,
            baselines,
            polarizations)
        {
            cudaHostRegister(m_visibilities.data(), m_visibilities.size() * sizeof(decltype(*m_visibilities.data())), cudaHostRegisterPortable);
            cudaHostRegister(m_UVW.data(), m_UVW.size() * sizeof(decltype(*m_UVW.data())), cudaHostRegisterPortable);
        }

        ~HostIntegration()
        {
            cudaHostUnregister(m_visibilities.data());
            cudaHostUnregister(m_UVW.data());
        }
    };
}
}
#endif // CUDA_ENABLED
