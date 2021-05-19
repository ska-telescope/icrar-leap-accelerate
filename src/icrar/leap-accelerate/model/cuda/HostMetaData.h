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

#include <icrar/leap-accelerate/model/cpu/MetaData.h>
#include <cuda_runtime.h>

namespace icrar
{
namespace cuda
{
    /**
     * @brief A cuda decorator for for cpu::Integration. 
     * This class stores data on the host using pinned memory to allow for asyncronous read and write with cuda.
     */
    class HostMetaData : public cpu::MetaData
    {
    public:
        HostMetaData(
            const icrar::MeasurementSet& ms,
            boost::optional<unsigned int> refAnt,
            double minimumBaselineThreshold,
            bool computeInverse,
            bool useCache)
        : MetaData(ms, refAnt, minimumBaselineThreshold, computeInverse, useCache)
        {
            cudaHostRegister(m_A.data(), m_A.size() * sizeof(decltype(*m_A.data())), cudaHostRegisterPortable);
            cudaHostRegister(m_I.data(), m_I.size() * sizeof(decltype(*m_I.data())), cudaHostRegisterPortable);
            cudaHostRegister(m_A1.data(), m_A1.size() * sizeof(decltype(*m_A1.data())), cudaHostRegisterPortable);
            cudaHostRegister(m_I1.data(), m_I1.size() * sizeof(decltype(*m_I1.data())), cudaHostRegisterPortable);
            std::cout << m_I1.size() << std::endl;
            if(m_Ad.size() != 0)
            {
                cudaHostRegister(m_Ad.data(), m_Ad.size() * sizeof(decltype(*m_Ad.data())), cudaHostRegisterPortable);
            }
            if(m_Ad1.size() != 0)
            {
                cudaHostRegister(m_Ad1.data(), m_Ad1.size() * sizeof(decltype(*m_Ad1.data())), cudaHostRegisterPortable);
            }
        }

        ~HostMetaData()
        {
            cudaHostUnregister(m_A.data());
            cudaHostUnregister(m_I.data());
            cudaHostUnregister(m_A1.data());
            cudaHostUnregister(m_I1.data());
            cudaHostUnregister(m_Ad.data());
            cudaHostUnregister(m_Ad1.data());
        }
    };
}
}
#endif // CUDA_ENABLED
