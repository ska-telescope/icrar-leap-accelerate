/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#ifdef CUDA_ENABLED

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace icrar
{
namespace cuda
{
    /**
     * @brief A typeless cuda device buffer allocated using cuda 
     * device memory.
     * @tparam T buffer pointer cast. Compatible with void.
     */
    template<typename T>
    class device_buffer : boost::noncopyable
    {
        size_t m_sizeInBytes;
        T* m_buffer; // Pointer to cuda malloc memory

    public:
        device_buffer(size_t sizeInBytes)
        : m_sizeInBytes(sizeInBytes)
        {
            static_assert(!std::is_pointer<T>::value,
                "buffer not recommended for pointer types");
            cudaStream_t stream = nullptr;
            checkCudaErrors(cudaMallocAsync((void**)&m_buffer, sizeInBytes, stream));
        }
        
        ~device_buffer()
        {
            cudaStream_t stream = nullptr;
            checkCudaErrors(cudaFreeAsync((void*)m_buffer, stream));
        }

        T* get() { return m_buffer; }
        const T* get() const { return m_buffer; }
        size_t size() const { return m_sizeInBytes; }
    };

    /**
     * @brief A typeless cuda host buffer using pinned/page-locked 
     * cpu memory required for async operations
     * (prevents memory moving to VRAM).
     * @tparam T buffer pointer cast. Compatible with void.
     */
    template<typename T>
    class host_buffer : boost::noncopyable
    {
        size_t m_sizeInBytes;
        T* m_buffer;

    public:
        host_buffer(size_t sizeInBytes)
        : m_sizeInBytes(sizeInBytes)
        , m_buffer(nullptr)
        {
            checkCudaErrors(cudaMallocHost((void**)&m_buffer, sizeInBytes));
            assert(m_buffer != nullptr);
        }

        ~host_buffer()
        {
            cudaFreeHost(m_buffer);
        }

        T* get() { return m_buffer; }
        const T* get() const { return m_buffer; }
        size_t size() const { return m_sizeInBytes; }
    };
} // namepace cuda
} // namespace icrar

#endif