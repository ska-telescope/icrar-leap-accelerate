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

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cuda_runtime.h>
#include <vector>

template<typename Tensor>
class cuda_mapped_tensor : public Tensor
{
public:
    cuda_mapped_tensor(const Tensor& ref) : Tensor(ref)
    {
        cudaHostRegister(this->data(), this->size() * sizeof(decltype(*(this->data()))), cudaHostRegisterPortable);
    }

    virtual ~cuda_mapped_tensor()
    {
        cudaHostUnregister(this->data());
    }
};

#endif // CUDA_ENABLED