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

#include "icrar/leap-accelerate/math/cuda/matrix_epsilon.h"


#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/common/eigen_stringutils.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>

#include <icrar/leap-accelerate/cuda/device_matrix.h>
#include <icrar/leap-accelerate/cuda/device_vector.h>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>
#include <limits>

namespace icrar
{
namespace cuda
{
    __global__ void g_epsilon(double threshold, size_t size, double* data)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if(index < size)
        {
            if(abs(data[index]) < threshold)
            {
                data[index] = 0;
            }
        }
    }

    void epsilon(device_matrix<double>& matrix, double threshold)
    {
        dim3 blockSize = dim3(1024, 1, 1);
        dim3 gridSize = dim3((int)ceil(static_cast<double>(matrix.GetCount()) / blockSize.x), 1, 1);
        g_epsilon<<<blockSize,gridSize>>>(threshold, matrix.GetCount(), matrix.Get());
    }
} // namespace cuda
} // namespace icrar
