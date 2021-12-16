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
 * MA  02110-1301  USA
 */

#pragma once
#ifdef CUDA_ENABLED

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

#include <Eigen/Core>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuComplex.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

namespace icrar
{
namespace cuda
{
    /**
     * @brief Copies the argument of the 1st column/polarization in avgData to phaseAnglesI1
     * 
     * @param I1 the index vector for unflagged antennas
     * @param avgData the averaged data matrix
     * @param phaseAnglesI1 the output phaseAngles vector
     */
    __host__ void AvgDataToPhaseAngles(
        const device_vector<int>& I1,
        const device_matrix<std::complex<double>>& avgData,
        device_vector<double>& phaseAnglesI1);
} // namespace cuda
} // namespace icrar
#endif // CUDA_ENABLED
