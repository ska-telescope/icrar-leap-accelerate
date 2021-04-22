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

#include "ComputePhaseDeltaKernel.h"
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
namespace cuda
{
    /**
     * @brief Computes the phase delta vector for the first polarization of avgData
     * 
     * @param A Antenna matrix
     * @param cal1 cal1 matrix
     * @param avgData averaged visibilities
     * @param deltaPhase output deltaPhase vector 
     */
    __global__ void g_CalcDeltaPhase(
        const Eigen::Map<const Eigen::MatrixXd> A,
        const Eigen::Map<const Eigen::VectorXd> cal1,
        const Eigen::Map<const Eigen::Matrix<thrust::complex<double>, -1, -1>> avgData,
        Eigen::Map<Eigen::VectorXd> deltaPhase);

    __host__ void CalcDeltaPhase(
        const device_matrix<double>& A,
        const device_vector<double>& cal1,
        const device_matrix<std::complex<double>>& avgData,
        device_matrix<double>& deltaPhase)
    {
        if(A.GetCols() != cal1.GetRows())
        {
            std::stringstream ss;
            ss << "a columns (" << A.GetCols() << ") does not match cal1 rows (" << cal1.GetRows() << ")";
            throw invalid_argument_exception(ss.str(), "A", __FILE__, __LINE__);
        }

        dim3 blockSize = dim3(1024, 1, 1);
        dim3 gridSize = dim3((int)ceil(static_cast<double>(A.GetRows()) / blockSize.x), 1, 1);

        auto AMap = Eigen::Map<const Eigen::MatrixXd>(A.Get(), A.GetRows(), A.GetCols());
        auto cal1Map = Eigen::Map<const Eigen::VectorXd>(cal1.Get(), cal1.GetRows());
        auto avgDataMap = Eigen::Map<const Eigen::Matrix<thrust::complex<double>, -1, -1>>(
            (thrust::complex<double>*)avgData.Get(), avgData.GetRows(), avgData.GetCols());
        auto deltaPhaseMap = Eigen::Map<Eigen::VectorXd>(deltaPhase.Get(), deltaPhase.GetRows());
        g_CalcDeltaPhase<<<blockSize,gridSize>>>(AMap, cal1Map, avgDataMap, deltaPhaseMap);
    }

    __global__ void g_CalcDeltaPhase(
        const Eigen::Map<const Eigen::MatrixXd> A,
        const Eigen::Map<const Eigen::VectorXd> cal1,
        const Eigen::Map<const Eigen::Matrix<thrust::complex<double>, -1, -1>> avgData,
        Eigen::Map<Eigen::VectorXd> deltaPhase)
    {
        constexpr double two_pi = 2 * CUDART_PI;
        int n = blockDim.x * blockIdx.x + threadIdx.x;

        if(n < A.rows())
        {
            double sum = A.row(n) * cal1;
            deltaPhase.row(n) = (thrust::exp(thrust::complex<double>(0, -two_pi * sum)) * avgData.row(n))
            .unaryExpr([](const thrust::complex<double>& v){ return thrust::arg(v); });
        }
    }
} // namespace cuda
} // namespace icrar
