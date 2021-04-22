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

#include "RotateVisibilitiesKernel.h"
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
namespace cuda
{
    __global__ void g_AvgDataToPhaseAngles(
        const Eigen::Map<const Eigen::VectorXi> I1,
        const Eigen::Map<const Eigen::Matrix<thrust::complex<double>, -1, -1>> avgData,
        Eigen::Map<Eigen::VectorXd> phaseAnglesI1)
    {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if(row < I1.rows())
        {
            phaseAnglesI1(row) = thrust::arg(avgData(I1(row), 0));
        }
    }

    __host__ void AvgDataToPhaseAngles(const device_vector<int>& I1, const device_matrix<std::complex<double>>& avgData, device_vector<double>& phaseAnglesI1)
    {
        if(I1.GetRows()+1 != phaseAnglesI1.GetRows())
        {
            throw invalid_argument_exception("incorrect number of columns", "phaseAnglesI1", __FILE__, __LINE__);
        }

        dim3 blockSize = dim3(1024, 1, 1);
        dim3 gridSize = dim3(static_cast<int>(ceil(static_cast<double>(I1.GetRows()) / blockSize.x)), 1, 1);

        using MatrixXcd = Eigen::Matrix<thrust::complex<double>, -1, -1>;
        auto I1Map = Eigen::Map<const Eigen::VectorXi>(I1.Get(), I1.GetRows());
        auto avgDataMap = Eigen::Map<const MatrixXcd>((thrust::complex<double>*)avgData.Get(), avgData.GetRows(), avgData.GetCols());
        auto phaseAnglesI1Map = Eigen::Map<Eigen::VectorXd>(phaseAnglesI1.Get(), phaseAnglesI1.GetRows());
        g_AvgDataToPhaseAngles<<<blockSize, gridSize>>>(I1Map, avgDataMap, phaseAnglesI1Map);
    }
} // namespace cuda
} // namespace icrar
