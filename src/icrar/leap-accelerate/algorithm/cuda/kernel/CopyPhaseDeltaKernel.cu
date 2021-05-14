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

#include "CopyPhaseDeltaKernel.h"
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
namespace cuda
{
    __global__ void g_GenerateDeltaPhaseColumn(
        const Eigen::Map<const Eigen::MatrixXd> deltaPhase,
        Eigen::Map<Eigen::VectorXd> deltaPhaseColumn);

    __host__ void GenerateDeltaPhaseColumn(
        const device_matrix<double>& deltaPhase,
        device_vector<double>& deltaPhaseColumn)
    {
        if(deltaPhase.GetRows()+1 != deltaPhaseColumn.GetRows())
        {
            throw invalid_argument_exception("incorrect number of columns", "deltaPhaseColumn", __FILE__, __LINE__);
        }

        dim3 blockSize = dim3(1024, 1, 1);
        dim3 gridSize = dim3(static_cast<int>(ceil(static_cast<double>(deltaPhaseColumn.GetRows()) / blockSize.x)), 1, 1);

        auto deltaPhaseMap = Eigen::Map<const Eigen::MatrixXd>(deltaPhase.Get(), deltaPhase.GetRows(), deltaPhase.GetCols());
        auto deltaPhaseColumnMap = Eigen::Map<Eigen::VectorXd>(deltaPhaseColumn.Get(), deltaPhaseColumn.GetRows());
        g_GenerateDeltaPhaseColumn<<<blockSize,gridSize>>>(deltaPhaseMap, deltaPhaseColumnMap);
    }

    __global__ void g_GenerateDeltaPhaseColumn(
        const Eigen::Map<const Eigen::MatrixXd> deltaPhase,
        Eigen::Map<Eigen::VectorXd> deltaPhaseColumn)
    {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if(row < deltaPhase.rows())
        {
            deltaPhaseColumn(row) = deltaPhase(row, 0); // 1st pol only
        }
        else if (row < deltaPhaseColumn.rows())
        {
            deltaPhaseColumn(row) = 0; // deltaPhaseColumn is of size deltaPhaseRows+1 where the last/extra row = 0
        }
    }
} // namespace cuda
} // namespace icrar
