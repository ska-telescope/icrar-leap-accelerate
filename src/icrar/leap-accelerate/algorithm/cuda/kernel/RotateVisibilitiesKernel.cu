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
#include <icrar/leap-accelerate/math/cuda/math.cuh>

namespace icrar
{
namespace cuda
{
    /**
     * @brief Rotates visibilities in parallel for baselines and channels
     * @note Atomic operator required for writing to @p pAvgData
     * 
     * @param constants measurement set constants
     * @param dd direction dependent rotation 
     * @param UVW unrotated uvws
     * @param integrationData inout integration data 
     * @param avgData output avgData to increment
     */
    __global__ void g_RotateVisibilities(
        const icrar::cpu::Constants constants,
        const Eigen::Matrix3d dd,
        const Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>> UVWs,
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 3>> integrationData,
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 2>> avgData);

    __host__ void RotateVisibilities(
        DeviceIntegration& integration,
        DeviceMetaData& metadata)
    {
        const auto& constants = metadata.GetConstants(); 
        assert(constants.channels == integration.GetChannels() && integration.GetChannels() == integration.GetVis().GetDimensionSize(2));
        assert(constants.nbaselines == metadata.GetAvgData().GetRows() && integration.GetBaselines() == integration.GetVis().GetDimensionSize(1));
        assert(constants.num_pols == integration.GetVis().GetDimensionSize(0));

        dim3 blockSize = dim3(128, 8, 1); // block size can be any value where the product is 1024
        dim3 gridSize = dim3(
            static_cast<int>(ceil((float)integration.GetBaselines() / blockSize.x)),
            static_cast<int>(ceil((float)integration.GetChannels() / blockSize.y)),
            1
        );

        auto integrationDataMap = Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 3>>(
            (cuDoubleComplex*)integration.GetVis().Get(),
            static_cast<int>(integration.GetVis().GetDimensionSize(0)), // inferring (const int) causes error
            static_cast<int>(integration.GetVis().GetDimensionSize(1)), // inferring (const int) causes error
            static_cast<int>(integration.GetVis().GetDimensionSize(2)) // inferring (const int) causes error
        );

        auto avgDataMap = Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 2>>(
            (cuDoubleComplex*)metadata.GetAvgData().Get(),
            static_cast<int>(metadata.GetAvgData().GetRows()), // inferring (const int) causes error
            static_cast<int>(metadata.GetAvgData().GetCols()) // inferring (const int) causes error
        );

        auto UVWMap = Eigen::Map<Eigen::Matrix<double, 3, -1>>(
            (double*)(metadata.GetUVW().Get()),
            3,
            metadata.GetUVW().GetCount()
        );

        g_RotateVisibilities<<<gridSize, blockSize>>>(
            constants,
            metadata.GetDD(),
            UVWMap,
            integrationDataMap,
            avgDataMap);
    }

    __global__ void g_RotateVisibilities(
        const icrar::cpu::Constants constants,
        const Eigen::Matrix3d dd,
        const Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>> UVWs,
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 3>> integrationData,
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 2>> avgData)
    {
        const int integration_baselines = integrationData.dimension(1);
        const int integration_channels = integrationData.dimension(2);
        const int md_baselines = constants.nbaselines; //metadata baselines
        const int polarizations = constants.num_pols;
        constexpr double two_pi = 2 * CUDART_PI;

        //parallel execution per channel
        int baseline = blockDim.x * blockIdx.x + threadIdx.x; //baseline amongst all time smeared baselines
        int channel = blockDim.y * blockIdx.y + threadIdx.y;

        if(baseline < integration_baselines && channel < integration_channels)
        {
            int md_baseline = baseline % md_baselines; 

            Eigen::Vector3d rotatedUVW = dd * UVWs.col(baseline);
            double shiftFactor = -two_pi * (rotatedUVW.z() - UVWs.col(baseline).z());

            // loop over channels
            double shiftRad = shiftFactor / constants.GetChannelWavelength(channel);
            cuDoubleComplex exp = cuCexp(make_cuDoubleComplex(0.0, shiftRad));

            for(int polarization = 0; polarization < polarizations; polarization++)
            {
                integrationData(polarization, baseline, channel) = cuCmul(integrationData(polarization, baseline, channel), exp);
            }
            bool hasNaN = false;
            for(int polarization = 0; polarization < polarizations; polarization++)
            {
                cuDoubleComplex n = integrationData(polarization, baseline, channel);
                hasNaN |= isnan(n.x) || isnan(n.y);
            }

            if(!hasNaN)
            {
                for(int polarization = 0; polarization < polarizations; ++polarization)
                {
                    atomicAdd(&avgData(md_baseline, polarization).x, integrationData(polarization, baseline, channel).x);
                    atomicAdd(&avgData(md_baseline, polarization).y, integrationData(polarization, baseline, channel).y);
                }
            }
        }
    }
} // namespace cuda
} // namespace icrar
