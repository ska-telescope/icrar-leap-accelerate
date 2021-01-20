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

#include "CudaLeapCalibrator.h"

#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/vector_extensions.h>

#include <icrar/leap-accelerate/model/cuda/HostIntegration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/math/cuda/math.cuh>
#include <icrar/leap-accelerate/math/cuda/matrix.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/math/cpu/vector.h>

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/common/eigen_extensions.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/cuda/cuda_mapped_matrix.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <thrust/complex.h>

#include <boost/math/constants/constants.hpp>

#include <complex>
#include <istream>
#include <iostream>
#include <iterator>
#include <string>
#include <queue>
#include <exception>
#include <memory>
#include <set>

using Radians = double;
using namespace boost::math::constants;

namespace icrar
{
namespace cuda
{
    CudaLeapCalibrator::CudaLeapCalibrator()
    : m_cublasContext(nullptr)
    {
        int deviceCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if(deviceCount < 1)
        {
            throw icrar::exception("CUDA error: no devices supporting CUDA.", __FILE__, __LINE__);
        }

        checkCudaErrors(cublasCreate(&m_cublasContext));
    }

    CudaLeapCalibrator::~CudaLeapCalibrator()
    {
        checkCudaErrors(cublasDestroy(m_cublasContext));

        // cuda calls may still occur outside of this instance
        //checkCudaErrors(cudaDeviceReset());
    }

    cpu::CalibrateResult CudaLeapCalibrator::Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<SphericalDirection>& directions,
        double minimumBaselineThreshold,
        bool isFileSystemCacheEnabled)
    {
        LOG(info) << "Starting Calibration using cuda";
        LOG(info)
        << "stations: " << ms.GetNumStations() << ", "
        << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "flagged baselines: " << ms.GetNumFlaggedBaselines() << ", "
        << "baseline threshold: " << minimumBaselineThreshold << ", "
        << "short baselines: " << ms.GetNumShortBaselines(minimumBaselineThreshold) << ", "
        << "filtered baselines: " << ms.GetNumFilteredBaselines(minimumBaselineThreshold) << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << ms.GetNumRows() / ms.GetNumBaselines();

        profiling::timer calibration_timer;
        profiling::timer integration_read_timer;
        auto output_integrations = std::vector<std::vector<cpu::IntegrationResult>>();
        auto output_calibrations = std::vector<std::vector<cpu::CalibrationResult>>();
        auto input_queue = std::vector<cuda::DeviceIntegration>();

        // Flooring to remove incomplete measurements
        int integrations = ms.GetNumRows() / ms.GetNumBaselines();
        if(integrations == 0)
        {
            std::stringstream ss;
            ss << "invalid number of rows, expected >" << ms.GetNumBaselines() << ", got " << ms.GetNumRows();
            throw icrar::file_exception(ms.GetFilepath().get_value_or("unknown"), ss.str(), __FILE__, __LINE__);
        }

        auto integration = cuda::HostIntegration(0, ms, 0, ms.GetNumChannels(), ms.GetNumRows(), ms.GetNumPols());
        {
            for(int i = 0; i < directions.size(); ++i)
            {
                output_integrations.emplace_back();
                output_calibrations.emplace_back();
            }
            LOG(info) << "Read integration data in " << integration_read_timer;

            profiling::timer metadata_read_timer;
            LOG(info) << "Loading MetaData";
            
            auto metadata = icrar::cpu::MetaData(ms, integration.GetUVW(), minimumBaselineThreshold, isFileSystemCacheEnabled);
            
            auto constantBuffer = std::make_shared<ConstantBuffer>(
                metadata.GetConstants(),
                metadata.GetA(),
                metadata.GetI(),
                metadata.GetAd(),
                metadata.GetA1(),
                metadata.GetI1(),
                metadata.GetAd1()
            );

            auto solutionIntervalBuffer = std::make_shared<SolutionIntervalBuffer>(metadata.GetUVW());
            
            auto directionBuffer = std::make_shared<DirectionBuffer>(
                metadata.GetDirection(),
                metadata.GetDD(),
                metadata.GetUVW().size(),
                metadata.GetAvgData().rows(),
                metadata.GetAvgData().cols());

            auto deviceMetadata = DeviceMetaData(constantBuffer, solutionIntervalBuffer, directionBuffer);

            // Emplace a single empty tensor
    #ifdef HIGH_GPU_MEMORY
            const auto deviceIntegration = DeviceIntegration(integration);
    #endif
            LOG(info) << "Metadata loaded in " << metadata_read_timer;

            // always use a single integration
            input_queue.emplace_back(0, integration.GetVis().dimensions());

            profiling::timer phase_rotate_timer;
            for(int i = 0; i < directions.size(); ++i)
            {
                LOG(info) << "Processing direction " << i;
                LOG(info) << "Setting Metadata";
                metadata.SetDirection(directions[i]);
                directionBuffer->SetDirection(metadata.GetDirection());
                directionBuffer->SetDD(metadata.GetDD());
                directionBuffer->GetAvgData().SetZeroAsync();

                LOG(info) << "Sending integration to device";
    #ifdef HIGH_GPU_MEMORY
                input_queue[0].Set(deviceIntegration);
    #else
                input_queue[0].Set(integration);
    #endif

                LOG(info) << "RotateUVW";
                RotateUVW(
                    directionBuffer->GetDD(),
                    solutionIntervalBuffer->GetUVW(),
                    directionBuffer->GetRotatedUVW());

                LOG(info) << "PhaseRotate";
                PhaseRotate(
                    metadata,
                    deviceMetadata,
                    directions[i],
                    input_queue,
                    output_integrations[i],
                    output_calibrations[i]);
            }
            LOG(info) << "Performed PhaseRotate in " << phase_rotate_timer;
            LOG(info) << "Finished calibration in " << calibration_timer;
        }
        cudaHostUnregister(integration.GetVis().data());

        return std::make_pair(std::move(output_integrations), std::move(output_calibrations));
    }

    void CudaLeapCalibrator::PhaseRotate(
        cpu::MetaData& metadata,
        DeviceMetaData& deviceMetadata,
        const SphericalDirection& direction,
        std::vector<cuda::DeviceIntegration>& input,
        std::vector<cpu::IntegrationResult>& output_integrations,
        std::vector<cpu::CalibrationResult>& output_calibrations)
    {
        for(DeviceIntegration& integration : input)
        {
            LOG(info) << "Rotating integration " << integration.GetIntegrationNumber();
            RotateVisibilities(integration, deviceMetadata);
        }

        LOG(info) << "Calibrating in cuda";
        auto devicePhaseAnglesI1 = device_vector<double>(metadata.GetI1().rows() + 1);
        auto deviceCal1 = device_vector<double>(metadata.GetAd1().rows());
        auto deviceDInt = device_matrix<double>(metadata.GetI().size(), metadata.GetAvgData().cols());
        auto deviceDeltaPhaseColumn = device_vector<double>(metadata.GetI().size() + 1);
        auto cal1 = Eigen::VectorXd(metadata.GetAd1().rows());

        AvgDataToPhaseAngles(deviceMetadata.GetConstantBuffer().GetI1(), deviceMetadata.GetAvgData(), devicePhaseAnglesI1);
        cuda::multiply(m_cublasContext, deviceMetadata.GetConstantBuffer().GetAd1(), devicePhaseAnglesI1, deviceCal1);
        CalcDInt(deviceMetadata.GetConstantBuffer().GetA(), deviceCal1, deviceMetadata.GetAvgData(), deviceDInt);
        GenerateDeltaPhaseColumn(deviceDInt, deviceDeltaPhaseColumn);
        cuda::multiply_add<double>(m_cublasContext, deviceMetadata.GetConstantBuffer().GetAd(), deviceDeltaPhaseColumn, deviceCal1);
        deviceCal1.ToHost(cal1);
        output_calibrations.emplace_back(direction, cal1);
    }

    /**
     * @brief Kernel for rotating UVWs
     * 
     * @param dd rotation matrix
     * @param UVWs unrotated UVW buffer
     * @param RotatedUVWs output rotated UVW buffer
     */
    __global__ void g_RotateUVW(
        const Eigen::Matrix3d dd,
        const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>> UVWs,
        Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>> RotatedUVWs)
    {
        // Compute rows in parallel
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if(row < UVWs.rows())
        {
            RotatedUVWs.col(row) = dd * UVWs.col(row);
        }
    }

    __host__ void CudaLeapCalibrator::RotateUVW(Eigen::Matrix3d dd, const device_vector<icrar::MVuvw>& UVWs, device_vector<icrar::MVuvw>& rotatedUVWs)
    {
        assert(UVWs.GetCount() == rotatedUVWs.GetCount());
        dim3 blockSize = dim3(1024, 1, 1);
        dim3 gridSize = dim3((int)ceil((float)UVWs.GetCount() / blockSize.x), 1, 1);

        auto UVWsMap = Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>>(UVWs.Get()->data(), 3, UVWs.GetCount());
        auto rotatedUVWsMap = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(rotatedUVWs.Get()->data(), 3, rotatedUVWs.GetCount());
        g_RotateUVW<<<blockSize, gridSize>>>(dd, UVWsMap, rotatedUVWsMap);
    }

    /**
     * @brief Rotates visibilities in parallel for baselines and channels
     * @note Atomic operator required for writing to @p pAvgData
     * 
     * @param constants measurement set constants
     * @param rotatedUVW rotated uvws
     * @param UVW unrotated uvws
     * @param integrationData inout integration data 
     * @param avgData output avgData to increment
     */
    __global__ void g_RotateVisibilities(
        const icrar::cpu::Constants constants,
        const Eigen::Map<Eigen::Matrix<double3, Eigen::Dynamic, 1>> rotatedUVW,
        const Eigen::Map<Eigen::Matrix<double3, Eigen::Dynamic, 1>> UVW,
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 3>> integrationData,
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 2>> avgData)
    {
        const int integration_baselines = integrationData.dimension(1);
        const int integration_channels = integrationData.dimension(2);
        const int md_baselines = constants.nbaselines; //metadata baselines
        const int polarizations = constants.num_pols;

        //parallel execution per channel
        int baseline = blockDim.x * blockIdx.x + threadIdx.x;
        int channel = blockDim.y * blockIdx.y + threadIdx.y;

        if(baseline < integration_baselines && channel < integration_channels)
        {
            int md_baseline = baseline % md_baselines;

            // loop over baselines
            constexpr double two_pi = 2 * CUDART_PI;
            double shiftFactor = -two_pi * (rotatedUVW[baseline].z - UVW[baseline].z);

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
                auto n = integrationData(polarization, baseline, channel);
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

    __host__ void CudaLeapCalibrator::RotateVisibilities(
        DeviceIntegration& integration,
        DeviceMetaData& metadata)
    {
        const auto& constants = metadata.GetConstants(); 
        assert(constants.channels == integration.GetChannels() && integration.GetChannels() == integration.GetVis().GetDimensionSize(2));
        assert(constants.nbaselines == metadata.GetAvgData().GetRows() && integration.GetBaselines() == integration.GetVis().GetDimensionSize(1));
        assert(constants.num_pols == integration.GetVis().GetDimensionSize(0));

        dim3 blockSize = dim3(128, 8, 1); // block size can be any value where the product is 1024
        dim3 gridSize = dim3(
            (int)ceil((float)integration.GetBaselines() / blockSize.x),
            (int)ceil((float)integration.GetChannels() / blockSize.y),
            1
        );

        auto integrationDataMap = Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 3>>(
            (cuDoubleComplex*)integration.GetVis().Get(),
            (int)integration.GetVis().GetDimensionSize(0), // inferring (const int) causes error
            (int)integration.GetVis().GetDimensionSize(1), // inferring (const int) causes error
            (int)integration.GetVis().GetDimensionSize(2) // inferring (const int) causes error
        );

        auto avgDataMap = Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 2>>(
            (cuDoubleComplex*)metadata.GetAvgData().Get(),
            (int)metadata.GetAvgData().GetRows(), // inferring (const int) causes error
            (int)metadata.GetAvgData().GetCols() // inferring (const int) causes error
        );

        auto rotatedUVWMap = Eigen::Map<Eigen::Matrix<double3, -1, 1>>(
            (double3*)metadata.GetRotatedUVW().Get(),
            metadata.GetRotatedUVW().GetCount()
        );

        auto UVWMap = Eigen::Map<Eigen::Matrix<double3, -1, 1>>(
            (double3*)metadata.GetUVW().Get(),
            metadata.GetUVW().GetCount()
        );

        g_RotateVisibilities<<<gridSize, blockSize>>>(
            constants,
            rotatedUVWMap,
            UVWMap,
            integrationDataMap,
            avgDataMap);
    }

    /**
     * @brief Copies the arg of the 1st column of avgData into phaseAnglesI1
     * 
     * @param I1 
     * @param avgData 
     * @param phaseAnglesI1 
     */
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

    __host__ void CudaLeapCalibrator::AvgDataToPhaseAngles(const device_vector<int>& I1, const device_matrix<std::complex<double>>& avgData, device_vector<double>& phaseAnglesI1)
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

    /**
     * @brief Computes dInt matrix
     * 
     * @param A 
     * @param cal1 
     * @param avgData 
     * @param dInt 
     */
    __global__ void g_CalcDInt(
        const Eigen::Map<const Eigen::MatrixXd> A,
        const Eigen::Map<const Eigen::VectorXd> cal1,
        const Eigen::Map<const Eigen::Matrix<thrust::complex<double>, -1, -1>> avgData,
        Eigen::Map<Eigen::VectorXd> dInt)
    {
        constexpr double two_pi = 2 * CUDART_PI;
        int n = blockDim.x * blockIdx.x + threadIdx.x;

        if(n < A.rows())
        {
            double sum = A.row(n) * cal1;
            dInt.row(n) = (thrust::exp(thrust::complex<double>(0, -two_pi * sum)) * avgData.row(n))
            .unaryExpr([](const thrust::complex<double>& v){ return thrust::arg(v); });
        }
    }

    __host__ void CudaLeapCalibrator::CalcDInt(
        const device_matrix<double>& A,
        const device_vector<double>& cal1,
        const device_matrix<std::complex<double>>& avgData,
        device_matrix<double>& dInt)
    {
        if(A.GetCols() != cal1.GetRows())
        {
            throw invalid_argument_exception("A.cols must equal cal1.rows", "cal1", __FILE__, __LINE__);
        }

        dim3 blockSize = dim3(1024, 1, 1);
        dim3 gridSize = dim3((int)ceil(static_cast<double>(A.GetRows()) / blockSize.x), 1, 1);

        auto AMap = Eigen::Map<const Eigen::MatrixXd>(A.Get(), A.GetRows(), A.GetCols());
        auto cal1Map = Eigen::Map<const Eigen::VectorXd>(cal1.Get(), cal1.GetRows());
        auto avgDataMap = Eigen::Map<const Eigen::Matrix<thrust::complex<double>, -1, -1>>(
            (thrust::complex<double>*)avgData.Get(), avgData.GetRows(), avgData.GetCols());
        auto dIntMap = Eigen::Map<Eigen::VectorXd>(dInt.Get(), dInt.GetRows());
        g_CalcDInt<<<blockSize,gridSize>>>(AMap, cal1Map, avgDataMap, dIntMap);
    }

    /**
     * @brief Copies the first column of dInt into deltaPhaseColumn
     * 
     * @param dInt 
     * @param deltaPhaseColumn 
     */
    __global__ void g_GenerateDeltaPhaseColumn(const Eigen::Map<const Eigen::MatrixXd> dInt, Eigen::Map<Eigen::VectorXd> deltaPhaseColumn)
    {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if(row < dInt.rows())
        {
            deltaPhaseColumn(row) = dInt(row, 0); // 1st pol only
        }
        else if (row < deltaPhaseColumn.rows())
        {
            deltaPhaseColumn(row) = 0; // deltaPhaseColumn is dIntRows+1 where last/extra row = 0
        }
    }

    __host__ void CudaLeapCalibrator::GenerateDeltaPhaseColumn(const device_matrix<double>& dInt, device_vector<double>& deltaPhaseColumn)
    {
        if(dInt.GetRows()+1 != deltaPhaseColumn.GetRows())
        {
            throw invalid_argument_exception("incorrect number of columns", "deltaPhaseColumn", __FILE__, __LINE__);
        }

        dim3 blockSize = dim3(1024, 1, 1);
        dim3 gridSize = dim3((int)ceil(static_cast<double>(deltaPhaseColumn.GetRows()) / blockSize.x), 1, 1);

        auto dIntMap = Eigen::Map<const Eigen::MatrixXd>(dInt.Get(), dInt.GetRows(), dInt.GetCols());
        auto deltaPhaseColumnMap = Eigen::Map<Eigen::VectorXd>(deltaPhaseColumn.Get(), deltaPhaseColumn.GetRows());
        g_GenerateDeltaPhaseColumn<<<blockSize,gridSize>>>(dIntMap, deltaPhaseColumnMap);
    }

} // namespace cuda
} // namespace icrar
