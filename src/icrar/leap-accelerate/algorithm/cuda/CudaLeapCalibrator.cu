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

#include <icrar/leap-accelerate/model/cpu/Integration.h>
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
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuComplex.h>
#include <cublasLt.h>
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
    {
        int deviceCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if(deviceCount < 1)
        {
            throw icrar::exception("CUDA error: no devices supporting CUDA.", __FILE__, __LINE__);
        }

        checkCudaErrors(cublasCreate(&m_cublasContext));
        checkCudaErrors(cublasLtCreate(&m_cublasLtContext));
    }

    CudaLeapCalibrator::~CudaLeapCalibrator()
    {
        checkCudaErrors(cublasLtDestroy(m_cublasLtContext));
        checkCudaErrors(cublasDestroy(m_cublasContext));

        // cuda calls may still occur outside of this instance
        //checkCudaErrors(cudaDeviceReset());
    }

    cpu::CalibrateResult CudaLeapCalibrator::Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<icrar::MVDirection>& directions,
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

        if(GetCudaDeviceCount() == 0)
        {
            throw std::runtime_error("Could not find CUDA device");
        }

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

        auto integration = cpu::Integration(0, ms, 0, ms.GetNumChannels(), ms.GetNumRows(), ms.GetNumPols());
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

        auto solutionIntervalBuffer = std::make_shared<SolutionIntervalBuffer>(metadata.GetOldUVW());
        
        auto directionBuffer = std::make_shared<DirectionBuffer>(
            metadata.GetDirection(),
            metadata.GetDD(),
            metadata.GetOldUVW().size(),
            metadata.GetAvgData().rows(),
            metadata.GetAvgData().cols());

        auto deviceMetadata = icrar::cuda::DeviceMetaData(constantBuffer, solutionIntervalBuffer, directionBuffer);

        // Emplace a single empty tensor
        input_queue.emplace_back(0, integration.GetVis().dimensions());
        LOG(info) << "Metadata loaded in " << metadata_read_timer;

        profiling::timer phase_rotate_timer;
        for(int i = 0; i < directions.size(); ++i)
        {
            LOG(info) << "Processing direction " << i;
            LOG(info) << "Setting Metadata";
            metadata.SetDirection(directions[i]);

            directionBuffer->SetDirection(metadata.GetDirection());
            directionBuffer->SetDD(metadata.GetDD());
            directionBuffer->GetAvgData().SetZeroSync();

            input_queue[0].SetData(integration);

            LOG(info) << "PhaseRotate";
            RotateUVW(
                deviceMetadata.GetDD(),
                solutionIntervalBuffer->GetOldUVW(),
                directionBuffer->GetUVW());

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
        return std::make_pair(std::move(output_integrations), std::move(output_calibrations));
    }

    __global__ void g_CalcDInt(
        const double* pA, int ARows, int ACols,
        const double* pCal1, int Cal1Rows,
        const thrust::complex<double>* pAvgData, int avgDataRows, int avgDataCols,
        double* pDInt, int dIntRows)
    {
        using MatrixXcd = Eigen::Matrix<thrust::complex<double>, -1, -1>;
        auto A = Eigen::Map<const Eigen::MatrixXd>(pA, ARows, ACols);
        auto cal1 = Eigen::Map<const Eigen::VectorXd>(pCal1, Cal1Rows);
        auto avgData = Eigen::Map<const MatrixXcd>(pAvgData, avgDataRows, avgDataCols);
        auto dInt = Eigen::Map<Eigen::VectorXd>(pDInt, dIntRows);

        constexpr double two_pi = 2 * CUDART_PI;
        int n = blockDim.x * blockIdx.x + threadIdx.x;

        if(n < ARows)
        {
            double sum = A.row(n) * cal1; //TODO: use a sum ColumnVector from a matmul kernel
            dInt.row(n) = (thrust::exp(thrust::complex<double>(0, -sum * two_pi)) * avgData.row(n))
            .unaryExpr([](const thrust::complex<double>& v){ return thrust::arg(v); });
        }
    }

    __host__ void CalcDInt(
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
        dim3 gridSize = dim3(
            (int)ceil(static_cast<double>(A.GetRows()) / blockSize.x),
            1,
            1
        );

        g_CalcDInt<<<blockSize,gridSize>>>(
            A.Get(), A.GetRows(), A.GetCols(),
            cal1.Get(), cal1.GetRows(),
            (thrust::complex<double>*)avgData.Get(), avgData.GetRows(), avgData.GetCols(),
            dInt.Get(), dInt.GetRows());
    }
    
    __global__ void g_GenDeltaPhaseColumn(
        const double* pDInt, int dIntRows, int dIntCols,
        double* pDeltaPhaseColumn, int deltaPhaseColumnRows)
    {
        auto dInt = Eigen::Map<const Eigen::MatrixXd>(pDInt, dIntRows, dIntCols);
        auto deltaPhaseColumn = Eigen::Map<Eigen::VectorXd>(pDeltaPhaseColumn, deltaPhaseColumnRows);

        deltaPhaseColumn = dInt.col(0); // 1st pol only
        deltaPhaseColumn(deltaPhaseColumn.size() - 1) = 0; // deltaPhaseColumn is dIntRows+1 where last/extra row = 0
    }

    __host__ void GenDeltaPhaseColumn(const device_matrix<double>& dInt, device_vector<double>& deltaPhaseColumn)
    {
        g_GenDeltaPhaseColumn<<<1,1>>>(dInt.Get(), dInt.GetRows(), dInt.GetCols(), deltaPhaseColumn.Get(), deltaPhaseColumn.GetRows());
    }

    __global__ void g_AvgDataToPhaseAngles(
        const int* pI1, int I1Rows,
        const thrust::complex<double>* pavgData, int avgDataRows, int avgDataCols,
        double* pPhaseAnglesI1, int phaseAnglesI1Rows)
    {
        using MatrixXcd = Eigen::Matrix<thrust::complex<double>, -1, -1>;
        auto I1 = Eigen::Map<const Eigen::VectorXi>(pI1, I1Rows);
        auto avgData = Eigen::Map<const MatrixXcd>(pavgData, avgDataRows, avgDataCols);
        auto phaseAnglesI1 = Eigen::Map<Eigen::VectorXd>(pPhaseAnglesI1, phaseAnglesI1Rows);

        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if(row < I1Rows)
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
        dim3 gridSize = dim3(
            (int)ceil(static_cast<double>(I1.GetRows()) / blockSize.x),
            1,
            1
        );

        g_AvgDataToPhaseAngles<<<blockSize, gridSize>>>(
            I1.Get(), I1.GetRows(),
            (thrust::complex<double>*)avgData.Get(), avgData.GetRows(), avgData.GetCols(),
            phaseAnglesI1.Get(), phaseAnglesI1.GetRows());
    }

    void CudaLeapCalibrator::PhaseRotate(
        cpu::MetaData& metadata,
        DeviceMetaData& deviceMetadata,
        const icrar::MVDirection& direction,
        std::vector<cuda::DeviceIntegration>& input,
        std::vector<cpu::IntegrationResult>& output_integrations,
        std::vector<cpu::CalibrationResult>& output_calibrations)
    {
        for(DeviceIntegration& integration : input)
        {
            LOG(info) << "Rotating integration " << integration.GetIntegrationNumber();
            this->RotateVisibilities(integration, deviceMetadata);
        }

        //CPU Phase Angle Calibration
        LOG(info) << "Calibrating in cuda";
        auto devicePhaseAnglesI1 = device_vector<double>(metadata.GetI1().rows() + 1);
        auto deviceCal1 = device_vector<double>(metadata.GetAd1().rows());
        auto deviceDInt = device_matrix<double>(metadata.GetI().size(), metadata.GetAvgData().cols());
        auto deviceDeltaPhaseColumn = device_vector<double>(metadata.GetI().size() + 1);
        auto cal1 = Eigen::VectorXd(metadata.GetAd1().rows());

        AvgDataToPhaseAngles(deviceMetadata.GetConstantBuffer().GetI1(), deviceMetadata.GetAvgData(), devicePhaseAnglesI1);
        cuda::multiply(m_cublasContext, deviceMetadata.GetConstantBuffer().GetAd1(), devicePhaseAnglesI1, deviceCal1);
        CalcDInt(deviceMetadata.GetConstantBuffer().GetA(), deviceCal1, deviceMetadata.GetAvgData(), deviceDInt);
        GenDeltaPhaseColumn(deviceDInt, deviceDeltaPhaseColumn);
        icrar::cuda::multiply_add<double>(m_cublasContext, deviceMetadata.GetConstantBuffer().GetAd(), deviceDeltaPhaseColumn, deviceCal1);
        deviceCal1.ToHost(cal1);
        output_calibrations.emplace_back(direction, cal1);
    }

    /**
     * @brief Kernel for rotating UVWs
     * 
     * @param dd rotation matrix
     * @param pOldUVW unrotated UVW buffer
     * @param pUVW output UVW buffer
     * @param uvwLength UVW buffer number of elements
     * @return __global__ 
     */
    __global__ void g_RotateUVW(
        Eigen::Matrix3d dd,
        const double* pOldUVW, // TODO: use double3*
        double* pUVW, // TODO: use double3*
        int uvwLength)
    {
        auto oldUVWs = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(pOldUVW, uvwLength, 3);
        auto UVWs = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(pUVW, uvwLength, 3);

        // Compute rows in parallel
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        auto oldUvw = Eigen::RowVector3d(oldUVWs(row, 0), oldUVWs(row, 1), oldUVWs(row, 2));
        Eigen::RowVector3d uvw = oldUvw * dd;
        UVWs(row, 0) = uvw(0);
        UVWs(row, 1) = uvw(1);
        UVWs(row, 2) = uvw(2);
    }

    /**
     * @brief Rotates visibilities in parallel for baselines and channels
     * @note Atomic operator required for writing to @p pAvgData
     */
    __global__ void g_RotateVisibilities(
        cuDoubleComplex* pIntegrationData, int integration_data_dim0, int integration_data_dim1, int integration_data_dim2,
        icrar::cpu::Constants constants,
        Eigen::Matrix3d dd, //TODO(cgray) remove
        double2 direction, //TODO(cgray) remove
        const double3* uvw, int uvwLength,
        const double3* oldUVW, int oldUVWLegth,
        cuDoubleComplex* pAvgData, int avgDataRows, int avgDataCols)
    {
        using Tensor2Xcucd = Eigen::Tensor<cuDoubleComplex, 2>;
        using Tensor3Xcucd = Eigen::Tensor<cuDoubleComplex, 3>;
        
        const int integration_baselines = integration_data_dim1;
        const int integration_channels = integration_data_dim2;
        const int md_baselines = constants.nbaselines; //metadata baselines
        const int polarizations = constants.num_pols;

        //parallel execution per channel
        int baseline = blockDim.x * blockIdx.x + threadIdx.x;
        int channel = blockDim.y * blockIdx.y + threadIdx.y;

        if(baseline < integration_baselines && channel < integration_channels)
        {
            auto integration_data = Eigen::TensorMap<Tensor3Xcucd>(pIntegrationData, integration_data_dim0, integration_data_dim1, integration_data_dim2);
            auto avg_data = Eigen::TensorMap<Tensor2Xcucd>(pAvgData, avgDataRows, avgDataCols);
    
            int md_baseline = baseline % md_baselines;

            // loop over baselines
            constexpr double two_pi = 2 * CUDART_PI;
            double shiftFactor = two_pi * (uvw[baseline].z - oldUVW[baseline].z);

            // loop over channels
            double shiftRad = shiftFactor / constants.GetChannelWavelength(channel);

            cuDoubleComplex exp = cuCexp(make_cuDoubleComplex(0.0, shiftRad));

            for(int polarization = 0; polarization < polarizations; polarization++)
            {
                 integration_data(polarization, baseline, channel) = cuCmul(integration_data(polarization, baseline, channel), exp);
            }

            bool hasNaN = false;
            for(int polarization = 0; polarization < polarizations; polarization++)
            {
                auto n = integration_data(polarization, baseline, channel);
                hasNaN |= isnan(n.x) || isnan(n.y);
            }

            if(!hasNaN)
            {
                for(int polarization = 0; polarization < polarizations; ++polarization)
                {
                    atomicAdd(&avg_data(md_baseline, polarization).x, integration_data(polarization, baseline, channel).x);
                    atomicAdd(&avg_data(md_baseline, polarization).y, integration_data(polarization, baseline, channel).y);
                }
            }
        }
    }

    __host__ void CudaLeapCalibrator::RotateUVW(Eigen::Matrix3d dd, const device_vector<icrar::MVuvw>& oldUVW, device_vector<icrar::MVuvw>& UVW)
    {
        assert(oldUVW.GetCount() != UVW.GetCount());
        dim3 blockSize = dim3(1024, 1, 1);
        dim3 gridSize = dim3((int)ceil((float)oldUVW.GetCount() / blockSize.x), 1, 1);
        g_RotateUVW<<<blockSize, gridSize>>>(dd, oldUVW.Get()->data(), UVW.Get()->data(), oldUVW.GetCount());
    }

    __host__ void CudaLeapCalibrator::RotateVisibilities(
        DeviceIntegration& integration,
        DeviceMetaData& metadata)
    {
        const auto& constants = metadata.GetConstants(); 
        assert(constants.channels == integration.GetChannels() && integration.GetChannels() == integration.GetVis().GetDimensionSize(2));
        assert(constants.nbaselines == metadata.GetAvgData().GetRows() && integration.GetBaselines() == integration.GetVis().GetDimensionSize(1));
        assert(constants.num_pols == integration.GetVis().GetDimensionSize(0));

        // block size can any value where the product is 1024
        dim3 blockSize = dim3(128, 8, 1);
        dim3 gridSize = dim3(
            (int)ceil((float)integration.GetBaselines() / blockSize.x),
            (int)ceil((float)integration.GetChannels() / blockSize.y),
            1
        );

        //TODO: store polar form in advance
        const auto polar_direction = icrar::ToPolar(metadata.GetDirection());
        g_RotateVisibilities<<<gridSize, blockSize>>>(
            (cuDoubleComplex*)integration.GetVis().Get(), integration.GetVis().GetDimensionSize(0), integration.GetVis().GetDimensionSize(1), integration.GetVis().GetDimensionSize(2),
            constants,
            metadata.GetDD(),
            make_double2(polar_direction(0), polar_direction(1)),
            (double3*)metadata.GetUVW().Get(), metadata.GetUVW().GetCount(),
            (double3*)metadata.GetOldUVW().Get(), metadata.GetOldUVW().GetCount(),
            (cuDoubleComplex*)metadata.GetAvgData().Get(), metadata.GetAvgData().GetRows(), metadata.GetAvgData().GetCols());
    }

    __host__ void CudaLeapCalibrator::PhaseAngleCalibration(
        const DeviceMetaData& deviceMetadata,
        size_t I1Length,
        size_t ILength,
        size_t Ad1Rows,
        size_t AvgDataCols,
        device_vector<double>& calibrationResult)
    {

        //TODO(calgray) ...

        //icrar::cuda::mat_mul_add_vector<double>(m_cublasLtContext, deviceMetadata.GetConstantBuffer().GetAd(), deltaPhaseColumn, cal1, calibrationResult);
    }

} // namespace cuda
} // namespace icrar
