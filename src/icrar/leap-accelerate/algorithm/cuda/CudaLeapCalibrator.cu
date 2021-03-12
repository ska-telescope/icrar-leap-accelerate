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

#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/model/cuda/HostMetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/model/cuda/HostIntegration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/math/cuda/math.cuh>
#include <icrar/leap-accelerate/math/cuda/matrix.h>
#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/common/eigen_cache.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include <boost/math/constants/constants.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/thread.hpp>

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
    , m_cusolverDnContext(nullptr)
    {
        int deviceCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if(deviceCount < 1)
        {
            throw icrar::exception("CUDA error: no devices supporting CUDA.", __FILE__, __LINE__);
        }

        checkCudaErrors(cublasCreate(&m_cublasContext));
        checkCudaErrors(cusolverDnCreate(&m_cusolverDnContext));
    }

    CudaLeapCalibrator::~CudaLeapCalibrator()
    {
        checkCudaErrors(cusolverDnDestroy(m_cusolverDnContext));
        checkCudaErrors(cublasDestroy(m_cublasContext));

        // cuda calls may still occur outside of this instance lifetime
        //checkCudaErrors(cudaDeviceReset());
    }

    void CudaLeapCalibrator::Calibrate(
            std::function<void(const cpu::Calibration&)> outputCallback,
            const icrar::MeasurementSet& ms,
            const std::vector<SphericalDirection>& directions,
            const Slice& solutionInterval,
            double minimumBaselineThreshold,
            boost::optional<unsigned int> referenceAntenna,
            const ComputeOptions computeOptions)
    {
        LOG(info) << "Starting Calibration using cuda";

        LOG(info)
        << "stations: " << ms.GetNumStations() << ", "
        << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "flagged baselines: " << ms.GetNumFlaggedBaselines() << ", "
        << "solutionInterval: " << "[" << solutionInterval.GetStart() << "," << solutionInterval.GetInterval() << "," << solutionInterval.GetEnd() << "], "
        << "reference antenna: " << referenceAntenna << ", "
        << "baseline threshold: " << minimumBaselineThreshold << ", "
        << "short baselines: " << ms.GetNumShortBaselines(minimumBaselineThreshold) << ", "
        << "filtered baselines: " << ms.GetNumFilteredBaselines(minimumBaselineThreshold) << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << ms.GetNumTimesteps() << ", "
        << "use intermediate cuda buffer: " << computeOptions.useIntermediateBuffer; 

        profiling::timer calibration_timer;

        auto output_calibrations = std::vector<cpu::Calibration>();
        auto input_queue = std::vector<cuda::DeviceIntegration>();

        size_t timesteps = ms.GetNumTimesteps();
        Range validatedSolutionInterval = solutionInterval.Evaluate(timesteps);
        std::vector<double> epochs = ms.GetEpochs();

        profiling::timer metadata_read_timer;
        auto metadata = icrar::cuda::HostMetaData(
            ms,
            referenceAntenna,
            minimumBaselineThreshold,
            false,
            computeOptions.isFileSystemCacheEnabled);
        
        auto deviceA = device_matrix<double>(0, 0, nullptr);
        auto deviceAd = device_matrix<double>(0, 0, nullptr);
        CalculateAd(metadata.GetA(), deviceA, metadata.GetAd(), deviceAd, computeOptions.isFileSystemCacheEnabled, computeOptions.useCusolver);

        auto deviceA1 = device_matrix<double>(0, 0, nullptr);
        auto deviceAd1 = device_matrix<double>(0, 0, nullptr);
        CalculateAd1(metadata.GetA1(), deviceA1, metadata.GetAd1(), deviceAd1);

        auto constantBuffer = std::make_shared<ConstantBuffer>(
            metadata.GetConstants(),
            std::move(deviceA),
            device_vector<int>(metadata.GetI()),
            std::move(deviceAd),
            std::move(deviceA1),
            device_vector<int>(metadata.GetI1()),
            std::move(deviceAd1)
        );

        auto solutionIntervalBuffer = std::make_shared<SolutionIntervalBuffer>(metadata.GetConstants().nbaselines * validatedSolutionInterval.GetInterval());
        auto directionBuffer = std::make_shared<DirectionBuffer>(
                metadata.GetConstants().nbaselines * validatedSolutionInterval.GetInterval(),
                metadata.GetAvgData().rows(),
                metadata.GetAvgData().cols());
        auto deviceMetadata = DeviceMetaData(constantBuffer, solutionIntervalBuffer, directionBuffer);
        LOG(info) << "Metadata Constants loaded in " << metadata_read_timer;

        size_t solutions = validatedSolutionInterval.GetSize();
        constexpr unsigned int integrationNumber = 0;
        for(int solution = 0; solution < solutions; solution++)
        {
            profiling::timer solution_timer;
            output_calibrations.emplace_back(
                epochs[solution * validatedSolutionInterval.GetInterval()],
                epochs[(solution+1) * validatedSolutionInterval.GetInterval() - 1]);
            input_queue.clear();

            // Flooring to remove incomplete measurements
            int integrations = ms.GetNumTimesteps();
            if(integrations == 0)
            {
                std::stringstream ss;
                ss << "invalid number of rows, expected >" << ms.GetNumBaselines() << ", got " << ms.GetNumRows();
                throw icrar::file_exception(ms.GetFilepath().get_value_or("unknown"), ss.str(), __FILE__, __LINE__);
            }
        
            profiling::timer integration_read_timer;
            auto integration = cuda::HostIntegration(
                integrationNumber,
                ms,
                solution * validatedSolutionInterval.GetInterval() * ms.GetNumBaselines(),
                ms.GetNumChannels(),
                validatedSolutionInterval.GetInterval() * ms.GetNumBaselines(),
                ms.GetNumPols());
            LOG(info) << "Read integration data in " << integration_read_timer;

            LOG(info) << "Loading Metadata UVW";
            solutionIntervalBuffer->SetUVW(integration.GetUVW());
            LOG(info) << "Cuda metadata loaded";

            boost::optional<DeviceIntegration> deviceIntegration;
            if(computeOptions.useIntermediateBuffer)
            {
                LOG(info) << "Copying integration to intermediate buffer on device";
                deviceIntegration = DeviceIntegration(integration);
            }

            size_t free;
            size_t total;
            checkCudaErrors(cudaMemGetInfo(&free, &total));
            LOG(trace) << "free device memory: " << memory_amount(free) << "/" << memory_amount(total);

            // Emplace a single zero'd tensor
            input_queue.emplace_back(0, integration.GetVis().dimensions());

            profiling::timer phase_rotate_timer;
            for(int i = 0; i < directions.size(); ++i)
            {
                LOG(info) << "Processing direction " << i;
                LOG(info) << "Setting Metadata Direction";
                
                directionBuffer->SetDirection(directions[i]);
                directionBuffer->SetDD(metadata.GenerateDDMatrix(directions[i]));
                directionBuffer->GetAvgData().SetZeroAsync();

                if(computeOptions.useIntermediateBuffer)
                {
                    input_queue[0].Set(deviceIntegration.get());
                }
                else
                {
                    LOG(info) << "Sending integration to device";
                    input_queue[0].Set(integration);
                }

                LOG(info) << "PhaseRotate";
                PhaseRotate(
                    metadata,
                    deviceMetadata,
                    directions[i],
                    input_queue,
                    output_calibrations[solution].GetBeamCalibrations());
            }
            LOG(info) << "Performed PhaseRotate in " << phase_rotate_timer;
            LOG(info) << "Calculated solution in " << solution_timer;
            outputCallback(output_calibrations[solution]);
        }
        LOG(info) << "Finished calibration in " << calibration_timer;
    }

    void CudaLeapCalibrator::CalculateAd(
        const Eigen::Matrix<double, -1, -1>& hostA,
        device_matrix<double>& deviceA,
        Eigen::Matrix<double, -1, -1>& hostAd,
        device_matrix<double>& deviceAd,
        bool isFileSystemCacheEnabled,
        bool useCuda)
    {
        if(hostA.rows() <= hostA.cols())
        {
            useCuda = false;
        }
        if(useCuda)
        {
            // Compute Ad using cuda
            if(isFileSystemCacheEnabled)
            {
                auto invertA = [&](const Eigen::MatrixXd& a)
                {
                    LOG(info) << "Inverting PhaseMatrix A with cuda (" << a.rows() << ":" << a.cols() << ")";
                    deviceA = device_matrix<double>(a);
                    deviceAd = cuda::PseudoInverse(m_cusolverDnContext, m_cublasContext, deviceA, JobType::S);
                    // Write to host to update disk cache
                    return deviceAd.ToHost();
                };

                // Load cache into hostAd then deviceAd,
                // or load hostA into deviceA, compute deviceAd then load into hostAd
                ProcessCache<Eigen::MatrixXd, Eigen::MatrixXd>(
                    matrix_hash<Eigen::MatrixXd>()(hostA),
                    hostA, hostAd,
                    "A.hash", "Ad.cache",
                    invertA);
            }
            else
            {
                // Compute deviceA then deviceAd and skip writing to hostAd and diskAd
                LOG(info) << "Inverting PhaseMatrix A with cuda (" << hostA.rows() << ":" << hostA.cols() << ")";
                deviceA = device_matrix<double>(hostA);
                deviceAd = cuda::PseudoInverse(m_cusolverDnContext, m_cublasContext, deviceA, JobType::S);
            }
        }
        else
        {
            //Compute Ad into host
            auto invertA = [](const Eigen::MatrixXd& a)
            {
                LOG(info) << "Inverting PhaseMatrix A with cpu (" << a.rows() << ":" << a.cols() << ")";
                return icrar::cpu::PseudoInverse(a);
            };

            if(isFileSystemCacheEnabled)
            {
                ProcessCache<Eigen::MatrixXd, Eigen::MatrixXd>(
                    matrix_hash<Eigen::MatrixXd>()(hostA),
                    hostA, hostAd,
                    "A.hash", "Ad.cache",
                    invertA);
            }
            else
            {
                hostAd = invertA(hostA);
            }

            deviceAd = device_matrix<double>(hostAd);
            deviceA = device_matrix<double>(hostA);
        }
    }

    void CudaLeapCalibrator::CalculateAd1(
        const Eigen::Matrix<double, -1, -1>& hostA1,
        device_matrix<double>& deviceA1,
        Eigen::Matrix<double, -1, -1>& hostAd1,
        device_matrix<double>& deviceAd1)
    {
        // This matrix is not always m > n, compute on cpu until cuda supports this
        deviceA1 = device_matrix<double>(hostA1);
        hostAd1 = cpu::PseudoInverse(hostA1);
        deviceAd1 = device_matrix<double>(hostAd1);
    }

    void CudaLeapCalibrator::PhaseRotate(
        const cpu::MetaData& metadata,
        DeviceMetaData& deviceMetadata,
        const SphericalDirection& direction,
        std::vector<cuda::DeviceIntegration>& input,
        std::vector<cpu::BeamCalibration>& output_calibrations)
    {
        for(DeviceIntegration& integration : input)
        {
            LOG(info) << "Rotating integration " << integration.GetIntegrationNumber();
            RotateVisibilities(integration, deviceMetadata);
        }

        LOG(info) << "Calibrating in cuda";
        auto devicePhaseAnglesI1 = device_vector<double>(metadata.GetI1().rows() + 1);
        auto deviceCal1 = device_vector<double>(metadata.GetA1().cols());
        auto devicedeltaPhase = device_matrix<double>(metadata.GetI().size(), metadata.GetAvgData().cols());
        auto deviceDeltaPhaseColumn = device_vector<double>(metadata.GetI().size() + 1);
        auto cal1 = Eigen::VectorXd(metadata.GetA1().cols());

        AvgDataToPhaseAngles(deviceMetadata.GetConstantBuffer().GetI1(), deviceMetadata.GetAvgData(), devicePhaseAnglesI1);
        LOG(info) << "devicePhaseAnglesI1";
        cuda::multiply(m_cublasContext, deviceMetadata.GetConstantBuffer().GetAd1(), devicePhaseAnglesI1, deviceCal1);
        CalcDeltaPhase(deviceMetadata.GetConstantBuffer().GetA(), deviceCal1, deviceMetadata.GetAvgData(), devicedeltaPhase);
        GenerateDeltaPhaseColumn(devicedeltaPhase, deviceDeltaPhaseColumn);
        LOG(info) << "deviceDeltaPhaseColumn";
        cuda::multiply_add<double>(m_cublasContext, deviceMetadata.GetConstantBuffer().GetAd(), deviceDeltaPhaseColumn, deviceCal1);
        deviceCal1.ToHost(cal1);
        output_calibrations.emplace_back(direction, cal1);
    }

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
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 2>> avgData)
    {
        const int integration_baselines = integrationData.dimension(1);
        const int integration_channels = integrationData.dimension(2);
        const int md_baselines = constants.nbaselines; //metadata baselines
        const int polarizations = constants.num_pols; //TODO

        //parallel execution per channel
        int baseline = blockDim.x * blockIdx.x + threadIdx.x; //baseline amongst all time smeared baselines
        int channel = blockDim.y * blockIdx.y + threadIdx.y;

        if(baseline < integration_baselines && channel < integration_channels)
        {
            int md_baseline = baseline % md_baselines; //baseline within

            // loop over baselines
            constexpr double two_pi = 2 * CUDART_PI;

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

        auto UVWMap = Eigen::Map<Eigen::Matrix<double, 3, -1>>(
            (double*)metadata.GetUVW().Get(),
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

    __host__ void CudaLeapCalibrator::CalcDeltaPhase(
        const device_matrix<double>& A,
        const device_vector<double>& cal1,
        const device_matrix<std::complex<double>>& avgData,
        device_matrix<double>& deltaPhase)
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
        auto deltaPhaseMap = Eigen::Map<Eigen::VectorXd>(deltaPhase.Get(), deltaPhase.GetRows());
        g_CalcDeltaPhase<<<blockSize,gridSize>>>(AMap, cal1Map, avgDataMap, deltaPhaseMap);
    }

    __global__ void g_GenerateDeltaPhaseColumn(const Eigen::Map<const Eigen::MatrixXd> deltaPhase, Eigen::Map<Eigen::VectorXd> deltaPhaseColumn)
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

    __host__ void CudaLeapCalibrator::GenerateDeltaPhaseColumn(const device_matrix<double>& deltaPhase, device_vector<double>& deltaPhaseColumn)
    {
        if(deltaPhase.GetRows()+1 != deltaPhaseColumn.GetRows())
        {
            throw invalid_argument_exception("incorrect number of columns", "deltaPhaseColumn", __FILE__, __LINE__);
        }

        dim3 blockSize = dim3(1024, 1, 1);
        dim3 gridSize = dim3((int)ceil(static_cast<double>(deltaPhaseColumn.GetRows()) / blockSize.x), 1, 1);

        auto deltaPhaseMap = Eigen::Map<const Eigen::MatrixXd>(deltaPhase.Get(), deltaPhase.GetRows(), deltaPhase.GetCols());
        auto deltaPhaseColumnMap = Eigen::Map<Eigen::VectorXd>(deltaPhaseColumn.Get(), deltaPhaseColumn.GetRows());
        g_GenerateDeltaPhaseColumn<<<blockSize,gridSize>>>(deltaPhaseMap, deltaPhaseColumnMap);
    }

} // namespace cuda
} // namespace icrar
