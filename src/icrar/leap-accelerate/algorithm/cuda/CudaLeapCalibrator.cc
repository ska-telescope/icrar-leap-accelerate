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

#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/common/eigen_stringutils.h>

#include <icrar/leap-accelerate/algorithm/cuda/CudaComputeOptions.h>
#include <icrar/leap-accelerate/algorithm/cuda/kernel/RotateVisibilitiesKernel.h>
#include <icrar/leap-accelerate/algorithm/cuda/kernel/PolarizationsToPhaseAnglesKernel.h>
#include <icrar/leap-accelerate/algorithm/cuda/kernel/ComputePhaseDeltaKernel.h>
#include <icrar/leap-accelerate/algorithm/cuda/kernel/CopyPhaseDeltaKernel.h>

#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/model/cuda/HostMetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/model/cuda/HostIntegration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/math/cuda/matrix.h>
#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/common/eigen_cache.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/cuda/device_matrix.h>
#include <icrar/leap-accelerate/cuda/device_vector.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <boost/optional/optional_io.hpp>

#include <string>

using namespace boost::math::constants;

namespace icrar
{
namespace cuda
{
    //__global__ void g_checkKernelSM() { }

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
        //g_checkKernelSM<<<1,1>>>();
        // cudaError_t smError = cudaGetLastError();
        // if(smError != cudaError_t::cudaSuccess)
        // {   
        //     CUdevice device;
        //     checkCudaErrors(cuDeviceGet(&device, 0));
        //     int major, minor;
        //     checkCudaErrors(cuDeviceGetAttribute(&major, CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        //     checkCudaErrors(cuDeviceGetAttribute(&minor, CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        //     LOG(warning) << "CUDA error: No suitable kernel found, hardware sm compatibility is sm_" << major << minor;
        // }
        // checkCudaErrors(smError);

        checkCudaErrors(cublasCreate(&m_cublasContext));
        checkCudaErrors(cusolverDnCreate(&m_cusolverDnContext));
    }

    CudaLeapCalibrator::~CudaLeapCalibrator()
    {
        //checkCudaErrors(cudaGetLastError());
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
            const ComputeOptionsDTO& computeOptions)
    {
        auto cudaComputeOptions = CudaComputeOptions(computeOptions, ms);

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
        << "use filesystem cache: " << cudaComputeOptions.isFileSystemCacheEnabled << ", "
        << "use intermediate cuda buffer: " << cudaComputeOptions.useIntermediateBuffer << ", "
        << "use cusolver: " << cudaComputeOptions.useCusolver;

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
            false);

        device_matrix<double> deviceA, deviceAd;
        
        CalculateAd(metadata.GetA(), deviceA, metadata.GetAd(), deviceAd, cudaComputeOptions.isFileSystemCacheEnabled, cudaComputeOptions.useCusolver);
        cudaHostRegister(metadata.GetAd().data(), metadata.GetAd().size() * sizeof(decltype(*metadata.GetAd().data())), cudaHostRegisterPortable);

        device_matrix<double> deviceA1, deviceAd1;
        CalculateAd1(metadata.GetA1(), deviceA1, metadata.GetAd1(), deviceAd1);
        cudaHostRegister(metadata.GetAd1().data(), metadata.GetAd1().size() * sizeof(decltype(*metadata.GetAd1().data())), cudaHostRegisterPortable);

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
                metadata.GetAvgData().rows(),
                metadata.GetAvgData().cols());
        auto deviceMetadata = DeviceMetaData(constantBuffer, solutionIntervalBuffer, directionBuffer);
        LOG(info) << "Metadata loaded in " << metadata_read_timer;

        size_t solutions = validatedSolutionInterval.GetSize();
        constexpr unsigned int integrationNumber = 0;
        for(size_t solution = 0; solution < solutions; solution++)
        {
            profiling::timer solution_timer;
            output_calibrations.emplace_back(
                epochs[solution * validatedSolutionInterval.GetInterval()],
                epochs[(solution+1) * validatedSolutionInterval.GetInterval() - 1]);
            input_queue.clear();

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
            if(cudaComputeOptions.useIntermediateBuffer)
            {
                LOG(info) << "Copying integration to intermediate buffer on device";
                deviceIntegration = DeviceIntegration(integration);
            }

            // Emplace a single zero'd tensor
            input_queue.emplace_back(0, integration.GetVis().dimensions());

            profiling::timer phase_rotate_timer;
            for(size_t i = 0; i < directions.size(); ++i)
            {
                LOG(info) << "Processing direction " << i;
                LOG(info) << "Setting Metadata Direction";
                
                directionBuffer->SetDirection(directions[i]);
                directionBuffer->SetDD(metadata.GenerateDDMatrix(directions[i]));
                directionBuffer->GetAvgData().SetZeroAsync();

                if(cudaComputeOptions.useIntermediateBuffer)
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

    inline bool IsDegenerate(const Eigen::MatrixXd& identity, double tolerance)
    {
        return icrar::cpu::near(identity, Eigen::MatrixXd::Identity(identity.rows(), identity.cols()), tolerance);
    }

    inline bool IsDiagonal(const Eigen::MatrixXd& diagonal, double tolerance)
    {
        // IsApprox uses frobenius L2 norm which can't be compared to a zero matrix
        //if(!diagonal.isApprox(Eigen::MatrixXd::Zero(diagonal.cols(), diagonal.cols()), tolerance))
        for(std::int64_t row = 0; row < diagonal.rows(); row++)
        {
            for(std::int64_t col = 0; col < diagonal.cols(); col++)
            {
                if(row != col && std::abs(diagonal(row, col)) > tolerance)
                {
                    LOG(trace) << "matrix differs at " << row << ":" << col;
                    return false;
                }
            }
        }
        return true;
    }

    void CudaLeapCalibrator::CalculateAd(
        const Eigen::MatrixXd& hostA,
        device_matrix<double>& deviceA,
        Eigen::MatrixXd& hostAd,
        device_matrix<double>& deviceAd,
        bool isFileSystemCacheEnabled,
        bool useCusolver)
    {
        if(hostA.rows() <= hostA.cols())
        {
            useCusolver = false;
        }
        if(useCusolver)
        {
            auto invertA = [&](const Eigen::MatrixXd& a)
            {
                LOG(info) << "Inverting PhaseMatrix A with cuda (" << a.rows() << ":" << a.cols() << ")";
                deviceA = device_matrix<double>(a);
                cudaDeviceSynchronize();
                deviceAd = cuda::pseudo_inverse(m_cusolverDnContext, m_cublasContext, deviceA, JobType::S);
                cudaDeviceSynchronize();
                // Write to host to update disk cache
                return deviceAd.ToHost();
            };

            // Compute Ad using Cusolver
            if(isFileSystemCacheEnabled)
            {
                // Load cache into hostAd then deviceAd,
                // or load hostA into deviceA, compute deviceAd then load into hostAd
                ProcessCache<Eigen::MatrixXd, Eigen::MatrixXd>(
                    matrix_hash<Eigen::MatrixXd>()(hostA),
                    hostA, hostAd,
                    "A.hash", "Ad.cache",
                    invertA);

                deviceAd = device_matrix<double>(hostAd);
                deviceA = device_matrix<double>(hostA);
                if(IsDegenerate(hostAd * hostA, 1e-5))
                {
                    LOG(warning) <<  "Ad is degenerate";
                }
            }
            else
            {
                // TODO(calgray): in some cases pseudo-inversion fails to produce a diagonal matrix
                // due to a cusolver bug. In testing this happens rarely (about 10% chance) and can
                // temporarily worked around by running again.
                auto CheckDiagonal = [&]()
                {
                    bool isDiagonal = IsDiagonal(hostAd * hostA, 1e-10);
                    if(!isDiagonal)
                    {
                        LOG(warning) << "Ad is non-diagonal";
                    }
                    return isDiagonal;
                };
                do
                {
                    hostAd = invertA(hostA);
                    cudaDeviceSynchronize();
                    deviceA = device_matrix<double>(hostA);
                    cudaDeviceSynchronize();
                }
                while(!CheckDiagonal());
            }
        }
        else
        {
            //Compute Ad on host
            auto invertA = [](const Eigen::MatrixXd& a)
            {
                LOG(info) << "Inverting PhaseMatrix A with cpu (" << a.rows() << ":" << a.cols() << ")";
                return icrar::cpu::pseudo_inverse(a);
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
            if(IsDegenerate(hostAd * hostA, 1e-5))
            {
                LOG(warning) << "Ad is degenerate";
            }
        }
    }

    void CudaLeapCalibrator::CalculateAd1(
        const Eigen::MatrixXd& hostA1,
        device_matrix<double>& deviceA1,
        Eigen::MatrixXd& hostAd1,
        device_matrix<double>& deviceAd1)
    {
        // This matrix is not always m > n, compute on cpu until cuda supports this
        deviceA1 = device_matrix<double>(hostA1);
        hostAd1 = cpu::pseudo_inverse(hostA1);
        deviceAd1 = device_matrix<double>(hostAd1);
        if(IsDegenerate(hostAd1 * hostA1, 1e-5))
        {
            LOG(warning) << "Ad1 is degenerate";
        }
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
        cuda::multiply(m_cublasContext, deviceMetadata.GetConstantBuffer().GetAd1(), devicePhaseAnglesI1, deviceCal1);
        CalcDeltaPhase(deviceMetadata.GetConstantBuffer().GetA(), deviceCal1, deviceMetadata.GetAvgData(), devicedeltaPhase);
        GenerateDeltaPhaseColumn(devicedeltaPhase, deviceDeltaPhaseColumn);
        cuda::multiply_add<double>(m_cublasContext, deviceMetadata.GetConstantBuffer().GetAd(), deviceDeltaPhaseColumn, deviceCal1);
        deviceCal1.ToHost(cal1);
        output_calibrations.emplace_back(direction, cal1);
    }
} // namespace cuda
} // namespace icrar
