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

#pragma once

#ifdef CUDA_ENABLED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <icrar/leap-accelerate/common/MVDirection.h>
#include <icrar/leap-accelerate/model/cpu/CalibrateResult.h>

#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>

//#define EIGEN_HAS_CXX11 1
//#define EIGEN_VECTORIZE_GPU 1
//#define EIGEN_CUDACC 1
#include <Eigen/Core>

#include <cublasLt.h>
#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

#include <boost/noncopyable.hpp>
#include <vector>


namespace icrar
{
    class MeasurementSet;
    namespace cpu
    {
        class Integration;
        class IntegrationResult;
        class CalibrationResult;
        class MetaData;
    }
    namespace cuda
    {
        class DeviceMetaData;
        class DeviceIntegration;
    }
}

namespace icrar
{
namespace cuda
{
    class CudaLeapCalibrator : public ILeapCalibrator
    {
        cublasHandle_t m_cublasContext;
        cublasLtHandle_t m_cublasLtContext;

    public:
        CudaLeapCalibrator();
        ~CudaLeapCalibrator() override;

        virtual cpu::CalibrateResult Calibrate(
            const icrar::MeasurementSet& ms,
            const std::vector<MVDirection>& directions,
            double minimumBaselineThreshold,
            bool isFileSystemCacheEnabled) override;

        /**
         * Performs only visibilities rotation on the GPU
         */
        void PhaseRotate(
            cpu::MetaData& hostMetadata,
            DeviceMetaData& deviceMetadata,
            const MVDirection& direction,
            std::vector<cuda::DeviceIntegration>& input,
            std::vector<cpu::IntegrationResult>& output_integrations,
            std::vector<cpu::CalibrationResult>& output_calibrations);

        __host__ void RotateVisibilities(
            DeviceIntegration& integration,
            DeviceMetaData& metadata);

        __host__ void PhaseAngleCalibration(
            const DeviceMetaData& deviceMetadata,
            size_t I1Length,
            size_t ILength,
            size_t Ad1Rows,
            size_t AvgDataCols,
            device_vector<double>& calibrationResult);

        __host__ void RotateUVW(
            Eigen::Matrix3d dd,
            const device_vector<icrar::MVuvw>& oldUVW,
            device_vector<icrar::MVuvw>& UVW);
    };
} // namespace cuda
} // namespace icrar
#endif
