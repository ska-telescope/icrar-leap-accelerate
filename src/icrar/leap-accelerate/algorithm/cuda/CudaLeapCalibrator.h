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

#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/model/cpu/CalibrateResult.h>

#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

//#define EIGEN_HAS_CXX11 1
//#define EIGEN_VECTORIZE_GPU 1
//#define EIGEN_CUDACC 1
#include <Eigen/Core>

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
    /**
     * @brief LEAP calibration object implemented using CUDA
     * 
     */
    class CudaLeapCalibrator : public ILeapCalibrator
    {
        cublasHandle_t m_cublasContext;

    public:
        CudaLeapCalibrator();
        ~CudaLeapCalibrator() override;

        virtual cpu::CalibrateResult Calibrate(
            const icrar::MeasurementSet& ms,
            const std::vector<SphericalDirection>& directions,
            const Range& solutionInterval,
            double minimumBaselineThreshold,
            boost::optional<unsigned int> referenceAntenna,
            bool isFileSystemCacheEnabled) override;

        /**
         * Performs only visibilities rotation on the GPU
         */
        void PhaseRotate(
            cpu::MetaData& hostMetadata,
            DeviceMetaData& deviceMetadata,
            const SphericalDirection& direction,
            std::vector<cuda::DeviceIntegration>& input,
            std::vector<cpu::IntegrationResult>& output_integrations,
            std::vector<cpu::CalibrationResult>& output_calibrations);

        /**
         * @brief Rotates oldUVW by dd into UVW
         * 
         * @param dd 
         * @param oldUVW 
         * @param UVW 
         * @return __host__ 
         */
        __host__ void RotateUVW(
            Eigen::Matrix3d dd,
            const device_vector<icrar::MVuvw>& oldUVW,
            device_vector<icrar::MVuvw>& UVW);

        /**
         * @brief Calculates metadata.avgData
         * 
         * @param integration 
         * @param metadata 
         * @return __host__ 
         */
        __host__ void RotateVisibilities(
            DeviceIntegration& integration,
            DeviceMetaData& metadata);

    private:
        /**
         * @brief Copies the arg of the 1st column of avgData into phaseAnglesI1
         * 
         * @param I1 
         * @param avgData 
         * @param phaseAnglesI1 
         * @return __host__ 
         */
        __host__ void AvgDataToPhaseAngles(
            const device_vector<int>& I1,
            const device_matrix<std::complex<double>>& avgData,
            device_vector<double>& phaseAnglesI1);

        /**
         * @brief Calculates dInt
         * 
         * @param A 
         * @param cal1 
         * @param avgData 
         * @param dInt 
         * @return __host__ 
         */
        __host__ void CalcDInt(
            const device_matrix<double>& A,
            const device_vector<double>& cal1,
            const device_matrix<std::complex<double>>& avgData,
            device_matrix<double>& dInt);

        /**
         * @brief Copies the first column of dInt into deltaPhaseColumn
         * 
         * @param dInt 
         * @param deltaPhaseColumn 
         * @return __host__ 
         */
        __host__ void GenerateDeltaPhaseColumn(
            const device_matrix<double>& dInt,
            device_vector<double>& deltaPhaseColumn);
    };
} // namespace cuda
} // namespace icrar
#endif
