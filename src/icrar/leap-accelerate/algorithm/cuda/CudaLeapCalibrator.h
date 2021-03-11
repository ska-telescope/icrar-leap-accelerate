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
#include <icrar/leap-accelerate/model/cpu/calibration/Calibration.h>

#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

//#define EIGEN_HAS_CXX11 1
//#define EIGEN_VECTORIZE_GPU 1
//#define EIGEN_CUDACC 1
#include <Eigen/Core>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <boost/noncopyable.hpp>
#include <vector>


namespace icrar
{
    class MeasurementSet;
    namespace cpu
    {
        class Integration;
        class IntegrationResult;
        class BeamCalibration;
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
        cusolverDnHandle_t m_cusolverDnContext;

    public:
        CudaLeapCalibrator();
        ~CudaLeapCalibrator() override;
        
        /**
         * @copydoc ILeapCalibrator
         * Calibrates by performing phase rotation for each direction in @p directions
         * by splitting uvws into integration batches per timestep.
         */
        void Calibrate(
            std::function<void(const cpu::Calibration&)> outputCallback,
            const icrar::MeasurementSet& ms,
            const std::vector<SphericalDirection>& directions,
            const Slice& solutionInterval,
            double minimumBaselineThreshold,
            boost::optional<unsigned int> referenceAntenna,
            bool isFileSystemCacheEnabled) override;

        /**
         * @brief Calculates Ad into deviceAd, writes to cache if @p isFileSystemCacheEnabled is true
         * 
         * @param hostA matrix to invert
         * @param deviceA output device memory of A
         * @param hostAd output host memory of Ad (optionally written to)
         * @param deviceAd output device memory of Ad
         * @param isFileSystemCacheEnabled whether to use file caching
         * @param useCuda whether to use cuda solvers
         */
        void CalculateAd(
            const Eigen::Matrix<double, -1, -1>& hostA,
            device_matrix<double>& deviceA,
            Eigen::Matrix<double, -1, -1>& hostAd,
            device_matrix<double>& deviceAd,
            bool isFileSystemCacheEnabled,
            bool useCuda);

        /**
         * @brief Calculates Ad1 into deviceAd1
         * 
         * @param hostA1 matrix to invert
         * @param deviceA1 output device memory of A1
         * @param hostAd1 output host memory of Ad1 (optionally written to)
         * @param deviceAd1 output device memory of Ad1
         */
        void CalculateAd1(
            const Eigen::Matrix<double, -1, -1>& hostA1,
            device_matrix<double>& deviceA1,
            Eigen::Matrix<double, -1, -1>& hostAd1,
            device_matrix<double>& deviceAd1);

        /**
         * Performs only visibilities rotation on the GPU
         */
        void PhaseRotate(
            const cpu::MetaData& hostMetadata,
            DeviceMetaData& deviceMetadata,
            const SphericalDirection& direction,
            std::vector<cuda::DeviceIntegration>& input,
            std::vector<cpu::BeamCalibration>& output_calibrations);

        /**
         * @brief Calculates avgData in metadata
         * 
         * @param integration the input visibilities to integrate
         * @param metadata the metadata container
         */
        __host__ void RotateVisibilities(
            DeviceIntegration& integration,
            DeviceMetaData& metadata);

    private:
        /**
         * @brief Copies the argument of the 1st polarization in avgData to phaseAnglesI1
         * 
         * @param I1 the index vector for unflagged antennas
         * @param avgData the averaged data matrix
         * @param phaseAnglesI1 the output phaseAngles vector
         */
        __host__ void AvgDataToPhaseAngles(
            const device_vector<int>& I1,
            const device_matrix<std::complex<double>>& avgData,
            device_vector<double>& phaseAnglesI1);

        /**
         * @brief Calculates the delta phase matrix
         * 
         * @param A Antenna matrix
         * @param cal1 cal1 matrix
         * @param avgData averaged visibilities
         * @param deltaPhase output deltaPhase matrix 
         */
        __host__ void CalcDeltaPhase(
            const device_matrix<double>& A,
            const device_vector<double>& cal1,
            const device_matrix<std::complex<double>>& avgData,
            device_matrix<double>& deltaPhase);

        /**
         * @brief Copies the first column of deltaPhase into deltaPhaseColumn
         * 
         * @param deltaPhase The delta phase matrix 
         * @param deltaPhaseColumn The output delta phase vector/column
         */
        __host__ void GenerateDeltaPhaseColumn(
            const device_matrix<double>& deltaPhase,
            device_vector<double>& deltaPhaseColumn);
    };
} // namespace cuda
} // namespace icrar
#endif
