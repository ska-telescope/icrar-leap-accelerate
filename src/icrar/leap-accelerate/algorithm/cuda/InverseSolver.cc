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
    void SolveInverse(const Eigen::Matrix<T, -1, -1>& matrix, bool enableCache, bool enableCuda)
    {

    }

        auto metadata = icrar::cuda::HostMetaData(
            ms,
            referenceAntenna,
            minimumBaselineThreshold,
            !highGpuMemory,
            isFileSystemCacheEnabled);
        
        auto Ad = device_matrix<double>(0, 0, nullptr);
#ifdef HIGH_GPU_MEMORY
        // Compute Ad using cuda
        if(isFileSystemCacheEnabled)
        {
            auto invertA = [](const Eigen::MatrixXd& a)
            {
                LOG(info) << "Inverting PhaseMatrix A " << a.rows() << ":" << a.cols();
                Ad = cuda::PseudoInverse(cusolverHandle, cublasHandle, metadata.GetA(), JobType::S));
                auto hostAd = Eigen::MatrixXd(deviceAd.GetRows(), deviceAd.GetCols());
                return Ad.ToHost();
            };

            ProcessCache<Eigen::MatrixXd, Eigen::MatrixXd>(
                matrix_hash<Eigen::MatrixXd>()(metadata.GetA()),
                metadata.GetA(), metadata.GetAd(),
                "A.hash", "Ad.cache",
                invertA);
        }
        else
        {
            // Skip writing cache to host and disk
            LOG(info) << "Inverting PhaseMatrix A " << a.rows() << ":" << a.cols();
            Ad = cuda::PseudoInverse(cusolverHandle, cublasHandle, metadata.GetA(), JobType::S));
        }
#else
        //Compute Ad via host
        auto invertA = [](const Eigen::MatrixXd& a)
        {
            LOG(info) << "Inverting PhaseMatrix A " << a.rows() << ":" << a.cols();
            return icrar::cpu::PseudoInverse(a);
        };

        if(isFileSystemCacheEnabled)
        {
            ProcessCache<Eigen::MatrixXd, Eigen::MatrixXd>(
                matrix_hash<Eigen::MatrixXd>()(metadata.GetA()),
                metadata.GetA(), metadata.GetAd(),
                "A.hash", "Ad.cache",
                invertA);
        }
        else
        {
            Ad = device_matrix<double>(invertA(metadata.GetA()));
        }
#endif


} // namespace cuda
} // namespace icrar
