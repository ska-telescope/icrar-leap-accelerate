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
#if CUDA_ENABLED
#include <icrar/leap-accelerate/algorithm/ComputeOptionsDTO.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/memory/ioutils.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>

#include <cuda_runtime.h>

#include <boost/optional.hpp>

namespace icrar
{
    class Range;
    
    /**
     * @brief Validates and determines the best compute features for calibration depending on measurement set data and hardware
     * configuration.
     */
    class CudaComputeOptions
    {
    public:
        bool isFileSystemCacheEnabled; ///< Enables caching of expensive calculations to the filesystem
        bool useIntermediateBuffer; ///< enables an intermediate buffer containing unrotated visibilities to improve per direction performance
        bool useCusolver; ///< Uses cusolver for Ad calculation

        /**
         * @brief Determines ideal calibration compute options for a given measurementSet 
         * 
         * @param computeOptions 
         * @param ms 
         */
        CudaComputeOptions(const ComputeOptionsDTO& computeOptions, const icrar::MeasurementSet& ms, const Range& solutionRange);
    };
} // namespace icrar

#endif // CUDA_ENABLED