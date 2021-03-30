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

#if CUDA_ENABLED

#include <icrar/leap-accelerate/algorithm/cuda/ValidatedCudaComputeOptions.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/ioutils.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>

#include <cuda_runtime.h>

#include <boost/optional.hpp>

namespace icrar
{
    ValidatedCudaComputeOptions::ValidatedCudaComputeOptions(const ComputeOptions& computeOptions, const icrar::MeasurementSet& ms)
    {
        LOG(info) << "Determining cuda compute options";

        size_t free = 0;
        size_t total = 0;
        if(GetCudaDeviceCount() != 0)
        {
            checkCudaErrors(cudaMemGetInfo(&free, &total));
        }
        size_t VisSize = ms.GetNumBaselines() * ms.GetNumChannels() * sizeof(std::complex<double>);
        size_t AdSize = ms.GetNumStations() * ms.GetNumBaselines() * sizeof(double);
        double safetyFactor = 1.3;

        if(computeOptions.isFileSystemCacheEnabled.is_initialized())
        {
            isFileSystemCacheEnabled = computeOptions.isFileSystemCacheEnabled.get();
        }
        else
        {
            isFileSystemCacheEnabled = false;
        }

        if(computeOptions.useCusolver.is_initialized())
        {
            useCusolver = computeOptions.useCusolver.get();
        }
        else // determine from available memory
        {
            //check Ad matrix size
            size_t required = 3 * AdSize * safetyFactor; // A, Ad and SVD buffers required to compute inverse
            if(required < free)
            {
                LOG(info) << memory_amount(free) << " > " << memory_amount(required) << ". Enabling Cusolver";
                useCusolver = true;
            }
            else
            {
                LOG(info) << memory_amount(free) << " < " << memory_amount(required) << ". Disabling Cusolver";
                useCusolver = false;
            }
        }

        if(computeOptions.useIntermediateBuffer.is_initialized())
        {
            useIntermediateBuffer = computeOptions.useIntermediateBuffer.get();
        }
        else // determine from available memory
        {
            // A, Ad and 2x visibilities required to calibrate
            size_t required = (2 * AdSize + 2 * VisSize) * safetyFactor;
            if(required < free)
            {
                LOG(info) << memory_amount(free) << " > " << memory_amount(required) << ". Enabling IntermediateBuffer";
                useIntermediateBuffer = true;
            }
            else
            {
                LOG(info) << memory_amount(free) << " < " << memory_amount(required) << ". Disabling IntermediateBuffer";
                useIntermediateBuffer = false;
            }
        }
    }
} // namespace icrar

#endif // CUDA_ENABLED