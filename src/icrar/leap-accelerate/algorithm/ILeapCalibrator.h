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

#include <icrar/leap-accelerate/algorithm/ComputeOptionsDTO.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/common/Slice.h>
#include <icrar/leap-accelerate/model/cpu/calibration/Calibration.h>
#include <boost/noncopyable.hpp>
#include <boost/optional.hpp>
#include <vector>
#include <functional>

namespace icrar
{
    class MeasurementSet;

    namespace cpu
    {
        class Integration;
        class IntegrationResult;
        class BeamCalibration;
        class Calibration;
    }

    /**
     * @brief Interface for Leap calibration implementations.
     * 
     */
    class ILeapCalibrator : boost::noncopyable
    {
    public:
        virtual ~ILeapCalibrator() = default;

        /**
         * @brief Performs Leap calibration for single or multiple solutions.
         * 
         * @param ms the mesurement set containing all input measurements
         * @param directions the directions to calibrate for
         * @param minimumBaselineThreshold the minimum baseline length to use in calibrations
         * @param solutionInterval the arbitrary interval to calculate solutions for
         * @param referenceAntenna the reference antenna of metrix A1
         * @param isFileSystemCacheEnabled enable to use the filesystem to cache data between calibration calls
         * @return CalibrationCollection the calibrationn result
         */
        virtual void Calibrate(
            std::function<void(const cpu::Calibration&)> outputCallback,
            const icrar::MeasurementSet& ms,
            const std::vector<SphericalDirection>& directions,
            const Slice& solutionInterval,
            double minimumBaselineThreshold,
            boost::optional<unsigned int> referenceAntenna,
            const ComputeOptionsDTO& computeOptions) = 0;
    };
} // namespace icrar