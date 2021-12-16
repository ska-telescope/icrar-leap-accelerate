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
 * MA  02110-1301  USA
 */

#pragma once

#include <icrar/leap-accelerate/config.h>
#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>

#include <icrar/leap-accelerate/common/Slice.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cpu/calibration/Calibration.h>

#include <casacore/ms/MeasurementSets.h>
#include <Eigen/Core>

#include <boost/noncopyable.hpp>
#include <boost/optional.hpp>

#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>

namespace icrar
{
namespace cpu
{
    class MetaData;

    /**
     * @brief Leap Calibration implementation using 
     * 
     */
    class CpuLeapCalibrator : public ILeapCalibrator
    {
    public:
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
            const ComputeOptionsDTO& computeOptions) override;

        /**
         * @brief Performs rotation, summing and calibration for @p direction
         * 
         * @param metadata metadata object containing data required for calibration
         * @param direction the direction to calibrate for 
         * @param input batches of uvws and visibilities to process
         * @param output_calibrations output calibration from summing a function of uvws and visibilities
         */
        static void PhaseRotate(
            MetaData& metadata,
            const SphericalDirection& direction,
            std::vector<Integration>& input,
            std::vector<BeamCalibration>& output_calibrations);

        /**
         * @brief Performs rotation and averaging over each baseline, channel and polarization.
         * 
         * @param integration The input integration batch of uvws and visibilities
         * @param metadata The metadata object where AverageData is written to
         */
        static void RotateVisibilities(
            Integration& integration,
            MetaData& metadata);
    };
} // namespace cpu
} // namespace icrar