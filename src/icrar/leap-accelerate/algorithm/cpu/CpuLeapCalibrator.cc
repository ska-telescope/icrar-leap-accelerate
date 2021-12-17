/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


#include "CpuLeapCalibrator.h"

#include <icrar/leap-accelerate/common/eigen_stringutils.h>

#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>
#include <icrar/leap-accelerate/algorithm/cpu/CpuComputeOptions.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cpu/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/ms/MeasurementSets/MSAntenna.h>

#include <boost/math/constants/constants.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/thread.hpp>

#include <istream>
#include <iostream>
#include <iterator>
#include <string>
#include <queue>
#include <exception>
#include <memory>
#include <sstream>

using Radians = double;
using namespace boost::math::constants;

namespace icrar
{
namespace cpu
{
    void CpuLeapCalibrator::Calibrate(
        std::function<void(const cpu::Calibration&)> outputCallback,
        const icrar::MeasurementSet& ms,
        const std::vector<SphericalDirection>& directions,
        const Slice& solutionInterval,
        double minimumBaselineThreshold,
        boost::optional<unsigned int> referenceAntenna,
        const ComputeOptionsDTO& computeOptions)
    {
        auto cpuComputeOptions = CpuComputeOptions(computeOptions, ms);

        LOG(info) << "Starting calibration using cpu";
        LOG(info)
        << "stations: " << ms.GetNumStations() << ", "
        << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "solutionInterval: [" << solutionInterval.GetStart() << "," << solutionInterval.GetInterval() << "," << solutionInterval.GetEnd() << "], "
        << "reference antenna: " << referenceAntenna << ", "
        << "flagged baselines: " << ms.GetNumFlaggedBaselines() << ", "
        << "baseline threshold: " << minimumBaselineThreshold << "m, "
        << "short baselines: " << ms.GetNumShortBaselines(minimumBaselineThreshold) << ", "
        << "filtered baselines: " << ms.GetNumFilteredBaselines(minimumBaselineThreshold) << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << ms.GetNumTimesteps();

        profiling::timer calibration_timer;
        Rangei validatedSolutionInterval = solutionInterval.Evaluate(boost::numeric_cast<int32_t>(ms.GetNumTimesteps()));
        std::vector<double> epochs = ms.GetEpochs();

        profiling::timer metadata_read_timer;
        LOG(info) << "Loading MetaData";
        auto metadata = icrar::cpu::MetaData(
            ms,
            referenceAntenna,
            minimumBaselineThreshold,
            true,
            cpuComputeOptions.IsFileSystemCacheEnabled());
        LOG(info) << "Metadata loaded in " << metadata_read_timer;

        int32_t solutions = validatedSolutionInterval.GetSize();
        auto output_calibrations = std::vector<cpu::Calibration>();
        output_calibrations.reserve(solutions);

        constexpr unsigned int integrationNumber = 0;
        for(int32_t solution = 0; solution < solutions; ++solution)
        {
            auto input_queues = std::vector<std::vector<cpu::Integration>>();
            profiling::timer solution_timer;
            output_calibrations.emplace_back(
                epochs[solution * validatedSolutionInterval.GetInterval()],
                epochs[(solution+1) * validatedSolutionInterval.GetInterval() - 1]);

            //Iterate solutions
            profiling::timer integration_read_timer;
            const auto integration = Integration(
                    integrationNumber,
                    ms,
                    solution * validatedSolutionInterval.GetInterval(),
                    validatedSolutionInterval.GetInterval(),
                    Slice(0, ms.GetNumPols(), ms.GetNumPols()-1)); // XX + YY pols
                    
            LOG(info) << "Read integration data in " << integration_read_timer;

            for(size_t direction = 0; direction < directions.size(); ++direction)
            {
                auto queue = std::vector<cpu::Integration>();
                queue.push_back(integration);
                input_queues.push_back(std::move(queue));
            }

            profiling::timer phase_rotate_timer;
            for(size_t direction = 0; direction < directions.size(); ++direction)
            {
                LOG(info) << "Processing direction " << direction;
                metadata.SetDirection(directions[direction]);
                metadata.GetAvgData().setConstant(std::complex<double>(0.0,0.0));
                PhaseRotate(
                    metadata,
                    directions[direction],
                    input_queues[direction],
                    output_calibrations[solution].GetBeamCalibrations());
            }

            LOG(info) << "Performed PhaseRotate in " << phase_rotate_timer;
            LOG(info) << "Calculated solution in " << solution_timer;

            profiling::timer write_timer;
            outputCallback(output_calibrations[solution]);
            LOG(info) << "Write out in " << write_timer;
        }
        LOG(info) << "Finished calibration in " << calibration_timer;
    }

    void CpuLeapCalibrator::PhaseRotate(
        cpu::MetaData& metadata,
        const SphericalDirection& direction,
        std::vector<cpu::Integration>& input,
        std::vector<cpu::BeamCalibration>& output_calibrations)
    {
        for(auto& integration : input)
        {
            LOG(info) << "Rotating Integration " << integration.GetIntegrationNumber();
            RotateVisibilities(integration, metadata);
        }

        LOG(info) << "Calculating Calibration";
        auto avgDataI1 = metadata.GetAvgData().wrapped_row_select(metadata.GetI1());
        Eigen::VectorXd phaseAnglesI1 = avgDataI1.arg();

        // Value at last index of phaseAnglesI1 must be 0 (which is the reference antenna phase value)
        phaseAnglesI1.conservativeResize(phaseAnglesI1.rows() + 1);
        phaseAnglesI1(phaseAnglesI1.rows() - 1) = 0;

        Eigen::VectorXd cal1 = metadata.GetAd1() * phaseAnglesI1;
        Eigen::VectorXd ACal1 = metadata.GetA() * cal1;

        Eigen::VectorXd deltaPhase = Eigen::VectorXd::Zero(metadata.GetI().size());
        for(int n = 0; n < metadata.GetI().size(); ++n)
        {
            deltaPhase(n) = std::arg(std::exp(std::complex<double>(0, -two_pi<double>() * ACal1(n))) * metadata.GetAvgData()(n));
        }

        Eigen::VectorXd deltaPhaseColumn = deltaPhase;
        deltaPhaseColumn.conservativeResize(deltaPhaseColumn.size() + 1);
        deltaPhaseColumn(deltaPhaseColumn.size() - 1) = 0;
        output_calibrations.emplace_back(direction, (metadata.GetAd() * deltaPhaseColumn) + cal1);
    }

    void CpuLeapCalibrator::RotateVisibilities(cpu::Integration& integration, cpu::MetaData& metadata)
    {
        using namespace std::literals::complex_literals;
        Eigen::Tensor<std::complex<double>, 4>& visibilities = integration.GetVis();

        for(size_t timestep = 0; timestep < integration.GetNumTimesteps(); ++timestep)
        {
            for(size_t baseline = 0; baseline < integration.GetNumBaselines(); baseline++)
            {
                Eigen::VectorXd uvw = ToVector(Eigen::Tensor<double, 1>(integration.GetUVW().chip(timestep, 2).chip(baseline, 1)));
                auto rotatedUVW = metadata.GetDD() * uvw;
                double shiftFactor = -two_pi<double>() * (rotatedUVW.z() - uvw.z());

                // Loop over channels
                for(uint32_t channel = 0; channel < integration.GetNumChannels(); channel++)
                {
                    double shiftRad = shiftFactor / metadata.GetConstants().GetChannelWavelength(channel);
                    for(uint32_t polarization = 0; polarization < integration.GetNumPolarizations(); ++polarization)
                    {
                        visibilities(polarization, channel, baseline, timestep) *= std::exp(std::complex<double>(0.0, shiftRad));
                    }

                    bool hasNaN = false;
                    for(uint32_t polarization = 0; polarization < integration.GetNumPolarizations(); ++polarization)
                    {
                        hasNaN |= std::isnan(visibilities(polarization, channel, baseline, timestep).real())
                               || std::isnan(visibilities(polarization, channel, baseline, timestep).imag());
                    }

                    if(!hasNaN)
                    {
                        // Averaging with XX and YY polarizations
                        metadata.GetAvgData()(baseline) += visibilities(0, channel, baseline, timestep);
                        metadata.GetAvgData()(baseline) += visibilities(visibilities.dimension(0) - 1, channel, baseline, timestep);
                    }
                }
            }
        }
    }
} // namespace cpu
} // namespace icrar
