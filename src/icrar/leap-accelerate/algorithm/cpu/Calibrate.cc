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

#include "Calibrate.h"

#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cpu/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/cpu/vector.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/common/eigen_extensions.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/ms/MeasurementSets/MSAntenna.h>

#include <boost/math/constants/constants.hpp>
#include <boost/optional.hpp>

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
    CalibrationCollection Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<SphericalDirection>& directions,
        const Slice& solutionInterval,
        double minimumBaselineThreshold,
        boost::optional<unsigned int> referenceAntenna,
		bool isFileSystemCacheEnabled)
    {
        LOG(info) << "Starting calibration using cpu";
        LOG(info)
        << "stations: " << ms.GetNumStations() << ", "
        << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "solutionInterval: [" << solutionInterval.start << "," << solutionInterval.interval << "," << solutionInterval.end << "], "
        << "flagged baselines: " << ms.GetNumFlaggedBaselines() << ", "
        << "baseline threshold: " << minimumBaselineThreshold << "m, "
        << "short baselines: " << ms.GetNumShortBaselines(minimumBaselineThreshold) << ", "
        << "filtered baselines: " << ms.GetNumFilteredBaselines(minimumBaselineThreshold) << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << ms.GetNumTimesteps();

        profiling::timer calibration_timer;

        auto output_calibrations = std::vector<cpu::Calibration>();
        auto input_queues = std::vector<std::vector<cpu::Integration>>();

        size_t timesteps = ms.GetNumTimesteps();
        Range validatedSolutionInterval = solutionInterval.Evaluate(timesteps);
        std::vector<double> epochs = ms.GetEpochs();

        profiling::timer metadata_read_timer;
        LOG(info) << "Loading MetaData";
        auto metadata = icrar::cpu::MetaData(
            ms,
            referenceAntenna,
            minimumBaselineThreshold,
            isFileSystemCacheEnabled);
        LOG(info) << "Read metadata in " << metadata_read_timer;

        size_t solutions = validatedSolutionInterval.GetSize();
        constexpr unsigned int integrationNumber = 0;
        for(size_t solution = 0; solution < solutions; ++solution)
        {
            profiling::timer solution_timer;
            output_calibrations.emplace_back(
                epochs[solution * validatedSolutionInterval.interval],
                epochs[(solution+1) * validatedSolutionInterval.interval - 1]);
            input_queues.clear();

            //Iterate solutions
            profiling::timer integration_read_timer;
            const Integration integration = Integration(
                    integrationNumber,
                    ms,
                    solution * validatedSolutionInterval.interval * ms.GetNumBaselines(),
                    ms.GetNumChannels(),
                    validatedSolutionInterval.interval * ms.GetNumBaselines(),
                    ms.GetNumPols());
            LOG(info) << "Read integration data in " << integration_read_timer;

            for(size_t direction = 0; direction < directions.size(); ++direction)
            {
                auto queue = std::vector<cpu::Integration>();
                queue.push_back(integration);
                input_queues.push_back(queue);
            }

            profiling::timer phase_rotate_timer;
            metadata.SetUVW(integration.GetUVW());
            for(size_t i = 0; i < directions.size(); ++i)
            {
                LOG(info) << "Processing direction " << i;
                metadata.SetDirection(directions[i]);
                metadata.CalcUVW();
                metadata.GetAvgData().setConstant(std::complex<double>(0.0,0.0));
                icrar::cpu::PhaseRotate(metadata, directions[i], input_queues[i], output_calibrations[solution].GetBeamCalibrations());
            }

            LOG(info) << "Performed PhaseRotate in " << phase_rotate_timer;
            LOG(info) << "Finished solution in " << solution_timer;
        }
        LOG(info) << "Finished calibration in " << calibration_timer;
        return CalibrationCollection(output_calibrations);
    }

    void PhaseRotate(
        cpu::MetaData& metadata,
        const SphericalDirection& direction,
        std::vector<cpu::Integration>& input,
        std::vector<cpu::BeamCalibration>& output_calibrations)
    {
        for(auto& integration : input)
        {
            LOG(info) << "Rotating Integration batch " << integration.GetIntegrationNumber();
            icrar::cpu::RotateVisibilities(integration, metadata);
        }

        LOG(info) << "Calculating Calibration";
        // PhaseAngles I1
        // Value at last index of phaseAnglesI1 must be 0 (which is the reference antenna phase value)
        Eigen::VectorXd phaseAnglesI1 = icrar::arg(icrar::cpu::VectorRangeSelect(metadata.GetAvgData(), metadata.GetI1(), 0)); // 1st pol only
        phaseAnglesI1.conservativeResize(phaseAnglesI1.rows() + 1);
        phaseAnglesI1(phaseAnglesI1.rows() - 1) = 0;

        Eigen::VectorXd cal1 = metadata.GetAd1() * phaseAnglesI1;
        Eigen::MatrixXd dInt = Eigen::MatrixXd::Zero(metadata.GetI().size(), metadata.GetAvgData().cols());

        Eigen::VectorXd ACal1 = metadata.GetA() * cal1;
        for(int n = 0; n < metadata.GetI().size(); ++n)
        {
            dInt.row(n) = icrar::arg(std::exp(std::complex<double>(0, -two_pi<double>() * ACal1(n))) * metadata.GetAvgData().row(n));
        }

        Eigen::VectorXd deltaPhaseColumn = dInt.col(0); // 1st pol only
        deltaPhaseColumn.conservativeResize(deltaPhaseColumn.size() + 1);
        deltaPhaseColumn(deltaPhaseColumn.size() - 1) = 0;

        output_calibrations.emplace_back(direction, (metadata.GetAd() * deltaPhaseColumn) + cal1);
    }

    void RotateVisibilities(cpu::Integration& integration, cpu::MetaData& metadata)
    {
        using namespace std::literals::complex_literals;
        Eigen::Tensor<std::complex<double>, 3>& integration_data = integration.GetVis();

        // loop over smeared baselines
        for(size_t baseline = 0; baseline < integration.GetBaselines(); ++baseline)
        {
            auto md_baseline = static_cast<int>(baseline % static_cast<size_t>(metadata.GetConstants().nbaselines)); // metadata baseline
            double shiftFactor = -two_pi<double>() * (metadata.GetRotatedUVW()[baseline](2) - metadata.GetUVW()[baseline](2));

            // Loop over channels
            for(uint32_t channel = 0; channel < metadata.GetConstants().channels; channel++)
            {
                double shiftRad = shiftFactor / metadata.GetConstants().GetChannelWavelength(channel);
                for(uint32_t polarization = 0; polarization < metadata.GetConstants().num_pols; ++polarization)
                {
                    integration_data(polarization, baseline, channel) *= std::exp(std::complex<double>(0.0, shiftRad));
                }

                bool hasNaN = false;
                const Eigen::Tensor<std::complex<double>, 1> polarizations = integration_data.chip(channel, 2).chip(baseline, 1);
                for(uint32_t polarization = 0; polarization < metadata.GetConstants().num_pols; ++polarization)
                {
                    hasNaN |= std::isnan(polarizations(polarization).real()) || std::isnan(polarizations(polarization).imag());
                }

                if(!hasNaN)
                {
                    for(uint32_t polarization = 0; polarization < metadata.GetConstants().num_pols; ++polarization)
                    {
                        metadata.GetAvgData()(md_baseline, polarization) += integration_data(polarization, baseline, channel);
                    }
                }
            }
        }
    }
} // namespace cpu
} // namespace icrar
