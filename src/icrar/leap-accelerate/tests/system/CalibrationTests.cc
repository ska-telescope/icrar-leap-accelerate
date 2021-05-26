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

#include <gtest/gtest.h>

#include <icrar/leap-accelerate/tests/math/eigen_helper.h>
#include <icrar/leap-accelerate/common/config/Arguments.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <icrar/leap-accelerate/model/cpu/calibration/Calibration.h>

#include <icrar/leap-accelerate/model/cpu/calibration/BeamCalibration.h>

#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>

#include <utility>
#include <iostream>
#include <array>
#include <sstream>
#include <streambuf>

using namespace icrar;

/**
 * @brief Contains system tests 
 * 
 */
class CalibrationTests : public testing::Test
{
    const double TOLERANCE = 1e-5;

    const std::string m_simulationDirections = "[\
        [0.0, -0.471238898],\
        [0.017453293, -0.4537856055]\
    ]";

public:

    /**
     * @brief Tests an AA3 simulated observation with no ionosphere
     * 
     * @param outputPath 
     */
    void TestAA3ClearCalibration(std::string impl, const std::vector<std::pair<double, double>>& expected)
    {
        auto rawArgs = CLIArgumentsDTO::GetDefaultArguments();
        rawArgs.filePath = std::string(TEST_DATA_DIR) + "/aa3/aa3-SS-300.ms";
        rawArgs.directions = m_simulationDirections;
        rawArgs.readAutocorrelations = false;
        rawArgs.referenceAntenna = 210;
        rawArgs.minimumBaselineThreshold = 1000.0;
        rawArgs.computeImplementation = impl;
        rawArgs.useFileSystemCache = false;
        auto args = ArgumentsValidated(std::move(rawArgs));

        auto calibrator = LeapCalibratorFactory::Create(args.GetComputeImplementation());
        std::vector<cpu::Calibration> calibrations;
        auto outputCallback = [&](const cpu::Calibration& cal)
        {
            calibrations.push_back(cal);
        };
        calibrator->Calibrate(
            outputCallback,
            args.GetMeasurementSet(),
            args.GetDirections(),
            args.GetSolutionInterval(),
            args.GetMinimumBaselineThreshold(),
            args.GetReferenceAntenna(),
            args.GetComputeOptions());

        const auto& calibration = calibrations[0];

        // 0 values are better
        EXPECT_NEAR(calibration.GetBeamCalibrations()[0].GetPhaseCalibration().mean(),                          std::get<0>(expected[0]), TOLERANCE);
        EXPECT_NEAR(icrar::cpu::standard_deviation(calibration.GetBeamCalibrations()[0].GetPhaseCalibration()), std::get<1>(expected[0]), TOLERANCE);
        EXPECT_NEAR(calibration.GetBeamCalibrations()[1].GetPhaseCalibration().mean(),                          std::get<0>(expected[1]), TOLERANCE);
        EXPECT_NEAR(icrar::cpu::standard_deviation(calibration.GetBeamCalibrations()[1].GetPhaseCalibration()), std::get<1>(expected[1]), TOLERANCE);
    }
};

TEST_F(CalibrationTests, TestAA3ClearCpuCalibration)
{
    TestAA3ClearCalibration(ComputeImplementationToString(ComputeImplementation::cpu),
    {
        { 0.0031753870255758166, 0.049746623893558904 },
        { -0.023484514309303432, 0.1534504270121704 }
    });
}
#if CUDA_ENABLED
TEST_F(CalibrationTests, TestAA3ClearCudaCalibration)
{
    TestAA3ClearCalibration(ComputeImplementationToString(ComputeImplementation::cuda),
    {
        { 0.0031753870255758166, 0.049746623893558939 },
        { -0.023484514309303432, 0.15345042701215036 }
    });
}
#endif // CUDA_ENABLED
