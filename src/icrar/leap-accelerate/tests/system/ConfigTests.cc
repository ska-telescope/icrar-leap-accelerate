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

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <iostream>
#include <array>
#include <sstream>
#include <streambuf>

#if __linux__
#include <string>
#include <limits.h>
#include <unistd.h>

boost::filesystem::path getexepath()
{
  char result[PATH_MAX];
  ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
  return std::string(result, (count > 0) ? count : 0);
}

boost::filesystem::path getexedir()
{
    return getexepath().parent_path();
}
#endif

using namespace icrar;

class ConfigTests : public testing::Test
{
    const double TOLERANCE = 0.0001;

public:
    ConfigTests() = default;

    void TestDefaultConfig(boost::filesystem::path outputPath)
    {

        std::string path = (getexedir() / outputPath).string();
        std::ifstream expectedStream(path);
        std::cout << path << std::endl;
        ASSERT_TRUE(expectedStream.good());
        auto expected = std::string(std::istreambuf_iterator<char>(expectedStream), std::istreambuf_iterator<char>());
    
        auto rawArgs = CLIArguments::GetDefaultArguments();
        rawArgs.filePath = std::string(TEST_DATA_DIR) + "/mwa/1197638568-split.ms";
        rawArgs.directions = "[[0,0]]";
        auto args = ArgumentsValidated(std::move(rawArgs));
        auto calibrator = LeapCalibratorFactory::Create(args.GetComputeImplementation());

        LOG(info) << "printing";
        std::stringstream output;
        auto outputCallback = [&](const cpu::Calibration& cal)
        {
            cal.Serialize(output);
        };
        calibrator->Calibrate(
            outputCallback,
            args.GetMeasurementSet(),
            args.GetDirections(),
            args.GetSolutionInterval(),
            args.GetMinimumBaselineThreshold(),
            args.GetReferenceAntenna(),
            args.IsFileSystemCacheEnabled());

        auto actualFile = std::ofstream(getexedir().append("testdata/DefaultOutput_ACTUAL.json").string());
        actualFile << output.str();
        ASSERT_EQ(expected, output.str());
    }
};

TEST_F(ConfigTests, TestDefaultConfig) { TestDefaultConfig("testdata/DefaultOutput.json"); }
