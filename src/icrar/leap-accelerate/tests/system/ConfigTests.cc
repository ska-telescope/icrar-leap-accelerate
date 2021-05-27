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

#include <icrar/leap-accelerate/core/memory/system_memory.h>
#include <icrar/leap-accelerate/core/memory/ioutils.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/dll/runtime_symbol_info.hpp>
#include <iostream>
#include <array>
#include <sstream>
#include <streambuf>

using namespace icrar;

/**
 * @brief Contains system tests 
 * 
 */
class ConfigTests : public testing::Test
{
    const std::string m_mwaDirections = "[\
        [-0.4606549305661674,-0.29719233792392513],\
        [-0.753231018062671,-0.44387635324622354],\
        [-0.6207547100721282,-0.2539086572881469],\
        [-0.41958660604621867,-0.03677626900108552],\
        [-0.41108685258900596,-0.08638012622791202],\
        [-0.7782459495668798,-0.4887860989684432],\
        [-0.17001324965728973,-0.28595644149463484],\
        [-0.7129444556035118,-0.365286407171852],\
        [-0.1512764129166089,-0.21161026349648748]\
    ]";

    const std::string m_simulationDirections = "[\
        [0.0, -0.471238898],\
        [0.017453293, -0.4537856055]\
    ]";

public:
    ConfigTests() = default;

    void TestDefaultConfig(const boost::filesystem::path& outputPath)
    {
        std::string path = (boost::dll::program_location().parent_path() / outputPath).string();
        std::ifstream expectedStream(path);
        ASSERT_TRUE(expectedStream.good());
        auto expected = std::string(std::istreambuf_iterator<char>(expectedStream), std::istreambuf_iterator<char>());
    
        auto rawArgs = CLIArgumentsDTO::GetDefaultArguments();
        rawArgs.filePath = std::string(TEST_DATA_DIR) + "/mwa/1197638568-split.ms";
        rawArgs.directions = "[[0,0]]";
        auto args = ArgumentsValidated(std::move(rawArgs));
        auto calibrator = LeapCalibratorFactory::Create(args.GetComputeImplementation());
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
            args.GetComputeOptions());

        auto actualPath = boost::dll::program_location().parent_path() / outputPath.parent_path()
        / (outputPath.stem().string() + "_ACTUAL" + outputPath.extension().string());
        auto actualFile = std::ofstream(actualPath.string());
        actualFile << output.str();
        ASSERT_EQ(expected, output.str());
    }

    void TestConfig(CLIArgumentsDTO&& rawArgs, double threshold)
    {
        ASSERT_TRUE(rawArgs.outputFilePath.is_initialized()) << "outputFilePath not set";
        boost::filesystem::path outputPath = rawArgs.outputFilePath.get();
        std::string path = (boost::dll::program_location().parent_path() / outputPath).string();
        std::ifstream expectedStream(path);
        auto args = ArgumentsValidated(std::move(rawArgs));
        auto calibrator = LeapCalibratorFactory::Create(args.GetComputeImplementation());
        std::stringstream output;
        auto outputCallback = [&](const cpu::Calibration& cal)
        {
            cal.Serialize(output, true);
        };
        calibrator->Calibrate(
            outputCallback,
            args.GetMeasurementSet(),
            args.GetDirections(),
            args.GetSolutionInterval(),
            args.GetMinimumBaselineThreshold(),
            args.GetReferenceAntenna(),
            args.GetComputeOptions());

        auto actualPath = boost::dll::program_location().parent_path() / outputPath.parent_path()
        / (outputPath.stem().string() + "_ACTUAL" + outputPath.extension().string());
        auto actualFile = std::ofstream(actualPath.string());
        actualFile << output.str();
        actualFile.flush();
        auto actual = cpu::Calibration::Parse(output.str());

        auto expectedStr = std::string(std::istreambuf_iterator<char>(expectedStream), std::istreambuf_iterator<char>());
        ASSERT_TRUE(expectedStream.good()) << path << " does not exist";

        auto expected = cpu::Calibration::Parse(expectedStr);
        ASSERT_TRUE(expected.IsApprox(actual, threshold)) << actualPath << " does not match " << path;
    }

    void TestMWACpuConfig()
    {
        CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
        rawArgs.filePath = std::string(TEST_DATA_DIR) + "/mwa/1197638568-split.ms";
        rawArgs.outputFilePath = "testdata/MWACpuOutput.json";
        rawArgs.directions = m_mwaDirections;
        rawArgs.computeImplementation = "cpu";
        rawArgs.useFileSystemCache = false;
        TestConfig(std::move(rawArgs), 1e-15);
    }

    void TestMWACudaConfig()
    {
        CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
        rawArgs.filePath = std::string(TEST_DATA_DIR) + "/mwa/1197638568-split.ms";
        rawArgs.outputFilePath = "testdata/MWACudaOutput.json";
        rawArgs.directions = m_mwaDirections;
        rawArgs.computeImplementation = "cuda";
        rawArgs.useCusolver = true;
        rawArgs.useFileSystemCache = false;
        TestConfig(std::move(rawArgs), 1e-10);
    }

    void TestAA3CpuConfig()
    {
        CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
        rawArgs.filePath = std::string(TEST_DATA_DIR) + "/aa3/aa3-SS-300.ms";
        rawArgs.outputFilePath = "testdata/AA3CpuOutput.json";
        rawArgs.directions = m_simulationDirections;
        rawArgs.computeImplementation = "cpu";
        rawArgs.useFileSystemCache = false;
        TestConfig(std::move(rawArgs), 1e-15);
    }

    void TestAA3CudaConfig()
    {
        CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
        rawArgs.filePath = std::string(TEST_DATA_DIR) + "/aa3/aa3-SS-300.ms";
        rawArgs.outputFilePath = "testdata/AA3CudaOutput.json";
        rawArgs.directions = m_simulationDirections;
        rawArgs.computeImplementation = "cuda";
        rawArgs.useCusolver = true;
        rawArgs.useFileSystemCache = false;
        TestConfig(std::move(rawArgs), 1e-10);
    }

    void TestAA4CpuConfig()
    {
        size_t availableMemory = GetTotalAvailableSystemVirtualMemory();

        size_t VisSize = 4 * 512*511/2 * 33 * sizeof(std::complex<double>); // polarizations * baselines * channels * sizeof(std::complex<double>)
        size_t AdSize = 512 * 512*511/2 * sizeof(double); // stations * baselines * sizeof(double);
        if(availableMemory < (VisSize + AdSize))
        {
            GTEST_SKIP() << memory_amount(VisSize + AdSize)
            << " system memory required but only " << memory_amount(availableMemory) << " available";
        }

        CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
        rawArgs.filePath = std::string(TEST_DATA_DIR) + "/aa4/aa4-SS-33-120.ms";
        rawArgs.outputFilePath = "testdata/AA4CpuOutput.json";
        rawArgs.directions = m_simulationDirections;
        rawArgs.computeImplementation = "cpu";
        rawArgs.useCusolver = true;
        rawArgs.useFileSystemCache = false;
        rawArgs.solutionInterval = "[0,1,1]";
        TestConfig(std::move(rawArgs), 1e-15);
    }

    void TestAA4CudaConfig()
    {
        size_t cudaAvailable = GetAvailableCudaPhysicalMemory();

        size_t VisSize = 4 * 512*511/2 * 33 * sizeof(std::complex<double>); // polarizations * baselines * channels * sizeof(std::complex<double>)
        size_t AdSize = 512 * 512*511/2 * sizeof(double); // stations * baselines * sizeof(double);
        size_t AdWorkSize = 1170 * 1024 * 1024;
        if(cudaAvailable < (VisSize + AdSize + AdWorkSize))
        {
            GTEST_SKIP() << memory_amount(VisSize + AdSize + AdWorkSize)
            << " device memory required but only " << memory_amount(cudaAvailable) << " available";
        }

        CLIArgumentsDTO rawArgs = CLIArgumentsDTO::GetDefaultArguments();
        rawArgs.filePath = std::string(TEST_DATA_DIR) + "/aa4/aa4-SS-33-120.ms";
        rawArgs.outputFilePath = "testdata/AA4CudaOutput.json";
        rawArgs.directions = m_simulationDirections;
        rawArgs.computeImplementation = "cuda";
        rawArgs.useCusolver = true;
        rawArgs.useFileSystemCache = false;
        rawArgs.solutionInterval = "[0,1,1]";
        TestConfig(std::move(rawArgs), 1e-5);
    }
};

TEST_F(ConfigTests, TestDefaultConfig) { TestDefaultConfig("testdata/DefaultOutput.json"); }
TEST_F(ConfigTests, TestMWACpuConfig) { TestMWACpuConfig(); }
TEST_F(ConfigTests, DISABLED_TestAA3CpuConfig) { TestAA3CpuConfig(); } // Large download
TEST_F(ConfigTests, DISABLED_TestAA4CpuConfig) { TestAA4CpuConfig(); } // Large download

#if CUDA_ENABLED
TEST_F(ConfigTests, TestMWACudaConfig) { TestMWACudaConfig(); }
TEST_F(ConfigTests, DISABLED_TestAA3CudaConfig) { TestAA3CudaConfig(); } // Large download
TEST_F(ConfigTests, DISABLED_TestAA4CudaConfig) { TestAA4CudaConfig(); } // Large download
#endif // CUDA_ENABLED