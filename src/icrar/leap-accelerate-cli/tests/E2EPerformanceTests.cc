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


#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>

#include <gtest/gtest.h>

#include <vector>
#include <set>
#include <unordered_map>

using namespace std::literals::complex_literals;

namespace icrar
{
    class E2EPerformanceTests : public ::testing::Test
    {
        std::unique_ptr<icrar::MeasurementSet> ms;

    protected:
        void SetUp() override
        {

        }

        void TearDown() override
        {
            
        }

        void MultiDirectionTest(ComputeImplementation impl, const std::string& msname, boost::optional<int> stations_override, bool readAutocorrelations)
        {
            std::string filepath = std::string(TEST_DATA_DIR) + msname;
            ms = std::make_unique<icrar::MeasurementSet>(filepath, stations_override, readAutocorrelations);

            std::vector<SphericalDirection> directions =
            {
                SphericalDirection(-0.4606549305661674,-0.29719233792392513),
                SphericalDirection(-0.753231018062671,-0.44387635324622354),
                SphericalDirection(-0.6207547100721282,-0.2539086572881469),
                SphericalDirection(-0.41958660604621867,-0.03677626900108552),
                SphericalDirection(-0.41108685258900596,-0.08638012622791202),
                SphericalDirection(-0.7782459495668798,-0.4887860989684432),
                SphericalDirection(-0.17001324965728973,-0.28595644149463484),
                SphericalDirection(-0.7129444556035118,-0.365286407171852),
                SphericalDirection(-0.1512764129166089,-0.21161026349648748)
            };

            std::vector<cpu::Calibration> calibrations;
            auto outputCallback = [&](const cpu::Calibration& cal)
            {
                calibrations.push_back(cal);
            };

            LeapCalibratorFactory::Create(impl)->AsyncCalibrate(
                outputCallback,
                *ms,
                directions,
                Slice(0,1,1), 0.0, 0, false); 
        }
    };

    // These measurements have flagged data removed and complete data for each timestep
    TEST_F(E2EPerformanceTests, MWACleanTestCpu) { MultiDirectionTest(ComputeImplementation::cpu, "/mwa/1197638568-split.ms", 102, true); }
#ifdef CUDA_ENABLED
    TEST_F(E2EPerformanceTests, MWACleanTestCuda) { MultiDirectionTest(ComputeImplementation::cuda, "/mwa/1197638568-split.ms", 102, true); }
#endif

    // These measurements are clean and use a single timestep
    TEST_F(E2EPerformanceTests, SKACleanTestCpu) { MultiDirectionTest(ComputeImplementation::cpu, "/ska/SKA_LOW_SIM_short_EoR0_ionosphere_off_GLEAM.0001.ms", boost::none, true); }
#ifdef CUDA_ENABLED
    TEST_F(E2EPerformanceTests, SKACleanTestCuda) { MultiDirectionTest(ComputeImplementation::cuda, "/ska/SKA_LOW_SIM_short_EoR0_ionosphere_off_GLEAM.0001.ms", boost::none, true); }
#endif
} // namespace icrar
