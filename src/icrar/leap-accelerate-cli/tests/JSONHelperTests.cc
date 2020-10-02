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

#include <icrar/leap-accelerate/json/json_helper.h>

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/model/casa/Integration.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>

#include <icrar/leap-accelerate/core/compute_implementation.h>

#include <casacore/casa/Quanta/MVDirection.h>

#include <gtest/gtest.h>

#include <vector>

using namespace std::literals::complex_literals;

namespace icrar
{
    class JSONHelperTests : public ::testing::Test
    {

    protected:

        JSONHelperTests() {

        }

        ~JSONHelperTests() override
        {

        }

        void SetUp() override
        {

        }

        void TearDown() override
        {
            
        }

        void TestParseDirections(const std::string input, const std::vector<icrar::MVDirection>& expected)
        {
            auto actual = icrar::ParseDirections(input);
            ASSERT_EQ(actual, expected);
        }

        void TestParseDirectionsException(const std::string input)
        {
            ASSERT_THROW(icrar::ParseDirections(input), icrar::exception);
        }

    };

    TEST_F(JSONHelperTests, TestParseDirectionsEmpty)
    {
        TestParseDirections("[]", std::vector<icrar::MVDirection>());
        TestParseDirectionsException("{}");
        TestParseDirectionsException("[[]]");
    }

    TEST_F(JSONHelperTests, TestParseDirectionsOne)
    {
        TestParseDirectionsException("[[1.0]]");
        TestParseDirectionsException("[[1.0,1.0,1.0]]");
        TestParseDirections(
            "[[-0.4606549305661674,-0.29719233792392513]]",
            std::vector<icrar::MVDirection>
            {
                ToDirection(casacore::MVDirection(-0.4606549305661674,-0.29719233792392513))
            });
    }

    TEST_F(JSONHelperTests, TestParseDirectionsFive)
    {
        TestParseDirections(
            "[[0.0,0.0],[1.0,1.0],[2.0,2.0],[3.0,3.0],[4.0,4.0]]",
            std::vector<icrar::MVDirection>
            {
                ToDirection(casacore::MVDirection(0.0,0.0)),
                ToDirection(casacore::MVDirection(1.0,1.0)),
                ToDirection(casacore::MVDirection(2.0,2.0)),
                ToDirection(casacore::MVDirection(3.0,3.0)),
                ToDirection(casacore::MVDirection(4.0,4.0)),
            });
    }
}
