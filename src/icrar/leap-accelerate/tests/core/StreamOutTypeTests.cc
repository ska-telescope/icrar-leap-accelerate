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

#include <gtest/gtest.h>

#include <icrar/leap-accelerate/core/stream_out_type.h>

namespace icrar
{
    class StreamOutTypeTests : public testing::Test
    {
    public:
        void TestParseStreamOutType()
        {
            ASSERT_EQ(StreamOutType::collection, ParseStreamOutType("c"));
            ASSERT_EQ(StreamOutType::collection, ParseStreamOutType("collection"));
            ASSERT_EQ(StreamOutType::singleFile, ParseStreamOutType("s"));
            ASSERT_EQ(StreamOutType::singleFile, ParseStreamOutType("single"));
            ASSERT_EQ(StreamOutType::multipleFiles, ParseStreamOutType("m"));
            ASSERT_EQ(StreamOutType::multipleFiles, ParseStreamOutType("multiple"));
        }
    };

    TEST_F(StreamOutTypeTests, TestParseStreamOutType) { TestParseStreamOutType(); }
} // namespace icrar
