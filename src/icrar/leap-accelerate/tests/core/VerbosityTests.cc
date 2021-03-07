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

#include <icrar/leap-accelerate/core/log/Verbosity.h>

namespace icrar
{
    class VerbosityTests : public testing::Test
    {
    public:
        void TestParseVerbosity()
        {
            using namespace log;
            ASSERT_EQ(Verbosity::fatal, ParseVerbosity("fatal"));
            ASSERT_EQ(Verbosity::error, ParseVerbosity("error"));
            ASSERT_EQ(Verbosity::warn, ParseVerbosity("warn"));
            ASSERT_EQ(Verbosity::info, ParseVerbosity("info"));
            ASSERT_EQ(Verbosity::debug, ParseVerbosity("debug"));
            ASSERT_EQ(Verbosity::trace, ParseVerbosity("trace"));
            ASSERT_EQ(Verbosity::trace, ParseVerbosity("Trace"));
            ASSERT_EQ(Verbosity::trace, ParseVerbosity("TRACE"));
        }
    };

    TEST_F(VerbosityTests, TestParseVerbosity) { TestParseVerbosity(); }
} // namespace icrar