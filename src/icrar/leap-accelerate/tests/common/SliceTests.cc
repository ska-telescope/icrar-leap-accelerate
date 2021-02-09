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

#include <icrar/leap-accelerate/common/Slice.h>

namespace icrar
{
    class SliceTests : public testing::Test
    {
    public:
        void TestConstructors()
        {
            ASSERT_NO_THROW(Slice());
            ASSERT_NO_THROW(Slice(-1));
            ASSERT_THROW(Slice(0), icrar::exception);
            ASSERT_NO_THROW(Slice(1));


            ASSERT_NO_THROW(Slice(-1, -1, -1));
            ASSERT_THROW(Slice(-1, -1,  0), icrar::exception);
            ASSERT_THROW(Slice(-1, -1,  1), icrar::exception);
            ASSERT_THROW(Slice(-1,  0, -1), icrar::exception);
            ASSERT_THROW(Slice(-1,  0,  0), icrar::exception);
            ASSERT_THROW(Slice(-1,  0,  1), icrar::exception);
            ASSERT_NO_THROW(Slice(-1,  1, -1));
            ASSERT_THROW(Slice(-1,  1,  0), icrar::exception);
            ASSERT_THROW(Slice(-1,  1,  1), icrar::exception);

            ASSERT_NO_THROW(Slice( 0, -1, -1));
            ASSERT_THROW(Slice( 0, -1,  0), icrar::exception);
            ASSERT_NO_THROW(Slice( 0, -1,  1));
            ASSERT_THROW(Slice( 0,  0, -1), icrar::exception);
            ASSERT_THROW(Slice( 0,  0,  0), icrar::exception);
            ASSERT_THROW(Slice( 0,  0,  1), icrar::exception);
            ASSERT_NO_THROW(Slice( 0,  1, -1));
            ASSERT_THROW(Slice( 0,  1,  0), icrar::exception);
            ASSERT_NO_THROW(Slice( 0,  1,  1));

            ASSERT_NO_THROW(Slice( 1, -1, -1));
            ASSERT_THROW(Slice( 1, -1,  0), icrar::exception);
            ASSERT_THROW(Slice( 1, -1,  1), icrar::exception);
            ASSERT_THROW(Slice( 1,  0, -1), icrar::exception);
            ASSERT_THROW(Slice( 1,  0,  0), icrar::exception);
            ASSERT_THROW(Slice( 1,  0,  1), icrar::exception);
            ASSERT_NO_THROW(Slice( 1,  1, -1));
            ASSERT_THROW(Slice( 1,  1,  0), icrar::exception);
            ASSERT_THROW(Slice( 1,  1,  1), icrar::exception);

            ASSERT_THROW(Slice(0, 1,  -2), icrar::exception);
        }
    };

    TEST_F(SliceTests, TestConstructors) { TestConstructors(); }
} // namespace icrar
