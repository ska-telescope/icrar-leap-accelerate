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

#include <icrar/leap-accelerate/common/eigen_stringutils.h>

namespace icrar
{
    class EigenStringUtilsTests : public testing::Test
    {
    public:
        void TestTallMatrix()
        {
            auto m = Eigen::MatrixXd(10,4);
            m.setConstant(1);
            std::string expected =
"Eigen::Matrix [ 10, 4]\n\
[           1            1            1            1]\n\
[           1            1            1            1]\n\
[           1            1            1            1]\n\
[         ...          ...          ...          ...]\n\
[           1            1            1            1]\n\
[           1            1            1            1]\n\
[           1            1            1            1]";
            ASSERT_EQ(expected , pretty_matrix(m));
        }

        void TestWideMatrix()
        {
            auto m = Eigen::MatrixXd(4,10);
            m.setConstant(1);
            std::string expected =
"Eigen::Matrix [ 4, 10]\n\
[           1            1            1          ...            1            1            1]\n\
[           1            1            1          ...            1            1            1]\n\
[           1            1            1          ...            1            1            1]\n\
[           1            1            1          ...            1            1            1]";
            ASSERT_EQ(expected , pretty_matrix(m));
        }

        void TestLargeMatrix()
        {
            auto m = Eigen::MatrixXd(10,10);
            m.setConstant(1);
            std::string expected =
"Eigen::Matrix [ 10, 10]\n\
[           1            1            1          ...            1            1            1]\n\
[           1            1            1          ...            1            1            1]\n\
[           1            1            1          ...            1            1            1]\n\
[         ...          ...          ...          ...          ...          ...          ...]\n\
[           1            1            1          ...            1            1            1]\n\
[           1            1            1          ...            1            1            1]\n\
[           1            1            1          ...            1            1            1]";
            ASSERT_EQ(expected , pretty_matrix(m));
        }
    };

    TEST_F(EigenStringUtilsTests, TestTallMatrix) { TestTallMatrix(); }
    TEST_F(EigenStringUtilsTests, TestWideMatrix) { TestWideMatrix(); }
    TEST_F(EigenStringUtilsTests, TestLargeMatrix) { TestLargeMatrix(); }
} // namespace icrar