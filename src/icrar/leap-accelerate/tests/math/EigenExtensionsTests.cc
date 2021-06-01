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
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>
#include <boost/math/constants/constants.hpp>

namespace icrar
{
    class EigenExtensionsTests : public testing::Test
    {
        double THRESHOLD = 0.00001;
    public:
        template<typename Index>
        void TestWrappedRowSelect()
        {
            auto m = Eigen::MatrixXd(3,3);
            m <<
            0, 1, 2,
            3, 4, 5,
            6, 7, 8;

            auto r = Eigen::Vector<Index, Eigen::Dynamic>(6);
            r << -3, -2, -1, 0, 1, 2;

            Eigen::MatrixXd v = cpu::WrappedRowSelect(m, r);
            auto expected = Eigen::MatrixXd(6,3);
            expected <<
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            0, 1, 2,
            3, 4, 5,
            6, 7, 8;
            ASSERT_MEQD(expected, v, THRESHOLD);

            // out of bounds
            auto rl = Eigen::Vector<Index, Eigen::Dynamic>(2);
            rl << -3, -4;
            ASSERT_THROW(cpu::WrappedRowSelect(m, rl), std::runtime_error);

            auto rh = Eigen::Vector<Index, Eigen::Dynamic>(2);
            rh << 2, 3;
            ASSERT_THROW(cpu::WrappedRowSelect(m, rh), std::runtime_error);
        }

        void TestArg()
        {
            using namespace std::complex_literals;
            auto m = Eigen::MatrixXcd(2,2);
            m <<
            0, 1,
            1i, -1i;

            Eigen::MatrixXd v = cpu::arg(m);

            auto expected = Eigen::MatrixXd(2,2);
            expected <<
            0, 0,
            boost::math::constants::pi<double>() / 2, -boost::math::constants::pi<double>() / 2;
            ASSERT_MEQD(expected, v, THRESHOLD);
        }
    };

    TEST_F(EigenExtensionsTests, TestWrappedRowSelect32) { TestWrappedRowSelect<int32_t>(); }
    TEST_F(EigenExtensionsTests, TestWrappedRowSelect64) { TestWrappedRowSelect<Eigen::Index>(); }
    TEST_F(EigenExtensionsTests, TestArg) { TestArg(); }
} // namespace icrar
