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

#include "eigen_extensions.h"
#include <icrar/leap-accelerate/core/log/logging.h>

namespace icrar
{
    namespace cpu
    {
        Eigen::MatrixXd arg(const Eigen::Ref<const Eigen::MatrixXcd>& a)
        {
            return a.unaryExpr([](std::complex<double> v){ return std::arg(v); });
        }

        bool near(const Eigen::Ref<const Eigen::MatrixXd> left, const Eigen::Ref<const Eigen::MatrixXd> right, double tolerance)
        {
            bool equal = left.rows() == right.rows() && left.cols() == right.cols();
            if(equal)
            { 
                for(std::int64_t row = 0; row < left.rows(); row++)
                {
                    for(std::int64_t col = 0; col < left.cols(); col++)
                    {
                        if(std::abs(left(row, col) - right(row, col)) > tolerance)
                        {
                            LOG(trace) << "matrix differs at " << row << "," << col;
                            #ifndef NDEBUG
                            equal = false;
                            #else
                            return false;
                            #endif
                        }
                    }
                }
            }
            return equal;
        }
    }
}