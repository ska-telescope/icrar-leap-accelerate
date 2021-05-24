/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#pragma once

#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/memory/ioutils.h>
#include <icrar/leap-accelerate/exception/exception.h>
#include <Eigen/Core>
#include <boost/numeric/conversion/cast.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <functional>
#include <type_traits>

namespace icrar
{
    namespace cpu
    {
        template<typename Matrix>
        Eigen::IndexedView<Matrix, Eigen::VectorXi, Eigen::internal::AllRange<-1>>
        WrappedRangeSelect(
            Eigen::MatrixBase<Matrix>& matrix,
            const Eigen::VectorXi& rowIndices)
        {
            Eigen::VectorXi correctedIndices = rowIndices;

            // wrap around
            for(int& i : correctedIndices)
            {
                if(i < 0)
                {
                    i = boost::numeric_cast<int>(i % matrix.rows());
                }
            }

            return matrix(correctedIndices, Eigen::all);
        }

        /**
         * @brief Selects a range of elements from matrix row indices and column index.
         * Negative indexes select from the bottom of the matrix with -1 representing the bottom row.
         * 
         * @tparam T 
         * @param matrix the referenced matrix to select from
         * @param rowIndices a range of row indices to select
         * @param column a valid column index 
         */
        template<typename Matrix>
        Eigen::IndexedView<Matrix, Eigen::VectorXi, Eigen::internal::SingleRange>
        VectorRangeSelect(
            Eigen::MatrixBase<Matrix>& matrix,
            const Eigen::VectorXi& rowIndices,
            unsigned int column)
        {
            Eigen::VectorXi correctedIndices = rowIndices;

            // wrap around
            for(int& i : correctedIndices)
            {
                if(i < 0)
                {
                    i = boost::numeric_cast<int>(i % matrix.rows());
                }
            }

            return matrix(correctedIndices, column);
        }

        /**
         * @brief Selects a range of elements from matrix row indices. Negative indexes
         * select from the bottom of the matrix with -1 representing the bottom row.
         * 
         * @tparam T 
         * @param matrix the referenced matrix to select from
         * @param rowIndices a range of row indices to select
         * @param column a valid column index 
         */
        template<typename Matrix>
        Eigen::IndexedView<Matrix, Eigen::VectorXi, Eigen::internal::AllRange<-1>>
        MatrixRangeSelect(
            Matrix& matrix,
            const Eigen::VectorXi& rowIndices)
        {
            Eigen::VectorXi correctedIndices = rowIndices;
            for(int& i : correctedIndices)
            {
                if(i < 0)
                {
                    i = boost::numeric_cast<int>(i % matrix.rows());
                }
            }

            return matrix(correctedIndices, Eigen::all);
        }

        /**
         * @brief Computes the element-wise standard deviation
         * 
         * @tparam T 
         * @param matrix 
         * @return double 
         */
        template<typename T>
        double standard_deviation(const Eigen::MatrixBase<T>& matrix)
        {
            double mean = matrix.sum() / (double)matrix.size();
            double sumOfSquareDifferences = 0;
            for(const double& e : matrix.reshaped())
            {
                sumOfSquareDifferences += std::pow(e - mean, 2);
            }
            return std::sqrt(sumOfSquareDifferences / (double)matrix.size());
        }

        /**
         * @brief Returns the component-wise arguments of a matrix
         * 
         * @param a 
         * @return Eigen::MatrixXd 
         */
        Eigen::MatrixXd arg(const Eigen::Ref<const Eigen::MatrixXcd>& a);

        /**
         * @brief Performs an elementwise comparison between matrices and returns
         * false if the absolute difference exceeds the tolerance.
         * 
         * @param left 
         * @param right 
         * @param tolerance 
         */
        bool near(const Eigen::Ref<const Eigen::MatrixXd> left, const Eigen::Ref<const Eigen::MatrixXd> right, double tolerance);
    }
} // namespace icrar
