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

/// See http://eigen.tuxfamily.org/dox-3.2/TopicCustomizingEigen.html
/// for details on extending Eigen3.

/**
 * @brief A numpythonic row selection operation that selects the rows
 * of a matrix using index wrap around. Negative indexes select from
 * the bottom of the matrix with -1 representing the last row.
 * 
 * @tparam Index
 * @param rowIndices a range of row indices to select
 * @param column a valid column index
 */
template<typename Index>
inline auto wrapped_row_select(const Matrix<Index, Eigen::Dynamic, 1>& rowIndices)
{
    Matrix<Index, Eigen::Dynamic, 1> correctedIndices = rowIndices;
    for(Index& index : correctedIndices)
    {
        if(index < -rows() || index >= rows())
        {
            throw std::runtime_error("index out of range");
        }
        if(index < 0)
        {
            index = rows() + index;
        }
    }
    return this->operator()(correctedIndices, Eigen::all);
}

template<typename Index>
inline auto wrapped_row_select(const Matrix<Index, Eigen::Dynamic, 1>& rowIndices) const
{
    Matrix<Index, Eigen::Dynamic, 1> correctedIndices = rowIndices;
    for(Index& index : correctedIndices)
    {
        if(index < -rows() || index >= rows())
        {
            throw std::runtime_error("index out of range");
        }
        if(index < 0)
        {
            index = rows() + index;
        }
    }
    return this->operator()(correctedIndices, Eigen::all);
}

/**
 * @brief Computes the element-wise standard deviation
 * 
 * @return Scalar 
 */
double standard_deviation() const
{
    double mean = this->sum() / static_cast<double>(size());
    double sumOfSquareDifferences = 0;
    for(const Scalar& e : this->reshaped())
    {
        sumOfSquareDifferences += std::pow(e - mean, 2);
    }
    return std::sqrt(sumOfSquareDifferences / static_cast<double>(size()));
}

/**
 * @brief Performs elementwise comparison of matrix elements to determine
 * near equality within the specified threshold.
 * 
 * @tparam OtherDerived 
 * @param other 
 * @param tolerance 
 * @return true 
 * @return false 
 */
template<typename OtherDerived>
inline bool near(const MatrixBase<OtherDerived>& other, double tolerance) const
{
    bool equal = rows() == other.rows() && cols() == other.cols();
    if(equal)
    { 
        for(std::int64_t row = 0; row < rows(); row++)
        {
            for(std::int64_t col = 0; col < cols(); col++)
            {
                if(std::abs(this->operator()(row, col) - other(row, col)) > tolerance)
                {
                    return false;
                }
            }
        }
    }
    return equal;
}

/**
 * @brief Computes a matrix of the component-wise angles/args from the respective complex values
 */
inline auto arg() const { return this->unaryExpr([](Scalar v){ return std::arg(v); }); }