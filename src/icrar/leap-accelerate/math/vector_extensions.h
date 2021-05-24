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

#pragma once

#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <vector>
#include <iostream>
#include <vector>
#include <functional>
#include <type_traits>

namespace icrar
{
    /**
     * @brief returns a linear sequence of values from start at step sized
     * intervals to the stop value inclusive
     * 
     * @tparam IntType 
     * @param start 
     * @param stop 
     * @param step 
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType start, IntType stop, IntType step)
    {
        if (step == IntType(0))
        {
            throw std::invalid_argument("step for range must be non-zero");
        }

        std::vector<IntType> result;
        IntType i = start;
        while ((step > 0) ? (i < stop) : (i > stop))
        {
            result.push_back(i);
            i += step;
        }

        return result;
    }

    /**
     * @brief 
     * 
     * @tparam IntType 
     * @param start 
     * @param stop 
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType start, IntType stop)
    {
        return range(start, stop, IntType(1));
    }

    /**
     * @brief 
     * 
     * @tparam IntType 
     * @param stop 
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType stop)
    {
        return range(IntType(0), stop, IntType(1));
    }

    /**
     * @brief Computes the sum of the the collection
     * 
     * @tparam T 
     * @param v 
     * @return double 
     */
    template<typename T>
    double sum(const std::vector<T>& v)
    {
        return std::accumulate(v.begin(), v.end(), 0);
    }

    /**
     * @brief Computes the mean of the the collection
     * 
     * @tparam T 
     * @param v 
     * @return double 
     */
    template<typename T>
    double mean(const std::vector<T>& v)
    {
        double sum = std::accumulate(v.begin(), v.end(), 0);
        return sum / v.size();
    }

    /**
     * @brief Computes the standard deviation of the collection
     * 
     * @tparam T 
     * @param v 
     * @return double 
     */
    template<typename T>
    double standard_deviation(const std::vector<T>& v)
    {
        double mean = std::accumulate(v.begin(), v.end(), 0);
        double sumOfSquareDifferences = 0;
        for(const double& e : v)
        {
            sumOfSquareDifferences += std::pow(e - mean, 2);
        }
        return std::sqrt(sumOfSquareDifferences / v.size());
    }

    /**
     * @brief Returns of true if all vector elements of @p lhs are within the tolerance
     * threshold difference to @p rhs
     * 
     * @tparam T numerically comparable type
     * @param lhs left collection
     * @param rhs right collection
     * @param tolerance numeric tolerance inclusive threshold for equality
	**/
    template<typename T>
    bool isApprox(const std::vector<T>& lhs, const std::vector<T>& rhs, T tolerance)
    {
        if(lhs.size() != rhs.size())
        {
            return false;
        }
        for(size_t i = 0; i < lhs.size(); ++i)
        {
            if(std::abs(lhs[i] - rhs[i]) >= tolerance)
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Performs a std::transform on a newly allocated std::vector 
     * 
     * @tparam T The input vector template type
     * @tparam Op function of signature R(const T&)
     * @param vector vector to perform an element-wise lambda transfomation to
     * @param lambda operation to perform for each element
     * @return std::vector<R> 
     */
    template<typename T, typename Op>
    std::vector<std::result_of_t<Op(const T&)>> vector_map(const std::vector<T>& vector, Op lambda)
    {
        using R = std::result_of_t<Op(const T&)>;
        static_assert(std::is_assignable<std::function<R(const T&)>, Op>::value, "lambda argument must be a function of signature R(const T&)");

        auto result = std::vector<R>();
        result.reserve(vector.size());
        std::transform(vector.cbegin(), vector.cend(), std::back_inserter(result), lambda);
        return result;
    }
} // namespace icrar
