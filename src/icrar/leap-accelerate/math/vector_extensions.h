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
#include <numeric>
#include <iostream>
#include <vector>
#include <functional>
#include <type_traits>

/**
 * @brief Provides stream operator for std::vector as
 * a json-like literal.
 * 
 * @tparam T streamable type
 * @param os output stream
 * @param v vector
 * @return std::ostream& 
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "{";
    for (size_t i = 0; i < v.size(); ++i)
    { 
        os << v[i]; 
        if (i != v.size() - 1)
        { 
            os << ", ";
        }
    } 
    os << "}\n"; 
    return os;
}

namespace icrar
{
    /**
     * @brief returns a linear sequence of values from start at step sized
     * intervals to the stop value inclusive
     * 
     * @tparam IntType integer type
     * @param start start index
     * @param stop exclusive end inex
     * @param step increment between generated elements
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType start, IntType stop, IntType step)
    {
        if (step == IntType(0))
        {
            throw std::invalid_argument("step must be non-zero");
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
     * @brief returns a linear sequence of values from start to stop
     * 
     * @tparam IntType integer type
     * @param start start index
     * @param stop exclusive end index
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType start, IntType stop)
    {
        return range(start, stop, IntType(1));
    }

    /**
     * @brief returns a linear sequence of values from 0 to stop
     * 
     * @tparam IntType integer type
     * @param stop exclusive end index
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType stop)
    {
        return range(IntType(0), stop, IntType(1));
    }

    /**
     * @brief Returns true if all vector elements of @p lhs are within the
     * tolerance threshold to @p rhs
     * 
     * @tparam T numeric type
     * @param lhs left hand side
     * @param rhs  right hand side
     * @param tolerance tolerance threshold
     */
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
     * @brief Performs a std::transform into a newly allocated std::vector
     * 
     * @tparam T The input vector template type
     * @tparam Op function of signature R(const T&)
     * @param vector vector to transform
     * @param lambda transformation of signature R(const T&)
     * @return std::vector<std::result_of_t<Op(const T&)>>
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
