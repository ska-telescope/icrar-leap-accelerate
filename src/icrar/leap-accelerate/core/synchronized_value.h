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

//#include <boost/thread/synchronized_value.hpp>
#include <iostream>
#include <mutex>

namespace icrar
{
    /**
     * @brief A data-mutex pairing that locks automatically
     * when constructed. Instances of this class can mutate data
     * with thread safety by using the scope of the synchronized
     * object as a unique lock 
     * 
     * @tparam T 
     */
    template<typename T,
            std::enable_if_t<std::is_reference<T>::value, bool> = true>
    class synchronized_value
    {
        T m_data;
        std::unique_lock<std::mutex> m_lock;

    public:
        synchronized_value(synchronized_value<T>&& other)
        : m_data(other.m_data)
        , m_lock(std::move(other.m_lock))
        {
        }

        synchronized_value(std::pair<T, std::mutex&>&& other)
        : m_data(other.first)
        , m_lock(other.second)
        {}

        synchronized_value(T data, std::mutex& mutex)
        : m_data(data)
        , m_lock(mutex)
        {}

        /**
         * @brief Gets the raw data member. Do not create copies
         * of this reference.
         * 
         * @return T 
         */
        T get() { return m_data; }

        /**
         * @brief Gets the raw data member. Do not create copies
         * of this reference.
         * 
         * @return T 
         */
        const T get() const { return m_data; }
    };
} // namespace icrar
