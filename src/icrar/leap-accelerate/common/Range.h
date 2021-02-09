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
#include <icrar/leap-accelerate/exception/exception.h>
#include <string>
#include <stdint.h>

namespace icrar
{
    /**
     * @brief Represents a forwards linear sequence of indexes for some finite collection
     * 
     */
    class Range
    {
        std::uint32_t m_start;
        std::uint32_t m_interval;
        std::uint32_t m_end;

    public:
        Range(int start, int interval, int end)
        {
            if(start < 0) throw icrar::exception("expected a positive integer", __FILE__, __LINE__);
            if(interval < 1) throw icrar::exception("expected a positive integer", __FILE__, __LINE__);
            if(end < 0) throw icrar::exception("expected a positive integer", __FILE__, __LINE__);
            if(start > end) throw icrar::exception("range start must be less than end", __FILE__, __LINE__);

            m_start = start;
            m_interval = interval;
            m_end = end;
        }

        uint32_t GetStart() const { return m_start; }
        uint32_t GetInterval() const { return m_interval; }
        uint32_t GetEnd() const { return m_end; }

        /**
         * @brief Gets the number of elements in the range
         * 
         * @return int 
         */
        int GetSize() const
        {
            return (m_end - m_start) / m_interval;
        }
    };
} // namespace icrar
