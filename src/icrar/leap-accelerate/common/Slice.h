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
#include <icrar/leap-accelerate/common/Range.h>
#include <icrar/leap-accelerate/exception/exception.h>
#include <rapidjson/document.h>
#include <string>
#include <stdint.h>

namespace icrar
{
    /**
     * @brief Represents a forwards linear sequence of indexes for some arbitrary collection.
     * Python equivalent is the slice operator [start:end:interval].
     * Eigen equivalent is Eigen::seq(start, end, interval).
     * Matlab equivalent is slice operator (start:interval:end)
     * TODO(cgray) no support for reverse order, e.g. (end:-1:0)
     * TODO(calgray): swap end <-> interval
     */
    class Slice
    {
        std::int32_t m_start;
        std::int32_t m_interval;
        std::int32_t m_end;

    public:
        Slice() = default;
        Slice(int interval);
        Slice(int start, int end);
        Slice(int start, int interval, int end);
        
        /**
         * @brief Gets the starting index of an arbitrary collection slice. -1 represents the end of the collection.
         */
        int32_t GetStart() const { return m_start; }

        /**
         * @brief Gets the interval betweeen indices of an arbitrary collection slice. -1 represents the end of the collection. 
         */
        int32_t GetInterval() const { return m_interval; }

        /**
         * @brief Gets the end exclusive index of an arbitrary collection slice. -1 represents the end of the collection.
         */
        int32_t GetEnd() const { return m_end; }

        Range Evaluate(int collectionSize) const;
    };

    Slice ParseSlice(const std::string& json);

    Slice ParseSlice(const rapidjson::Value& doc);
} // namespace icrar
