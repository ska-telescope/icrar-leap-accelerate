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
     * @brief Represents a linear sequence of indexes for some arbitrary collection
     * 
     */
    struct Slice
    {
        std::int32_t start;
        std::int32_t interval;
        std::int32_t end;

        Slice() = default;
        Slice(int interval);
        Slice(int start, int end);
        Slice(int start, int interval, int end);
        
        Range Evaluate(int collectionSize) const
        {
            return Range(
                (start == -1) ? collectionSize : start,
                (interval == -1) ? collectionSize : interval,
                (end == -1) ? collectionSize : end
            );
        }
    };

    Slice ParseSlice(const std::string& json);

    Slice ParseSlice(const rapidjson::Value& doc);
} // namespace icrar
