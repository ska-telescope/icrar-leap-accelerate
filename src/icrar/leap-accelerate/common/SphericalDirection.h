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

#include <Eigen/Core>
#include <vector>

// NOTE: nvcc 11.4 no longer includes rapidjson successfully. Forward
// declarations are taken from rapidjson headers.
namespace rapidjson
{
    class CrtAllocator;
    template <typename BaseAllocator>
    class MemoryPoolAllocator;
    template <typename Encoding, typename Allocator>
    class GenericValue;
    template<typename CharType>
    struct UTF8;
}

namespace icrar
{
    using SphericalDirection = Eigen::Vector2d;

    /**
     * @brief Parses a json string to a collection of MVDirections
     * 
     * @param json 
     * @return std::vector<SphericalDirection> 
     */
    std::vector<SphericalDirection> ParseDirections(const std::string& json);

    /**
     * @brief Parses a json object to a collection of MVDirections
     * 
     */
    std::vector<SphericalDirection> ParseDirections(
        const rapidjson::GenericValue<rapidjson::UTF8<char>, rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator>>& doc);
} // namespace icrar
