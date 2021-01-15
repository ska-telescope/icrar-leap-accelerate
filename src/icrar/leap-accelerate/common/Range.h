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
#include <rapidjson/document.h>
#include <string>
#include <stdint.h>

namespace icrar
{
    // template<typename Rows, typename Numeric>
    // struct PositionVector
    // {
    //     Eigen::Vector<Rows, 1, Numeric> Position;
    //     Eigen::Vector<Rows, 1, Numeric> Direction;
    // }
    // using PositionVector1d = PositionVector<1,double>;
    // using PositionVector2d = PositionVector<2,double>;
    // using PositionVector3d = PositionVector<3,double>;

    struct Range
    {
        std::int32_t start;
        std::int32_t interval;
        std::int32_t end;

        Range() = default;
        Range(int interval);
        Range(int start, int end);
        Range(int start, int interval, int end);
    };

    Range ParseRange(const std::string& json);

    Range ParseRange(const rapidjson::Value& doc);
} // namespace icrar
