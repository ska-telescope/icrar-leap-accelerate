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

#include "SphericalDirection.h"

#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/exception/exception.h>
#include <rapidjson/document.h>

namespace icrar
{
    std::vector<SphericalDirection> ParseDirections(const std::string& json)
    {
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        return ParseDirections(doc);
    }

    std::vector<SphericalDirection> ParseDirections(const rapidjson::Value& doc)
    {
        //Validate Schema
        if(!doc.IsArray())
        {
            throw icrar::exception("expected an array", __FILE__, __LINE__);
        }
        
        //Parse
        auto result = std::vector<SphericalDirection>();
        for(auto it = doc.Begin(); it != doc.End(); it++)
        {
            if(!it->IsArray())
            {
                throw icrar::exception("expected an array of 2 numbers", __FILE__, __LINE__);
            }
            if(it->Size() != 2)
            {
                throw icrar::exception("expected an array of 2 numbers", __FILE__, __LINE__);
            }

            auto& array = *it;
            result.emplace_back(array[0].GetDouble(), array[1].GetDouble());
        }
        return result;
    }
} // namespace icrar
