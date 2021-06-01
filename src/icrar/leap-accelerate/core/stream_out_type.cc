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

#include "stream_out_type.h"
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
    StreamOutType ParseStreamOutType(const std::string& value)
    {
        StreamOutType e;
        if(!TryParseStreamOutType(value, e))
        {
            throw invalid_argument_exception(value, "value", __FILE__, __LINE__);
        }
        return e;
    }

    bool TryParseStreamOutType(const std::string& value, StreamOutType& out)
    {
        if(value == "c" || value == "collection")
        {
            out = StreamOutType::collection;
            return true;
        }
        else if(value == "s" || value == "single")
        {
            out = StreamOutType::singleFile;
            return true;
        }
        else if(value == "m" || value == "multiple")
        {
            out = StreamOutType::multipleFiles;
            return true;
        }
        return false;
    }

    bool IsImmediateMode(StreamOutType streamOutType)
    {
        switch(streamOutType)
        {
            case StreamOutType::collection:
                return false;
            case StreamOutType::singleFile:
                return true;
            case StreamOutType::multipleFiles:
                return true;
            default:
                throw std::invalid_argument("invalid stream out type");
        }
    }
} // namespace icrar
