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

#include "Range.h"
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
    Range::Range(int interval)
    : Range(0, interval, -1)
    {}

    Range::Range(int start, int end)
    : Range(start, end == -1 ? -1 : end - start, end)
    {}

    Range::Range(int start, int interval, int end)
    {
        if(start < -1) throw icrar::exception("expected a positive integer or -1", __FILE__, __LINE__);
        if(interval < -1) throw icrar::exception("expected a positive integer or -1", __FILE__, __LINE__);
        if(interval == 0) throw icrar::exception("expected a non zero integer", __FILE__, __LINE__);
        if(end < -1) throw icrar::exception("expected a positive integer or -1", __FILE__, __LINE__);

        //forward sequences only
        if(end != -1 && start >= end)
        {
            throw icrar::exception("range start must be greater than end", __FILE__, __LINE__);
        }
        if(interval > (end - start))
        {
            throw icrar::exception("range increment out of bounds", __FILE__, __LINE__);
        }
        if(interval == -1)
        {
            interval = end - start;
        }

        this->start = start;
        this->interval = interval;
        this->end = end;
    }

    Range ParseRange(const std::string& json)
    {
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        return ParseRange(doc);
    }

    Range ParseRange(const rapidjson::Value& doc)
    {
        //Validate Schema
        if(!doc.IsArray())
        {
            throw icrar::exception("expected an array", __FILE__, __LINE__);
        }

        if(doc.Size() != 3)
        {
            throw icrar::exception("expected 3 integers", __FILE__, __LINE__);
        }

        return Range(doc[0].GetInt(), doc[1].GetInt(), doc[2].GetInt());
    }
} // namespace icrar
