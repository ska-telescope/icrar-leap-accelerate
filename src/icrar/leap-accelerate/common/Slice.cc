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

#include "Slice.h"
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
    Slice::Slice(int interval)
    : Slice(0, interval, -1)
    {}

    Slice::Slice(int start, int end)
    : Slice(start, end == -1 ? -1 : end - start, end)
    {}

    Slice::Slice(int start, int interval, int end)
    {
        if(start < -1) throw icrar::exception("expected a positive integer start", __FILE__, __LINE__);
        if(interval < -1) throw icrar::exception("expected a positive integer or -1 interval", __FILE__, __LINE__);
        if(end < -1) throw icrar::exception("expected a positive or -1 integer end", __FILE__, __LINE__);

        //forward sequences only
        if(end != -1)
        {
            if(start == -1)
            {
                throw icrar::exception("range start must be greater than end", __FILE__, __LINE__);
            }
            if(start > end)
            {
                throw icrar::exception("range start must be greater than end", __FILE__, __LINE__);
            }
            if(interval == -1)
            {
                interval = end - start;
            }
            if(interval > (end - start))
            {
                throw icrar::exception("range increment out of bounds", __FILE__, __LINE__);
            }
        }
        if(interval == 0) throw icrar::exception("expected a non zero integer interval", __FILE__, __LINE__);

        m_start = start;
        m_interval = interval;
        m_end = end;
    }

    Range Slice::Evaluate(int collectionSize) const
    {
        return Range
        {
            (m_start == -1) ? collectionSize : m_start,
            (m_interval == -1) ? collectionSize : m_interval,
            (m_end == -1) ? collectionSize : m_end
        };
    }

    Slice ParseSlice(const std::string& json)
    {
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        return ParseSlice(doc);
    }

    Slice ParseSlice(const rapidjson::Value& doc)
    {
        Slice result = {};

        //Validate Schema
        if(doc.IsInt())
        {
            result = Slice(doc.GetInt());
        }
        else if(doc.IsArray())
        {
            if(doc.Size() == 2)
            {
                result = Slice(doc[0].GetInt(), doc[1].GetInt());
            }
            if(doc.Size() == 3)
            {
                result = Slice(doc[0].GetInt(), doc[1].GetInt(), doc[2].GetInt());
            }
            else
            {
                throw icrar::json_exception("expected 3 integers", __FILE__, __LINE__);
            }
            
        }
        else
        {
            throw icrar::json_exception("expected an integer or array of integers", __FILE__, __LINE__);
        }

        return result;
    }
} // namespace icrar
