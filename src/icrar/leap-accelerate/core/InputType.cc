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

#include "InputType.h"
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
    InputType ParseInputType(const std::string& value)
    {
        InputType e;
        if(!TryParseInputType(value, e))
        {
            throw invalid_argument_exception(value, "value", __FILE__, __LINE__);
        }
        return e;
    }

    bool TryParseInputType(const std::string& value, InputType& out)
    {
        bool handled = false;
        if(value == "f" || value == "file")
        {
            out = InputType::file;
            handled = true;
        }
        else if(value == "s" || value == "stream")
        {
            out = InputType::stream;
            handled = true;
        }
        return handled;
    }
} // namespace icrar
