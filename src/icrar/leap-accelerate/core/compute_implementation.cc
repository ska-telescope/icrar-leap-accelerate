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

#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/core/log/logging.h>

namespace icrar
{
    std::string ComputeImplementationToString(ComputeImplementation value)
    {
        switch(value)
        {
            case ComputeImplementation::cpu:
                return "cpu";
            case ComputeImplementation::cuda:
                return "cuda";
            default:
                throw invalid_argument_exception("ComputeImplementation", "value", __FILE__, __LINE__);
                return "";
        }
    }

    ComputeImplementation ParseComputeImplementation(const std::string& value)
    {
        ComputeImplementation e;
        if(!TryParseComputeImplementation(value, e))
        {
            throw invalid_argument_exception(value, "value", __FILE__, __LINE__);
        }
        return e;
    }

    bool TryParseComputeImplementation(const std::string& value, ComputeImplementation& out)
    {
        if(value == "casa")
        {
            LOG(warning) << "argument 'casa' deprecated, use 'cpu' instead";
            out = ComputeImplementation::cpu;
            return true;
        }
        else if(value == "eigen")
        {
            LOG(warning) << "argument 'eigen' deprecated, use 'cpu' instead";
            out = ComputeImplementation::cpu;
            return true;
        }
        else if(value == "cpu")
        {
            out = ComputeImplementation::cpu;
            return true;
        }
        else if(value == "cuda")
        {
            out = ComputeImplementation::cuda;
            return true;
        }
        return false;
    }
} // namespace icrar