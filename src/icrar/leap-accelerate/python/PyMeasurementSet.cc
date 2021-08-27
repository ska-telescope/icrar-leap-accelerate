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

#if PYTHON_ENABLED

#include "PyMeasurementSet.h"

#include <future>

namespace np = boost::python::numpy;
namespace bp = boost::python;

template<typename T>
inline T ternary(bool condition, T trueValue, T falseValue)
{
    return condition ? trueValue : falseValue;
}

namespace icrar
{
namespace python
{
    PyMeasurementSet::PyMeasurementSet(std::string msPath)
    {
        
    }

    np::ndarray PyMeasurementSet::ReadCoords(
        std::uint32_t startTimestep,
        std::uint32_t intervalTimesteps)
    {

    }

    np::ndarray PyMeasurementSet::ReadVis(
        std::uint32_t startTimestep,
        std::uint32_t intervalTimesteps)
    {

    }
} // namespace python
} // namespace icrar

#endif // PYTHON_ENABLED