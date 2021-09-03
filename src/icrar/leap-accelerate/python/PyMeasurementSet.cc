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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <future>

// namespace np = boost::python::numpy;
// namespace bp = boost::python;

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
    : m_measurementSet(std::make_shared<MeasurementSet>(msPath))
    {
    }

    Eigen::Tensor<double, 3> PyMeasurementSet::ReadCoords(
        std::uint32_t startTimestep,
        std::uint32_t intervalTimesteps)
    {
        const auto coords = m_measurementSet->ReadCoords(startTimestep, intervalTimesteps);
        Eigen::DSizes<long, 3> shape = coords.dimensions();
        Eigen::DSizes<long, 3> offset = Eigen::DSizes<long, 3>();
        //bp::object owner;
        //return np::from_data(coords.data(), np::dtype::get_builtin<decltype(*coords.data())>(), shape, offset, owner);
        return coords;
    }

    Eigen::Tensor<std::complex<double>, 4> PyMeasurementSet::ReadVis(
        std::uint32_t startTimestep,
        std::uint32_t intervalTimesteps)
    {
        const auto vis = m_measurementSet->ReadVis(0, 1, Slice(0, boost::none, 1));

        // convert to numpy row-major order (boost::numpy column-major not supported)
        auto shape = Eigen::DSizes<long, 4> { vis.dimension(3), vis.dimension(2), vis.dimension(1), vis.dimension(0) };
        auto strides = Eigen::DSizes<long, 4>
        { 
            vis.dimension(2) * vis.dimension(1) * vis.dimension(0) * sizeof(double),
            vis.dimension(1) * vis.dimension(0) * sizeof(double),
            vis.dimension(0) * sizeof(double),
            sizeof(double)
        };
        return vis;
        //return np::from_data(vis.data(), np::dtype::get_builtin<std::complex<double>>(), shape, strides, bp::object()).copy();
    }
} // namespace python
} // namespace icrar

#endif // PYTHON_ENABLED
