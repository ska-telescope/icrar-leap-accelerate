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

#include "PyLeapCalibrator.h"
#include "PyMeasurementSet.h"

// #include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

//namespace np = boost::python::numpy;
//namespace bp = boost::python;

namespace py = pybind11;

PYBIND11_MODULE(LeapAccelerate, m)
{
    m.doc() = "Linear Execision of the Atmosphere in Parallel";
    
    py::enum_<icrar::ComputeImplementation>(m, "compute_implementation")
        .value("cpu", icrar::ComputeImplementation::cpu)
        .value("cuda", icrar::ComputeImplementation::cuda)
        .export_values();

    py::class_<icrar::python::PyLeapCalibrator>(m, "LeapCalibrator")
        .def(py::init<icrar::ComputeImplementation>())
        .def(py::init<std::string>())
        .def("calibrate", &icrar::python::PyLeapCalibrator::PythonCalibrate,
            py::arg("ms_path"),
            py::arg("directions").noconvert(),
            py::arg("solution_interval")=py::slice(0,1,1),
            py::arg("output_path")
        );

    py::class_<icrar::python::PyMeasurementSet>(m, "MeasurementSet")
        .def(py::init<std::string>())
        .def("read_coords", &icrar::python::PyMeasurementSet::ReadCoords,
            py::arg("start_timestep"),
            py::arg("num_timesteps")
        )
        .def("read_vis", &icrar::python::PyMeasurementSet::ReadVis,
            py::arg("start_timestep"),
            py::arg("num_timesteps")
        );
}

/**
BOOST_PYTHON_MODULE(LeapAccelerate)
{
    bp::numpy::initialize();

    bp::class_<icrar::python::PyLeapCalibrator>("LeapCalibrator", bp::init<icrar::ComputeImplementation>())
        .def(bp::init<std::string>())
        .def("calibrate", &icrar::python::PyLeapCalibrator::PythonCalibrate, (
            bp::arg("ms_path"),
            bp::arg("directions"),
            bp::arg("solution_interval")=bp::slice(0,1,1),
            bp::arg("output_path")=bp::object()
        ));
        //.def("async_calibrate")

    bp::class_<icrar::python::PyMeasurementSet>("MeasurementSet", bp::init<std::string>())
        .def("read_coords", &icrar::python::PyMeasurementSet::ReadCoords, (
            bp::arg("start_timestep"),
            bp::arg("num_timesteps")
        ))
        .def("read_vis", &icrar::python::PyMeasurementSet::ReadVis, (
            bp::arg("start_timestep"),
            bp::arg("num_timesteps")
        ));

    bp::enum_<icrar::ComputeImplementation>("compute_implementation")
        .value("cpu", icrar::ComputeImplementation::cpu)
        .value("cuda", icrar::ComputeImplementation::cuda);
}
**/
#endif // PYTHON_ENABLED