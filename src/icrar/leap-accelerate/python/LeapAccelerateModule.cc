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
#include "PyTensor.h"

// #include <Eigen/Core>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

template<typename Scalar, size_t Dims>
std::vector<long int> DimensionsVector(const typename Eigen::DSizes<long int, Dims>& dimensions)
{
    std::vector<long int> result;
    result.assign(dimensions.begin(), dimensions.end());
    return result;
}

/**
 * @brief Creates a class binding for an Eigen Tensor template. Supports both
 * buffer protocol and eigen array wrappers to python types.
 * 
 * @tparam Scalar scalar datatype
 * @tparam Dims number of dimensions
 * @tparam InitArgs constructor argument types
 * @param m module
 * @param name class name
 */
template<typename Scalar, size_t Dims, typename... InitArgs>
void PybindEigenTensor(py::module& m, const char* name)
{
    py::class_<Eigen::Tensor<Scalar, Dims>>(m, name, py::buffer_protocol())
        .def(py::init<InitArgs...>())
        .def_buffer([](Eigen::Tensor<Scalar, Dims>& t) -> py::buffer_info {
            const auto shape = DimensionsVector<Scalar, Dims>(t.dimensions());
            return py::buffer_info(
                t.data(),
                sizeof(Scalar),
                py::format_descriptor<Scalar>::format(),
                Dims,
                shape,
                py::detail::f_strides(shape, sizeof(Scalar))
            );
        })
        .def_property_readonly("numpy_view", [](Eigen::Tensor<Scalar, Dims>& t) {
            return py::array_t<Scalar, py::array::f_style>(t.dimensions(), t.data());
        });
}

PYBIND11_MODULE(LeapAccelerate, m)
{
    m.doc() = "Linear Execision of the Atmosphere in Parallel";
    
    // See https://stackoverflow.com/questions/39995149/expand-a-type-n-times-in-template-parameter
    // for automatically generating parameter packs (requires a wrapper type)
    PybindEigenTensor<double, 3, int, int, int>(m, "Tensor3d");
    PybindEigenTensor<double, 4, int, int, int, int>(m, "Tensor4d");
    PybindEigenTensor<std::complex<double>, 3, int, int, int>(m, "Tensor3cd");
    PybindEigenTensor<std::complex<double>, 4, int, int, int, int>(m, "Tensor4cd");

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
    
    m.def("create_matrix", []()
    {
        return Eigen::MatrixXd::Zero(5,5);
    });

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

#endif // PYTHON_ENABLED