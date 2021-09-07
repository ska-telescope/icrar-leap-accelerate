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

#include "async.h"

#include <Eigen/Core>

#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <future>
#include <string>

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
    //TODO: see pybind11/functional.h for simple type caster to
    // convert to pytypes
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
        // TODO: this appears to do a copy and not provide a view
        // or take pointer ownership, use capsules for this
        // https://github.com/pybind/pybind11/issues/1042#issuecomment-325941022
        .def_property_readonly("numpy_view", [](Eigen::Tensor<Scalar, Dims>& t) {
            
            // pybind11 already wraps the lifetime of class instances. Capsule
            // is required for python to know the memory will not go out of scope
            auto capsule = py::capsule(&t, [](void *p) {
                //delete reinterpret_cast<Eigen::Tensor<Scalar, Dims>*>(p);
            });
            return py::array_t<Scalar, py::array::f_style>(t.dimensions(), t.data(), capsule);
        });
}

PYBIND11_MODULE(LeapAccelerate, m)
{
    m.doc() = "Linear Execision of the Atmosphere in Parallel";
    
    // py::class_<std::future<void>>(m, "Future")
    //     .def(py::init<>())
    //     .def("done", [](const std::future<void>& f) -> bool {
    //         return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    //     })
    //     .def("wait", [](const std::future<void>& f) -> void {
    //         f.wait();
    //     })
    //     .def("result", [](std::future<void>& f) -> void {
    //         f.get();
    //     });

    // py::class_<py::Awaitable<void>, std::shared_ptr<py::Awaitable<void>>>(m, "Awaitable")
    //     .def(py::init<>())
    //     .def("__iter__", &py::Awaitable<void>::__iter__)
    //     .def("__await__", &py::Awaitable<void>::__await__)
    //     .def("__next__", &py::Awaitable<void>::__next__);
    py::async::enable_async(m);

    // See https://stackoverflow.com/questions/39995149/expand-a-type-n-times-in-template-parameter
    // for automatically generating parameter packs (requires a wrapper type)
    PybindEigenTensor<double, 3, int, int, int>(m, "Tensor3d");
    PybindEigenTensor<double, 4, int, int, int, int>(m, "Tensor4d");
    PybindEigenTensor<std::complex<double>, 3, int, int, int>(m, "Tensor3cd");
    PybindEigenTensor<std::complex<double>, 4, int, int, int, int>(m, "Tensor4cd");

    py::class_<icrar::cpu::BeamCalibration>(m, "BeamCalibration")
        .def(py::init<Eigen::Vector2d, Eigen::MatrixXd>());

    py::class_<icrar::cpu::Calibration>(m, "Calibration")
        .def(py::init<int, int>());

    py::enum_<icrar::ComputeImplementation>(m, "ComputeImplementation")
        .value("cpu", icrar::ComputeImplementation::cpu)
        .value("cuda", icrar::ComputeImplementation::cuda)
        .export_values();

    // def_async extension on available on class_async, need to cast def() return type to move elsewhere
    // or integrate into 
    py::async::class_async<icrar::python::PyLeapCalibrator>(m, "LeapCalibrator")
        .def_async("calibrate_async", &icrar::python::PyLeapCalibrator::PythonCalibrateAsync,
            py::arg("ms"),
            py::arg("directions").noconvert(),
            py::arg("solution_interval")=py::slice(0,1,1),
            py::arg("callback")
        )
        .def(py::init<icrar::ComputeImplementation>())
        .def(py::init<std::string>())
        .def("calibrate", &icrar::python::PyLeapCalibrator::PythonCalibrate,
            py::arg("ms_path"),
            py::arg("directions").noconvert(),
            py::arg("solution_interval")=py::slice(0,1,1),
            py::arg("output_path")
        )
        .def("calibrate_callback", &icrar::python::PyLeapCalibrator::PythonCalibrateAsync,
            py::arg("ms"),
            py::arg("directions").noconvert(),
            py::arg("solution_interval")=py::slice(0,1,1),
            py::arg("callback")
        )
        .def("calibrate_async2", &icrar::python::PyLeapCalibrator::PythonCalibrateAsync2,
            py::arg("ms"),
            py::arg("directions").noconvert(),
            py::arg("solution_interval")=py::slice(0,1,1),
            py::arg("callback")
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
            //py::arg("polarizationSlice")=py::slice(0,-1,1)
        );
}

#endif // PYTHON_ENABLED