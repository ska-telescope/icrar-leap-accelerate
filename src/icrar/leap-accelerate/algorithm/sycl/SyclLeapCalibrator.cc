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

#ifdef SYCL_ENABLED

#include "SyclLeapCalibrator.h"

#include <icrar/leap-accelerate/common/eigen_stringutils.h>

#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>
#include <icrar/leap-accelerate/algorithm/cpu/CpuComputeOptions.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cpu/MetaData.h>
#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>

#include <CL/sycl.hpp>
// #include <ml/eigen/eigen.hpp>
// #include <ml/math/mat_mul.hpp>
// #include <ml/math/svd.hpp>
#include <sycl_blas.h>

#include <boost/math/constants/constants.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/thread.hpp>

#include <istream>
#include <iostream>
#include <iterator>
#include <string>
#include <queue>
#include <exception>
#include <memory>
#include <sstream>

using Radians = double;
using namespace boost::math::constants;

namespace icrar
{
namespace sycl
{
    SyclLeapCalibrator::SyclLeapCalibrator()
    {

    }

    SyclLeapCalibrator::~SyclLeapCalibrator()
    {

    }

    void SyclLeapCalibrator::Calibrate(
        std::function<void(const cpu::Calibration&)> outputCallback,
        const icrar::MeasurementSet& ms,
        const std::vector<SphericalDirection>& directions,
        const Slice& solutionInterval,
        double minimumBaselineThreshold,
        boost::optional<unsigned int> referenceAntenna,
        const ComputeOptionsDTO& computeOptions)
    {
        auto cpuComputeOptions = CpuComputeOptions(computeOptions, ms);

        LOG(info) << "Starting calibration using sycl";
        LOG(info)
        << "stations: " << ms.GetNumStations() << ", "
        << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "solutionInterval: [" << solutionInterval.GetStart() << "," << solutionInterval.GetInterval() << "," << solutionInterval.GetEnd() << "], "
        << "reference antenna: " << referenceAntenna << ", "
        << "flagged baselines: " << ms.GetNumFlaggedBaselines() << ", "
        << "baseline threshold: " << minimumBaselineThreshold << "m, "
        << "short baselines: " << ms.GetNumShortBaselines(minimumBaselineThreshold) << ", "
        << "filtered baselines: " << ms.GetNumFilteredBaselines(minimumBaselineThreshold) << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << ms.GetNumTimesteps();

        profiling::timer calibration_timer;

        auto output_calibrations = std::vector<cpu::Calibration>();

        std::vector<double> epochs = ms.GetEpochs();
        
        profiling::timer metadata_read_timer;
        auto metadata = icrar::cpu::MetaData(
            ms,
            referenceAntenna,
            minimumBaselineThreshold,
            false,
            false);

        // device_matrix<double> deviceA, deviceAd;
        // CalculateAd(metadata, deviceA, deviceAd, cudaComputeOptions.isFileSystemCacheEnabled, cudaComputeOptions.useCusolver);

        // device_matrix<double> deviceA1, deviceAd1;
        // CalculateAd1(metadata, deviceA1, deviceAd1);

        std::vector<float> a = { 1.0, 2.0, 3.0, 4.0 };
        std::vector<float> b = { 4.0, 3.0, 2.0, 1.0 };
        std::vector<float> c = { 0.0, 0.0, 0.0, 0.0 };

        cl::sycl::default_selector device_selector;

        cl::sycl::queue queue(device_selector);
        std::cout << "Running on " << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";
        
        {
            cl::sycl::buffer<float, 2> a_sycl(&a[0], cl::sycl::range<2>(2,2));
            cl::sycl::buffer<float, 2> b_sycl(&b[0], cl::sycl::range<2>(2,2));
            cl::sycl::buffer<float, 2> c_sycl(&c[0], cl::sycl::range<2>(2,2));
        
            queue.submit([&](cl::sycl::handler& cgh) {
                auto a_acc = a_sycl.get_access<cl::sycl::access::mode::read>(cgh);
                auto b_acc = b_sycl.get_access<cl::sycl::access::mode::read>(cgh);
                auto c_acc = c_sycl.get_access<cl::sycl::access::mode::discard_write>(cgh);

                // Single Task
                // cgh.single_task<class vector_addition>([=] () {
                // c_acc[0] = a_acc[0] + b_acc[0];
                // });

                // Parallel Task
                // https://intel.github.io/llvm-docs/doxygen/classcl_1_1sycl_1_1handler.html
                cgh.parallel_for<class vector_addition>(cl::sycl::range<2>(2,2), [=](cl::sycl::id<2> idx)
                {
                    auto a_m = Eigen::Map<const Eigen::Matrix2f>(&a_acc[0][0], a_acc.get_range()[0], a_acc.get_range()[1]);
                    auto b_m = Eigen::Map<const Eigen::Matrix2f>(&b_acc[0][0], b_acc.get_range()[0], b_acc.get_range()[1]);
                    auto c_m = Eigen::Map<Eigen::Matrix2f>(&c_acc[0][0], c_acc.get_range()[0], c_acc.get_range()[1]);

                    c_m(idx[0],idx[1]) = a_m(idx[0],idx[1]) + b_m(idx[0],idx[1]);
                    //c_acc[idx[0]][idx[1]] = a_acc[idx[0]][idx[1]] + b_acc[idx[0]][idx[1]];
                });
            });
        }
        std::cout << "  A { " << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << " }\n"
                << "+ B { " << b[0] << ", " << b[1] << ", " << b[2] << ", " << b[3] << " }\n"
                << "------------------\n"
                << "= C { " << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << " }"
                << std::endl;
    }

    void SyclLeapCalibrator::PhaseRotate(
        cpu::MetaData& metadata,
        const SphericalDirection& direction,
        std::vector<cpu::Integration>& input,
        std::vector<cpu::BeamCalibration>& output_calibrations)
    {

    }

    void SyclLeapCalibrator::RotateVisibilities(cpu::Integration& integration, cpu::MetaData& metadata)
    {
       
    }
} // namespace cpu
} // namespace icrar
#endif // SYCL_ENABLED
