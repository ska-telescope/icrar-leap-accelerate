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

#include "PhaseRotateTestCaseData.h"

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/tests/math/eigen_helper.h>

#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>

#include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/cpu/CpuLeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/cuda/CudaLeapCalibrator.h>

#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>
#include <icrar/leap-accelerate/model/cpu/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>

#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <gtest/gtest.h>

#if CUDA_ENABLED
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <cuda_runtime.h>
#endif

#include <boost/log/trivial.hpp>

#include <functional>
#include <vector>
#include <set>
#include <unordered_map>

using namespace std::literals::complex_literals;

namespace icrar
{
    /**
     * @brief Test suite for PhaseRotate.cc functionality
     * 
     */
    class PhaseRotateTests : public ::testing::Test
    {
        std::unique_ptr<icrar::MeasurementSet> ms;

    protected:
        void SetUp() override
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/mwa/1197638568-split.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 102, true);
            std::cout << std::setprecision(15);
        }

        void TearDown() override
        {
#if CUDA_ENABLED
            checkCudaErrors(cudaDeviceReset());
#endif
        }

        void PhaseRotateTest(ComputeImplementation impl, const Slice solutionInterval, std::function<cpu::CalibrationCollection()> getExpected)
        {
            const double THRESHOLD = 1e-11;

            auto metadata = icrar::cpu::MetaData(*ms, ToUVWVector(ms->GetCoords(0, ms->GetNumRows())));
            std::vector<casacore::MVDirection> directions = //TODO: use icrar Direction
            {
                { -0.4606549305661674,-0.29719233792392513 },
                { -0.753231018062671,-0.44387635324622354 },
                { -0.4606549305661674,-0.29719233792392513 },
                { -0.753231018062671,-0.44387635324622354 },
            };

            const auto& expected = getExpected();
            ASSERT_LT(0, expected.GetCalibrations().size());
            
            auto calibrations = LeapCalibratorFactory::Create(impl)->Calibrate(
                *ms,
                ToDirectionVector(directions),
                solutionInterval,
                0.0,
                0,
                false);

            ASSERT_LT(0, calibrations.GetCalibrations().size());
            ASSERT_EQ(expected.GetCalibrations().size(), calibrations.GetCalibrations().size());

            for(size_t calibrationIndex = 0; calibrationIndex < expected.GetCalibrations().size(); calibrationIndex++)
            {
                const auto& calibration = calibrations.GetCalibrations()[calibrationIndex];
                const auto& expectedCalibration = expected.GetCalibrations()[calibrationIndex];

                ASSERT_EQ(directions.size(), calibration.GetBeamCalibrations().size());
                // This supports expected calibrations to be an incomplete collection
                size_t totalDirections = expectedCalibration.GetBeamCalibrations().size();
                for(size_t directionIndex = 0; directionIndex < totalDirections; directionIndex++)
                {
                    const cpu::BeamCalibration& expectedBeamCalibration = expectedCalibration.GetBeamCalibrations()[directionIndex];
                    const cpu::BeamCalibration& actualBeamCalibration = calibration.GetBeamCalibrations()[directionIndex];

                    ASSERT_EQ(expectedBeamCalibration.GetDirection()(0), actualBeamCalibration.GetDirection()(0));
                    ASSERT_EQ(expectedBeamCalibration.GetDirection()(1), actualBeamCalibration.GetDirection()(1));

                    ASSERT_EQ(expectedBeamCalibration.GetPhaseCalibration().rows(), actualBeamCalibration.GetPhaseCalibration().rows());
                    ASSERT_EQ(expectedBeamCalibration.GetPhaseCalibration().cols(), actualBeamCalibration.GetPhaseCalibration().cols());
                    if(!expectedBeamCalibration.GetPhaseCalibration().isApprox(actualBeamCalibration.GetPhaseCalibration(), THRESHOLD))
                    {
                        std::cout << directionIndex+1 << "/" << totalDirections << " got:\n" << actualBeamCalibration.GetPhaseCalibration() << std::endl;
                    }
                    ASSERT_MEQD(expectedBeamCalibration.GetPhaseCalibration(), actualBeamCalibration.GetPhaseCalibration(), THRESHOLD);
                }
            }
        }

        void RotateVisibilitiesTest(ComputeImplementation impl)
        {
            using namespace std::complex_literals;
            const double THRESHOLD = 0.0001;
            
            auto direction = casacore::MVDirection(-0.4606549305661674, -0.29719233792392513);

            boost::optional<icrar::cpu::MetaData> metadataOptionalOutput;
            if(impl == ComputeImplementation::cpu)
            {
                
                auto integration = cpu::Integration(
                    0,
                    *ms,
                    0,
                    ms->GetNumChannels(),
                    ms->GetNumBaselines(),
                    ms->GetNumPols());

                auto hostMetadata = icrar::cpu::MetaData(*ms, ToDirection(direction), integration.GetUVW());
                cpu::CpuLeapCalibrator::RotateVisibilities(integration, hostMetadata);

                metadataOptionalOutput = hostMetadata;
            }
#ifdef CUDA_ENABLED
            if(impl == ComputeImplementation::cuda)
            {
                auto integration = icrar::cpu::Integration(
                    0,
                    *ms,
                    0,
                    ms->GetNumChannels(),
                    ms->GetNumBaselines(),
                    ms->GetNumPols());
                auto deviceIntegration = icrar::cuda::DeviceIntegration(integration);
                auto hostMetadata = icrar::cpu::MetaData(*ms, ToDirection(direction), integration.GetUVW());
                auto deviceMetadata = icrar::cuda::DeviceMetaData(hostMetadata);

                cuda::CudaLeapCalibrator().RotateVisibilities(deviceIntegration, deviceMetadata);
                deviceMetadata.ToHost(hostMetadata);
                metadataOptionalOutput = hostMetadata;
            }
#endif // CUDA_ENABLED

            ASSERT_TRUE(metadataOptionalOutput.is_initialized());
            icrar::cpu::MetaData& metadataOutput = metadataOptionalOutput.get();

            // =======================
            // Build expected results
            // Test case generic
            auto expectedConstants = icrar::cpu::Constants();
            expectedConstants.nbaselines = 5253;
            expectedConstants.channels = 48;
            expectedConstants.num_pols = 4;
            expectedConstants.stations = 102;
            expectedConstants.rows = 73542;
            expectedConstants.freq_start_hz = 1.39195e+08;
            expectedConstants.freq_inc_hz = 640000;
            expectedConstants.phase_centre_ra_rad = 0.57595865315812877;
            expectedConstants.phase_centre_dec_rad = 0.10471975511965978;
            expectedConstants.dlm_ra = -1.0366135837242962;
            expectedConstants.dlm_dec = -0.40191209304358488;
            auto expectedDD = Eigen::Matrix3d();
            expectedDD <<
             0.50913780874486769, -0.089966081772685239,  0.85597009050371897,
             -0.2520402307174327,   0.93533988977932658,  0.24822371499818516,
            -0.82295468514759529,  -0.34211897743046571,  0.45354182990718139;

            //========
            // ASSERT
            //========
            EXPECT_DOUBLE_EQ(expectedConstants.nbaselines, metadataOutput.GetConstants().nbaselines);
            EXPECT_DOUBLE_EQ(expectedConstants.channels, metadataOutput.GetConstants().channels);
            EXPECT_DOUBLE_EQ(expectedConstants.num_pols, metadataOutput.GetConstants().num_pols);
            EXPECT_DOUBLE_EQ(expectedConstants.stations, metadataOutput.GetConstants().stations);
            EXPECT_DOUBLE_EQ(expectedConstants.rows, metadataOutput.GetConstants().rows);
            EXPECT_DOUBLE_EQ(expectedConstants.freq_start_hz, metadataOutput.GetConstants().freq_start_hz);
            EXPECT_DOUBLE_EQ(expectedConstants.freq_inc_hz, metadataOutput.GetConstants().freq_inc_hz);
            EXPECT_DOUBLE_EQ(expectedConstants.phase_centre_ra_rad, metadataOutput.GetConstants().phase_centre_ra_rad);
            EXPECT_DOUBLE_EQ(expectedConstants.phase_centre_dec_rad, metadataOutput.GetConstants().phase_centre_dec_rad);
            EXPECT_DOUBLE_EQ(expectedConstants.dlm_ra, metadataOutput.GetConstants().dlm_ra);
            EXPECT_DOUBLE_EQ(expectedConstants.dlm_dec, metadataOutput.GetConstants().dlm_dec);
            ASSERT_TRUE(expectedConstants == metadataOutput.GetConstants());
            
            EXPECT_DOUBLE_EQ(expectedDD(0,0), metadataOutput.GetDD()(0,0));
            EXPECT_DOUBLE_EQ(expectedDD(0,1), metadataOutput.GetDD()(0,1));
            EXPECT_DOUBLE_EQ(expectedDD(0,2), metadataOutput.GetDD()(0,2));
            EXPECT_DOUBLE_EQ(expectedDD(1,0), metadataOutput.GetDD()(1,0));
            EXPECT_DOUBLE_EQ(expectedDD(1,1), metadataOutput.GetDD()(1,1));
            EXPECT_DOUBLE_EQ(expectedDD(1,2), metadataOutput.GetDD()(1,2));
            EXPECT_DOUBLE_EQ(expectedDD(2,0), metadataOutput.GetDD()(2,0));
            EXPECT_DOUBLE_EQ(expectedDD(2,1), metadataOutput.GetDD()(2,1));
            EXPECT_DOUBLE_EQ(expectedDD(2,2), metadataOutput.GetDD()(2,2));

            ASSERT_EQ(5253, metadataOutput.GetAvgData().rows());
            ASSERT_EQ(4, metadataOutput.GetAvgData().cols());
            ASSERT_EQCD(-527.143090304241 + -89.9946133549982i, metadataOutput.GetAvgData()(1,0), THRESHOLD);
            ASSERT_EQCD(55.4651357651697 + 60.5911844734979i, metadataOutput.GetAvgData()(1,1), THRESHOLD);
            ASSERT_EQCD(0.800572867605558 + -9.05853723508164i, metadataOutput.GetAvgData()(1,2), THRESHOLD);
            ASSERT_EQCD(-251.31739125869 + 39.6303072927434i, metadataOutput.GetAvgData()(1,3), THRESHOLD);
        }

        void PhaseMatrixFunction0Test(ComputeImplementation impl)
        {
            int refAnt = 0;

            try
            {
                if(impl == ComputeImplementation::cpu)
                {
                    auto a1 = Eigen::VectorXi();
                    auto a2 = Eigen::VectorXi();
                    auto fg = Eigen::Matrix<bool, Eigen::Dynamic, 1>();
                    icrar::cpu::PhaseMatrixFunction(a1, a2, fg, refAnt);
                }
                else
                {
                    throw icrar::invalid_argument_exception("invalid PhaseMatrixFunction implementation", "impl", __FILE__, __LINE__);
                }
            }
            catch(invalid_argument_exception& e)
            {
                SUCCEED();
            }
            catch(...)
            {
                FAIL() << "Expected icrar::invalid_argument_exception";
            }
        }

        void PhaseMatrixFunctionDataTest(ComputeImplementation impl)
        {
            auto msmc = ms->GetMSMainColumns();

            //select the first epoch only
            casacore::Vector<double> time = msmc->time().getColumn();
            double epoch = time[0];
            int epochRows = 0;
            for(size_t i = 0; i < time.size(); i++)
            {
                if(time[i] == epoch) epochRows++;
            }

            const int aSize = epochRows;
            auto epochIndices = casacore::Slice(0, aSize);
            casacore::Vector<std::int32_t> a1 = msmc->antenna1().getColumn()(epochIndices); 
            casacore::Vector<std::int32_t> a2 = msmc->antenna2().getColumn()(epochIndices);

            // Selects only the flags of the first channel and polarization
            auto flagSlice = casacore::Slicer(
                casacore::IPosition(2, 0, 0),
                casacore::IPosition(2, 1, 1),
                casacore::IPosition(2, 1, 1));
            casacore::Vector<bool> flags = msmc->flag().getColumnRange(epochIndices, flagSlice);

            //Start calculations

            //output
            Eigen::MatrixXd A;
            Eigen::VectorXi I;
            Eigen::MatrixXd Ad;
            Eigen::MatrixXd A1;
            Eigen::VectorXi I1;
            Eigen::MatrixXd Ad1;

            if(impl == ComputeImplementation::cpu)
            {
                auto ea1 = ToVector(a1);
                auto ea2 = ToVector(a2);
                auto efg = ToVector(flags);
                std::tie(A, I) = cpu::PhaseMatrixFunction(ea1, ea2, efg, boost::none);
                Ad = icrar::cpu::PseudoInverse(A);

                std::tie(A1, I1) = cpu::PhaseMatrixFunction(ea1, ea2, efg, 0);
                Ad1 = icrar::cpu::PseudoInverse(A1);
            }
            else
            {
                throw icrar::invalid_argument_exception("invalid PhaseMatrixFunction implementation", "impl", __FILE__, __LINE__);
            }

            double TOLERANCE = 0.00001;

            // A
            const int aRows = 4754;
            const int aCols = 128;
            ASSERT_EQ(aRows, A.rows());
            ASSERT_EQ(aCols, A.cols());
            EXPECT_EQ(1.00, A(0,0));
            EXPECT_EQ(-1.00, A(0,1));
            EXPECT_EQ(0.00, A(0,2));
            //...
            EXPECT_NEAR(0.00, A(aRows-2, 125), TOLERANCE);
            EXPECT_NEAR(1.00, A(aRows-2, 126), TOLERANCE);
            EXPECT_NEAR(-1.00, A(aRows-2, 127), TOLERANCE);
            EXPECT_NEAR(0.00, A(aRows-1, 125), TOLERANCE);
            EXPECT_NEAR(0.00, A(aRows-1, 126), TOLERANCE);
            EXPECT_NEAR(0.00, A(aRows-1, 127), TOLERANCE);

            // I
            const int nBaselines = 4753;
            ASSERT_EQ(nBaselines, I.size());
            EXPECT_EQ(1.00, I(0));
            EXPECT_EQ(3.00, I(1));
            EXPECT_EQ(4.00, I(2));
            //...
            EXPECT_EQ(5248, I(nBaselines-3));
            EXPECT_EQ(5249, I(nBaselines-2));
            EXPECT_EQ(5251, I(nBaselines-1));

            // Ad
            ASSERT_EQ(aCols, Ad.rows());
            ASSERT_EQ(aRows, Ad.cols());
            // EXPECT_NEAR(2.62531368e-15, Ad(0,0), TOLERANCE); // TODO(calgray): emergent
            // EXPECT_NEAR(2.04033520e-15, Ad(0,1), TOLERANCE); // TODO(calgray): emergent
            // EXPECT_NEAR(3.25648083e-16, Ad(0,2), TOLERANCE); // TODO(calgray): emergent
            // //...
            // EXPECT_NEAR(-1.02040816e-02, Ad(127,95), TOLERANCE); // TODO(calgray): emergent
            // EXPECT_NEAR(-0.020408163265312793, Ad(127,96), TOLERANCE); // TODO(calgray): emergent
            // EXPECT_NEAR(-8.9737257304377696e-16, Ad(127,97), TOLERANCE); // TODO(calgray): emergent

            ASSERT_EQ(Ad.cols(), I.size() + 1);
            ASSERT_MEQD(A, A * Ad * A, TOLERANCE);

            //A1
            const int a1Rows = 98;
            const int a1Cols = 128;
            ASSERT_EQ(a1Rows, A1.rows());
            ASSERT_EQ(a1Cols, A1.cols());
            EXPECT_DOUBLE_EQ(1.0, A1(0,0));
            EXPECT_DOUBLE_EQ(-1.0, A1(0,1));
            EXPECT_DOUBLE_EQ(0.0, A1(0,2));
            //...
            EXPECT_NEAR(0.00, A1(a1Rows-2,125), TOLERANCE);
            EXPECT_NEAR(0.00, A1(a1Rows-2,126), TOLERANCE);
            EXPECT_NEAR(-1.00, A1(a1Rows-2,127), TOLERANCE);
            EXPECT_NEAR(0.00, A1(a1Rows-1,125), TOLERANCE);
            EXPECT_NEAR(0.00, A1(a1Rows-1,126), TOLERANCE);
            EXPECT_NEAR(0.00, A1(a1Rows-1,127), TOLERANCE);

            //I1
            ASSERT_EQ(a1Rows-1, I1.size());
            EXPECT_DOUBLE_EQ(1.00, I1(0));
            EXPECT_DOUBLE_EQ(3.00, I1(1));
            EXPECT_DOUBLE_EQ(4.00, I1(2));
            //...
            EXPECT_DOUBLE_EQ(99.00, I1(a1Rows-4));
            EXPECT_DOUBLE_EQ(100.00, I1(a1Rows-3));
            EXPECT_DOUBLE_EQ(101.00, I1(a1Rows-2));

            //Ad1
            ASSERT_EQ(a1Rows, Ad1.cols());
            ASSERT_EQ(a1Cols, Ad1.rows());

            // EXPECT_DOUBLE_EQ(-9.8130778667735933e-18, Ad1(0,0)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(6.3742385976163974e-17, Ad1(0,1)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(3.68124219034074e-19, Ad1(0,2)); // TODO: emergent
            // //...
            // EXPECT_DOUBLE_EQ(5.4194040934156436e-17, Ad1(127,95)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(-1.0, Ad1(127,96)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(1.0, Ad1(127,97)); // TODO: emergent
            
            ASSERT_EQ(Ad1.cols(), I1.size() + 1);
            ASSERT_MEQD(A1, A1 * Ad1 * A1, TOLERANCE);
        }
    };

    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCpu) { PhaseMatrixFunction0Test(ComputeImplementation::cpu); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCpu) { PhaseMatrixFunctionDataTest(ComputeImplementation::cpu); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCpu) { RotateVisibilitiesTest(ComputeImplementation::cpu); }
#ifdef CUDA_ENABLED
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCuda) { RotateVisibilitiesTest(ComputeImplementation::cuda); }
#endif

    TEST_F(PhaseRotateTests, PhaseRotateFirstTimestepTestCpu) { PhaseRotateTest(ComputeImplementation::cpu, Slice(0, 1), &GetFirstTimestepMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateAllTimesteps0TestCpu) { PhaseRotateTest(ComputeImplementation::cpu, Slice(0,14), &GetAllTimestepsMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateAllTimesteps1TestCpu) { PhaseRotateTest(ComputeImplementation::cpu, Slice(0,-1), &GetAllTimestepsMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateEachTimestepTestCpu) { PhaseRotateTest(ComputeImplementation::cpu, Slice(1), &GetEachTimestepMWACalibration); }
#ifdef CUDA_ENABLED
    TEST_F(PhaseRotateTests, PhaseRotateFirstTimestepsTestCuda) { PhaseRotateTest(ComputeImplementation::cuda, Slice(0,1), &GetFirstTimestepMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateAllTimesteps0TestCuda) { PhaseRotateTest(ComputeImplementation::cuda, Slice(0,14), &GetAllTimestepsMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateAllTimesteps1TestCuda) { PhaseRotateTest(ComputeImplementation::cuda, Slice(0,-1), &GetAllTimestepsMWACalibration); }
    TEST_F(PhaseRotateTests, PhaseRotateEachTimestepTestCuda) { PhaseRotateTest(ComputeImplementation::cuda, Slice(1), &GetEachTimestepMWACalibration); }
#endif
} // namespace icrar
