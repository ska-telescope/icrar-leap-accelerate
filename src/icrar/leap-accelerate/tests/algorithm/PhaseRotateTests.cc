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

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <icrar/leap-accelerate/algorithm/casa/PhaseMatrixFunction.h>
#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
//#include <icrar/leap-accelerate/algorithm/cuda/PhaseMatrixFunction.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/model/casa/Integration.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <gtest/gtest.h>

#include <boost/log/trivial.hpp>

#include <vector>
#include <set>
#include <unordered_map>

using namespace std::literals::complex_literals;

namespace icrar
{
    class PhaseRotateTests : public ::testing::Test
    {
        std::unique_ptr<icrar::MeasurementSet> ms;

    protected:

        PhaseRotateTests() {

        }

        ~PhaseRotateTests() override
        {

        }

        void SetUp() override
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 126, true);
            std::cout << std::setprecision(15);
        }

        void TearDown() override
        {
            
        }

        void PhaseRotateTest(ComputeImplementation impl)
        {
            const double THRESHOLD = 1e-12;

            auto metadata = icrar::casalib::MetaData(*ms);

            std::vector<casacore::MVDirection> directions =
            {
                { -0.4606549305661674,-0.29719233792392513 },
                { -0.753231018062671,-0.44387635324622354 },
                { -0.6207547100721282,-0.2539086572881469 },
                { -0.41958660604621867,-0.03677626900108552 },
                { -0.41108685258900596,-0.08638012622791202 },
                { -0.7782459495668798,-0.4887860989684432 },
                //{ -0.17001324965728973,-0.28595644149463484 },
                //{ -0.7129444556035118,-0.365286407171852 },
                //{ -0.1512764129166089,-0.21161026349648748 }

            };

            std::vector<std::vector<cpu::IntegrationResult>> integrations;
            std::vector<std::vector<cpu::CalibrationResult>> calibrations;
            if(impl == ComputeImplementation::casa)
            {
                auto pair = icrar::casalib::Calibrate(*ms, directions);
                std::tie(integrations, calibrations) = cpu::ToCalibrateResult(pair);
            }
            else if(impl == ComputeImplementation::cpu)
            {
                std::tie(integrations, calibrations) = cpu::Calibrate(*ms, ToDirectionVector(directions));
            }
            else if(impl == ComputeImplementation::cuda)
            {
                std::tie(integrations, calibrations) = cuda::Calibrate(*ms, ToDirectionVector(directions));
            }
            else
            {
                throw std::invalid_argument("impl");
            }

            auto expected = GetExpectedCalibration();

            ASSERT_EQ(directions.size(), calibrations.size());
            for(size_t i = 0; i < expected.size(); i++)
            {
                casacore::MVDirection expectedDirection;
                std::vector<double> expectedCalibration;
                std::tie(expectedDirection, expectedCalibration) = expected[i];

                ASSERT_EQ(1, calibrations[i].size());
                const auto& result = calibrations[i].front();
                ASSERT_EQ(1, result.GetData().size());

                //TODO: assert with LEAP-Cal
                ASSERT_EQ(expectedDirection(0), result.GetDirection()(0));
                ASSERT_EQ(expectedDirection(1), result.GetDirection()(1));

                if(!ToVector(expectedCalibration).isApprox(ToMatrix(result.GetData()[0]), THRESHOLD))
                {
                    std::cout << i+1 << "/" << expected.size() << " got:\n" << ToMatrix(result.GetData()[0]) << std::endl;
                }
                ASSERT_MEQD(ToVector(expectedCalibration), ToMatrix(result.GetData()[0]), THRESHOLD);
            }
        }

        void RotateVisibilitiesTest(ComputeImplementation impl)
        {
            using namespace std::complex_literals;
            const double THRESHOLD = 0.01;
            
            auto direction = casacore::MVDirection(-0.4606549305661674, -0.29719233792392513);

            boost::optional<icrar::cpu::Integration> integrationOptionalOutput;
            boost::optional<icrar::cpu::MetaData> metadataOptionalOutput;
            if(impl == ComputeImplementation::casa)
            {
                auto metadata = casalib::MetaData(*ms);
                auto integration = casalib::Integration(
                    0,
                    *ms,
                    0,
                    ms->GetNumChannels(),
                    ms->GetNumBaselines(),
                    ms->GetNumPols());

                icrar::casalib::RotateVisibilities(integration, metadata, direction);
                integrationOptionalOutput = icrar::cpu::Integration(integration);
                metadataOptionalOutput = icrar::cpu::MetaData(metadata);
            }
            if(impl == ComputeImplementation::cpu)
            {
                
                auto integration = cpu::Integration(
                    0,
                    *ms,
                    0,
                    ms->GetNumChannels(),
                    ms->GetNumBaselines(),
                    ms->GetNumPols());

                auto metadatahost = icrar::cpu::MetaData(*ms, ToDirection(direction), integration.GetUVW());
                icrar::cpu::RotateVisibilities(integration, metadatahost);

                integrationOptionalOutput = integration;
                metadataOptionalOutput = metadatahost;
            }
            if(impl == ComputeImplementation::cuda)
            {
                auto integration = icrar::cpu::Integration(
                    0,
                    *ms,
                    0,
                    ms->GetNumChannels(),
                    ms->GetNumBaselines(),
                    ms->GetNumPols());

                auto metadatahost = icrar::cpu::MetaData(*ms, ToDirection(direction), integration.GetUVW());
                auto metadatadevice = icrar::cuda::DeviceMetaData(metadatahost);
                auto deviceIntegration = icrar::cuda::DeviceIntegration(integration);
                icrar::cuda::RotateVisibilities(deviceIntegration, metadatadevice);
                metadatadevice.ToHost(metadatahost);
                integrationOptionalOutput = integration;
                metadataOptionalOutput = metadatahost;
            }
            ASSERT_TRUE(integrationOptionalOutput.is_initialized());
            //icrar::cpu::Integration& integrationOutput = integrationOptionalOutput.get();

            ASSERT_TRUE(metadataOptionalOutput.is_initialized());
            icrar::cpu::MetaData& metadataOutput = metadataOptionalOutput.get();

            // =======================
            // Build expected results
            // Test case generic
            auto expectedIntegration = icrar::casalib::Integration(0, *ms, 0, ms->GetNumChannels(), ms->GetNumBaselines(), ms->GetNumPols());
            expectedIntegration.uvw = ToCasaUVWVector(ms->GetCoords());

            auto expectedConstants = icrar::cpu::Constants();
            expectedConstants.nantennas = 0;
            expectedConstants.nbaselines = 8001;
            expectedConstants.channels = 48;
            expectedConstants.num_pols = 4;
            expectedConstants.stations = 126;
            expectedConstants.rows = 63089;
            expectedConstants.freq_start_hz = 1.39195e+08;
            expectedConstants.freq_inc_hz = 640000;
            expectedConstants.phase_centre_ra_rad = 0.57595865315812877;
            expectedConstants.phase_centre_dec_rad = 0.10471975511965978;
            expectedConstants.dlm_ra = -1.0366135837242962;
            expectedConstants.dlm_dec = -0.40191209304358488;
            auto expectedDD = Eigen::Matrix3d();
            expectedDD <<
            0.46856701307821974, 0.860685013060222, -0.19916390874975543,
            -0.792101075276669, 0.509137808744868, 0.336681716539552,
            0.39117878367889541, 0, 0.920314706608288;

            //========
            // ASSERT
            //========
            EXPECT_DOUBLE_EQ(expectedConstants.nantennas, metadataOutput.GetConstants().nantennas);
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
            
            EXPECT_DOUBLE_EQ(expectedDD(0,0), metadataOutput.dd(0,0));
            EXPECT_DOUBLE_EQ(expectedDD(0,1), metadataOutput.dd(0,1));
            EXPECT_DOUBLE_EQ(expectedDD(0,2), metadataOutput.dd(0,2));
            EXPECT_DOUBLE_EQ(expectedDD(1,0), metadataOutput.dd(1,0));
            EXPECT_DOUBLE_EQ(expectedDD(1,1), metadataOutput.dd(1,1));
            EXPECT_DOUBLE_EQ(expectedDD(1,2), metadataOutput.dd(1,2));
            EXPECT_DOUBLE_EQ(expectedDD(2,0), metadataOutput.dd(2,0));
            EXPECT_DOUBLE_EQ(expectedDD(2,1), metadataOutput.dd(2,1));
            EXPECT_DOUBLE_EQ(expectedDD(2,2), metadataOutput.dd(2,2));

            ASSERT_EQ(8001, metadataOutput.avg_data.rows());
            ASSERT_EQ(4, metadataOutput.avg_data.cols());
            ASSERT_EQCD(152.207482222774 + 157.780854994143i, metadataOutput.avg_data(1,0), THRESHOLD);
            ASSERT_EQCD(237.735520799299 + 123.628127794715i, metadataOutput.avg_data(1,1), THRESHOLD);
            ASSERT_EQCD(3.57682429815259 + -75.3381937487565i, metadataOutput.avg_data(1,2), THRESHOLD);
            ASSERT_EQCD(-168.342543770758 + -87.1917020804175i, metadataOutput.avg_data(1,3), THRESHOLD);
        }

        void PhaseMatrixFunction0Test(ComputeImplementation impl)
        {
            int refAnt = 0;

            try
            {
                if(impl == ComputeImplementation::casa)
                {
                    const casacore::Vector<int32_t> a1;
                    const casacore::Vector<int32_t> a2;
                    const casacore::Vector<bool> fg;
                    icrar::casalib::PhaseMatrixFunction(a1, a2, fg, refAnt);
                }
                if(impl == ComputeImplementation::cpu)
                {
                    auto a1 = Eigen::VectorXi();
                    auto a2 = Eigen::VectorXi();
                    auto fg = Eigen::Matrix<bool, Eigen::Dynamic, 1>();
                    icrar::cpu::PhaseMatrixFunction(a1, a2, fg, refAnt);
                }
                if(impl == ComputeImplementation::cuda)
                {
                    const Eigen::VectorXi a1;
                    const Eigen::VectorXi a2;
                    auto fg = Eigen::Matrix<bool, Eigen::Dynamic, 1>();
                    icrar::cuda::PhaseMatrixFunction(a1, a2, fg, refAnt);
                }
            }
            catch(std::invalid_argument& e)
            {
                
            }
            catch(...)
            {
                FAIL() << "Expected std::invalid_argument";
            }
        }

        /**
         * @brief Get the Expected Calibration object validated the output of LEAP-Cal:ported
         * 
         * @return a vector of direction and antenna calibration pairs
         */
        std::vector<std::pair<casacore::MVDirection, std::vector<double>>> GetExpectedCalibration()
        {
            std::vector<std::pair<casacore::MVDirection, std::vector<double>>> output;

            output.push_back(std::make_pair(casacore::MVDirection(-0.4606549305661674,-0.29719233792392513), std::vector<double>{
                    -0.29346540760013,
                    0.162321460721177,
                -1.30559416735717e-14,
                 8.32778309091414e-15,
                    0.269606270341435,
                     0.20555314891351,
                    -0.13850411737849,
                   -0.177400814402018,
                   -0.118592813538623,
                 1.57210059411765e-14,
                -4.18704735998426e-14,
                 2.25377226335441e-14,
                 -0.00537855097148199,
                   0.0260838416120084,
                   -0.136004032810747,
                -4.38965957376933e-14,
                   0.0814959693772079,
                   -0.131946679542235,
                -4.15368780815132e-14,
                 -1.5939735379172e-14,
                   0.0451905719791519,
                     0.16464587667749,
                   -0.118416005593518,
                  -0.0210894481145574,
                    0.416754265149017,
                    0.158683541693398,
                -1.26868585009014e-15,
                     0.14728866463714,
                   -0.211363067980175,
                   -0.144043373268725,
                -5.04390711036177e-15,
                   0.0318638003104375,
                   -0.234413536145573,
                    0.218691154147818,
                -2.37183122195024e-16,
                  -0.0307920301481439,
                    0.122662057313104,
                   -0.125146878845156,
                 2.63487678833672e-15,
                 1.56698732960517e-15,
                    0.137947080357575,
                    0.178304769177944,
                    -0.10316651984165,
                     0.16712234603666,
                    0.319586971163098,
                  -0.0311898017041778,
                  -0.0508129813935232,
                  -0.0920353222606466,
                   -0.227130058473377,
                 4.89528328739461e-15,
                  -0.0900832241318265,
                 1.12352419717917e-15,
                   -0.156653182021156,
                     0.14191673388099,
                   -0.136463103269851,
                  -0.0261246280981386,
                   -0.262965760152825,
                   -0.161458094607725,
                   -0.122608138493968,
                   -0.170549297026716,
                 8.13731119812781e-15,
                -4.91505250124211e-16,
                -3.88259560735496e-16,
                 3.51510787986649e-15,
                -8.44812589224067e-16,
                   -0.140757680724244,
                   -0.142811251436261,
                    -0.30092691476669,
                    -0.16679189058729,
                   -0.172968525595542,
                 1.24930397068098e-16,
                -1.41507190180168e-15,
                   -0.443696372810859,
                    0.286709614135526,
                   -0.137098014070699,
                  -0.0330654924615472,
                   0.0131598007528737,
                    0.107425870650681,
                    0.343798271348206,
                    0.276794851415818,
                 1.82744408684572e-15,
                  0.00891717849025175,
                    0.295784667933972,
                   0.0979936487432179,
                   -0.109150849509262,
                  -0.0763496112567561,
                    0.271257542812337,
                 1.19139350463809e-15,
                   -0.220997196806752,
                   0.0999623758509942,
                   -0.387787315759087,
                 2.64275000745485e-15,
                   0.0638291999339503,
                 8.85418036438295e-16,
                -4.18990924127071e-16,
                -1.99906056539683e-16,
                   0.0244693351312313,
                    0.110169171965504,
                       0.106977681308,
                 -0.00965431999540034,
                  -0.0739867469036498,
                   0.0655816213936653,
                     -0.2900015400092,
                   0.0993659090629067,
                  -0.0271102473577411,
                   0.0159572494485105,
                   -0.006596683703746,
                   0.0277782779185491,
                    0.161873941537783,
                  -0.0889212921991527,
                  -0.0548048046623849,
                -5.18408577909698e-16,
                   -0.267955659213956,
                    0.255192956616248,
                     0.33889186726645,
                 4.90451749928798e-16,
                     0.33766672253983,
                  -0.0326692550466925,
                    0.384301454939388,
                  -0.0550315603204559,
                   -0.435371857750121,
                   0.0085904470003384,
                   -0.119852786514056,
                    0.360911321756097,
                    -0.14779483020627,
                   -0.154600693735746,
                -0.000396873777773976,
                    0.455867631554942
            }));

            // output.push_back(std::make_pair(casacore::MVDirection(-0.753231018062671,-0.44387635324622354), std::vector<double>{
            //     2.66940707100503,
            //      2.2970261355245,
            //                    0,
            //                    0,
            //     2.37286363595303,
            //     2.63169072474505,
            //     2.72339153492401,
            //      2.4787964771248,
            //     2.56266530776803,
            //                    0,
            //                    0,
            //                    0,
            //      2.3498107749363,
            //     2.60812066866685,
            //     2.71155466852463,
            //                    0,
            //     2.53239692415824,
            //     2.59639746435725,
            //                    0,
            //                    0,
            //     2.68815507344987,
            //     2.42280642919287,
            //     2.38381829083625,
            //     2.17897172810935,
            //     2.59142871786394,
            //     2.45175822369597,
            //                    0,
            //     2.33658232492272,
            //     2.62501940297269,
            //     2.50845523780901,
            //                    0,
            //     2.45800221101849,
            //     2.11690589644253,
            //     2.54346336226205,
            //                    0,
            //     2.77596376574356,
            //     2.59632501770404,
            //     2.58941712994333,
            //                    0,
            //                    0,
            //     2.47954340799536,
            //     2.78908831120321,
            //     2.58855227550415,
            //     2.34748400404489,
            //     2.84913863482906,
            //     2.12753070636066,
            //     2.05852397500549,
            //     2.26811193090935,
            //     2.67369416525684,
            //                    0,
            //      2.7507484091406,
            //                    0,
            //     2.58484600698607,
            //     2.26696309266383,
            //     2.39715817905781,
            //     2.74973915060334,
            //     2.51109323679793,
            //     2.58314298840669,
            //     2.45463216062136,
            //     2.29311157172345,
            //                    0,
            //                    0,
            //                    0,
            //                    0,
            //                    0,
            //     2.73450051928645,
            //     2.33848299978127,
            //     2.49046430796644,
            //     2.49370429585425,
            //     2.45176370212733,
            //                    0,
            //                    0,
            //     2.33338397930632,
            //     2.64931914296968,
            //     2.41145476301061,
            //     2.55940951035906,
            //     2.44728828799146,
            //     2.65642398536674,
            //      2.5245266383623,
            //     2.40560958368183,
            //                    0,
            //     2.11783575121744,
            //     2.49265370256483,
            //     2.33702500242863,
            //     2.22231181478633,
            //     2.59083932865882,
            //     2.55754303688411,
            //                    0,
            //     2.43892301312037,
            //     2.60771128744822,
            //     2.20792262714726,
            //                    0,
            //     2.56329491748639,
            //                    0,
            //                    0,
            //                    0,
            //       2.622212205822,
            //     2.56653379895624,
            //     2.66296476018488,
            //     2.56776882414028,
            //     2.56062742370982,
            //     2.82979723755799,
            //     2.43247977790005,
            //     2.52792625983831,
            //     2.31189781420646,
            //     2.68471348792089,
            //     2.52144021308095,
            //     2.81641633041315,
            //     2.66123518865772,
            //     2.37457923334427,
            //     2.83433133594982,
            //                    0,
            //      2.6096572778671,
            //     2.54270676897643,
            //     2.86993859813482,
            //                    0,
            //     2.70824021037875,
            //     2.43279237964071,
            //     2.39891158028436,
            //     2.38212304452993,
            //     2.39291435669244,
            //     2.25098998284473,
            //     2.63989589493289,
            //       2.431640263158,
            //     2.62321928177316,
            //     2.50927244546791,
            //     2.41362292454067,
            //     2.76440352155434,
            // }));

            // output.push_back(std::make_pair(casacore::MVDirection(-0.6207547100721282,-0.2539086572881469), std::vector<double>{
            //     3.02649528654214,
            //     3.10777887684405,
            //                    0,
            //                    0,
            //     2.81212632484272,
            //     2.88622529752106,
            //     3.20642545507654,
            //     2.82456265927589,
            //     3.02575915917859,
            //                    0,
            //                    0,
            //                    0,
            //      3.1468930917008,
            //     3.41648740549913,
            //     2.81273653786727,
            //                    0,
            //     2.95389528330968,
            //     2.95948996876507,
            //                    0,
            //                    0,
            //     2.90832680822659,
            //     3.15549292896688,
            //     2.96158096881739,
            //     3.14581570375654,
            //     2.94261706130072,
            //      3.3571399345161,
            //                    0,
            //     2.99301301501561,
            //     3.32630143618634,
            //     3.13819551153512,
            //                    0,
            //     3.05566904198689,
            //     3.14975711830892,
            //     2.89412773329797,
            //                    0,
            //     2.91326740419561,
            //     2.78177196652179,
            //     3.10098779461005,
            //                    0,
            //                    0,
            //     3.02818851094753,
            //     3.07908014019085,
            //     2.92759146580687,
            //     2.98047661247693,
            //     3.17173850491018,
            //     2.97402493811016,
            //     2.65521788178272,
            //     3.00233761241966,
            //     2.63263034009139,
            //                    0,
            //     2.91383760331526,
            //                    0,
            //     2.93930253762704,
            //     2.86873308968946,
            //     2.96328606254811,
            //     3.14720054483507,
            //     3.15999757850614,
            //     3.01573011395736,
            //     2.91230524663411,
            //     2.82302605259917,
            //                    0,
            //                    0,
            //                    0,
            //                    0,
            //                    0,
            //     3.11736370155431,
            //     3.02641965575357,
            //      3.1974744023122,
            //     3.07507911838769,
            //     2.72356631729012,
            //                    0,
            //                    0,
            //     3.06277693715546,
            //     2.89659778792819,
            //      3.1604058232977,
            //     2.86765938016416,
            //      2.9715891719881,
            //     3.18701757291299,
            //     3.03832735949693,
            //     2.93024942866138,
            //                    0,
            //     2.78778505030499,
            //     3.03617080794071,
            //     3.29384451343836,
            //     2.91872345871414,
            //     2.74264674796115,
            //     3.15189951906102,
            //                    0,
            //     3.10524831580181,
            //     2.93001675089563,
            //     2.95582807470878,
            //                    0,
            //     3.00883801697744,
            //                    0,
            //                    0,
            //                    0,
            //     2.80220655192013,
            //     2.85734363385934,
            //      3.1022812874959,
            //     3.01633464585403,
            //     3.06104307479965,
            //     2.84158971978347,
            //     2.67950574434861,
            //     3.08206033468821,
            //     2.69357120801818,
            //     2.66772364538293,
            //     2.96552386218769,
            //     3.04092420222566,
            //     2.95456394545774,
            //      3.1103778624671,
            //     3.26807067488729,
            //                    0,
            //     3.24518992619425,
            //      2.7046261731372,
            //     3.12206182399197,
            //                    0,
            //     2.99429114041317,
            //     2.77900859216216,
            //     2.94083236294281,
            //     3.04405219905266,
            //     2.91723739526017,
            //      2.6466651567884,
            //     2.88554172053832,
            //     3.24423565983191,
            //     2.93071313385988,
            //     2.92675261197971,
            //     3.02495480201382,
            //     3.08042209050104,
            // }));

            // output.push_back(std::make_pair(casacore::MVDirection(-0.41958660604621867,-0.03677626900108552), std::vector<double>{
            //     2.09405670300514,
            //     2.24973265632178,
            //                    0,
            //                    0,
            //     2.19814491003888,
            //     1.98577745175181,
            //     1.79954717619448,
            //       2.375570947815,
            //     1.96192668848045,
            //                    0,
            //                    0,
            //                    0,
            //     2.30337558595535,
            //     2.07818019810956,
            //      2.2444819218512,
            //                    0,
            //     2.32663604895753,
            //     2.17930895554649,
            //                    0,
            //                    0,
            //     2.26311144882305,
            //     2.38321075715048,
            //     2.37140844870669,
            //      2.0979621334619,
            //     2.23308944656673,
            //     1.67361839723636,
            //                    0,
            //      2.4803843973393,
            //     1.93453981153673,
            //     2.16705104199178,
            //                    0,
            //     2.20742262459279,
            //     2.37353217394698,
            //     2.37859400915076,
            //                    0,
            //      2.4394637411842,
            //      2.0328638105279,
            //     2.28405537297735,
            //                    0,
            //                    0,
            //     1.94772924219363,
            //     2.43682195394374,
            //     2.02707166314488,
            //      2.1497004813737,
            //     2.22460505098086,
            //     2.04703407589749,
            //      2.0334889934639,
            //     2.18527367359791,
            //     1.98228479907915,
            //                    0,
            //      2.3315541351602,
            //                    0,
            //     2.28303319068191,
            //     1.96740369544605,
            //     2.36242832181987,
            //     2.06570314724028,
            //     2.02575000573757,
            //     2.20320929024714,
            //     2.28709092639526,
            //     2.40266193725367,
            //                    0,
            //                    0,
            //                    0,
            //                    0,
            //                    0,
            //      2.1493691505881,
            //     1.92442968518846,
            //     1.77336393210406,
            //     2.16373879913863,
            //     2.33362254461566,
            //                    0,
            //                    0,
            //      2.0579547235385,
            //     2.15238576718379,
            //     2.11355895034468,
            //     2.03559362218414,
            //     2.39790919746143,
            //      2.0580525191644,
            //     2.05420556600375,
            //     2.02407580569069,
            //                    0,
            //     2.17577991673266,
            //      1.9937266688611,
            //      1.8768316594321,
            //     2.19972627612467,
            //     2.36440819952834,
            //     2.02857839961879,
            //                    0,
            //     2.25667910985571,
            //     1.88301437871967,
            //     1.99932852462542,
            //                    0,
            //     1.92429679062874,
            //                    0,
            //                    0,
            //                    0,
            //     1.96492474429743,
            //     2.09041892569708,
            //     1.83411862358093,
            //      2.3316965560831,
            //     1.99441690246242,
            //     2.04991828686767,
            //      2.0706823318835,
            //     2.06337436773907,
            //     2.28941978927821,
            //     2.22215350498702,
            //     2.32674373682571,
            //     1.88901431693404,
            //     2.41964518104566,
            //     1.93843961245602,
            //     2.39197820612532,
            //                    0,
            //     2.19622982806276,
            //      2.1235354712062,
            //     2.29497953216884,
            //                    0,
            //     2.17389078599935,
            //     2.38290169020514,
            //     2.37641930189825,
            //     2.28004005265543,
            //     2.09104778481841,
            //       2.262270476462,
            //     1.75961070307986,
            //     2.11913834969633,
            //      2.1298777398892,
            //     2.16337478303729,
            //     2.44521328711319,
            //     2.25694596891932,
            // }));

            return output;
        }

        void PhaseMatrixFunctionDataTest(ComputeImplementation impl)
        {
            auto msmc = ms->GetMSMainColumns();

            //select the first epoch only
            casacore::Vector<double> time = msmc->time().getColumn();
            double epoch = time[0];
            int nEpochs = 0;
            for(size_t i = 0; i < time.size(); i++)
            {
                if(time[i] == epoch) nEpochs++;
            }

            const int aSize = nEpochs;
            auto epochIndices = casacore::Slice(0, aSize, 1); //TODO assuming epoch indices are sorted

            casacore::Vector<std::int32_t> a1 = msmc->antenna1().getColumn()(epochIndices); 
            casacore::Vector<std::int32_t> a2 = msmc->antenna2().getColumn()(epochIndices);
            
            auto flagSlice = casacore::Slicer(
                casacore::IPosition(3,0,0,0),
                casacore::IPosition(3,1,1,aSize),
                casacore::IPosition(3,1,1,1));
            casacore::Vector<bool> fg = msmc->flag().getColumn()
            (flagSlice).reform(casacore::IPosition(1, aSize))
            (epochIndices);

            //Start calculations

            //output
            Eigen::MatrixXd A;
            Eigen::VectorXi I;
            Eigen::MatrixXd Ad;
            Eigen::MatrixXd A1;
            Eigen::VectorXi I1;
            Eigen::MatrixXd Ad1;

            if(impl == ComputeImplementation::casa)
            {
                casacore::Matrix<double> casaA;
                casacore::Array<std::int32_t> casaI;
                std::tie(casaA, casaI) = casalib::PhaseMatrixFunction(a1, a2, fg, -1);
                Ad = ToMatrix(icrar::casalib::PseudoInverse(casaA));

                casacore::Matrix<double> casaA1;
                casacore::Array<std::int32_t> casaI1;
                std::tie(casaA1, casaI1) = casalib::PhaseMatrixFunction(a1, a2, fg, 0);
                Ad1 = ToMatrix(icrar::casalib::PseudoInverse(casaA1));

                A = ToMatrix(casaA);
                I = ToVector(casaI);
                A1 = ToMatrix(casaA1);
                I1 = ToVector(casaI1);
            }
            if(impl == ComputeImplementation::cpu)
            {
                auto ea1 = ToVector(a1);
                auto ea2 = ToVector(a2);
                auto efg = ToVector(fg);
                std::tie(A, I) = cpu::PhaseMatrixFunction(ea1, ea2, efg, -1);
                Ad = icrar::cpu::PseudoInverse(A);

                std::tie(A1, I1) = cpu::PhaseMatrixFunction(ea1, ea2, efg, 0);
                Ad1 = icrar::cpu::PseudoInverse(A1);
            }
            if(impl == ComputeImplementation::cuda)
            {
                auto ea1 = ToVector(a1);
                auto ea2 = ToVector(a2);
                auto efg = ToVector(fg);
                std::tie(A, I) = cuda::PhaseMatrixFunction(ea1, ea2, efg, -1);
                Ad = icrar::cpu::PseudoInverse(A);

                std::tie(A1, I1) = cuda::PhaseMatrixFunction(ea1, ea2, efg, 0);
                Ad1 = icrar::cpu::PseudoInverse(A1);
            }

            double TOLERANCE = 0.00001;

            // A
            ASSERT_DOUBLE_EQ(4753, A.rows()); //-32=4754, -split=5152
            ASSERT_DOUBLE_EQ(128, A.cols());
            EXPECT_EQ(1.00, A(0,0));
            EXPECT_EQ(-1.00, A(0,1));
            EXPECT_EQ(0.00, A(0,2));
            //...
            EXPECT_NEAR(0.00, A(4752,125), TOLERANCE);
            EXPECT_NEAR(1.00, A(4752,126), TOLERANCE);
            EXPECT_NEAR(-1.00, A(4752,127), TOLERANCE);


            // I
            const int nBaselines = 4753;
            ASSERT_DOUBLE_EQ(nBaselines, I.size());
            ASSERT_EQ(4753, I.size());
            EXPECT_EQ(1.00, I(0));
            EXPECT_EQ(2.00, I(1));
            EXPECT_EQ(3.00, I(2));
            //...
            EXPECT_EQ(4848, I(4750));
            EXPECT_EQ(4849, I(4751));
            EXPECT_EQ(4851, I(4752));

            // Ad
            ASSERT_DOUBLE_EQ(128, Ad.rows());
            ASSERT_DOUBLE_EQ(4753, Ad.cols());
            // EXPECT_NEAR(2.62531368e-15, Ad(0,0), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(2.04033520e-15, Ad(0,1), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(3.25648083e-16, Ad(0,2), TOLERANCE); // TODO: emergent
            // //...
            // EXPECT_NEAR(-1.02040816e-02, Ad(127,95), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(-0.020408163265312793, Ad(127,96), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(-8.9737257304377696e-16, Ad(127,97), TOLERANCE); // TODO: emergent
            ASSERT_MEQD(A, A * Ad * A, TOLERANCE);

            //A1
            ASSERT_DOUBLE_EQ(97, A1.rows()); //-32=98, -split=102
            ASSERT_DOUBLE_EQ(128, A1.cols());
            EXPECT_DOUBLE_EQ(1.0, A1(0,0));
            EXPECT_DOUBLE_EQ(-1.0, A1(0,1));
            EXPECT_DOUBLE_EQ(0.0, A1(0,2));
            //...
            EXPECT_NEAR(0.00, A1(96,125), TOLERANCE);
            EXPECT_NEAR(0.00, A1(96,126), TOLERANCE);
            EXPECT_NEAR(-1.00, A1(96,127), TOLERANCE);

            //I1
            ASSERT_DOUBLE_EQ(97, I1.size());
            EXPECT_DOUBLE_EQ(1.00, I1(0));
            EXPECT_DOUBLE_EQ(2.00, I1(1));
            EXPECT_DOUBLE_EQ(3.00, I1(2));
            //...
            EXPECT_DOUBLE_EQ(95.00, I1(94));
            EXPECT_DOUBLE_EQ(96.00, I1(95));
            EXPECT_DOUBLE_EQ(97.00, I1(96));

            //Ad1
            ASSERT_DOUBLE_EQ(97, Ad1.cols());
            ASSERT_DOUBLE_EQ(128, Ad1.rows());
            //TODO: Ad1 not identical
            // EXPECT_DOUBLE_EQ(-9.8130778667735933e-18, Ad1(0,0)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(6.3742385976163974e-17, Ad1(0,1)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(3.68124219034074e-19, Ad1(0,2)); // TODO: emergent
            // //...
            // EXPECT_DOUBLE_EQ(5.4194040934156436e-17, Ad1(127,95)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(-1.0, Ad1(127,96)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(1.0, Ad1(127,97)); // TODO: emergent
            ASSERT_MEQD(A1, A1 * Ad1 * A1, TOLERANCE);
        }
    };

    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCasa) { PhaseMatrixFunction0Test(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCpu) { PhaseMatrixFunction0Test(ComputeImplementation::cpu); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCuda) { PhaseMatrixFunction0Test(ComputeImplementation::cuda); }

    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCasa) { PhaseMatrixFunctionDataTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCpu) { PhaseMatrixFunctionDataTest(ComputeImplementation::cpu); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCuda) { PhaseMatrixFunctionDataTest(ComputeImplementation::cuda); }

    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCasa) { RotateVisibilitiesTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCpu) { RotateVisibilitiesTest(ComputeImplementation::cpu); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCuda) { RotateVisibilitiesTest(ComputeImplementation::cuda); }
    
    TEST_F(PhaseRotateTests, PhaseRotateTestCasa) { PhaseRotateTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseRotateTestCpu) { PhaseRotateTest(ComputeImplementation::cpu); }
    TEST_F(PhaseRotateTests, PhaseRotateTestCuda) { PhaseRotateTest(ComputeImplementation::cuda); }
}
