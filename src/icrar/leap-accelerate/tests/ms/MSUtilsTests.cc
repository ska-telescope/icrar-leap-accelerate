/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#include <icrar/leap-accelerate/ms/utils.h>
#include <icrar/leap-accelerate/common/stream_extensions.h>
#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/tests/math/eigen_helper.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include <gtest/gtest.h>

class MSUtilsTests : public testing::Test
{
    const double TOLERANCE = 0.0001;
    
    casacore::MeasurementSet msMwa;
    casacore::MeasurementSet msAa4;

public:
    void SetUp() override
    {
        msMwa = casacore::MeasurementSet(std::string(TEST_DATA_DIR) + "/mwa/1197638568-split.ms");
        msAa4 = casacore::MeasurementSet(std::string(TEST_DATA_DIR) + "/aa4/aa4-SS-33-120.ms");
    }

    void TearDown() override
    {

    }

    void TestReadRecords()
    {
        unsigned int start_row = 0;
        unsigned int num_baselines = 196;

        auto uu = std::vector<double>(num_baselines);
        auto ww = std::vector<double>(num_baselines);
        auto vv = std::vector<double>(num_baselines);

        icrar::ms_read_coords(msMwa,
            start_row,
            num_baselines,
            uu.data(),
            vv.data(),
            ww.data());

        auto expectedUu = GetExpectedUU();
        auto expectedVv = GetExpectedVV();
        auto expectedWw = GetExpectedWW();

        ASSERT_VEQD(expectedUu, uu, TOLERANCE);
        ASSERT_VEQD(expectedVv, vv, TOLERANCE);
        ASSERT_VEQD(expectedWw, ww, TOLERANCE);
    }

    template<typename T>
    void TestReadVis()
    {
        using namespace std::complex_literals;
        unsigned int start_row = 0;
        unsigned int start_channel = 0;

        unsigned int slice_channels = 2;
        unsigned int slice_baselines = 3;
        unsigned int slice_pols = 4;

        auto rms = casacore::MeasurementSet(msMwa);
        auto msc = std::make_unique<casacore::MSColumns>(rms);

        //const size_t num_stations = (size_t) icrar::ms_num_stations(&msMwa);
        //slice_baselines = num_stations * (num_stations - 1) / 2;
        //slice_pols = msc->polarization().numCorr().get(0);
        //slice_channels = msc->spectralWindow().numChan().get(0);

        auto visibilities = Eigen::Tensor<std::complex<T>, 3>(slice_pols, slice_baselines, slice_channels);

        icrar::ms_read_vis(msMwa,
            start_row,
            start_channel,
            slice_channels,
            slice_baselines,
            slice_pols,
            "DATA",
            reinterpret_cast<T*>(visibilities.data())); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)

        ASSERT_EQ(slice_pols, visibilities.dimension(0));
        ASSERT_EQ(slice_baselines, visibilities.dimension(1));
        ASSERT_EQ(slice_channels, visibilities.dimension(2));
        ASSERT_EQCD(0.0+0.0i, visibilities(0,0,0), TOLERANCE);
        ASSERT_EQCD(0.0+0.0i, visibilities(1,0,0), TOLERANCE);
        ASSERT_EQCD(0.0+0.0i, visibilities(2,0,0), TOLERANCE);
        ASSERT_EQCD(0.0+0.0i, visibilities(3,0,0), TOLERANCE);
        ASSERT_EQCD(-0.703454494476318 + -24.7045249938965i, visibilities(0,1,0), TOLERANCE);
        ASSERT_EQCD(5.16687202453613 + -1.57053351402283i, visibilities(1,1,0), TOLERANCE);
        ASSERT_EQCD(-10.9083280563354 + 11.3552942276001i, visibilities(2,1,0), TOLERANCE);
        ASSERT_EQCD(-28.7867774963379 + 20.7210712432861i, visibilities(3,1,0), TOLERANCE); 

        //TODO(calgray): Column major reading
        //ASSERT_TEQ(GetExpectedVis(), visibilities, TOLERANCE);
    }

    void TestVisPerformance1()
    {
        //mwa
        // uint32_t num_pols = 4;
        // uint32_t num_channels = 48;
        // uint32_t num_baselines = 5253;
        // uint32_t num_timesteps = 14;

        //aa4
        uint32_t num_pols = 4;
        uint32_t num_channels = 33;
        uint32_t num_baselines = 130816;
        uint32_t num_timesteps = 1;

        auto visibilities = Eigen::Tensor<std::complex<double>, 3>(num_pols, num_baselines * num_timesteps, num_channels);
        icrar::ms_read_vis(msAa4,
            0,
            0,
            num_channels,
            num_baselines * num_timesteps,
            num_pols,
            "DATA",
            reinterpret_cast<double*>(visibilities.data())); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        EXPECT_EQ(num_pols, visibilities.dimension(0));
        EXPECT_EQ(num_baselines * num_timesteps, visibilities.dimension(1));
        EXPECT_EQ(num_channels, visibilities.dimension(2));
    }

    template<typename T>
    void TestVisPerformance2()
    {
        //mwa
        // uint32_t num_pols = 4;
        // uint32_t num_channels = 48;
        // uint32_t num_baselines = 5253;
        // uint32_t num_timesteps = 14;

        //aa4
        uint32_t num_pols = 4;
        uint32_t num_channels = 33;
        uint32_t num_baselines = 130816;
        uint32_t num_timesteps = 1;

        Eigen::Tensor<std::complex<T>, 3> visibilities = icrar::ms_read_vis1<std::complex<T>>(msAa4,
            0,
            0,
            num_channels,
            num_baselines * num_timesteps,
            num_pols,
            "DATA");

        EXPECT_EQ(2, visibilities.dimension(0));
        EXPECT_EQ(num_baselines * num_timesteps, visibilities.dimension(1));
        EXPECT_EQ(num_channels, visibilities.dimension(2));
    }

    template<typename T>
    void TestVisPerformance3()
    {
        //mwa
        // uint32_t num_pols = 4;
        // uint32_t num_channels = 48;
        // uint32_t num_baselines = 5253;
        // uint32_t num_timesteps = 14;

        //aa4
        uint32_t num_pols = 4;
        uint32_t num_channels = 33;
        uint32_t num_baselines = 130816;
        uint32_t num_timesteps = 1;

        Eigen::Tensor<std::complex<T>, 4> visibilities = icrar::ms_read_vis2<std::complex<T>>(msAa4,
            0,
            0,
            num_channels,
            num_baselines,
            num_pols,
            num_timesteps,
            "DATA");
        EXPECT_EQ(2, visibilities.dimension(0));
        EXPECT_EQ(num_channels, visibilities.dimension(1));
        EXPECT_EQ(num_baselines, visibilities.dimension(2));
        EXPECT_EQ(num_timesteps, visibilities.dimension(3));
    }

private:
    std::vector<double> GetExpectedUU()
    {
        return std::vector<double>
        {
            0, -213.234574834057, -126.130233053304, -171.088059584787, 24.8832968787911, 75.9030116816415, 31.245674509303, 131.830257530978,
            401.958029973594, 400.081348645036, 297.869416811878, 409.233549684089, 384.100584883511, 584.903582556051, 609.706162817685,
            1006.99336853124, 912.026587829394, 138.127819921149, 100.479896312256, 130.26521840149, 80.8504201109847, 137.198986319818, 733.459486795056,
            -869.239479202946, -380.056059687537, -338.953745657986, -175.475837483463, -56.9845612182013, -953.365320135896, -1272.29364709852,
            -1378.64886953924, -609.290318352972, -634.568642438031, -559.413564178326, -608.414011892997, -474.108230109677, -1134.52854769092,
            -508.174225421552, -402.897460266823, -568.196979552495, -539.171313039116, -562.682107740621, -591.616091685183, -403.097170078042,
            59.4834065277692, -43.5280601988355, -335.731172157637, 133.316296154809, 499.616982347651, 597.122982982116, 673.835856340046, 597.274536789626,
            536.821276382862, 1899.07135788141, 1687.22102336035, 1530.8731095236, 1417.71549188342, 1337.04067536405, 1156.82291048554, 1097.79627442992,
            897.425527655708, 544.453807573374, 31.4666531727684, 223.908871230464, -104.71195303456, -180.429491317839, -1028.5247700206, -761.437176404545,
            -535.411523281294, -1094.18257680237, -1054.65386184022, -1884.30406453937, -2022.96371820072, -2214.56145126538, -2591.94620847363, -2594.18848000359,
            -2463.63501821009, -2927.70876308412, -3232.15662459114, -2987.02319024555, -1738.71402963224, -2053.31670359623, -2414.8051724614, -2272.28486972627,
            -2769.99609315878, -2710.39517135088, -2402.07729503966, -757.731497251641, -1500.48317313861, -1135.39745012848, -1487.88972981907, -1876.4044263105,
            -2259.72310840189, -1910.10104366261, -117.236467034634, -411.920936461338, -190.954541027847, -707.33161853491, -1025.51807814974, -780.496734909115,
            -1186.5352909369, -1553.06453862741, 0, 87.1043417807526, 42.14651524927, 238.117871712848, 289.137586515699, 244.48024934336, 345.064832365035,
            615.192604807651, 613.315923479093, 511.103991645935, 622.468124518147, 597.335159717568, 798.138157390108, 822.940737651742, 1220.2279433653,
            1125.26116266345, 351.362394755206, 313.714471146313, 343.499793235547, 294.084994945042, 350.433561153875, 946.694061629113, -656.004904368889,
            -166.82148485348, -125.719170823929, 37.7587373505943, 156.250013615856, -740.130745301839, -1059.05907226446, -1165.41429470518, -396.055743518915,
            -421.334067603974, -346.178989344268, -395.17943705894, -260.873655275619, -921.293972856864, -294.939650587495, -189.662885432766, -354.962404718438,
            -325.936738205058, -349.447532906564, -378.381516851126, -189.862595243985, 272.717981361826, 169.706514635222, -122.49659732358, 346.550870988866,
            712.851557181708, 810.357557816174, 887.070431174103, 810.509111623683, 750.05585121692, 2112.30593271546, 1900.45559819441, 1744.10768435766,
            1630.95006671748, 1550.27525019811, 1370.05748531959, 1311.03084926398, 1110.66010248976, 757.688382407431, 244.701228006825, 437.143446064521,
            108.522621799498, 32.8050835162184, -815.290195186547, -548.202601570488, -322.176948447237, -880.948001968315, -841.419287006162, -1671.06948970531,
            -1809.72914336667, -2001.32687643132, -2378.71163363957, -2380.95390516953, -2250.40044337603, -2714.47418825006, -3018.92204975708, -2773.78861541149,
            -1525.47945479818, -1840.08212876218, -2201.57059762734, -2059.05029489222, -2556.76151832472, -2497.16059651682, -2188.8427202056, -544.496922417584,
            -1287.24859830456, -922.162875294422, -1274.65515498501, -1663.16985147644, -2046.48853356783, -1696.86646882855, 95.9981077994236
        };
    }
    std::vector<double> GetExpectedVV()
    {
        return std::vector<double>
        {
            0, 135.473926784922, 169.064851738458, 319.068968204233, 123.7967645586, 139.580031204316, 0.491465408359545, -51.4308327921503, 126.269542566514,
            -42.8597480028707, -42.8593608206351, -419.459225570206, -396.399063614196, -2.82362540122222, -61.1007154345691, 54.4037749472961, -358.727726924461,
            -1289.83050714666, -648.846640780999, -643.146324819389, -535.504093373364, -508.215327180105, -796.027387768997, -1095.94867737627, -423.813165630257,
            -591.73538413285, -584.182036555205, -578.010799914816, -750.342020109981, -647.292076247665, -394.291310972808, -116.613444496752, -213.398821609818,
            -266.696876358163, -363.803648564412, -455.45518276327, -52.264911616887, 313.664119005009, 126.695976974399, 166.448170959927, 93.2486672977004,
            26.521250004884, 959.890753862305, 794.75206562539, 528.378245175097, 442.466650229463, 353.59366144431, 458.914121804042, 801.763325670822,
            842.853994466165, 655.603095460038, 323.331959951853, 332.356960956254, -28.4597963289403, 96.3207104820781, 192.380766171604, 299.433772412922,
            514.688725693068, 608.062105567031, 714.647912322484, 778.395095245909, -2039.64225840185, -2272.55451997827, -1606.00576013879, -2119.66115011181,
            -1700.74648667834, -2047.13037319047, -1887.75864103111, -1198.89853447853, -1751.34455586467, -1336.26192087977, -1743.43616406626, -988.806206843199,
            -1434.78429120802, -1810.50627803479, -1452.58305241504, -1099.62450890646, -1560.32424206788, -1345.10096513654, -1184.258568485, -646.284074118204,
            -471.720628962821, -812.456444217966, -292.294715817491, -741.51538984482, -439.766362371569, 27.1035802644985, 767.290194486131, -83.6896878065603,
            655.641352100391, 849.241201456278, 147.685068959358, 260.768273206925, 845.556317182374, 931.320943503773, 1339.90036497811, 1829.44571695761,
            1267.00501673714, 1035.23298198641, 1740.36535450099, 1640.51039372462, 1202.59542169983, 0, 33.5909249535359, 183.595041419311, -11.6771622263227,
            4.106104419394, -134.982461376563, -186.904759577073, -9.20438421840836, -178.333674787793, -178.333287605558, -554.933152355129, -531.872990399119,
            -138.297552186145, -196.574642219491, -81.0701518376263, -494.201653709383, -1425.30443393159, -784.320567565921, -778.620251604311, -670.978020158287,
            -643.689253965028, -931.50131455392, -1231.42260416119, -559.287092415179, -727.209310917772, -719.655963340128, -713.484726699739, -885.815946894903,
            -782.766003032587, -529.765237757731, -252.087371281675, -348.87274839474, -402.170803143085, -499.277575349334, -590.929109548193, -187.738838401809,
            178.190192220087, -8.77794981052358, 30.9742441750046, -42.2252594872219, -108.952676780038, 824.416827077382, 659.278138840468, 392.904318390175,
            306.992723444541, 218.119734659388, 323.44019501912, 666.289398885899, 707.380067681242, 520.129168675116, 187.858033166931, 196.883034171332,
            -163.933723113863, -39.1532163028443, 56.906839386682, 163.959845627999, 379.214798908145, 472.588178782108, 579.173985537561, 642.921168460987,
            -2175.11618518677, -2408.0284467632, -1741.47968692372, -2255.13507689673, -1836.22041346326, -2182.60429997539, -2023.23256781603, -1334.37246126345,
            -1886.81848264959, -1471.73584766469, -1878.91009085118, -1124.28013362812, -1570.25821799294, -1945.98020481971, -1588.05697919996, -1235.09843569138,
            -1695.7981688528, -1480.57489192146, -1319.73249526992, -781.758000903126, -607.194555747744, -947.930371002888, -427.768642602413, -876.989316629742,
            -575.240289156491, -108.370346520424, 631.816267701208, -219.163614591483, 520.167425315468, 713.767274671356, 12.2111421744355, 125.294346422003, 710.082390397451, 795.847016718851
        };
    }
    std::vector<double> GetExpectedWW()
    {
        return std::vector<double>
        {
            0, 136.990822255294, 139.291586460673, 246.436375079676, 76.529474707634, 76.0106037503559, -6.69782470535193,
            -62.5921328122801, -1.93119143118611, -111.93733392341, -91.0785532503598, -358.125841400359, -337.95658113004,  -125.230602553655,
            -168.503274083297, -181.906711457168, -428.570375281322, -868.195143937159, -443.160467302938, -445.606841515537,  -365.12444285657,
            -359.039995276986, -672.196727897135, -528.025361835215, -193.721822111275, -313.393971943387, -344.0300186785, -364.546598418132,
            -281.044550287589, -143.982423850562, 46.9340672948513, 60.5489517181626, 2.29762540407233, -49.4090873148754, -102.262093538964,
            -193.09040413516, 219.982124576062, 319.594313512271, 172.975620558296, 236.557838564276, 182.330007192696, 143.901436052993, 
            763.749001106886, 613.400484918897, 338.188397815304, 302.750229367421, 306.232189788561, 276.011421719851, 418.972146525812,
            425.10138365446, 286.043434606862, 84.547379060888, 103.870446621154, -423.761040346968, -298.108623896573, -202.746934802232,
            -108.99078687786, 51.0887316996343, 151.347342388149, 234.731510906064, 319.399745789019, -1445.42730855565, -1492.05111370257,
            -1093.78480387467, -1363.17385629258, -1071.34171731899, -1120.81917006958, -1072.36521867605, -667.573928723549, -912.002311893606,
            -647.855962531558, -741.095135653962, -209.711024065352, -464.606065358386, -632.159006014165, -392.852167828113, -187.643866712285,
            -392.803012776843, -184.064462303605, -131.123848157709, -42.7857028896351, 137.717316544794, -8.58659539046886, 305.240007044374,
            116.126845039312, 302.780511141426, 545.195346459592, 673.404609748124, 278.148008903684, 685.048461001437, 889.096244299655,
            512.106687534055, 669.319513135958, 980.541649122212, 643.588356351719, 976.848267515202, 1253.36410537354, 993.880334842916,
            907.410971369358, 1322.93505594728, 1344.16700919014, 1136.60831243451, 0, 2.30076420537912, 109.445552824382,
            -60.4613475476598, -60.9802185049379, -143.688646960646, -199.582955067574, -138.92201368648, -248.928156178704, -228.069375505654,
            -495.116663655653, -474.947403385334, -262.221424808949, -305.49409633859, -318.897533712461, -565.561197536615, -1005.18596619245,
            -580.151289558232, -582.59766377083, -502.115265111863, -496.03081753228, -809.187550152429, -665.016184090509, -330.712644366568,
            -450.384794198681, -481.020840933794, -501.537420673426, -418.035372542883, -280.973246105856, -90.0567549604425, -76.4418705371312,
            -134.693196851221, -186.399909570169, -239.252915794258, -330.081226390454, 82.9913023207684, 182.603491256977, 35.9847983030019,
            99.567016308982, 45.3391849374026, 6.9106137976992, 626.758178851592, 476.409662663603, 201.19757556001, 165.759407112127,
            169.241367533267, 139.020599464557, 281.981324270518, 288.110561399166, 149.052612351569, -52.4434431944058, -33.1203756341403,
            -560.751862602261, -435.099446151867, -339.737757057525, -245.981609133154, -85.9020905556596, 14.3565201328553, 97.7406886507698,
            182.408923533725, -1582.41813081094, -1629.04193595786, -1230.77562612997, -1500.16467854788, -1208.33253957429, -1257.80999232487,
            -1209.35604093134, -804.564750978843, -1048.9931341489, -784.846784786852, -878.085957909256, -346.701846320646, -601.59688761368,
            -769.149828269459, -529.842990083406, -324.634688967579, -529.793835032137, -321.055284558899, -268.114670413003, -179.776525144929,
            0.726494289500636, -145.577417645763, 168.24918478908, -20.8639772159819, 165.789688886132, 408.204524204298, 536.41378749283,
            141.15718664839, 548.057638746143, 752.105422044361, 375.115865278761, 532.328690880665, 843.550826866919, 506.597534096425
        };
    }

    Eigen::Tensor<std::complex<float>, 3> GetExpectedVis()
    {
        using namespace std::complex_literals;
        auto v = Eigen::Tensor<std::complex<float>, 3>(4, 3, 2);
        v.setValues({
            {
                {{0,0}, {-10.9083f,11.3553f}, {0,0}, {-21.3128f,-9.91422f}},
                {{0,0}, {7.89942f,33.7481f}, {0,0}, {1.80109f,4.9f}},
                {{-0.703454f,-24.7045f}, {-25.9081f,8.35707f}, {24.1003f,17.5731f}, {33.0251f,34.0921f}},
            },
            {
                {{0,0}, {-28.7868f,20.7211f}, {0,0}, {-9.91203f,-28.6091f}},
                {{0,0}, {36.844f,9.9157f}, {0,0}, {8.72433f,-6.79423f}},
                {{5.16687f,-1.57053f}, {-28.6226f,17.7872f}, {-13.3485f,23.2287f}, {-22.3512f,23.9786f}},
            }
        });
        return v;
    }
};

TEST_F(MSUtilsTests, TestReadRecords) { TestReadRecords(); }
TEST_F(MSUtilsTests, TestReadVis) { TestReadVis<float>(); }

TEST_F(MSUtilsTests, TestVisPerformance11) { TestVisPerformance1(); }
TEST_F(MSUtilsTests, TestVisPerformance12) { TestVisPerformance1(); }
TEST_F(MSUtilsTests, TestVisPerformance13) { TestVisPerformance1(); }
TEST_F(MSUtilsTests, TestVisPerformance21f) { TestVisPerformance2<float>(); }
TEST_F(MSUtilsTests, TestVisPerformance22f) { TestVisPerformance2<float>(); }
TEST_F(MSUtilsTests, TestVisPerformance23f) { TestVisPerformance2<float>(); }
TEST_F(MSUtilsTests, TestVisPerformance21d) { TestVisPerformance2<double>(); }
TEST_F(MSUtilsTests, TestVisPerformance22d) { TestVisPerformance2<double>(); }
TEST_F(MSUtilsTests, TestVisPerformance23d) { TestVisPerformance2<double>(); }
TEST_F(MSUtilsTests, TestVisPerformance31f) { TestVisPerformance3<float>(); }
TEST_F(MSUtilsTests, TestVisPerformance32f) { TestVisPerformance3<float>(); }
TEST_F(MSUtilsTests, TestVisPerformance33f) { TestVisPerformance3<float>(); }
TEST_F(MSUtilsTests, TestVisPerformance31d) { TestVisPerformance3<double>(); }
TEST_F(MSUtilsTests, TestVisPerformance32d) { TestVisPerformance3<double>(); }
TEST_F(MSUtilsTests, TestVisPerformance33d) { TestVisPerformance3<double>(); }
