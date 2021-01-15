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

#pragma once

#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>

#include <vector>
#include <utility>

namespace icrar
{
    cpu::CalibrationCollection GetEachTimestepMWACalibration()
    {
      return cpu::CalibrationCollection(std::vector<std::vector<cpu::BeamCalibration>>());
    }

    /**
     * @brief Gets the expected calibration output averaging over all timesteps. From LEAP-Cal:ported
     * 
     * @return a vector of direction and antenna calibration pairs
     */
    cpu::CalibrationCollection GetAllTimestepsMWACalibration()
    {
        std::vector<std::pair<SphericalDirection, std::vector<double>>> output;
        output.push_back(std::make_pair(SphericalDirection(-0.4606549305661674,-0.29719233792392513), std::vector<double>
        {
           2.89283703997894e-14,
               1.29172837617279,
          -1.10195708726992e-14,
          -6.02408004172075e-15,
              0.615639516380927,
              -2.41522929076744,
              -1.96419036519531,
              0.752903665858852,
               0.43923239802379,
           1.85418581679986e-14,
           1.00347183886339e-14,
          -9.99968543056696e-15,
               2.84474142899086,
              0.423204561650745,
              -1.32299769087009,
           9.16345434543845e-16,
               2.78886001032021,
                2.6513281038066,
           1.26974146565845e-15,
            1.7708279195666e-15,
               2.38887583822479,
             -0.469354066935844,
              0.195548598592461,
               1.00488539772672,
              -1.74317946909799,
             -0.762819912838224,
          -6.36265305800664e-15,
             -0.817083934832943,
               1.93309971438933,
               2.02353164600282,
           4.10894528924106e-15,
              -1.60237672048941,
             -0.343719184462975,
              -1.27230548155094,
          -2.98924335825357e-15,
              0.178406493754494,
            -0.0610574885241317,
              0.453888676570319,
          -9.50559017484861e-16,
           1.89896233972081e-15,
             -0.857647215415693,
          -0.000131971073708693,
              -2.56944937287897,
               1.44296117035914,
               1.55123013273198,
              -1.15909536459586,
              0.303140354844319,
               -2.3255844881271,
            -0.0331824923426527,
           -4.5766388959511e-16,
              -1.60761172665587,
          -1.63185387770227e-15,
               1.60391376380987,
              -1.23628128153521,
               1.43641896761173,
             -0.115806207909914,
                  1.57264127708,
             -0.620614303001209,
               1.52412937087027,
               2.64999118636214,
            2.4513732560664e-16,
           5.08293347220226e-16,
           1.30108812285813e-15,
          -1.27401451469353e-15,
           1.06954632796716e-15,
               2.10015847656664,
               1.86577910111364,
              -1.47202131520786,
               -1.8522084465661,
              -1.29799132932937,
           -1.1473004417941e-16,
           -5.0366436627436e-16,
                2.9785973168015,
              -2.61723467811308,
               2.47556527971796,
               3.23711625555856,
             -0.862371218773955,
                1.7096505326303,
               -1.7394791891572,
               2.51521807159921,
            3.7470629382584e-16,
             -0.915291113636368,
               -0.7536396204446,
               1.22360420919851,
              0.463789362905379,
              0.699917282929976,
             -0.284957811584928,
          -4.78842310630486e-16,
              -0.82025488707225,
              -2.69107736177905,
               0.92119943960637,
           7.49783849123814e-17,
              0.849329088662967,
           -3.8514373961025e-16,
           9.35307240377321e-16,
           1.08785930700025e-16,
              -1.94314172584193,
                 1.639450532543,
             0.0528463135769317,
              0.854349004247568,
             -0.779362475456103,
               2.42370772716365,
             -0.951797892429537,
               1.56978439558617,
               1.48854402048733,
               1.52992251302733,
              -2.99717818392389,
                2.6790859393487,
             -0.181448711286958,
              -1.37334068824988,
               1.13379123800312,
           2.37934926188629e-16,
             -0.253279090688323,
               1.54190103196744,
               2.70264715599737,
           -1.3861801699076e-16,
              -1.77104353813479,
              0.850835258943672,
               2.80189974305559,
             -0.472848527966362,
               2.98668985630865,
             -0.421962452926599,
               2.65672828325552,
               3.18228423999102,
              -2.37288585159801,
             -0.564285207909852,
               1.12449080448426,
              -2.53451552736576,
        }));

        output.push_back(std::make_pair(SphericalDirection(-0.753231018062671,-0.44387635324622354), std::vector<double>
        {
           3.83351589782366e-14,
               1.16256032787742,
           1.70664763371784e-14,
          -9.36371818750463e-15,
              -1.75851123061503,
               3.47476986781203,
              -1.59625067526502,
               1.56608399729856,
               2.56632551387209,
          -3.40691564734199e-15,
          -2.51320440917344e-15,
          -9.84070615659185e-15,
              -1.02114670382977,
              -2.27620220800839,
              0.443030487523898,
          -9.62077237366401e-15,
               2.53949216686423,
              -2.20721010081191,
          -1.92785119800563e-15,
          -3.16031730720504e-15,
               1.18703585178308,
             -0.540123312423631,
               2.44690051810241,
               2.63274377839799,
               2.88971278566055,
               2.65885046022196,
          -6.15687349163142e-15,
              0.494443685112384,
              -1.31555740139746,
               -2.8034942451224,
           2.75729963482722e-16,
               -0.6659452768819,
               2.70034357421901,
               2.55201295707778,
          -1.61803856397729e-15,
              -2.22895625196703,
            -0.0585598971002885,
              -2.59363587999907,
           1.92973455116138e-15,
           9.48348395620031e-16,
              -1.33115469136966,
               2.60636948958735,
               1.96919599016174,
             -0.979830408956914,
               1.41198653251218,
              -1.34851696849211,
             -0.661045645071848,
              -1.89759658876537,
              -1.83229844665204,
           -9.1873964603917e-16,
              -2.89184030638364,
           2.60141869373703e-15,
                -0.716179739692,
              -2.65833221024708,
               -2.7901611591455,
              0.519683250865405,
               1.53536459948124,
               1.04395917713711,
               2.03997931720425,
               1.24136377020829,
          -9.72984950216887e-16,
          -7.85101022560361e-16,
          -3.38608490308313e-16,
           5.41524536903711e-16,
           6.86529877921258e-16,
               2.88792968162268,
            -0.0494909065235808,
              -2.56350981005506,
               -2.6274142536339,
              0.536386228612682,
          -1.04620801114444e-16,
          -7.07388718770592e-17,
               2.64369081083792,
               0.93577572353844,
              -2.92283601811635,
              -1.67630094909374,
              -2.20511936224048,
              -2.09766345910252,
              0.348597329797571,
              -1.20848351196646,
           1.70792063843241e-16,
              0.374689784851185,
              -1.31150658425182,
              0.844863048694103,
              0.537084525326734,
              -1.25926700777318,
              -1.40811631752702,
           1.10966080061759e-16,
               2.73767430628344,
              -1.48030954735188,
             -0.179776215982521,
          -1.73094378817194e-16,
              0.130117080537245,
           1.15748926120206e-16,
           6.67241587999602e-16,
          -8.91694469173964e-17,
             -0.689895565165254,
                2.4982911296107,
              -1.30866520811601,
               2.64267976036319,
               2.31784629390906,
             -0.438376802258681,
             -0.380009686022328,
              0.407550587595973,
              -1.48002850465093,
              -1.83278206347306,
              -3.20176395185475,
              -3.04212403729609,
               -2.0180875555315,
              -2.23686843929735,
              -2.74328745468605,
          -4.34313188197761e-17,
             -0.194624285389279,
               -2.4934117293877,
              0.510821203073761,
          -1.73807866530522e-16,
               2.49370964020408,
               1.77873947619319,
             -0.758925178834453,
               1.78709211294679,
              -3.23068652100652,
              0.264128374042389,
             -0.100968233652098,
               2.32512864409159,
              -2.03546457404303,
              -1.06663951065391,
                1.1708114845029,
              0.621967188914773,
        }));
        return cpu::CalibrationCollection(std::vector<std::vector<cpu::BeamCalibration>>());
    }

    /**
     * @brief Gets the expected calibration output averaging over the first timestep.
     * 
     * @return std::vector<std::pair<SphericalDirection, std::vector<double>>> 
     */
    cpu::CalibrationCollection GetFirstTimestepMWACalibration()
    {
        std::vector<std::pair<SphericalDirection, std::vector<double>>> output;
        output.push_back(std::make_pair(SphericalDirection(-0.4606549305661674,-0.29719233792392513), std::vector<double>
        {
           6.25039512024925e-15,
               2.58353835690993,
          -1.81107505054403e-14,
          -1.64199987572405e-14,
             -0.252472710942695,
              -2.28861688865054,
              0.228732006874436,
              0.686597293381812,
               1.77804991928562,
           1.37849930565368e-14,
           5.18227608409766e-15,
          -6.82977964653949e-15,
               1.58507785224764,
               3.07223349928847,
             -0.434261577124824,
          -6.35663525887017e-16,
             -0.352323879706292,
              0.768511536592427,
          -1.62446514833804e-14,
           2.16177284572499e-14,
              0.751967713142276,
              -1.72867270249562,
              -2.91716283801553,
               1.37674388762445,
              -1.24418342056317,
             -0.313685258160726,
          -1.22732236813955e-14,
                -1.762037419329,
              -3.26620529262608,
              0.901368846793849,
          -1.57445236020344e-16,
              -2.09893634305654,
             0.0864018749811722,
              -2.32727594563197,
          -2.69720747296311e-15,
              0.470229067270681,
             -0.751762876841271,
               -2.9249033091111,
           -2.4345224048514e-15,
           1.31244534811873e-15,
              -3.06053409584614,
              0.208794242934082,
               1.37606785079465,
               2.02983657601205,
                 2.208700687447,
              0.158077697355951,
              0.540061087938519,
                1.8702280877249,
             0.0363170520990879,
           2.83013902408683e-15,
              -0.95109480788268,
          -8.11325021875138e-16,
               1.57390437128947,
              -2.02253352801162,
               2.18512515718179,
               2.04802525734083,
              0.188576980562485,
               1.87244993573164,
                 1.229397792827,
              0.910636434227151,
          -4.96436330903456e-17,
           4.13838478716039e-16,
           6.66672554944726e-16,
          -4.03180327535415e-16,
           2.63507774058517e-16,
              -1.27905468769428,
               2.48296822953083,
              -2.94209750669712,
              -3.05910664109189,
             -0.932367037941525,
           6.56762565879835e-16,
          -2.13578542705132e-16,
               2.69012825462836,
              -2.21541992541925,
               2.05400808513996,
              -2.89978964807358,
               1.06661542098682,
              -2.17387037300521,
               2.40913390784897,
             -0.823532093539714,
          -9.05598590922489e-16,
              -2.09059041603227,
              -1.93958694444698,
              -2.23814458176794,
              0.311958773383149,
              0.275885833710506,
               1.36171665986458,
            5.1442216320638e-16,
               -2.2416703004565,
             -0.291473013454579,
               1.66149737027794,
          -3.19573676749043e-16,
               2.68418387605794,
          -1.62073110480294e-16,
           -1.4755098907761e-16,
          -1.17471736173718e-16,
             -0.473873995595682,
              -1.47557615055401,
                1.8498914066787,
             0.0463338754262231,
             0.0594792265752265,
              -1.69176610625886,
              -2.12487235615182,
               1.45558687815288,
               1.61972607029488,
               2.00270938543747,
              0.383126758761379,
              -2.74628151930774,
             -0.256950527108853,
               2.69475991753122,
              0.303645280984064,
           5.36048009923462e-17,
             -0.410716088573861,
              -1.22451141984037,
              -2.55201827261388,
          -1.04364060422462e-16,
             -0.873744576906659,
              0.979952921590555,
               2.44787739128521,
              0.533779611529558,
              -2.44282737108892,
              0.654199541595436,
             -0.678890264085049,
               2.57396940953619,
              -1.92562096029658,
              -2.87751126558295,
               1.31861515904311,
             -0.340505317898034
        }));

        output.push_back(std::make_pair(SphericalDirection(-0.753231018062671,-0.44387635324622354), std::vector<double>
        {
          -4.59114312747322e-16,
              -1.38317740431689,
           4.18428173439873e-15,
           2.94292129983241e-14,
              -2.08854815671236,
               1.15559477572133,
               -2.5493831417105,
               1.71420419275696,
              0.759579727986383,
          -3.54146528530426e-14,
          -8.50579216027767e-16,
          -3.54520976176802e-15,
                -0.916183292255,
              -2.97029407754923,
                1.7882227483281,
          -1.79069697277257e-14,
               3.07918100413635,
              -1.71148704925558,
           2.01243970416388e-14,
          -8.21145113961271e-15,
              0.330810499597021,
              -1.15143389209663,
               3.22525607619763,
               1.95819121064189,
             -0.339348011374135,
               1.36770920395569,
           5.69811415739305e-15,
             -0.473069650650574,
                2.2818360742147,
               1.97981886777905,
          -2.07800681641081e-15,
               2.76350005786059,
               1.75760615879099,
               2.64386823152505,
             3.175919223258e-16,
             -0.726451203705964,
               1.59693965951415,
                2.5812404571729,
          -7.80429192547327e-16,
          -2.22797741998133e-16,
             -0.346492370395025,
              -2.72310373875308,
               2.60780592592165,
              -2.92561290505777,
              0.473579469073129,
               0.16072013177686,
              -1.92932983277484,
             -0.495414641851889,
               1.66371678916241,
           1.23279449483387e-15,
               2.95188822778956,
           1.04080662369492e-15,
             -0.200826983279155,
              -2.99277151437142,
              -2.69893667859373,
              0.514333278225677,
               2.43445397214826,
               1.34870310079376,
              -3.07754792950629,
              -2.60698715943064,
          -7.89601072705704e-16,
          -5.12280225314279e-16,
           5.19177740569753e-17,
            9.2507553361377e-16,
           1.11701535664638e-15,
              -2.48869988044042,
              0.238227275711812,
               2.87162789218186,
               2.02577063699946,
               1.47116105198316,
          -3.56566807786671e-16,
           4.83495517229494e-16,
               2.09645469702107,
              0.449830567768782,
              -2.68670555629483,
              -1.21765438614944,
               2.35184147791499,
              -2.04891012634857,
               1.26857049316465,
               1.02679700437764,
             1.940781025311e-16,
               2.68838553303412,
               -2.3180177752967,
              -1.02967170816098,
              -2.80830241512263,
              -0.52247882149343,
                -1.623642296974,
            7.7690782892932e-17,
              0.856928222073718,
              0.690146152851834,
               1.52426216154134,
          -5.31754493301204e-17,
              -1.26951430767456,
           1.18186535464914e-18,
           2.49277642183565e-16,
           1.29044891606317e-16,
              0.483810225059247,
                1.5069286485975,
             -0.617692239810053,
              0.518425229160385,
               1.64748271809938,
              -3.19100614456074,
              -2.62061095847826,
              0.263010000020195,
               -1.3208692115465,
              -1.82375789762513,
               2.26733182664494,
              0.222674212030279,
              -1.76456907576111,
             -0.575042393895367,
               2.84411860324511,
           4.66008431003833e-17,
               1.89283381000264,
              -2.11302315788024,
              0.267994718616895,
          -1.67419366153516e-16,
               2.32252853343022,
              0.861814073929316,
               2.94758737178243,
               2.82441244119669,
              -1.63933943054983,
            -0.0503748335113503,
               2.69690453006316,
               3.25750715440819,
              -2.17892381464458,
             -0.757316793432348,
               2.91996434066302,
               1.19117497840134,
        }));
        //return cpu::CalibrationCollection(std::vector<cpu::Calibration>{output});
        return cpu::CalibrationCollection(std::vector<std::vector<cpu::BeamCalibration>>());
    }
}