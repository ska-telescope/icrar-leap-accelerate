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

#include <icrar/leap-accelerate/common/config/Arguments.h>

#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
#include <icrar/leap-accelerate/algorithm/cpu/CpuLeapCalibrator.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/core/stream_out_type.h>
#include <icrar/leap-accelerate/core/git_revision.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/UsageReporter.h>
#include <icrar/leap-accelerate/core/version.h>

#include <boost/program_options.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

#include <iostream>
#include <queue>
#include <string>
#include <exception>

#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>

using namespace icrar;
namespace po = boost::program_options;

/**
 * @brief Combines command line arguments into a formatted string
 * 
 */
std::string arg_string(int argc, char** argv)
{
    std::stringstream ss;
    for(int i = 0; i < argc; i++)
    {
        ss << argv[i] << " "; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
    return ss.str();
}

/**
 * @brief Displays project version information including git info
 * 
 * @param name 
 * @return std::string 
 */
std::string version_information(const char *name)
{
    std::ostringstream os;
    os << name << " version " << version() << '\n'
       << "git revision " << git_sha1() << (git_has_local_changes() ? "-dirty\n" : "\n");
    os << name << " built on " << __DATE__ << ' ' << __TIME__;
    return os.str();
}

int main(int argc, char** argv)
{
    auto appName = "LeapAccelerateCLI";

    po::options_description desc(appName);

    CLIArgumentsDTO rawArgs;
    desc.add_options()
        ("help,h", "display help message")
        ("version,v", "display version information")
        ("config,c", po::value<boost::optional<std::string>>(&rawArgs.configFilePath), "Configuration file relative path")
        // TODO(calgray): app.add_option("-i,--input-type", rawArgs.source, "Input source type");
        ("filepath,f", po::value<boost::optional<std::string>>(&rawArgs.filePath), "Measurement set file path")
        ("streamOut", po::value<boost::optional<std::string>>(&rawArgs.streamOutType), "Stream out setting (c=collection, s=singlefile, m=multiplefile)")
        ("output,o", po::value<boost::optional<std::string>>(&rawArgs.outputFilePath), "Calibration output file path")
        ("directions,d", po::value<boost::optional<std::string>>(&rawArgs.directions), "Directions to calibrations")
        ("stations,s", po::value<boost::optional<int>>(&rawArgs.stations), "Overrides number of stations in measurement set")
        ("referenceAntenna,r", po::value<boost::optional<unsigned int>>(&rawArgs.referenceAntenna), "Specifies the reference antenna, defaults to the last antenna")
        ("implementation,i", po::value<boost::optional<std::string>>(&rawArgs.computeImplementation), "Compute implementation type (cpu, cuda)")
        ("solutionInterval,n", po::value<boost::optional<std::string>>(&rawArgs.solutionInterval), "Sets the intervals to generate solutions for, [start, interval, end]")
        ("autoCorrelations,a", po::value<boost::optional<bool>>(&rawArgs.readAutocorrelations), "Set to true if measurement set rows store autocorrelations")
        ("minimumBaselineThreshold,m", po::value<boost::optional<double>>(&rawArgs.minimumBaselineThreshold), "Minimum baseline length in meters")
        ("useFileSystemCache,u", po::value<boost::optional<bool>>(&rawArgs.useFileSystemCache), "Use filesystem caching between calls")
        ("useCusolver", po::value<boost::optional<bool>>(&rawArgs.useCusolver), "Use cusolver for fast matrix inversion")
        ("useIntermediateBuffer", po::value<boost::optional<bool>>(&rawArgs.useIntermediateBuffer), "Use extra device buffers as cache")
        ("verbosity", po::value<boost::optional<int>>(&rawArgs.verbosity), "Verbosity (0=fatal, 1=error, 2=warn, 3=info, 4=debug, 5=trace), defaults to info");

    try
    {
        po::variables_map variablesMap;
        po::store(po::parse_command_line(argc, argv, desc), variablesMap);
        po::notify(variablesMap);

        if(variablesMap.count("help"))
        {
            std::cout << desc << std::endl;
        }
        else if (variablesMap.count("version"))
        {
            std::cout << version_information(appName) << std::endl;
        }
        else
        {
            icrar::profiling::UsageReporter _;
            ArgumentsValidated args = { ArgumentsDTO(std::move(rawArgs)) };

            LOG(info) << version_information(argv[0]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            LOG(info) << arg_string(argc, argv);

            if(IsImmediateMode(args.GetStreamOutType()))
            {
                auto calibrator = LeapCalibratorFactory::Create(args.GetComputeImplementation());

                auto outputCallback = [&](const cpu::Calibration& cal)
                {
                    cal.Serialize(*args.CreateOutputStream(cal.GetStartEpoch()));
                };

                calibrator->Calibrate(
                    outputCallback,
                    args.GetMeasurementSet(),
                    args.GetDirections(),
                    args.GetSolutionInterval(),
                    args.GetMinimumBaselineThreshold(),
                    args.GetReferenceAntenna(),
                    args.GetComputeOptions());
            }
            else
            {
                auto calibrator = LeapCalibratorFactory::Create(args.GetComputeImplementation());

                std::vector<cpu::Calibration> calibrations;
                auto outputCallback = [&](const cpu::Calibration& cal)
                {
                    calibrations.push_back(cal);
                };
                
                calibrator->Calibrate(
                    outputCallback,
                    args.GetMeasurementSet(),
                    args.GetDirections(),
                    args.GetSolutionInterval(),
                    args.GetMinimumBaselineThreshold(),
                    args.GetReferenceAntenna(),
                    args.GetComputeOptions());
                
                auto calibrationCollection = cpu::CalibrationCollection(std::move(calibrations));
                calibrationCollection.Serialize(*args.CreateOutputStream());
            }
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "error: " << e.what() << '\n';
        return -1;
    }
}

