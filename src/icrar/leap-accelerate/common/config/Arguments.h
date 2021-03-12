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

#include <icrar/leap-accelerate/algorithm/ComputeOptions.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/common/Slice.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/stream_out_type.h>

#include <boost/optional.hpp>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <iostream>

namespace icrar
{
    class MeasurementSet;

    enum class InputType
    {
        MEASUREMENT_SET,
        STREAM,
        //APACHE_ARROW
    };

    /**
     * @brief Raw arguments received via CLI interface
     * 
     */
    struct CLIArguments
    {
        boost::optional<InputType> sourceType;
        boost::optional<std::string> filePath;
        boost::optional<std::string> configFilePath;
        boost::optional<std::string> streamOutType;
        boost::optional<std::string> outputFilePath;

        boost::optional<int> stations;
        boost::optional<unsigned int> referenceAntenna;
        boost::optional<std::string> directions;
        boost::optional<std::string> computeImplementation;
        boost::optional<std::string> solutionInterval;
        boost::optional<double> minimumBaselineThreshold;
        boost::optional<bool> useFileSystemCache;
        boost::optional<bool> mwaSupport;
        boost::optional<bool> readAutocorrelations;
        boost::optional<int> verbosity;

        static CLIArguments GetDefaultArguments();
    };

    /**
     * @brief Typed arguments of \c CLIArguments 
     * 
     */
    struct Arguments
    {
        Arguments() = default;
        Arguments(CLIArguments&& args);

        boost::optional<InputType> sourceType; // MeasurementSet source type
        boost::optional<std::string> filePath; // MeasurementSet filepath
        boost::optional<std::string> configFilePath; // Optional config filepath
        boost::optional<StreamOutType> streamOutType;
        boost::optional<std::string> outputFilePath; // Calibration output file, print to terminal if empty

        boost::optional<int> stations;
        boost::optional<unsigned int> referenceAntenna;
        boost::optional<std::vector<SphericalDirection>> directions;
        boost::optional<ComputeImplementation> computeImplementation;
        boost::optional<Slice> solutionInterval;
        boost::optional<double> minimumBaselineThreshold;
        boost::optional<bool> readAutocorrelations;
        boost::optional<bool> mwaSupport;
        boost::optional<bool> useFileSystemCache;
        boost::optional<icrar::log::Verbosity> verbosity;
    };

    /**
     * Validated set of command line arguments required to perform leap calibration
     */
    class ArgumentsValidated
    {
        /**
         * Constants
         */
        InputType m_sourceType;
        boost::optional<std::string> m_filePath; // MeasurementSet filepath
        boost::optional<std::string> m_configFilePath; // Config filepath
        StreamOutType m_streamOutType;
        boost::optional<std::string> m_outputFilePath; // Calibration output filepath

        boost::optional<int> m_stations; // Overriden number of stations (will be removed in a later release)
        boost::optional<unsigned int> m_referenceAntenna; // Index of the reference antenna
        std::vector<SphericalDirection> m_directions; // Calibration directions
        ComputeImplementation m_computeImplementation; // Specifies the implementation for calibration computation
        Slice m_solutionInterval; // Specifies the interval to calculate solutions for
        double m_minimumBaselineThreshold; // Minimum baseline length otherwise flagged at runtime
        bool m_readAutocorrelations; // Adjusts the number of baselines calculation to include autocorrelations
        bool m_mwaSupport; // Negates baselines when enabled
        icrar::log::Verbosity m_verbosity; // Defines logging level for std::out

        ComputeOptions m_computeOptions; // Defines options for compute performance to be determined based on harware configuration
        
        /**
         * Resources
         */
        std::unique_ptr<MeasurementSet> m_measurementSet;


    public:
        ArgumentsValidated(Arguments&& cliArgs);

        /**
         * @brief Overwrites the stored set of arguments.
         * 
         * @param args 
         */
        void ApplyArguments(Arguments&& args);

        void Validate() const;

        boost::optional<std::string> GetOutputFilePath() const;

        std::unique_ptr<std::ostream> CreateOutputStream(double startEpoch = 0.0) const;

        /**
         * @brief Gets the configuration for output stream type
         * 
         * @return StreamOutType 
         */
        StreamOutType GetStreamOutType() const;

        /**
         * @brief Gets the user specifified measurement set
         * 
         * @return MeasurementSet& 
         */
        MeasurementSet& GetMeasurementSet();

        const std::vector<SphericalDirection>& GetDirections() const;

        ComputeImplementation GetComputeImplementation() const;

        Slice GetSolutionInterval() const;

        boost::optional<unsigned int> GetReferenceAntenna() const;

        /**
         * @brief Gets the minimum baseline threshold in meteres. Baselines
         * of length beneath the threshold are to be filtered/flagged.
         * 
         * @return double baseline threshold length in meters
         */
        double GetMinimumBaselineThreshold() const;

        ComputeOptions GetComputeOptions() const;

        bool IsFileSystemCacheEnabled() const;

        icrar::log::Verbosity GetVerbosity() const;

    private:
        /**
         * @brief Converts a JSON file to a config
         * 
         * @param configFilepath 
         * @return Config 
         */
        Arguments ParseConfig(const std::string& configFilepath);
        
        /**
         * @brief Converts a JSON file to a config
         * 
         * @param configFilepath 
         * @param args 
         */
        void ParseConfig(const std::string& configFilepath, Arguments& args);
    };
} // namespace icrar
