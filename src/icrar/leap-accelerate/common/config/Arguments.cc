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

#include "Arguments.h"

#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/exception/exception.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>

namespace icrar
{
    /**
     * Default set of command line interface arguments
     */
    CLIArguments CLIArguments::GetDefaultArguments()
    {
        auto args = CLIArguments();
        args.sourceType = InputType::MEASUREMENT_SET;
        args.filePath = boost::none;
        args.configFilePath = boost::none;
        args.streamOutType = "single";
        args.outputFilePath = boost::none;

        args.stations = boost::none;
        args.referenceAntenna = boost::none;
        args.directions = boost::none;
        args.computeImplementation = std::string("cpu");
        args.solutionInterval = std::string("[0,-1,-1]"); // Average over all timesteps
        args.readAutocorrelations = true;
        args.minimumBaselineThreshold = 0.0;
        args.mwaSupport = false;
        args.useFileSystemCache = true;
        args.verbosity = static_cast<int>(log::DEFAULT_VERBOSITY);
        return args;
    }

    Arguments::Arguments(CLIArguments&& args)
        : sourceType(args.sourceType)
        , filePath(std::move(args.filePath))
        , configFilePath(std::move(args.configFilePath))
        , outputFilePath(std::move(args.outputFilePath))
        , stations(std::move(args.stations))
        , referenceAntenna(std::move(args.referenceAntenna))
        , minimumBaselineThreshold(args.minimumBaselineThreshold)
        , readAutocorrelations(args.readAutocorrelations)
        , mwaSupport(args.mwaSupport)
        , useFileSystemCache(args.useFileSystemCache)
    {
        //Perform type conversions
        if(args.computeImplementation.is_initialized())
        {
            computeImplementation.reset(ComputeImplementation()); //Defualt value ignored
            if(!TryParseComputeImplementation(args.computeImplementation.get(), computeImplementation.get()))
            {
                throw std::invalid_argument("invalid compute implementation argument");
            }
        }

        if(args.streamOutType.is_initialized())
        {
            streamOutType.reset(StreamOutType());
            if(!TryParseStreamOutType(args.streamOutType.get(), streamOutType.get()))
            {
                throw icrar::invalid_argument_exception("invalid stream out type argument", args.streamOutType.get(), __FILE__, __LINE__);
            }
        }

        if(args.directions.is_initialized())
        {
            directions = ParseDirections(args.directions.get());
        }

        if(args.solutionInterval.is_initialized())
        {
            solutionInterval = ParseSlice(args.solutionInterval.get());
        }

        if(args.verbosity.is_initialized())
        {
            verbosity = static_cast<icrar::log::Verbosity>(args.verbosity.get());
        }
    }

    ArgumentsValidated::ArgumentsValidated(Arguments&& cliArgs)
    : m_sourceType(InputType::MEASUREMENT_SET)
    , m_computeImplementation(ComputeImplementation::cpu)
    , m_solutionInterval()
    , m_minimumBaselineThreshold(0)
    , m_readAutocorrelations(false)
    , m_mwaSupport(false)
    , m_useFileSystemCache(false)
    , m_verbosity(icrar::log::Verbosity::trace) //These values are overwritten
    {
        // Initialize default arguments first
        ApplyArguments(CLIArguments::GetDefaultArguments());

        // Read the config argument second and apply the config arguments over the default arguments
        if(cliArgs.configFilePath.is_initialized())
        {
            // Configuration via json config
            ApplyArguments(ParseConfig(cliArgs.configFilePath.get()));
        }

        // OVerride the config args with the remaining cli arguments
        ApplyArguments(std::move(cliArgs));
        Validate();

        // Load resources
        icrar::log::Initialize(GetVerbosity()); //TODO: Arguments tests already intiializes verbosity
        switch (m_sourceType)
        {
        case InputType::MEASUREMENT_SET:
            if (m_filePath.is_initialized())
            {
                m_measurementSet = std::make_unique<MeasurementSet>(
                    m_filePath.get(),
                    m_stations,
                    m_readAutocorrelations);
            }
            else
            {
                throw std::invalid_argument("measurement set filename not provided");
            }
            break;
        case InputType::STREAM:
            throw std::runtime_error("only measurement set input is currently supported");
            break;
        default:
            throw std::runtime_error("only measurement set input is currently supported");
            break;
        }
    }

    void ArgumentsValidated::ApplyArguments(Arguments&& args)
    {
        if(args.sourceType.is_initialized())
        {
            m_sourceType = std::move(args.sourceType.get());
        }

        if(args.filePath.is_initialized())
        {
            m_filePath = std::move(args.filePath.get());
        }

        if(args.configFilePath.is_initialized())
        {
            m_configFilePath = std::move(args.configFilePath.get());
        }

        if(args.streamOutType.is_initialized())
        {
            m_streamOutType = std::move(args.streamOutType.get());
        }

        if(args.outputFilePath.is_initialized())
        {
            m_outputFilePath = std::move(args.outputFilePath.get());
        }

        if(args.stations.is_initialized())
        {
            m_stations = std::move(args.stations.get());
        }

        if(args.referenceAntenna.is_initialized())
        {
            m_referenceAntenna = std::move(args.referenceAntenna.get());
        }

        if(args.directions.is_initialized())
        {
            m_directions = std::move(args.directions.get());
        }

        if(args.computeImplementation.is_initialized())
        {
            m_computeImplementation = std::move(args.computeImplementation.get());
        }

        if(args.solutionInterval.is_initialized())
        {
            m_solutionInterval = std::move(args.solutionInterval.get());
        }

        if(args.minimumBaselineThreshold.is_initialized())
        {
            m_minimumBaselineThreshold = std::move(args.minimumBaselineThreshold.get());
        }
        
        if(args.readAutocorrelations.is_initialized())
        {
            m_readAutocorrelations = std::move(args.readAutocorrelations.get());
        }

        if(args.mwaSupport.is_initialized())
        {
            m_mwaSupport = std::move(args.mwaSupport.get());
        }

        if(args.useFileSystemCache.is_initialized())
        {
            m_useFileSystemCache = std::move(args.useFileSystemCache.get());
        }

        if(args.verbosity.is_initialized())
        {
            m_verbosity = std::move(args.verbosity.get());
        }
    }


    void ArgumentsValidated::Validate() const
    {
        if(m_directions.size() == 0)
        {
            throw std::invalid_argument("directions argument not provided");
        }
    }

    boost::optional<std::string> ArgumentsValidated::GetOutputFilePath() const
    {
        return m_outputFilePath;
    }

    StreamOutType ArgumentsValidated::GetStreamOutType() const
    {
        return m_streamOutType;
    }

    std::unique_ptr<std::ostream> ArgumentsValidated::CreateOutputStream(double startEpoch) const
    {
        if(!m_outputFilePath.is_initialized())
        {
            return std::make_unique<std::ostream>(std::cout.rdbuf());
        }
        if(m_streamOutType == StreamOutType::collection)
        {
            return std::make_unique<std::ostream>(std::cout.rdbuf());
        }
        else if(m_streamOutType == StreamOutType::singleFile)
        {
            auto path = m_outputFilePath.get();
            return std::make_unique<std::ofstream>(path);
        }
        else if(m_streamOutType == StreamOutType::multipleFiles)
        {
            auto path = m_outputFilePath.get() + "." + std::to_string(startEpoch) + ".json";
            return std::make_unique<std::ofstream>(path);
        }
        else
        {
            throw std::runtime_error("invalid output stream type");
        }
    }

    MeasurementSet& ArgumentsValidated::GetMeasurementSet()
    {
        return *m_measurementSet;
    }

    const std::vector<SphericalDirection>& ArgumentsValidated::GetDirections() const
    {
        return m_directions;
    }

    ComputeImplementation ArgumentsValidated::GetComputeImplementation() const
    {
        return m_computeImplementation;
    }

    Slice ArgumentsValidated::GetSolutionInterval() const
    {
        return m_solutionInterval;
    }

    boost::optional<unsigned int> ArgumentsValidated::GetReferenceAntenna() const
    {
        return m_referenceAntenna;
    }

    double ArgumentsValidated::GetMinimumBaselineThreshold() const
    {
        return m_minimumBaselineThreshold;
    }
	
	bool ArgumentsValidated::IsFileSystemCacheEnabled() const
    {
        return m_useFileSystemCache;
    }

    icrar::log::Verbosity ArgumentsValidated::GetVerbosity() const
    {
        return m_verbosity;
    }

    Arguments ArgumentsValidated::ParseConfig(const std::string& configFilepath)
    {
        Arguments args;
        ParseConfig(configFilepath, args);
        return args;
    }

    void ArgumentsValidated::ParseConfig(const std::string& configFilepath, Arguments& args)
    {
        auto ifs = std::ifstream(configFilepath);
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::Document doc;
        doc.ParseStream(isw);

        if(!doc.IsObject())
        {
            throw json_exception("expected config to be an object", __FILE__, __LINE__);
        }
        for(auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it)
        {
            if(!it->name.IsString())
            {
                throw json_exception("config keys must be of type string", __FILE__, __LINE__);
            }
            else
            {
                std::string key = it->name.GetString();
                if(key == "sourceType")
                {
                    //args.sourceType = it->value.GetInt(); //TODO: use string
                }
                else if(key == "filePath")
                {
                    args.filePath = it->value.GetString();
                    if(it->value.IsString())
                    {
                        args.filePath = it->value.GetString();
                    }
                    else
                    {
                        throw json_exception("filePath must be of type string", __FILE__, __LINE__);
                    }
                }
                else if(key == "configFilePath")
                {
                    throw json_exception("recursive config detected", __FILE__, __LINE__);
                }
                else if(key == "streamOutType")
                {
                    StreamOutType e;
                    if(TryParseStreamOutType(it->value.GetString(), e))
                    {
                        args.streamOutType = e;
                    }
                    else
                    {
                        throw json_exception("invalid stream out type string", __FILE__, __LINE__);
                    }
                }
                else if(key == "outputFilePath")
                {
                    if(it->value.IsString())
                    {
                        args.outputFilePath = it->value.GetString();
                    }
                    else
                    {
                        throw json_exception("outFilePath must be of type string", __FILE__, __LINE__);
                    }
                }
                else if(key == "stations")
                {
                    if(it->value.IsInt())
                    {
                        args.stations = it->value.GetInt();
                    }
                    else
                    {
                        throw json_exception("outFilePath must be of type int", __FILE__, __LINE__);
                    }
                }
                else if(key == "solutionInterval")
                {
                    if(it->value.IsArray())
                    {
                        args.solutionInterval = ParseSlice(it->value);
                    }
                }
                else if(key == "referenceAntenna")
                {
                    if(it->value.IsInt())
                    {
                        args.referenceAntenna = it->value.GetInt();
                    }
                    else
                    {
                        throw json_exception("referenceAntenna must be of type unsigned int", __FILE__, __LINE__);
                    }
                }
                else if(key == "directions")
                {
                    args.directions = ParseDirections(it->value);
                }
                else if(key == "computeImplementation")
                {
                    ComputeImplementation e;
                    if(TryParseComputeImplementation(it->value.GetString(), e))
                    {
                        args.computeImplementation = e;
                    }
                    else
                    {
                        throw json_exception("invalid compute implementation string", __FILE__, __LINE__);
                    }
                }
                else if(key == "minimumBaselineThreshold")
                {
                    if(it->value.IsDouble())
                    {
                        args.minimumBaselineThreshold = it->value.GetDouble();
                    }
                    else
                    {
                        throw json_exception("minimumBaselineThreshold must be of type double", __FILE__, __LINE__);
                    }
                }
                else if(key == "useFileSystemCache")
                {
                    if(it->value.IsBool())
                    {
                        args.useFileSystemCache = it->value.GetBool();
                    }
                    else
                    {
                        throw json_exception("useFileSystemCache must be of type bool", __FILE__, __LINE__);
                    }
                }
                else if(key == "mwaSupport")
                {
                    if(it->value.IsBool())
                    {
                        args.mwaSupport = it->value.GetBool();
                    }
                    else
                    {
                        throw json_exception("mwaSupport must be of type bool", __FILE__, __LINE__);
                    }
                }
                else if(key == "autoCorrelations")
                {
                    if(it->value.IsBool())
                    {
                        args.readAutocorrelations = it->value.GetBool();
                    }
                    else
                    {
                        throw json_exception("readAutoCorrelations must be of type bool", __FILE__, __LINE__);
                    }
                }
                else if(key == "verbosity")
                {
                    if(it->value.IsInt())
                    {
                        args.verbosity = static_cast<log::Verbosity>(it->value.GetInt());
                    }
                    if(it->value.IsString())
                    {
                        log::Verbosity e;
                        if(TryParseVerbosity(it->value.GetString(), e))
                        {
                            args.verbosity = e;
                        }
                        else
                        {
                            throw json_exception("invalid verbosity string", __FILE__, __LINE__);
                        }
                    }
                    else
                    {
                        throw json_exception("verbosity must be of type int or string", __FILE__, __LINE__);
                    }
                }
                else
                {
                    std::stringstream ss;
                    ss << "invalid config key: " << key; 
                    throw json_exception(ss.str(), __FILE__, __LINE__);
                }
            }
        }
    }
} // namespace icrar