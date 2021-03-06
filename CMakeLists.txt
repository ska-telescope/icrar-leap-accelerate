cmake_minimum_required(VERSION 3.8)

set(SOLUTION_NAME LEAP-Accelerate)
set(SOLUTION_NAME_LOWER leap-accelerate)

file(STRINGS version.txt SOLUTION_VERSION)
message(STATUS "Building ${SOLUTION_NAME} version ${SOLUTION_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(CUDA_ENABLED "Build with cuda support" OFF)
option(HIGH_GPU_MEMORY "Improves device performance at the cost of extra device memory" OFF)
option(WERROR "Warnings as Errors" OFF)
option(WCONVERSION "Conversion warnings" OFF)
option(TRACE "Trace Data Values" OFF)
option(CMAKE_RUN_CLANG_TIDY "Run clang-tidy with the compiler." OFF)

# Override config via environment variables
if(DEFINED ENV{CUDA_ENABLED})
  set(CUDA_ENABLED $ENV{CUDA_ENABLED})
  message("-- Environment variable override: CUDA_ENABLED=${CUDA_ENABLED}")
endif()
if(DEFINED ENV{WERROR})
  set(WERROR $ENV{WERROR})
  message("-- Environment variable override: WERROR=${WERROR}")
endif()
if(DEFINED ENV{TRACE})
  set(TRACE $ENV{TRACE})
  message("-- Environment variable override: TRACE=${TRACE}")
endif()
if(DEFINED ENV{CMAKE_RUN_CLANG_TIDY})
  set(CMAKE_RUN_CLANG_TIDY $ENV{CMAKE_RUN_CLANG_TIDY})
  message("-- Environment variable override: CMAKE_RUN_CLANG_TIDY=${CMAKE_RUN_CLANG_TIDY}")
endif()
if(DEFINED ENV{HIGH_GPU_MEMORY})
  set(HIGH_GPU_MEMORY $ENV{HIGH_GPU_MEMORY})
  message("-- Environment variable override: HIGH_GPU_MEMORY=${HIGH_GPU_MEMORY}")
endif()

if(TRACE)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DTRACE")
endif()

# Enable Linting with build
if(CMAKE_RUN_CLANG_TIDY)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "CMake_RUN_CLANG_TIDY requires an out-of-source build!")
  endif()
  find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
  if(NOT CLANG_TIDY_COMMAND)
    message(WARNING "CMake_RUN_CLANG_TIDY is ON but clang-tidy is not found!")
    set(CMAKE_CXX_CLANG_TIDY "" CACHE STRING "" FORCE)
  else()
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
  endif()
  file(SHA1 ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy clang_tidy_sha1)
  set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
  unset(clang_tidy_sha1)
endif()
configure_file(.clang-tidy .clang-tidy COPYONLY)

# Project configuration, specifying version, languages,
# and the C++ standard to use for the whole project
project(${SOLUTION_NAME} LANGUAGES CXX VERSION ${SOLUTION_VERSION})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  if(WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
  if(WCONVERSION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")
  endif()
endif()

# Load Modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Seperate lint target
include(lint)

# Testing
add_definitions(-DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata")
include(CTest)

# Gather External Projects
find_package(Git)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed, see https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
      message(STATUS "Submodule update")
      execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      RESULT_VARIABLE GIT_SUBMOD_RESULT)
      if(NOT GIT_SUBMOD_RESULT EQUAL "0")
          message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
      endif()
  endif()
else()
  message(WARNING "git not found!")
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/eigen-3.3.90/CMakeLists.txt")
  message(FATAL_ERROR "Submodule eigen-3.3.90 was not downloaded! Please update submodules and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/gtest-1.8.1/CMakeLists.txt")
  message(FATAL_ERROR "Submodule gtest-1.8.1 was not downloaded! Please update submodules and try again.")
endif()

# Global Libraries

find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/config-file.doxygen)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(doxygen
        #ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen needs to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

if(CUDA_ENABLED)
  find_package(CUDA 9.0)
  enable_language(CUDA)
endif()
#modify exported compile commands for linting
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
set(CMAKE_CUDA_COMPILER ${CMAKE_CXX_COMPILER})
  # Clang IWYU Configuration
  # Clang will ignore explicit includes of its preconfigured include directory
  #include_directories(SYSTEM "/usr/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/include")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /usr/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/include")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem /usr/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/include")
endif()

add_definitions(-DBOOST_ALL_DYN_LINK)

find_package(Boost REQUIRED COMPONENTS log log_setup program_options)

set(CASACORE_FIND_QUIETLY ON)
find_package(Casacore REQUIRED)

find_package(LAPACK)

# Testing Libraries
if (BUILD_TESTING)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # required for Win32
  add_subdirectory(external/gtest-1.8.1 googletest EXCLUDE_FROM_ALL)
  enable_testing()
  
  include(GoogleTest)
  mark_as_advanced(
      BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
      gmock_build_tests gtest_build_samples gtest_build_tests
      gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  )
  set_target_properties(gtest PROPERTIES FOLDER external)
  set_target_properties(gtest_main PROPERTIES FOLDER external)
  set_target_properties(gmock PROPERTIES FOLDER external)
  set_target_properties(gmock_main PROPERTIES FOLDER external)
endif()

# Local Libraries
#add_subdirectory(external/eigen-3.3.90) # builds eigen3 tests
set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/eigen-3.3.90)
find_package(Eigen3 REQUIRED)

find_package(RapidJSON REQUIRED)
message(STATUS "RapidJSON: ${RapidJSON_INCLUDE_DIRS}")

# Local Targets
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(src/icrar/leap-accelerate)
add_subdirectory(src/icrar/leap-accelerate-cli)

# Install cmake config + version + target files
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/${SOLUTION_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${SOLUTION_NAME}Config.cmake"
  INSTALL_DESTINATION
    share/${SOLUTION_NAME_LOWER}/cmake
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${SOLUTION_NAME}Version.cmake"
  COMPATIBILITY
    AnyNewerVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${SOLUTION_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${SOLUTION_NAME}Version.cmake
  DESTINATION
    share/${SOLUTION_NAME_LOWER}/cmake
  COMPONENT
    dev
)
install(EXPORT ${SOLUTION_NAME_LOWER}-targets
  FILE
  ${SOLUTION_NAME}Targets.cmake
  DESTINATION
    share/${SOLUTION_NAME_LOWER}/cmake
  COMPONENT
    dev
  NAMESPACE
  ${SOLUTION_NAME_LOWER}::
)
