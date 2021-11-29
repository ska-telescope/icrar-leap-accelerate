# Configure a header file to pass some of the CMake settings
# to the source code
configure_file(config.h.in config.h)

include("${CMAKE_SOURCE_DIR}/cmake/CUDAConfig.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/GetGitRevisionDescription.cmake")
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_local_changes(GIT_HAS_LOCAL_CHANGES)
set(git_revision_cc "${CMAKE_CURRENT_BINARY_DIR}/core/git_revision.cc")
configure_file("core/git_revision.cc.in" "${git_revision_cc}" @ONLY)

set(TARGET_NAME LeapAccelerate)

set(public_headers
    model/visibility.h
    model/band_data.h
)
set(private_headers
    math/casacore_helper.h
    math/math_conversion.h
    math/vector_extensions.h
)
set(sources
    ${git_revision_cc}

    config.cc
    exception/exception.cc

    ms/MeasurementSet.cc

    core/compute_implementation.cc
    core/InputType.cc
    core/stream_out_type.cc
    core/log/logging.cc
    core/log/Verbosity.cc
    core/memory/system_memory.cc
    core/profiling/resource_usage.cc
    core/profiling/UsageReporter.cc
    core/version.cc

    model/cpu/MVuvw.cc
    
    common/SphericalDirection.cc
    common/Slice.cc
    common/config/Arguments.cc

    math/math_conversion.cc
    math/cpu/eigen_extensions.cc

    model/cpu/Integration.cc
    model/cpu/MetaData.cc
    model/cpu/calibration/Calibration.cc
    model/cpu/calibration/BeamCalibration.cc
    model/cuda/DeviceMetaData.cc
    model/cuda/DeviceIntegration.cc

    algorithm/LeapCalibratorFactory.cc
    algorithm/cpu/CpuLeapCalibrator.cc
    algorithm/cpu/PhaseMatrixFunction.cc
    algorithm/cuda/CudaComputeOptions.cc
    algorithm/cuda/CudaLeapCalibrator.cc

    python/LeapAccelerateModule.cc
    python/PyLeapCalibrator.cc
    python/PyMeasurementSet.cc
)
set(cuda_headers
    cuda/cuda_info.h
    cuda/helper_cuda.cuh
    cuda/helper_string.cuh
)
set(cuda_sources
    cuda/cuda_info.cu
    
    math/cuda/matrix_op.cu
    math/cuda/matrix_multiply.cu
    math/cuda/matrix_invert.cu
    
    algorithm/cuda/kernel/EmptyKernel.cu
    algorithm/cuda/kernel/RotateVisibilitiesKernel.cu
    algorithm/cuda/kernel/PolarizationsToPhaseAnglesKernel.cu
    algorithm/cuda/kernel/ComputePhaseDeltaKernel.cu
    algorithm/cuda/kernel/CopyPhaseDeltaKernel.cu
)

if(CUDA_ENABLED)
  list(APPEND private_headers ${cuda_headers})
  list(APPEND sources ${cuda_sources})
endif()

#python_add_library(
#PYTHON_ADD_MODULE(

# if(PYTHON_ENABLED)
#   pybind11_add_module(
#     ${TARGET_NAME} SHARED
#     ${sources}
#     ${private_headers}
#     ${public_headers}
#   )
# else()
# endif()

add_library(
    ${TARGET_NAME} SHARED
    ${sources}
    ${private_headers}
    ${public_headers}
  )
if(PYTHON_ENABLED)
  pybind11_extension(${TARGET_NAME})
  pybind11_strip(${TARGET_NAME})
endif()

# Libraries
option(CASA_ENABLED "Casa Enabled" TRUE)
option(GSL_ENABLED "GSL Enabled" OFF)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


if(HIPSYCL_ENABLED)
  add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${sources})
endif()

set_target_properties(${TARGET_NAME}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(USE_PCH)
  target_precompile_headers(${TARGET_NAME}
    PRIVATE
    [[pch.h]]
)
endif()

target_link_libraries(${TARGET_NAME} PUBLIC Threads::Threads)
target_link_libraries(${TARGET_NAME} PUBLIC ${Boost_LIBRARIES})
target_link_libraries(${TARGET_NAME} PUBLIC Boost::log)
target_link_libraries(${TARGET_NAME} PUBLIC Boost::log_setup)
target_link_libraries(${TARGET_NAME} PUBLIC Boost::program_options)
target_link_libraries(${TARGET_NAME} PUBLIC Boost::filesystem)
target_link_libraries(${TARGET_NAME} PUBLIC Eigen3::Eigen)
target_link_libraries(${TARGET_NAME} PUBLIC RapidJSON::RapidJSON)
#target_link_libraries(${TARGET_NAME} PUBLIC xtensor)


if(CUDA_ENABLED)
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    message(STATUS "generating clang-tidy cuda compatible configuration")
    configure_clang_cuda_compiler(${TARGET_NAME})
  else()
    configure_nvcc_cuda_compiler(${TARGET_NAME})
    configure_cuda_warnings(${TARGET_NAME})
  endif()

  #Set Cuda Stream Setting
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo --default-stream per-thread)
  #message("CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}")
  target_include_directories(${TARGET_NAME} SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${TARGET_NAME} PUBLIC cuda)  # libcuda.so may only be located in lib64/stubs
  target_link_libraries(${TARGET_NAME} PUBLIC ${CUDA_LIBRARIES})
  target_link_libraries(${TARGET_NAME} PUBLIC ${CUDA_CUBLAS_LIBRARIES})
  if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "10.0")
    target_link_libraries(${TARGET_NAME} PUBLIC ${CUDA_cusolver_LIBRARY})
    #target_link_libraries(${TARGET_NAME} PUBLIC cublasLt)
  endif()
endif()

if(SYCL_ENABLED)
  target_compile_definitions(${TARGET_NAME} PUBLIC SYCL_ENABLED)
endif()

if(PYTHON_ENABLED)
  target_link_options(${TARGET_NAME} PUBLIC "-fPIE")
  #target_link_options(${TARGET_NAME} PUBLIC "-fPIC")
  target_compile_definitions(${TARGET_NAME} PUBLIC PYTHON_ENABLED)
  #target_link_libraries(${TARGET_NAME} PUBLIC Boost::python)
  #target_link_libraries(${TARGET_NAME} PUBLIC Boost::numpy)
  target_link_libraries(${TARGET_NAME} PUBLIC Python3::Python)
  #target_link_libraries(${TARGET_NAME} PUBLIC Python3::NumPy)
  target_link_libraries(${TARGET_NAME} PUBLIC pybind11::pybind11)
  target_link_libraries(${TARGET_NAME} PUBLIC pybind11::module)
  #target_link_libraries(${TARGET_NAME} PUBLIC pybind11::thin_lto)
  #target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/external/xtl-0.7.2/include)
  #target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/external/xtensor-0.23.10/include)
  if(NOT Python3_NumPy_FOUND)
    target_include_directories(${TARGET_NAME} SYSTEM PUBLIC /usr/lib/python3/dist-packages/numpy/core/include)
    target_include_directories(${TARGET_NAME} SYSTEM PUBLIC /usr/lib/python3.9/site-packages/numpy/core/include)
  endif()
endif()

if(CASA_ENABLED)
  target_include_directories(${TARGET_NAME} PUBLIC ${CASACORE_INCLUDES})
  target_link_libraries(${TARGET_NAME} PUBLIC ${CASACORE_LIBRARIES})
endif()
if(GSL_ENABLED)
  #target_link_libraries(${TARGET_NAME} ${BLAS_LIBRARIES})
  #target_link_libraries(${TARGET_NAME} ${LAPACK_LIBRARIES})
  target_link_libraries(${TARGET_NAME} GSL::gsl)
endif()

target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
# get_target_property(PUBLIC_HEADER_DIR LeapAccelerate INTERFACE_INCLUDE_DIRECTORIES)
# message(${PUBLIC_HEADER_DIR})
# file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADER_DIR} *.h)
# message(${PUBLIC_HEADERS})

add_library(${SOLUTION_NAME_LOWER}::${TARGET_NAME} ALIAS ${TARGET_NAME})

install(
  TARGETS
    ${TARGET_NAME}
  EXPORT
    ${SOLUTION_NAME_LOWER}-targets
  LIBRARY DESTINATION
    lib
  ARCHIVE DESTINATION
    lib
)

install(
  FILES
    ${public_headers}
  DESTINATION
    include/icrar/${SOLUTION_NAME_LOWER}
)

# TODO: Developer Use
# Move python dll
add_custom_target(${TARGET_NAME}.DynLib ALL
  COMMAND ${CMAKE_COMMAND}
  -Dsrc_dir=${PROJECT_BINARY_DIR}/lib
  -Ddst_dir=${PROJECT_SOURCE_DIR}/python/leap-accelerate-python
  -P ${PROJECT_SOURCE_DIR}/cmake/update_directory.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_dependencies(${TARGET_NAME}.DynLib ${TARGET_NAME})

unset(CASA_ENABLED CACHE)
unset(GSL_ENABLED CACHE)

message(STATUS "Configured ${TARGET_NAME}")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
