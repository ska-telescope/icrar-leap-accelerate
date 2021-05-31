# Configure a header file to pass some of the CMake settings
# to the source code
configure_file(config.h.in config.h)

include("${PROJECT_SOURCE_DIR}/cmake/CUDAConfig.cmake")

include("${PROJECT_SOURCE_DIR}/cmake/GetGitRevisionDescription.cmake")
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

    exception/exception.cc

    ms/utils.cc
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

    algorithm/LeapCalibratorFactory.cc
    algorithm/cpu/CpuLeapCalibrator.cc
    algorithm/cpu/PhaseMatrixFunction.cc
    algorithm/cuda/CudaComputeOptions.cc
    algorithm/cuda/CudaLeapCalibrator.cc

    python/PyLeapCalibrator.cc
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
    
    model/cuda/DeviceIntegration.cu
    algorithm/cuda/kernel/EmptyKernel.cu
    algorithm/cuda/kernel/RotateVisibilitiesKernel.cu
    algorithm/cuda/kernel/PolarizationsToPhaseAnglesKernel.cu
    algorithm/cuda/kernel/ComputePhaseDeltaKernel.cu
    algorithm/cuda/kernel/CopyPhaseDeltaKernel.cu
)

# Libraries
option(CASA_ENABLED "Casa Enabled" TRUE)
option(GSL_ENABLED "GSL Enabled" OFF)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# if(CUDA_ENABLED)
  PYTHON_ADD_MODULE(
  #python_add_library(
    ${TARGET_NAME} SHARED
      ${sources}
      ${cuda_sources}
      ${private_headers}
      ${cuda_headers}
      ${public_headers}
  )
# else()
#   python_add_library(
#     ${TARGET_NAME} SHARED
#       ${sources}
#       ${private_headers}
#       ${public_headers}
#   )
# endif()
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

#target_compile_definitions(${TARGET_NAME} PUBLIC PRETTY_WRITER)

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
  target_link_libraries(${TARGET_NAME} PUBLIC cuda)
  target_link_libraries(${TARGET_NAME} PUBLIC ${CUDA_LIBRARIES})
  target_link_libraries(${TARGET_NAME} PUBLIC ${CUDA_CUBLAS_LIBRARIES})
  if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "10.0")
    target_link_libraries(${TARGET_NAME} PUBLIC ${CUDA_cusolver_LIBRARY})
    #target_link_libraries(${TARGET_NAME} PUBLIC cublasLt)
  endif()
endif()

target_link_libraries(${TARGET_NAME} PUBLIC Threads::Threads)
target_link_libraries(${TARGET_NAME} PUBLIC ${Boost_LIBRARIES})
target_link_libraries(${TARGET_NAME} PUBLIC Eigen3::Eigen)

target_link_libraries(${TARGET_NAME} PUBLIC Python3::Python)
if(Python3_NumPy_FOUND)
  target_link_libraries(${TARGET_NAME} PUBLIC Python3::NumPy)
else()
  target_include_directories(${TARGET_NAME} SYSTEM PUBLIC /usr/lib/python3/dist-packages/numpy/core/include)
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

unset(CASA_ENABLED CACHE)
unset(GSL_ENABLED CACHE)

message(STATUS "Configured ${TARGET_NAME}")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
