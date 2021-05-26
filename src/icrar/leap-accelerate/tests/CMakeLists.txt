
set(TEST_TARGET_NAME LeapAccelerate.Tests)
set(TEST_OUTPUT_DIR ${PROJECT_BINARY_DIR}/bin/tests)

set(sources
  main.cc
  
  test_helper.cc

  core/VerbosityTests.cc
  core/StreamOutTypeTests.cc

  common/SliceTests.cc
  common/RangeTests.cc
  common/SphericalDirectionTests.cc
  model/cpu/MVuvwTests.cc
  common/EigenStringUtilsTests.cc

  math/eigen_helper.cc
  math/MatrixTests.cc

  math/MathConversionTests.cc
  math/EigenExtensionsTests.cc
  math/cuda/CudaMatrixEigenTests.cc

  ms/MSUtilsTests.cc

  model/MetaDataTests.cc
  model/IntegrationTests.cc
  
  algorithm/PhaseRotateTests.cc
  system/ConfigTests.cc
)

set(cuda_sources
  math/cuda/CudaMatrixTests.cu
)

# Only CUDA 10+ is capable of compiling gtest
if(CUDA_VERSION_MAJOR GREATER 9)
  list(APPEND sources ${cuda_sources})
endif()

add_executable(${TEST_TARGET_NAME}
  ${sources}
)

set_target_properties(${TEST_TARGET_NAME}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}
)

if(USE_PCH)
  target_precompile_headers(${TEST_TARGET_NAME}
    PRIVATE
    [[pch.h]]
  )
endif()

# CUDA Options
if(CUDA_ENABLED)
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    message(STATUS "generating clang-tidy cuda compatible configuration")
    configure_clang_cuda_compiler(${TEST_TARGET_NAME})
  else()
    configure_nvcc_cuda_compiler(${TEST_TARGET_NAME})
    configure_cuda_warnings(${TEST_TARGET_NAME})
  endif()
endif()

target_compile_definitions(${TEST_TARGET_NAME} PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata")
target_link_libraries(${TEST_TARGET_NAME} LeapAccelerate gtest)

add_custom_target(${TEST_TARGET_NAME}.Data ALL
  COMMAND ${CMAKE_COMMAND}
  -Dsrc_dir=${CMAKE_CURRENT_SOURCE_DIR}/testdata
  -Ddst_dir=${TEST_OUTPUT_DIR}/testdata
  -P ${PROJECT_SOURCE_DIR}/cmake/update_directory.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME ${TEST_TARGET_NAME}
  COMMAND ${TEST_TARGET_NAME}
)

add_dependencies(LeapAccelerate.Tests LeapAccelerate.Tests.Data)
  
#gtest_add_tests(
#  TARGET ${TEST_TARGET_NAME}
#  TEST_PREFIX "*"
#  EXTRA_ARGS --gtest_stack_trace_depth=10
#)

message(STATUS "Configured ${TEST_TARGET_NAME}")