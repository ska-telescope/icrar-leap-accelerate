
set(TEST_TARGET_NAME LeapAccelerate.Tests)

set(sources
  main.cc
  
  test_helper.cc

  core/VerbosityTests.cc

  common/SliceTests.cc
  common/RangeTests.cc
  common/SphericalDirectionTests.cc
  common/MVuvwTests.cc
  
  math/eigen_helper.cc
  math/MatrixTests.cc
  math/MatrixTests.cc
  math/MathConversionTests.cc
  math/EigenExtensionsTests.cc

  ms/MSUtilsTests.cc

  model/MetaDataTests.cc
  model/IntegrationTests.cc
  
  algorithm/PhaseRotateTests.cc
)

set(cuda_sources
  math/cuda/CudaVectorTests.cu
  math/cuda/CudaMatrixTests.cu
  math/cuda/CudaMatrixEigenTests.cu
)

# Only CUDA 10+ is capable of compiling gtest
if(CUDA_VERSION_MAJOR GREATER 9)
  list(APPEND sources ${cuda_sources})
endif()

add_executable(${TEST_TARGET_NAME}
  ${sources}
)

if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
# TODO(calgray): clang-tidy tries to analyze pch systems headers
#  target_precompile_headers(${TEST_TARGET_NAME}
#    PRIVATE
#    [[pch.h]]
#  )
endif()

# CUDA Options
if(CUDA_ENABLED)
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    message("generating clang-tidy cuda compatible configuration")
    configure_clang_cuda_compiler(${TEST_TARGET_NAME})
  else()
    configure_nvcc_cuda_compiler(${TEST_TARGET_NAME})
    configure_cuda_warnings(${TEST_TARGET_NAME})
  endif()
endif()

target_link_libraries(${TEST_TARGET_NAME} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET_NAME}
  COMMAND ${TEST_TARGET_NAME}
)

#gtest_add_tests(
#  TARGET ${TEST_TARGET_NAME}
#  TEST_PREFIX "*"
#  EXTRA_ARGS --gtest_stack_trace_depth=10
#)