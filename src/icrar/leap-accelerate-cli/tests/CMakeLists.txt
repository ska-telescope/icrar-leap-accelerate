
set(TEST_TARGET_NAME LeapAccelerateCLI.Tests)
set(TEST_OUTPUT_DIR ${PROJECT_BINARY_DIR}/bin/tests)

set(sources
  main.cc

  CommandLineTests.cc
  JSONHelperTests.cc
  E2EPerformanceTests.cc
)

add_executable(${TEST_TARGET_NAME}
  ${sources}
)

set_target_properties(${TEST_TARGET_NAME}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}
)

if(${CMAKE_VERSION} VERSION_GREATER "3.16.0" AND NOT CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  # clang-tidy analysis does not support precompiled headers  
  target_precompile_headers(${TEST_TARGET_NAME}
    PRIVATE
    [[pch.h]]
  )
endif()

if(CUDA_ENABLED)
  configure_nvcc_cuda_compiler(${TEST_TARGET_NAME})
  configure_cuda_warnings(${TEST_TARGET_NAME})
endif()

target_link_libraries(${TEST_TARGET_NAME} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET_NAME}
  COMMAND ${TEST_TARGET_NAME}
)

message(STATUS "Configured ${TEST_TARGET_NAME}")