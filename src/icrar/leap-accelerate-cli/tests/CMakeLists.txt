
set(TEST_TARGET_NAME LeapAccelerateCLI.Tests)

set(sources
  main.cc

  CommandLineTests.cc
  JSONHelperTests.cc    
  E2EPerformanceTests.cc
)

add_executable(${TEST_TARGET_NAME}
  ${sources}
)

if(USE_PCH)
  target_precompile_headers(${TEST_TARGET_NAME}
    PRIVATE
    [[pch.h]]
  )
endif()

if(CUDA_ENABLED)
  target_compile_options(${TEST_TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_60>)
  configure_cuda_warnings(${TEST_TARGET_NAME})
endif()

target_link_libraries(${TEST_TARGET_NAME} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET_NAME}
  COMMAND ${TEST_TARGET_NAME}
)
