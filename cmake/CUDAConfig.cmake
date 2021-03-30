
# Configure NVCC cuda compiler
function(configure_nvcc_cuda_compiler TARGET_NAME)
  target_compile_definitions(${TARGET_NAME} PUBLIC CUDA_ENABLED)

  # Request that the target be built with -std=c++14
  # As this is a public compile feature anything that links to the target
  # will also build with -std=c++14
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_14)
  
  set_target_properties(${TARGET_NAME} PROPERTIES CUDA_STANDARD 14)
  # We need to explicitly state that we need all CUDA files in the target
  # library to be built with -dc as the member functions could be called by
  # other libraries and executables
  #set(CUDA_PROPAGATE_HOST_FLAGS ON)
  #set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  #set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if(CUDA_SM STREQUAL "")
    # Statically compile all compatible compute shaders
    set(CUDA_SM_60 1)
    set(CUDA_SM_61 1)
    set(CUDA_SM_70 1)
    set(CUDA_SM_72 1)
    set(CUDA_SM_75 1)
    set(CUDA_SM_80 1)
    set(CUDA_SM_86 1)
  else()
    # Read specified shader models to compile. This speeds up compile time for development
    list(FIND CUDA_SM "60" CUDA_SM_60)
    list(FIND CUDA_SM "61" CUDA_SM_61)
    list(FIND CUDA_SM "70" CUDA_SM_70)
    list(FIND CUDA_SM "72" CUDA_SM_72)
    list(FIND CUDA_SM "75" CUDA_SM_75)
    list(FIND CUDA_SM "80" CUDA_SM_80)
    list(FIND CUDA_SM "86" CUDA_SM_86)
  endif()

  if(CUDA_SM_60 GREATER_EQUAL 0 AND CUDA_VERSION VERSION_GREATER_EQUAL "8")
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_60,code=sm_60>)
    list(APPEND CUDA_FINAL_SM "sm_60")
  endif()
  if(CUDA_SM_61 GREATER_EQUAL 0 AND CUDA_VERSION VERSION_GREATER_EQUAL "8")
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61>)
    list(APPEND CUDA_FINAL_SM "sm_61")
  endif()
  if(CUDA_SM_70 GREATER_EQUAL 0 AND CUDA_VERSION VERSION_GREATER_EQUAL "9")
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_70,code=sm_70>)
    list(APPEND CUDA_FINAL_SM "sm_70")
  endif()
  if(CUDA_SM_72 GREATER_EQUAL 0 AND CUDA_VERSION VERSION_GREATER_EQUAL "9")
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_72,code=sm_72>)
    list(APPEND CUDA_FINAL_SM "sm_72")
  endif()
  if(CUDA_SM_75 GREATER_EQUAL 0 AND CUDA_VERSION VERSION_GREATER_EQUAL "10")
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=sm_75>)
    list(APPEND CUDA_FINAL_SM "sm_75")
  endif()
  if(CUDA_SM_80 GREATER_EQUAL 0 AND CUDA_VERSION VERSION_GREATER_EQUAL "11")
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_80,code=sm_80>)
    list(APPEND CUDA_FINAL_SM "sm_80")
  endif()
  if(CUDA_SM_86 GREATER_EQUAL 0 AND CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86>)
    list(APPEND CUDA_FINAL_SM "sm_86")
  endif()
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number>)

  message(STATUS "cuda shader config: ${CUDA_FINAL_SM}")
endfunction()

# Configure Clang++ cuda compiler
function(configure_clang_cuda_compiler TARGET_NAME)
  target_compile_definitions(${TARGET_NAME} PUBLIC CUDA_ENABLED)
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_14)
  set_target_properties(${TARGET_NAME} PROPERTIES CUDA_STANDARD 14)
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--cuda-gpu-arch=sm_60>)
endfunction()

# Configure Cuda Warning Options
function(configure_cuda_warnings TARGET_NAME)
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number>)
  if(CUDA_VERSION_STRING VERSION_GREATER 8)
    # 2829 annotation on a defaulted function is ignored
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=2829">)
  endif()
  if(CUDA_VERSION_STRING VERSION_GREATER 10)
    # 3057 annotation is ignored on a function that is explicitly defaulted
    # 2929 annotation is ignored on a function that is explicitly defaulted
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=3057,2929">)
    else()
  endif()
endfunction()