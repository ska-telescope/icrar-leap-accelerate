
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

  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_60,code=sm_60>)
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_80,code=sm_80>)
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number>)
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
  if(CUDA_VERSION_STRING VERSION_GREATER 10)
    # 2829 annotation on a defaulted function is ignored
    # 3057 annotation is ignored on a function that is explicitly defaulted
    # 2929 annotation is ignored on a function that is explicitly defaulted
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=3057,2929">)
  else()
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=2829">)
  endif()
endfunction()