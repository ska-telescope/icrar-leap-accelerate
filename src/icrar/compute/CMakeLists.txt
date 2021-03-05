set(TARGET_NAME Compute)
set(TARGET_EXPORT compute-target)

set(sources
  main.cc
)


if(CUDA_ENABLED)
add_library(
  ${TARGET_NAME} STATIC
    ${sources}
    ${cuda_sources}
    ${private_headers}
    ${public_headers}
    ${cuda_headers}
)
else()
add_library(
  ${TARGET_NAME} STATIC
    ${sources}
    ${private_headers}
    ${public_headers}
)
endif()


# CUDA Config
if(CUDA_ENABLED)
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    message("generating clang-tidy cuda compatible configuration")
    configure_clang_cuda_compiler(${TARGET_NAME})
  else()
    configure_nvcc_cuda_compiler(${TARGET_NAME})
    configure_cuda_warnings(${TARGET_NAME})
  endif()
  
  #Set Cuda Stream Setting
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo --default-stream per-thread)
  #message("CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}")
  target_include_directories(${TARGET_NAME} SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${TARGET_NAME} ${CUDA_LIBRARIES})
  target_link_libraries(${TARGET_NAME} cublas)
  if(CUDA_VERSION_MAJOR VERSION_GREATER 9)
    target_link_libraries(${TARGET_NAME} cublasLt)
    target_link_libraries(${TARGET_NAME} ${CUDA_cusolver_LIBRARY})
  endif()
endif()

target_link_libraries(${TARGET_NAME} Threads::Threads)
target_link_libraries(${TARGET_NAME} ${Boost_LIBRARIES})
target_link_libraries(${TARGET_NAME} Eigen3::Eigen)

set_target_properties(${TARGET_NAME}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if (BUILD_TESTING)
  add_subdirectory(tests)
endif()

install (
  TARGETS
    ${TARGET_NAME}
  EXPORT
    ${TARGET_EXPORT}
  DESTINATION
    bin
)
