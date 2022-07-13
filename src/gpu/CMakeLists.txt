# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.

# Generate GPU specific library relying upon OpenCL.

# Compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -fabi-version=0 -O1 -Wall -Wno-comment -Wno-maybe-uninitialized")

# Add the EXBLAS library.
#
option (USE_EXBLAS "Use ExBLAS provided reproducible and accurate implementations of the BLAS routines" ON)
if (USE_EXBLAS)
  include_directories ("${PROJECT_SOURCE_DIR}/include")
  include_directories ("${PROJECT_SOURCE_DIR}/src/common")
  include_directories ("${PROJECT_SOURCE_DIR}/src/gpu/blas")
  include_directories ("${PROJECT_BINARY_DIR}/include")
  set (EXTRA_LIBS ${EXTRA_LIBS} exblas)
endif (USE_EXBLAS)

# Architecture specific flags.
#
option (EXBLAS_GPU_AMD "By setting this option we enable compilation for AMD GPUs" OFF)
option (EXBLAS_GPU_NVIDIA "By setting this option we enable compilation for NVIDIA GPUs" OFF)
if (EXBLAS_GPU_AMD)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=att -DAMD")
    include_directories( "/opt/sdks/amd/current/include/" )
    link_directories ( "/opt/sdks/amd/current/lib/x86_64/" )
elseif (EXBLAS_GPU_NVIDIA)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=intel")
    include_directories( "$ENV{CUDA_HOME}/include/" )
    link_directories ( "$ENV{CUDA_HOME}/lib64/" )
endif (EXBLAS_GPU_AMD)

set (EXTRA_LIBS ${EXTRA_LIBS} OpenCL)

# Traverse library and examples.
#
add_subdirectory (blas)
add_subdirectory (examples)