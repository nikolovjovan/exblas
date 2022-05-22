# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.

# Compiler flags.
#
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -fabi-version=0 -O1 -Wall -fopenmp -masm=intel")

# Add TBB.
#
set (EXTRA_LIBS ${EXTRA_LIBS} tbb -L/opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8/ /opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8/libtbb.so)

# Add the EXBLAS library.
#
option (USE_EXBLAS "Use ExBLAS provided reproducible and accurate implementations of the BLAS routines" ON)
if (USE_EXBLAS)
  include_directories ("${PROJECT_SOURCE_DIR}/include")
  include_directories ("${PROJECT_SOURCE_DIR}/src/common")
  include_directories ("${PROJECT_BINARY_DIR}/include")
  set (EXTRA_LIBS ${EXTRA_LIBS} exblas)
endif (USE_EXBLAS)

# Traverse library and examples.
#
add_subdirectory (blas)
add_subdirectory (examples)