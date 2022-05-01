# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.

# Recurse into the "cpu" and "gpu" subdirectories to build architecture specific library

# Compile for CPUs
option (EXBLAS_CPU "Create a CPU specific library" OFF)
# Compile for GPUs
option (EXBLAS_GPU "Create a GPU specific library" OFF)

if (EXBLAS_GPU)
   add_subdirectory (gpu)
else (EXBLAS_CPU)
   add_subdirectory (cpu)
endif (EXBLAS_GPU)
