# Link math and OpenCL library.
#
set (EXTRA_LIBS ${EXTRA_LIBS} m OpenCL)

# Set target name base.
#
set (EXAMPLE_TARGET_NAME_BASE mri-gridding)

# Set install target directory.
#
set (EXAMPLE_TARGET_DIR ${PROJECT_BINARY_DIR}/examples/gpu/${EXAMPLE_TARGET_NAME_BASE})

# Traverse example versions.
#
add_subdirectory (StrongReproducibility)
add_subdirectory (CPUStrongReproducibility)

# Copy input data files during install.
#
install (DIRECTORY data DESTINATION ${EXAMPLE_TARGET_DIR})
