# Add OpenCL. For some reason does not work if I add it to the top level directory ("gpu")...
#
set (EXTRA_LIBS ${EXTRA_LIBS} OpenCL)

# Traverse examples.
#
add_subdirectory (RNGExample)
add_subdirectory ("kmeans (Rodinia)")
# add_subdirectory ("mri-gridding (Parboil)")
add_subdirectory ("spmv (Parboil)")