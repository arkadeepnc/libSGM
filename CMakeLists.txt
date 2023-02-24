cmake_minimum_required(VERSION 3.18)
# cmake .. -D CMAKE_CUDA_COMPILER=$(which nvcc)  
option(ENABLE_ZED_DEMO      "Build a Demo using ZED Camera" OFF)
option(ENABLE_SAMPLES       "Build samples" ON)
option(ENABLE_TESTS         "Test library" OFF)
option(LIBSGM_SHARED        "Build a shared library" ON)
option(BUILD_PYTHON_WRAPPER "Build pybind11 wrappers" ON)
option(BUILD_OPENCV_WRAPPER "Make library compatible with cv::Mat and cv::cuda::GpuMat of OpenCV" ON) # needed for python wrapper



set(CUDA_ARCHS "52;61;72;75;86" CACHE STRING "List of architectures to generate device code for")

project(libSGM VERSION 3.0.0)

configure_file(
${PROJECT_SOURCE_DIR}/include/libsgm_config.h.in
${PROJECT_SOURCE_DIR}/include/libsgm_config.h
)

add_subdirectory(src)

if(BUILD_PYTHON_WRAPPER)
    if (LIBSGM_SHARED)
        add_subdirectory(pysgm)
    else()
        message(WARNING "Python wrappers requires LIBSGM_SHARED=ON")
    endif()
endif()

if(ENABLE_SAMPLES)
	add_subdirectory(sample)
endif()

if(ENABLE_TESTS)
	add_subdirectory(test)
endif()