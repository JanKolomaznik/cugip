CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( test_suite )

find_path(CUB_INCLUDE_DIRECTORY "cub/cub.cuh" "../../cub")

set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options filesystem system timer chrono unit_test_framework)
MESSAGE(  ${Boost_LIBRARIES} )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} ${CUB_INCLUDE_DIRECTORY} "../include")

SET(CUDA_HOST_COMPILER "/usr/bin/gcc-4.8")
SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math --generate-line-info")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 --use_fast_math")

FIND_PACKAGE(CUDA)

CUDA_ADD_EXECUTABLE(
		test_suite 
		#utils_test.cu 
		graph_cut_test.cu)
TARGET_LINK_LIBRARIES(test_suite ${Boost_LIBRARIES} ${ITK_LIBRARIES})

