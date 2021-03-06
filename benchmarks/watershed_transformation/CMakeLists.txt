CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(watershed)

set(CMAKE_CXX_FLAGS --std=c++11 )
add_definitions("-DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
# Due to error in CUDA 7.0 we need this:
add_definitions("-DBOOST_RESULT_OF_USE_TR1")
#find_path(CUB_INCLUDE_DIRECTORY "cub/cub.cuh" "../../cub")

set(Boost_USE_STATIC_LIBS   OFF)
set(Boost_USE_MULTITHREADED ON)
#add_definitions(-DBOOST_LOG_DYN_LINK)
add_definitions(-DBOOST_ALL_DYN_LINK)
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options filesystem chrono system log log_setup timer)
MESSAGE(  ${Boost_LIBRARIES} )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} ${CUB_INCLUDE_DIRECTORY} "../../include" )

SET(CUDA_HOST_COMPILER "/usr/bin/gcc-4.9")
#SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_20,code=sm_20 --use_fast_math --generate-line-info")
#SET(CUDA_NVCC_FLAGS "-G --std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math -lineinfo")
#SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math -Xptxas=-v")
SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math --keep")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 --use_fast_math")

#set(CUDA_VERBOSE_BUILD ON)
FIND_PACKAGE(CUDA)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

cuda_add_executable(${PROJECT_NAME} main.cpp watershed_transformation.cu)
target_link_libraries(${PROJECT_NAME} ${ITK_LIBRARIES} ${Boost_LIBRARIES})
