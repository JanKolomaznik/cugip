CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( cli-tools )


set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options filesystem system timer chrono)
MESSAGE(  ${Boost_LIBRARIES} )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} "../include")

SET(CUDA_HOST_COMPILER "/usr/bin/gcc-4.8")
SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math --generate-line-info --ptxas-options=-v")
#SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math --generate-line-info --maxrregcount=31 --ptxas-options=-v")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 --use_fast_math")

FIND_PACKAGE(CUDA)


find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
CUDA_ADD_EXECUTABLE(denoising denoising.cpp denoising.cu )
TARGET_LINK_LIBRARIES(denoising ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(gradient gradient.cpp gradient.cu)
TARGET_LINK_LIBRARIES(gradient ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(ce_diffusion ce_diffusion.cpp ce_diffusion.cu)
TARGET_LINK_LIBRARIES(ce_diffusion ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(gauss gauss.cpp gauss.cu)
TARGET_LINK_LIBRARIES(gauss ${Boost_LIBRARIES} ${ITK_LIBRARIES})
