CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( cli-tools )


set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options filesystem system timer chrono)
MESSAGE(  ${Boost_LIBRARIES} )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} "../include")

SET(CUDA_PROPAGATE_HOST_FLAGS OFF )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC  -DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
#SET(CUDA_HOST_COMPILER "/usr/bin/gcc-4.8")
SET(CUDA_NVCC_FLAGS "-std=c++11 -gencode arch=compute_61,code=sm_61 --use_fast_math --generate-line-info")
#SET(CUDA_NVCC_FLAGS "-std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math --generate-line-info")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math --generate-line-info --ptxas-options=-v")
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

CUDA_ADD_EXECUTABLE(normalize_labeling normalize_labeling.cpp )
TARGET_LINK_LIBRARIES(normalize_labeling ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(watershed_transformation watershed_transformation.cpp watershed_transformation.cu)
TARGET_LINK_LIBRARIES(watershed_transformation ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(create_sample_dataset create_sample_dataset.cpp)
TARGET_LINK_LIBRARIES(create_sample_dataset ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(neighborhood_graph neighborhood_graph.cpp)
TARGET_LINK_LIBRARIES(neighborhood_graph ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(region_markers_from_masks region_markers_from_masks.cpp)
TARGET_LINK_LIBRARIES(region_markers_from_masks ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(mask_difference mask_difference.cpp)
TARGET_LINK_LIBRARIES(mask_difference ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(print_masked_labels print_masked_labels.cpp)
TARGET_LINK_LIBRARIES(print_masked_labels ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(ccl ccl.cpp ccl.cu)
TARGET_LINK_LIBRARIES(ccl ${Boost_LIBRARIES} ${ITK_LIBRARIES})

CUDA_ADD_EXECUTABLE(image_info image_info.cpp)
TARGET_LINK_LIBRARIES(image_info ${Boost_LIBRARIES} ${ITK_LIBRARIES})
