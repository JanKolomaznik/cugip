CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( tests )


set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options filesystem system timer chrono)
MESSAGE(  ${Boost_LIBRARIES} )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} "../include")

SET(CUDA_HOST_COMPILER "/usr/bin/gcc-4.8")
SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 --use_fast_math")

FIND_PACKAGE(CUDA)


SET(TEST1_SOURCES "test1.cpp" )
SET(TEST2_SOURCES "test2.cpp" )
SET(GENERATE_SOURCES "generate_image.cpp" )
SET( ${PROJECT_NAME}_CUDA_SOURCES "processing.cu" )

#CUDA_ADD_EXECUTABLE(test1 ${TEST1_SOURCES} ${${PROJECT_NAME}_CUDA_SOURCES} )
#TARGET_LINK_LIBRARIES(test1 jpeg png ${Boost_LIBRARIES})

CUDA_ADD_EXECUTABLE(test2 ${TEST2_SOURCES} ${${PROJECT_NAME}_CUDA_SOURCES} )
TARGET_LINK_LIBRARIES(test2 jpeg png ${Boost_LIBRARIES})

#CUDA_ADD_EXECUTABLE(generate_image ${GENERATE_SOURCES} ${${PROJECT_NAME}_CUDA_SOURCES} )
#TARGET_LINK_LIBRARIES(generate_image jpeg png ${Boost_LIBRARIES})

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
CUDA_ADD_EXECUTABLE(denoising denoising.cpp denoising.cu )
TARGET_LINK_LIBRARIES(denoising ${Boost_LIBRARIES} ${ITK_LIBRARIES})

