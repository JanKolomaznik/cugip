CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( cugip_test_suite )

enable_testing()

#find_path(CUB_INCLUDE_DIRECTORY "cub/cub.cuh" "../../cub")

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

set(Boost_USE_STATIC_LIBS   ON)
#set(Boost_USE_STATIC_LIBS   OFF)
set(Boost_USE_MULTITHREADED ON)
#add_definitions(-DBOOST_ALL_DYN_LINK)
find_package(Boost REQUIRED COMPONENTS program_options filesystem system timer chrono)
#add_definitions(-DBOOST_TEST_DYN_LINK)

find_path(CUB_INCLUDE_DIRECTORY "cub/cub.cuh" "../../cub" "../../../cub" "../../../../cub")
include_directories( ${Boost_INCLUDE_DIRS} ${CUB_INCLUDE_DIRECTORY} "../include" ${CUB_INCLUDE_DIRECTORY})

SET(CUDA_PROPAGATE_HOST_FLAGS OFF )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC  -DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
#set(CUDA_HOST_COMPILER "/usr/bin/gcc-4.9")
SET(CUDA_NVCC_FLAGS "-std=c++14 -gencode arch=compute_61,code=sm_61 --use_fast_math --generate-line-info --expt-extended-lambda")
#SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math --generate-line-info --expt-extended-lambda")
#set(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_50,code=sm_50 --use_fast_math")
#set(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math")
#SET(CUDA_NVCC_FLAGS "-G --std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math -lineinfo")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 --use_fast_math")

#ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
find_package(CUDA)

set(TEST_SUITE_TARGETS
	cellular_automata_test
	eigen_test
	for_each_test
	scan_test
	reduce_test
	geometry_transformation_test
	graph_cut_test
	procedural_view_test
	subview_test
	shared_memory_test
	transform_test
	utils_test
	vector_test
	)

#set(TEST_SUITE_TARGETS utils_test graph_cut_test procedural_view_test cellular_automata_test shared_memory_test transform_test for_each_test vector_test)
foreach(TEST_TARGET ${TEST_SUITE_TARGETS})
	message(STATUS "Adding test: ${TEST_TARGET}")
	CUDA_ADD_EXECUTABLE(
		${TEST_TARGET}
		${TEST_TARGET}.cu
		)
		TARGET_LINK_LIBRARIES(${TEST_TARGET} ${Boost_LIBRARIES} ${Boost_LIBRARIES})
	add_test(NAME "cugip_${TEST_TARGET}" COMMAND ${TEST_TARGET})
endforeach(TEST_TARGET ${TEST_SUITE_TARGETS})
