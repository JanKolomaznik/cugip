CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(graph_cut_benchmarks)

#set(CMAKE_CXX_FLAGS --std=c++11 )
add_definitions("-DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
# Due to error in CUDA 7.0 we need this:
add_definitions("-DBOOST_RESULT_OF_USE_TR1")
find_path(CUB_INCLUDE_DIRECTORY "cub/cub.cuh" "../../cub")

find_path(BK301_DIRECTORY "maxflow.cpp" "../../BK301")
set(BK301_SOURCES
	"${BK301_DIRECTORY}/maxflow.cpp"
	"${BK301_DIRECTORY}/graph.cpp"
	)

find_path(GRIDCUT_DIRECTORY "GridCut/GridGraph_3D_6C.h" "../../GridCut")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

set(Boost_USE_STATIC_LIBS   OFF)
set(Boost_USE_MULTITHREADED ON)
#add_definitions(-DBOOST_LOG_DYN_LINK)
add_definitions(-DBOOST_ALL_DYN_LINK)
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options filesystem chrono system log log_setup timer)
MESSAGE(  ${Boost_LIBRARIES} )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} ${CUB_INCLUDE_DIRECTORY} "../../include" "${BK301_DIRECTORY}/.." ${GRIDCUT_DIRECTORY})

SET(CUDA_PROPAGATE_HOST_FLAGS OFF )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC  -DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
#set(CUDA_SEPARABLE_COMPILATION TRUE)
#SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_20,code=sm_20 --use_fast_math --generate-line-info")
#SET(CUDA_NVCC_FLAGS "-G --std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math -lineinfo")
#SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math -Xptxas=-v")
#SET(CUDA_NVCC_FLAGS "-rdc=true -std=c++11 -gencode arch=compute_52,code=sm_52 --use_fast_math --generate-line-info -D_FORCE_INLINES")
#SET(CUDA_NVCC_FLAGS "-rdc=true -std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math --generate-line-info -D_FORCE_INLINES")
SET(CUDA_NVCC_FLAGS "-std=c++11 -gencode arch=compute_61,code=sm_61 --use_fast_math --generate-line-info -D_FORCE_INLINES")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 --use_fast_math")

message("Cuda libs: ${CUDA_LIBRARIES}")

#set(CUDA_VERBOSE_BUILD ON)
FIND_PACKAGE(CUDA)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})


message("CUDA_LIBRARIES ${CUDA_LIBRARIES}")

#cuda_add_library2(${PROJECT_NAME}_cuda_cut graph_cut.cu)
#target_link_libraries(${PROJECT_NAME}_cuda_cut ${Boost_LIBRARIES})

#nvcc -gencode arch=compute_52,code=sm_52 --device-c "${CMAKE_CURRENT_SOURCE_DIR}/graph_cut.cu"
#nvcc –arch=sm_20 –dlink a.o b.o –o link.o
#g++ a.o b.o link.o –L<path> -lcudart

#nvcc --gpu-architecture=sm_35 --device-c a.cu b.cu
#nvcc --gpu-architecture=sm_35 --device-link a.o b.o --output-file link.o
#nvcc --lib --output-file libgpu.a a.o b.o link.o
#g++ host.o --library=gpu --library-path=<path> \
#    --library=cudadevrt --library=cudart

set(GK_INCLUDE_DIRS
	-I/usr/local/cuda/include
	-I/usr/include
	-I/home/johny/programming/git/cub
	-I/home/johny/programming/git/MedV4D/extern/cugip/benchmarks/graph_cut/../../include
	-I/home/johny/programming/Libs/BK301/..
	-I/home/johny/programming/Libs/GridCut/include
	-I/usr/local/cuda/include
	)


cuda_add_executable(grid_segmentation segmentation.cpp boykov_kolmogorov.cpp grid_cut.cpp graph_cut.cpp graph_cut.cu ${BK301_SOURCES})
cuda_add_executable(region_segmentation segmentation_regions.cpp boykov_kolmogorov.cpp grid_cut.cpp graph_cut.cpp graph_cut.cu ${BK301_SOURCES})

#cuda_add_executable(${PROJECT_NAME} segmentation.cpp boykov_kolmogorov.cpp grid_cut.cpp graph_cut.cpp graph_cut.cu ${BK301_SOURCES})
target_link_libraries(grid_segmentation
	#${PROJECT_NAME}_cuda_cut
#	"${CMAKE_CURRENT_BINARY_DIR}/libgraph_cut.so"
	${ITK_LIBRARIES}
	${Boost_LIBRARIES}
	${CUDA_LIBRARIES}
	)

target_link_libraries(region_segmentation
	#${PROJECT_NAME}_cuda_cut
#	"${CMAKE_CURRENT_BINARY_DIR}/libgraph_cut.so"
	${ITK_LIBRARIES}
	${Boost_LIBRARIES}
	${CUDA_LIBRARIES}
	)

#add_executable(region_graph region_graph.cpp regions_graph_construction.cpp)
#target_link_libraries(region_graph ${ITK_LIBRARIES} ${Boost_LIBRARIES})
