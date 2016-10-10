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
SET(CUDA_NVCC_FLAGS "-rdc=true -std=c++11 -gencode arch=compute_61,code=sm_61 --use_fast_math --generate-line-info -D_FORCE_INLINES")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 --use_fast_math")

message("Cuda libs: ${CUDA_LIBRARIES}")

#set(CUDA_VERBOSE_BUILD ON)
FIND_PACKAGE(CUDA)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

macro(CUDA_ADD_LIBRARY2 cuda_target)

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  CUDA_BUILD_SHARED_LIBRARY(_cuda_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_cuda_shared_flag}
    OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  MESSAGE("_sources: ${_sources}")
  MESSAGE("_cmake_options: ${_cmake_options}")
  MESSAGE("_options: ${_options}")
  MESSAGE("_cuda_shared_flag: ${_cuda_shared_flag}")
  MESSAGE("link_file: ${link_file}")
  # Add the library.
  add_library(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  MESSAGE("aaaae: ${link_file} ${cuda_target} ${_options} ${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")
  target_link_libraries(${cuda_target}
    ${CUDA_LIBRARIES}
    "/usr/local/cuda-7.5/targets/x86_64-linux/lib/libcudadevrt.a"
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endmacro()


#CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})

#CUDA_WRAP_SRCS(${PROJECT_NAME}_cuda_cut OBJ _generated_files graph_cut.cu OPTIONS "-afafanfjk" )

#CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${PROJECT_NAME}_cuda_cut "${${PROJECT_NAME}_cuda_cut_SEPARABLE_COMPILATION_OBJECTS}")

#add_library(${PROJECT_NAME}_cuda_cut
#	    ${_generated_files}
#	    graph_cut.cu
#	    ${_sources}
#	    ${link_file}
#	    )

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

#set(CUSTOM_COMMAND_ARCH "-gencode arch=compute_52,code=sm_52")
set(CUSTOM_COMMAND_ARCH -gencode arch=compute_30,code=sm_30)

add_custom_command(
		OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/graph_cut.o"
		COMMAND nvcc
		ARGS
			${GK_INCLUDE_DIRS}
			${CUSTOM_COMMAND_ARCH}
			-std=c++11
			--device-c "${CMAKE_CURRENT_SOURCE_DIR}/graph_cut.cu"
			--output-file "${CMAKE_CURRENT_BINARY_DIR}/graph_cut.o"
			-ccbin /usr/bin/gcc-4.9
			--generate-line-info
			-Xcompiler "-O3 -fPIC"
			-DBOOST_RESULT_OF_USE_TR1
			-DBOOST_ALL_DYN_LINK
		MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/graph_cut.cu"
	)

add_custom_command(
		OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libgraph_cut.so"
		COMMAND nvcc
		ARGS
			--shared
			--link
			--output-file "${CMAKE_CURRENT_BINARY_DIR}/libgraph_cut.so"
			-L/usr/local/cuda/targets/x86_64-linux/lib/
			-lcudadevrt
			"${CMAKE_CURRENT_BINARY_DIR}/graph_cut.o"
		MAIN_DEPENDENCY "${CMAKE_CURRENT_BINARY_DIR}/graph_cut.o"
	)

#add_custom_command(OUTPUT a.txt
#	           COMMAND touch ARGS a.txt
#		   MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/graph_cut.cu"
#	   )

add_custom_target(run ALL
DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libgraph_cut.so")



add_executable(grid_segmentation segmentation.cpp boykov_kolmogorov.cpp grid_cut.cpp graph_cut.cpp ${BK301_SOURCES})
add_executable(region_segmentation segmentation_regions.cpp boykov_kolmogorov.cpp grid_cut.cpp graph_cut.cpp ${BK301_SOURCES})

#cuda_add_executable(${PROJECT_NAME} segmentation.cpp boykov_kolmogorov.cpp grid_cut.cpp graph_cut.cpp graph_cut.cu ${BK301_SOURCES})
target_link_libraries(grid_segmentation
	#${PROJECT_NAME}_cuda_cut
	"${CMAKE_CURRENT_BINARY_DIR}/libgraph_cut.so"
	${ITK_LIBRARIES}
	${Boost_LIBRARIES}
	${CUDA_LIBRARIES}
	)

target_link_libraries(region_segmentation
	#${PROJECT_NAME}_cuda_cut
	"${CMAKE_CURRENT_BINARY_DIR}/libgraph_cut.so"
	${ITK_LIBRARIES}
	${Boost_LIBRARIES}
	${CUDA_LIBRARIES}
	)

#add_executable(region_graph region_graph.cpp regions_graph_construction.cpp)
#target_link_libraries(region_graph ${ITK_LIBRARIES} ${Boost_LIBRARIES})
