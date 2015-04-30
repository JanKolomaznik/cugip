CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(benchmarks)


set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options filesystem timer chrono system)
MESSAGE(  ${Boost_LIBRARIES} )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} "../include")

SET(CUDA_HOST_COMPILER "/usr/bin/gcc-4.8")
#SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_20,code=sm_20 --use_fast_math --generate-line-info")
SET(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_30,code=sm_30 --use_fast_math --generate-line-info")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math")
#SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 --use_fast_math")

FIND_PACKAGE(CUDA)
SET(CUGIP_HEADERS
	../include/cugip/detail/include.hpp
	../include/cugip/device_flag.hpp
	../include/cugip/advanced_operations/detail/graph_cut_implementation.hpp
	../include/cugip/advanced_operations/nonlocal_means.hpp
	../include/cugip/advanced_operations/graph_to_graphml.hpp
	../include/cugip/advanced_operations/graph_cut.hpp
	../include/cugip/advanced_operations/coherence_enhancing_diffusion.hpp
	../include/cugip/neighborhood.hpp
	../include/cugip/fill.hpp
	../include/cugip/utils.hpp
	../include/cugip/traits.hpp
	../include/cugip/access_utils.hpp
	../include/cugip/copy.hpp
	../include/cugip/functors.hpp
	../include/cugip/meta_algorithm.hpp
	../include/cugip/gil_utils.hpp
	../include/cugip/math.hpp
	../include/cugip/image.hpp
	../include/cugip/filter.hpp
	../include/cugip/image_locator.hpp
	../include/cugip/memory.hpp
	../include/cugip/algebra/arithmetics.hpp
	../include/cugip/transform.hpp
	../include/cugip/kernel_execution_utils.hpp
	../include/cugip/exception.hpp
	../include/cugip/memory_view.hpp
	../include/cugip/basic_filters/local_extremas.hpp
	../include/cugip/basic_filters/gaussian.hpp
	../include/cugip/basic_filters/local_minima_detection.hpp
	../include/cugip/basic_filters/thresholding.hpp
	../include/cugip/basic_filters/convolution.hpp
	../include/cugip/basic_filters/gradient.hpp
	../include/cugip/basic_filters/watershed_transformation.hpp
	../include/cugip/basic_filters/connected_component_labeling.hpp
	../include/cugip/static_memory_block.hpp
	../include/cugip/image_view.hpp
	../include/cugip/union_find.hpp
	../include/cugip/for_each.hpp
	../include/cugip/scan.hpp
)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} bench-CUGIP.cpp bench-CUGIP.cu mfi.cpp timer.cpp ${CUGIP_HEADERS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Boost_LIBRARIES})

