CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( CUGIP )

if(UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif(UNIX)


SET(${PROJECT_NAME}_HEADERS
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/parallel_queue.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/device_flag.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/image_locator.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/static_memory_block.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/math/symmetric_tensor.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/math/matrix.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/math/eigen.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/tuple.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/image.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/utils.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/fill.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/nonlocal_means.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/coherence_enhancing_diffusion.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/graph_to_graphml.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/graph_cut.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/detail/graph_cut_push.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/detail/graph_cut_defines.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/detail/edge_record.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/detail/graph_cut_dynpar.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/detail/graph_cut_data.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/detail/graph_cut_host_utils.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/detail/graph_cut_implementation.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/detail/graph_cut_relabeling.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/advanced_operations/detail/graph_cut_policies.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/transform.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/subview.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/procedural_views.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/view_arithmetics.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/exception.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/host_image.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/neighborhood_accessor.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/host_image_view.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/for_each.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/meta_algorithm.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/scan.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/texture.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/cuda_utils.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/equivalence_management.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/reduce.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/memory.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/cellular_automata/async_cellular_automata.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/cellular_automata/cellular_automata.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/cellular_automata/global_state.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/cellular_automata/rules.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/filter.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/kernel_execution_utils.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/functors.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/image_view.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/itk_utils.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/neighborhood.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/copy.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/memory_view.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/timers.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/basic_filters/gaussian.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/basic_filters/watershed_transformation.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/basic_filters/connected_component_labeling.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/basic_filters/local_extremas.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/basic_filters/convolution.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/basic_filters/thresholding.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/basic_filters/local_minima_detection.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/basic_filters/gradient.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/math.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/algebra/arithmetics.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/static_int_sequence.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/region.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/gil_utils.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/traits.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/union_find.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/element.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/access_utils.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/detail/transform_device_implementation.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/detail/for_each_device_implementation.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/detail/transform_host_implementation.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/detail/defines.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/detail/include.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/detail/error_codes.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/detail/for_each_host_implementation.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/detail/shared_memory.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/detail/logging.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/cugip/detail/view_declaration_utils.hpp"
	)


SET(${PROJECT_NAME}_FILES ${${PROJECT_NAME}_HEADERS} PARENT_SCOPE)

option(CUGIP_ENABLE_TESTS "Enable building of the unit tests" ON)
option(CUGIP_ENABLE_CLI_TOOLS "Enable building of the cli utils" ON)

if(CUGIP_ENABLE_TESTS)
	add_subdirectory(unit_tests)
endif(CUGIP_ENABLE_TESTS)

if(CUGIP_ENABLE_CLI_TOOLS)
	add_subdirectory(cli-tools)
endif(CUGIP_ENABLE_CLI_TOOLS)
