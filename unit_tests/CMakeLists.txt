CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(cugip_test_suite  LANGUAGES CUDA CXX)

enable_testing()

set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.65 REQUIRED COMPONENTS program_options filesystem system timer chrono)


############################################################################
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)

# Find the QtWidgets library
find_package(Qt5Widgets CONFIG REQUIRED)

#Workaround for fPIC being passed to nvcc from Qt
# https://gitlab.kitware.com/cmake/cmake/issues/16915
if ( TARGET Qt5::Core )
	get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
	string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
	#set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
	#set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
endif()


###########################################################################






set(TEST_SUITE_TARGETS
#	cellular_automata_test
	eigen_test
	for_each_test
	scan_test
	reduce_test
#	geometry_transformation_test
#	graph_cut_test
	procedural_view_test
	subview_test
	shared_memory_test
	transform_test
	utils_test
	vector_test
	)

set(cugip_TEST_SUITE_DEPENDENCIES cugip Boost::program_options Boost::system Boost::timer Boost::chrono Qt5::Widgets)

#set(TEST_SUITE_TARGETS utils_test graph_cut_test procedural_view_test cellular_automata_test shared_memory_test transform_test for_each_test vector_test)
foreach(TEST_TARGET ${TEST_SUITE_TARGETS})
	message(STATUS "Adding test: ${TEST_TARGET}")
	ADD_EXECUTABLE(
		${TEST_TARGET}
		${TEST_TARGET}.cu
		)
	# Workaround
	set_source_files_properties(${TEST_TARGET}.cu PROPERTIES COMPILE_FLAGS " -Xcompiler=-fPIC")
	TARGET_LINK_LIBRARIES(${TEST_TARGET} ${cugip_TEST_SUITE_DEPENDENCIES} )
	add_test(NAME "cugip_${TEST_TARGET}" COMMAND ${TEST_TARGET})
endforeach(TEST_TARGET ${TEST_SUITE_TARGETS})
