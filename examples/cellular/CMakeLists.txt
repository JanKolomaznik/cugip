CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(cellular)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

if(UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif(UNIX )

if(MSVC)
	add_definitions(/MP)
endif(MSVC)

set(CUDA_HOST_COMPILER "/usr/bin/gcc-4.8")
set(CUDA_NVCC_FLAGS "--std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --use_fast_math --generate-line-info")

find_package(CUDA)

set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost REQUIRED COMPONENTS program_options filesystem system timer chrono)

find_package(OpenGL REQUIRED)

# Find the QtWidgets library
find_package(Qt5Widgets)

INCLUDE_DIRECTORIES(
	${Boost_INCLUDE_DIRS}
	"../../include"
	${${PROJECT_NAME}_BINARY_DIR}
	)

SET(${PROJECT_NAME}_HEADERS
	"AutomatonWrapper.hpp"
	)

SET(${PROJECT_NAME}_CUDA_SOURCES
	"automata.cu"
	)

SET(${PROJECT_NAME}_SOURCES
	"MainWindow.cpp"
	"main.cpp"
	)

qt5_wrap_ui(${PROJECT_NAME}_UIS
	"MainWindow.ui"
	)

CUDA_ADD_LIBRARY(${PROJECT_NAME}_cuda
	${${PROJECT_NAME}_CUDA_SOURCES}
	)


add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS} ${${PROJECT_NAME}_UIS})
IF(WIN32)
	SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES STATIC_LIBRARY_FLAGS /NODEFAULTLIB)
ENDIF(WIN32)

# Use the Widgets module from Qt 5.
qt5_use_modules(${PROJECT_NAME} Widgets OpenGL)

target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARY} ${PROJECT_NAME}_cuda ${Boost_LIBRARIES})
