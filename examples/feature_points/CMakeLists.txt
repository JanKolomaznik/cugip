cmake_minimum_required(VERSION 3.7.0)

project(FeaturePoints)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_executable(feature_points
    mainwindow.ui
    mainwindow.cpp
    mainwindow.h
    main.cpp
    feature_point_item.cpp
    feature_point_item.h
)

target_link_libraries(feature_points Qt5::Widgets)
