cmake_minimum_required (VERSION 3.8)

project ("Lab6")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable (Tracking util.h tracker.h tracker.cpp Lab6.h Lab6.cpp)
target_link_libraries(Tracking ${OpenCV_LIBS})