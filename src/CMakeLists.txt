cmake_minimum_required(VERSION 2.8)
project(lab6)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp Tracker.cpp Matches.h)

target_link_libraries(main ${OpenCV_LIBS})
