cmake_minimum_required(VERSION 3.12)
project(rf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
#include_directories(include)
add_executable(DecisionTree src/main.cpp
        src/load_data.cpp
        src/attribute_list.cpp
        src/tree.cpp
        src/tree_node.cpp
        src/tools.cpp
        src/class_list.cpp
        src/rf.cpp)