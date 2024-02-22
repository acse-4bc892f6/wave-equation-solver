cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(ExampleCode)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Host system processor: ${CMAKE_HOST_SYSTEM_PROCESSOR}.")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}.")
message(STATUS "C++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}.")

find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

add_executable(main main.cpp)
target_link_libraries(main ${MPI_LIBRARIES})