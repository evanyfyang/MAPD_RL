project(lifelong)

cmake_minimum_required (VERSION 3.12)
set(CMAKE_CXX_STANDARD 14)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "DEBUG")
    # SET(CMAKE_BUILD_TYPE "RELEASE")
    #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
    #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")

include_directories("inc")
file(GLOB SOURCES "src/*.cpp")

# Find Boost
find_package( Boost 1.49.0 REQUIRED COMPONENTS program_options system filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )

find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

# add_executable(lifelong ${SOURCES} )
# target_link_libraries(lifelong ${Boost_LIBRARIES})

pybind11_add_module(mapf_solver ${SOURCES})
target_link_libraries(mapf_solver PRIVATE ${Boost_LIBRARIES})

