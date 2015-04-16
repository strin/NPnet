CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT (CIBP-NET)

set(CMAKE_PREFIX_PATH /opt/local ${CMAKE_PREFIX_PATH}) 
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules) 

find_package(Boost COMPONENTS random python REQUIRED)
message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
message("Libs of boost: " ${Boost_LIBRARIES} )

find_package(PythonLibs REQUIRED)
message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
message("Libs of Python: " ${PYTHON_LIBRARIES} )

include_directories(
  "inc"
  "usr/include"
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

message(STATUS ${LIBS})

IF(CMAKE_COMPILER_IS_GNUCXX)
  message("Linux: Found g++ version " ${CMAKE_CXX_COMPILER_VERSION})
  ADD_DEFINITIONS("-std=c++11" "-g" "-O3" "-fPIC" "-Wno-deprecated-register" "-Wno-overloaded-virtual" "-Wno-write-strings")
ELSE("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  message("Mac OS X: Found clang++ version" ${CMAKE_CXX_COMPILER_VERSION})
  ADD_DEFINITIONS("-std=c++11" "-g" "-O0" "-fPIC" "-Wno-deprecated-register" "-Wno-overloaded-virtual" "-Wno-write-strings")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/include")

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/*.cpp)
foreach(testSrc ${TEST_SRCS})
        #Extract the filename without an extension (NAME_WE)
        get_filename_component(testName ${testSrc} NAME_WE)

        #Add compile target
        add_executable(${testName} ${testSrc})

        #link to Boost libraries AND your targets and dependencies
        target_link_libraries(${testName} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} armadillo)

        #I like to move testing binaries into a testBin directory
        set_target_properties(${testName} PROPERTIES 
            RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/test-bin)

        #Finally add it to test execution - 
        #Notice the WORKING_DIRECTORY and COMMAND
        add_test(NAME ${testName} 
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test-bin 
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-bin/${testName} )
endforeach(testSrc)

ADD_EXECUTABLE(cibp
  main.cpp
)

target_link_libraries(cibp
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  armadillo
)

