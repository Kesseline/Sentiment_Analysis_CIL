cmake_minimum_required (VERSION 3.0)
#include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
include(DownloadProject.cmake)

project (RNTN)

download_project(
		PROJ         eigen
		SOURCE_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/eigen
		DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.cache/eigen
		QUIET
		URL           http://bitbucket.org/eigen/eigen/get/3.2.10.tar.gz
		URL_MD5       8ad10ac703a78143a4062c9bda9d8fd3
	)
	


FILE(GLOB SRCFILES src/*.cpp)
FILE(GLOB HDRFILES src/*.h)

include_directories(${PROJECT_NAME} SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/eigen>
    $<INSTALL_INTERFACE:eigen>)

add_executable(${PROJECT_NAME} ${SRCFILES} ${HDRFILES})