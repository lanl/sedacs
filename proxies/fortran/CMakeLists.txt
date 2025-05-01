cmake_minimum_required(VERSION 3.10.0)
project(appendix C CXX Fortran)

set(dir ${CMAKE_CURRENT_SOURCE_DIR}/build/)
set(CMAKE_BUILD_DIRECTORY ${dir})
set(CMAKE_CURRENT_BINARY_DIR ${dir})

include(FindPkgConfig)

find_package(BML CONFIG QUIET)
pkg_check_modules(BML REQUIRED bml)
list(APPEND LINK_LIBRARIES BML::bml)
list(APPEND LINK_LIBRARIES ${BML_LDFLAGS})
message(STATUS "Found bml: ${BML_LDFLAGS}")

find_package(PROGRESS CONFIG QUIET)
pkg_check_modules(PROGRESS REQUIRED progress)
message(STATUS "Found progress: ${PROGRESS_LDFLAGS}")
list(APPEND LINK_LIBRARIES ${PROGRESS_LDFLAGS})

find_library(FOUND_METIS metis)
  if(NOT FOUND_METIS)
    message(FATAL_ERROR "Could not find metis library")
  endif()
  message(STATUS "Found metis: ${FOUND_METIS}")
  add_definitions(-DDO_GRAPHLIB)
  list(APPEND LINK_LIBRARIES ${FOUND_METIS})

  get_filename_component(METIS_LIB ${FOUND_METIS} DIRECTORY)
  list(APPEND LINK_LIBRARIES "-L/${METIS_LIB} -lmetis")

if(PROGRESS_MPI)
  message(STATUS "Will build with MPI")
  add_definitions(-DDO_MPI)
endif()

if(PROGRESS)
  message(STATUS "Will build with MPI")
  add_definitions(-DUSEPROGRESS)
endif()

if(SANITY_CHECK)
  message(STATUS "Will build with SANITY CHECK")
  add_definitions(-DSANITY_CHECK)
endif()
	
message(STATUS "Extra FC Flags ${EXTRA_FCFLAGS}")

if(DEFINED EXTRA_FCFLAGS)
   set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${EXTRA_FCFLAGS}")
endif()


if(LIB)
  list(APPEND SHAREDLIB "-fPIC -shared")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${EXTRA_FCFLAGS} ${SHAREDLIB}")
endif()

message(STATUS "Project sources = " ${PROJECT_SOURCE_DIR} )
include_directories(${CMAKE_BINARY_DIR}/)
include_directories(${BML_INCLUDEDIR})
include_directories(${PROGRESS_INCLUDEDIR})

list(
  APPEND common_sources 
"proxy_a_lib.F90"
"proxy_a_mod.F90"
                            )


function(progress_appendix myappendix main_and_srcs)
list(GET main_and_srcs 0 main)
include_directories(${PROGRESS_INCLUDEDIR})
add_executable(${myappendix} ${main} ${common_sources} ${extras})
target_sources(${myappendix} PRIVATE ${ARGN})
target_link_libraries(${myappendix} PUBLIC
    ${LINK_LIBRARIES})
  set_target_properties(${myappendix}
    PROPERTIES
    LINK_FLAGS "")
  #add_subdirectory(hamiltonian)
endfunction(progress_appendix)

function(progress_appendix_library myappendix main_and_srcs)
list(GET main_and_srcs 0 main)
include_directories(${PROGRESS_INCLUDEDIR})
add_library(${myappendix} SHARED ${main} ${common_sources} ${extras})
target_sources(${myappendix} PRIVATE ${ARGN})
target_link_libraries(${myappendix} PUBLIC
    ${LINK_LIBRARIES})
  set_target_properties(${myappendix}
    PROPERTIES
    LINK_FLAGS "")
endfunction(progress_appendix_library)

if(LIB)
  progress_appendix_library( proxya_fortran proxy_a_lib.F90 proxy_a_mod.F90 )
  #progress_appendix( gpmd_a gpmd_secuential.F90 gpmdcov_lib_mod.F90)
else()
  progress_appendix( proxya proxy_a.F90 )
endif()

SET(DESTINATION ${PROJECT_BINARY_DIR})
install(FILES ${CMAKE_BINARY_DIR}/progress.pc
  DESTINATION ${PROJECT_BINARY_DIR})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
