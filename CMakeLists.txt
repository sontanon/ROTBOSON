cmake_minimum_required(VERSION 3.20)
project(ROTBOSON LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# pardiso = Intel oneMKL/PARDISO (default); umfpack = SuiteSparse/UMFPACK
# (MKL-free fallback; 64-bit indices throughout).
set(ROTBOSON_SOLVER_BACKEND "pardiso" CACHE STRING "Sparse direct solver backend")
set_property(CACHE ROTBOSON_SOLVER_BACKEND PROPERTY STRINGS pardiso umfpack)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCONFIG REQUIRED IMPORTED_TARGET libconfig)

# ---------------------------------------------------------------------------
# Solver backend
# ---------------------------------------------------------------------------
set(ROTBOSON_PARDISO_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pardiso_start.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pardiso_stop.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pardiso_solve.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/low_rank.c)

if(ROTBOSON_SOLVER_BACKEND STREQUAL "pardiso")
  find_package(MKL)
  if(NOT MKL_FOUND)
    message(FATAL_ERROR
      "Intel oneMKL not found. Set MKLROOT (source /opt/intel/oneapi/setvars.sh) "
      "or install oneMKL, or set -DROTBOSON_SOLVER_BACKEND=umfpack.")
  endif()
  set(ROTBOSON_BACKEND_LIBS MKL::ILP64)
  set(ROTBOSON_BACKEND_SOURCES ${ROTBOSON_PARDISO_SOURCES})
elseif(ROTBOSON_SOLVER_BACKEND STREQUAL "umfpack")
  find_path(SUITESPARSE_INCLUDE_DIR umfpack.h PATH_SUFFIXES suitesparse)
  find_library(UMFPACK_LIBRARY umfpack)
  find_library(CHOLMOD_LIBRARY cholmod)
  find_library(AMD_LIBRARY amd)
  find_library(COLAMD_LIBRARY colamd)
  find_library(CCOLAMD_LIBRARY ccolamd)
  find_library(CAMD_LIBRARY camd)
  find_library(SUITESPARSE_CONFIG_LIBRARY suitesparseconfig)
  if(NOT UMFPACK_LIBRARY OR NOT SUITESPARSE_INCLUDE_DIR)
    message(FATAL_ERROR
      "SuiteSparse/UMFPACK not found. Install it (dnf: suitesparse-devel; "
      "apt: libsuitesparse-dev) or use -DROTBOSON_SOLVER_BACKEND=pardiso.")
  endif()
  set(ROTBOSON_BACKEND_LIBS
    ${UMFPACK_LIBRARY} ${CHOLMOD_LIBRARY} ${AMD_LIBRARY}
    ${COLAMD_LIBRARY} ${CCOLAMD_LIBRARY} ${CAMD_LIBRARY}
    ${SUITESPARSE_CONFIG_LIBRARY} m)
  set(ROTBOSON_BACKEND_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/umfpack_backend.c)
else()
  message(FATAL_ERROR "Unknown ROTBOSON_SOLVER_BACKEND=${ROTBOSON_SOLVER_BACKEND}")
endif()

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
file(GLOB ROTBOSON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
list(FILTER ROTBOSON_SOURCES EXCLUDE REGEX "/deprecated/")
list(REMOVE_ITEM ROTBOSON_SOURCES
  ${ROTBOSON_PARDISO_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/umfpack_backend.c)
list(APPEND ROTBOSON_SOURCES ${ROTBOSON_BACKEND_SOURCES})

add_executable(ROTBOSON ${ROTBOSON_SOURCES})

target_include_directories(ROTBOSON PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  ${SUITESPARSE_INCLUDE_DIR})

# MKL shim header: only for the OSS build (must not shadow the real <mkl.h>).
if(ROTBOSON_SOLVER_BACKEND STREQUAL "umfpack")
  target_include_directories(ROTBOSON PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/compat")
endif()

target_link_libraries(ROTBOSON PRIVATE
  ${ROTBOSON_BACKEND_LIBS}
  PkgConfig::LIBCONFIG
  OpenMP::OpenMP_C
)

# gnu_threaded MKL needs --no-as-needed so its dependencies are not dropped.
target_link_options(ROTBOSON PRIVATE "LINKER:--no-as-needed")

if(MSVC)
  target_compile_options(ROTBOSON PRIVATE /W4)
else()
  target_compile_options(ROTBOSON PRIVATE -Wall -Wextra)
endif()
