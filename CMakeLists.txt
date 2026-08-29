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

# tomlc99 (vendored TOML parser) replaces libconfig for parameter files.
set(TOMLC99_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlc99")

# ---------------------------------------------------------------------------
# HDF5 output backend (Phase 5)
# ---------------------------------------------------------------------------
option(ROTBOSON_HDF5 "Enable the HDF5 output backend" ON)
if(ROTBOSON_HDF5)
  find_package(HDF5 COMPONENTS C QUIET)
  if(HDF5_FOUND)
    message(STATUS "HDF5 found: ${HDF5_VERSION}")
  else()
    message(WARNING
      "HDF5 not found; building without the HDF5 output backend "
      "(outputFormat=\"hdf5\" will fail at runtime).")
    set(ROTBOSON_HDF5 OFF)
  endif()
endif()

# ---------------------------------------------------------------------------
# Build provenance: record the git hash for the HDF5 attributes.
# ---------------------------------------------------------------------------
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE ROTBOSON_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()
if(NOT ROTBOSON_GIT_HASH)
  set(ROTBOSON_GIT_HASH "unknown")
endif()
message(STATUS "ROTBOSON git hash: ${ROTBOSON_GIT_HASH}")

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
# Targets
# ---------------------------------------------------------------------------
file(GLOB ROTBOSON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
list(FILTER ROTBOSON_SOURCES EXCLUDE REGEX "/deprecated/")
list(REMOVE_ITEM ROTBOSON_SOURCES
  ${ROTBOSON_PARDISO_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/umfpack_backend.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c)
list(APPEND ROTBOSON_SOURCES ${ROTBOSON_BACKEND_SOURCES} ${TOMLC99_DIR}/toml.c)

# Core library: everything except the main() driver, so tests can link it.
add_library(rotboson_core STATIC ${ROTBOSON_SOURCES})

target_include_directories(rotboson_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  ${TOMLC99_DIR}
  ${SUITESPARSE_INCLUDE_DIR})

# MKL shim header: only for the OSS build (must not shadow the real <mkl.h>).
if(ROTBOSON_SOLVER_BACKEND STREQUAL "umfpack")
  target_include_directories(rotboson_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/compat")
endif()

target_link_libraries(rotboson_core PUBLIC
  ${ROTBOSON_BACKEND_LIBS}
  OpenMP::OpenMP_C
)

if(ROTBOSON_HDF5)
  target_compile_definitions(rotboson_core PUBLIC ROTBOSON_HDF5=1)
  target_link_libraries(rotboson_core PUBLIC HDF5::HDF5)
endif()

target_compile_definitions(rotboson_core PUBLIC ROTBOSON_GIT_HASH="${ROTBOSON_GIT_HASH}")

# gnu_threaded MKL needs --no-as-needed so its dependencies are not dropped.
target_link_options(rotboson_core PUBLIC "LINKER:--no-as-needed")

if(MSVC)
  target_compile_options(rotboson_core PRIVATE /W4)
else()
  target_compile_options(rotboson_core PRIVATE -Wall -Wextra)
endif()

# Executable driver.
add_executable(ROTBOSON "${CMAKE_CURRENT_SOURCE_DIR}/src/main.c")
target_link_libraries(ROTBOSON PRIVATE rotboson_core)

# ---------------------------------------------------------------------------
# Tests (Phase 3)
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
