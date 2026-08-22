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
# Phase 1 uses PARDISO/MKL by default; the OpenBLAS + SuiteSparse/UMFPACK
# fallback lands in a later step (ROTBOSON_SOLVER_BACKEND=umfpack).
set(ROTBOSON_SOLVER_BACKEND "pardiso" CACHE STRING "Sparse direct solver backend (pardiso)")
set_property(CACHE ROTBOSON_SOLVER_BACKEND PROPERTY STRINGS pardiso umfpack)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(MKL)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCONFIG REQUIRED IMPORTED_TARGET libconfig)

# ---------------------------------------------------------------------------
# Solver backend
# ---------------------------------------------------------------------------
if(ROTBOSON_SOLVER_BACKEND STREQUAL "pardiso")
  if(NOT MKL_FOUND)
    message(FATAL_ERROR
      "Intel oneMKL not found. Set MKLROOT (source /opt/intel/oneapi/setvars.sh) "
      "or install oneMKL, or set -DROTBOSON_SOLVER_BACKEND=umfpack once supported.")
  endif()
  set(ROTBOSON_BACKEND_LIBS MKL::ILP64)
  set(ROTBOSON_BACKEND_DEFS "")
elseif(ROTBOSON_SOLVER_BACKEND STREQUAL "umfpack")
  message(FATAL_ERROR "UMFPACK backend not yet implemented (Phase 1 step 2).")
else()
  message(FATAL_ERROR "Unknown ROTBOSON_SOLVER_BACKEND=${ROTBOSON_SOLVER_BACKEND}")
endif()

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
file(GLOB ROTBOSON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
list(FILTER ROTBOSON_SOURCES EXCLUDE REGEX "/deprecated/")

add_executable(ROTBOSON ${ROTBOSON_SOURCES})

target_include_directories(ROTBOSON PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

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
