# Locate Intel oneMKL and provide an imported target `MKL::ILP64` that links
# the ILP64 (64-bit integer) interface with the GNU-threaded runtime.
#
# Detection order:
#   1. CMake's oneMKL config package (find_package(MKL CONFIG))
#   2. the MKLROOT environment variable (set by /opt/intel/oneapi/setvars.sh)
#
# Result:
#   MKL::ILP64   imported interface target (include dirs, lib dirs, libs,
#                compile definition MKL_ILP64)
#   MKL_FOUND    TRUE/FALSE
if(TARGET MKL::ILP64)
  set(MKL_FOUND TRUE)
  return()
endif()

find_package(MKL CONFIG QUIET)

set(MKL_ROOT "" CACHE PATH "Intel oneMKL installation root")

if(NOT MKL_ROOT)
  if(DEFINED ENV{MKLROOT})
    set(MKL_ROOT "$ENV{MKLROOT}")
  elseif(TARGET MKL::MKL)
    get_target_property(_mkl_incs MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
    list(GET _mkl_incs 0 _mkl_inc)
    string(REGEX REPLACE "/include$" "" MKL_ROOT "${_mkl_inc}")
  endif()
endif()

if(MKL_ROOT AND EXISTS "${MKL_ROOT}/include")
  add_library(MKL::ILP64 INTERFACE IMPORTED)
  target_include_directories(MKL::ILP64 INTERFACE "${MKL_ROOT}/include")
  target_link_directories(MKL::ILP64 INTERFACE "${MKL_ROOT}/lib/intel64")
  target_compile_definitions(MKL::ILP64 INTERFACE MKL_ILP64)
  target_link_libraries(MKL::ILP64 INTERFACE
    mkl_intel_ilp64 mkl_gnu_thread mkl_core m dl pthread)
  set(MKL_FOUND TRUE)
else()
  set(MKL_FOUND FALSE)
endif()
