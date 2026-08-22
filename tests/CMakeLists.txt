# Phase 3 C tests (run via CTest). Each test is a self-contained executable
# that returns non-zero on failure and prints a one-line summary.

add_executable(test_derivatives
  test_derivatives.c
  fornberg.c)
target_link_libraries(test_derivatives PRIVATE rotboson_core)

add_test(NAME derivatives COMMAND test_derivatives)
