# Phase 3 C tests (run via CTest). Each test is a self-contained executable
# that returns non-zero on failure and prints a one-line summary.

add_executable(test_derivatives
  test_derivatives.c
  fornberg.c)
target_link_libraries(test_derivatives PRIVATE rotboson_core)

add_test(NAME derivatives COMMAND test_derivatives)

add_executable(test_residual test_residual.c)
target_link_libraries(test_residual PRIVATE rotboson_core)

add_test(NAME residual COMMAND test_residual)

add_executable(test_output test_output.c)
target_link_libraries(test_output PRIVATE rotboson_core)

add_test(NAME output COMMAND test_output)
