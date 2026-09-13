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

# Exit-code contract test: runs the real binary with valid and
# invalid parameter files and pins the process exit codes.
add_executable(test_exit_codes test_exit_codes.c)
target_compile_definitions(
    test_exit_codes PRIVATE ROTBOSON_BIN="$<TARGET_FILE:ROTBOSON>"
                            ROTBOSON_SMOKE_CONFIG="${CMAKE_SOURCE_DIR}/configs/l1_smoke_ci.toml")

add_test(NAME exit_codes COMMAND test_exit_codes)
