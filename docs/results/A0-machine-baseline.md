# Machine baseline (A0)

Date: 2026-11-27

## Machine
- CPU: Model name:                              13th Gen Intel(R) Core(TM) i7-13700K
- Threads: 24
- RAM: 31 GB
- OS: Linux 7.2.15-76070215-generic
- Compiler: gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
- CMake: cmake version 3.28.3
- MKL: 2026.1 2026.2 latest
- HDF5: 1.10.10
- Rust: rustc 1.100.0 (0ef1bc386 2026-11-09)

## C oracle tests
```
4/4 Test #4: exit_codes .......................   Passed    1.44 sec

100% tests passed, 0 tests failed out of 4

Total Test time (real) =   1.47 sec
```

## Smoke solve
```
14:15:02 INFO    smoke: running ROTBOSON with l1_smoke_ci.toml ...
14:15:04 INFO    smoke: ROTBOSON exited with code 0; log -> /home/santiago/Code/rotboson/out/smoke_run.log

[smoke] ROTBOSON solve took 1.3s (wall)

[smoke] solution directory: l=1,w=9.50000E-01,dr=4.16667E-02,N=0384
[smoke] error_code = N/A

[smoke] DONE
```

Note: the solver ran and exited 0. The "solution directory" line above is wrong: `tools/smoke.py` prints the last solution directory in `out/` sorted by name, here an old N=384 run from 2026-09-14, so `error_code = N/A` belongs to that old run. This run wrote `out/l=1,w=9.50000E-01,dr=2.50000E-01,N=0064/`. The helper bug is tracked separately.

## Rust HDF5 tests
```
test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```
