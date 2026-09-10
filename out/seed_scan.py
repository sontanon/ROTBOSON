#!/usr/bin/env python3
"""Find from-scratch seed solutions for l=3..6 at fixed omega.

Recipe (from the l=2 seeds, out/l2_seedrun/): the analytic ansatz amplitude
psi0 must sit in a convergence window at the given w:
  - too strong -> Newton diverges / PARDISO -4 (exit != 0),
  - too weak   -> Newton converges to the trivial flat solution,
  - right      -> converges to the genuine branch solution.
Classify each run from its exit code + output-dir files (no log parsing).
"""
import re, subprocess, shutil, sys
from pathlib import Path

BIN = "/workspace/rotboson/build/release/ROTBOSON"
WORK = Path("/workspace/rotboson/out/seed_scan")
TEMPLATE = """# seed scan: l={l} w={w:.6E} psi0={p:.6E}
dr = 1.25000E-01
dz = 1.25000E-01
NrInterior = 128
NzInterior = 128
order = 4

l = {l}
m = 1.0
psi0 = {p:.6E}
sigmaR = 4.0
sigmaZ = 4.0
rExt = 12.0

readInitialData = 0
w0 = {w:.6E}

fixedPhi = 0
fixedPhiR = 0
fixedPhiZ = 0
fixedOmega = 1

solverType = 1
localSolver = 1
epsilon = 1.0E-8
maxNewtonIter = 60
lambda0 = 1.0E-03
lambdaMin = 1.0E-06
useLowRank = 0
"""

PSI_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6]


def run(l, w, p):
    d = WORK / f"l{l}_w{w:.3f}_p{p:.0e}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.toml").write_text(TEMPLATE.format(l=l, w=w, p=p))
    out = f"l={l},w={w:.5E},dr=1.25000E-01,N=0128"
    shutil.rmtree(d / out, ignore_errors=True)
    log = d / "run.log"
    try:
        with log.open("w") as lf:
            rc = subprocess.run([BIN, "params.toml"], cwd=d, stdout=lf,
                                stderr=subprocess.STDOUT, timeout=300).returncode
    except subprocess.TimeoutExpired:
        return rc if False else {"rc": -999}
    sdir = d / out
    res = {"rc": rc}
    def tailval(name):
        f = sdir / f"{name}.asc"
        if f.exists():
            try:
                return float(f.read_text().split()[-1])
            except Exception:
                return None
        return None
    res["phi_max"] = tailval("phi_max")
    res["r99"] = tailval("r99")
    res["hwl"] = tailval("hwl_resolution")
    pf = sdir / "psi_f.asc"
    if pf.exists():
        try:
            res["psi0"] = float(pf.read_text().split("\n")[2].split()[0])
        except Exception:
            res["psi0"] = None
    return res


def classify(res):
    rc, phi, r99, hwl = res.get("rc"), res.get("phi_max"), res.get("r99"), res.get("hwl")
    if rc != 0:
        return "DIVERGED"
    if phi is None or abs(phi) < 1e-3:
        return "TRIVIAL"
    if r99 is not None and r99 > 16.0:
        return "DOMAIN"
    return "REAL"


for l in [int(x) for x in sys.argv[1:]] or [3, 4, 5, 6]:
    print(f"=== l={l} ===", flush=True)
    found = False
    for w in (0.9, 0.8):
        if found:
            break
        for p in PSI_GRID:
            res = run(l, w, p)
            cls = classify(res)
            print(f"  w={w:.3f} psi0={p:.0e} -> {cls:9s} phi_max={res.get('phi_max')} hwl={res.get('hwl')} r99={res.get('r99')}")
            if cls == "REAL":
                print(f"  >>> SEED: out/seed_scan/l{l}_w{w:.3f}_p{p:.0e}/l={l},w={w:.5E},dr=1.25000E-01,N=0128")
                found = True
                break
    if not found:
        print(f"  >>> NO seed found for l={l} on the scanned grid")
