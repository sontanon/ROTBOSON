import sys
src=open("seed_scan.py").read()
ns={}; exec(src.split("for l in [int")[0], ns)
run=ns["run"]; classify=ns["classify"]
jobs={
 5:[(w,p) for w in (0.8,0.7) for p in (1e-5,2e-5,3e-5,5e-5,1e-4,2e-4,3e-4,5e-4)],
 6:[(w,p) for w in (0.8,0.7) for p in (1e-6,3e-6,1e-5,3e-5,1e-4,3e-4)],
}
for l,wl in jobs.items():
    print(f"=== l={l} mid-branch scan ===", flush=True)
    hit=False
    for w,p in wl:
        res=run(l,w,p); cls=classify(res)
        print(f"  w={w:.2f} p={p:.0e} -> {cls:9s} phi={res.get('phi_max')} hwl={res.get('hwl')} r99={res.get('r99')}")
        if cls=="REAL":
            print(f"  >>> SEED: out/seed_scan/l{l}_w{w:.3f}_p{p:.0e}/l={l},w={w:.5E},dr=1.25000E-01,N=0128"); hit=True; break
    if not hit: print(f"  no window found for l={l}")
