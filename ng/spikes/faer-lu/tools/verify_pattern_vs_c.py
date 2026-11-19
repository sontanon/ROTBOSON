#!/usr/bin/env python3
"""Verify the Rust pattern tables against src/csr_vars.c (SAN-54).

Usage: python3 tools/verify_pattern_vs_c.py [repo-root]

Parses the column lists of the four
jacobian_4th_order_variable_omega_{cc,cs,sc,ss} kernels out of the C source
and diffs them, entry by entry, against the CC/CS/SC/SS tables in
src/pattern.rs. Prints ALL TABLES MATCH on success.
"""
import re, sys, os

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join('..', '..', '..')
C_SRC = os.path.join(ROOT, 'src', 'csr_vars.c')
RUST_SRC = os.path.join(ROOT, 'ng', 'spikes', 'faer-lu', 'src', 'pattern.rs')

src = open(C_SRC).read()
kdefs = [(m.start(), m.group(1)) for m in re.finditer(r'void (jacobian_\w+)\(', src)]
kdefs.append((len(src), 'END'))
def parse_one(e):
    e = e.replace(' ','')
    if e in ('i','j'): return 0
    m = re.match(r'i([+-]\d+)$', e)
    if m: return int(m.group(1))
    m = re.match(r'j([+-]\d+)$', e)
    if m: return int(m.group(1))
    raise ValueError(e)
def parse_idx(expr):
    a,b = expr.split(',')
    return (parse_one(a), parse_one(b))
c_tables = {}
for ki in range(len(kdefs)-1):
    name = kdefs[ki][1]
    if '4th_order_variable_omega_' not in name:
        continue
    body = src[kdefs[ki][0]:kdefs[ki+1][0]]
    for grid in range(6):
        rows = {}
        for m in re.finditer(r'ja\[offset%d \+ (\d+)\] = BASE \+ (\d+) \* dim \+ IDX\(([^;]+?)\);' % (grid+1), body):
            di, dj = parse_idx(m.group(3))
            rows[int(m.group(1))] = (int(m.group(2)), di, dj)
        for m in re.finditer(r'ja\[offset%d \+ (\d+)\] = BASE \+ w_idx;' % (grid+1), body):
            rows[int(m.group(1))] = (6, 0, 0)
        c_tables[(name[-2:], grid)] = [rows[s] for s in sorted(rows)]


r = open(RUST_SRC).read()
def rust_table(tag):
    mm = re.search(r'const %s: \[&\[\(u8, i8, i8\)\]; 6\] = \[(.*?)\n\];' % tag, r, re.S)
    assert mm, "no table " + tag
    fields = re.findall(r'&\[(.*?)\],', mm.group(1), re.S)
    assert len(fields) == 6, (tag, len(fields))
    return [[(int(a), int(b), int(c)) for a, b, c in re.findall(r'\((\d+), (-?\d+), (-?\d+)\)', f)] for f in fields]

R = {t.upper(): rust_table(t.upper()) for t in ('cc', 'cs', 'sc', 'ss')}
ok = True
for ct in ('cc', 'cs', 'sc', 'ss'):
    for g in range(6):
        c = c_tables[(ct, g)]
        mm = R[ct.upper()][g]
        if c != mm:
            ok = False
            print(f"{ct} g{g}: C {len(c)} vs rust {len(mm)}; missing {[x for x in c if x not in mm][:5]} extra {[x for x in mm if x not in c][:5]}")
print("ALL TABLES MATCH" if ok else "MISMATCH")
