# rbs-evolve

Time evolution of the initial data — the staged 1D-first program
(placeholder).

**Role:** evolve the elliptic initial data forward: 1D spherical
full-GR Einstein–Klein–Gordon first, validated by holding the stationary
solution (the elliptic solve) fixed in time, then measuring collapse
thresholds; axisymmetric evolution follows, and 3D/AMR stays gated
(ADR 0001 decision 4).

**Parity target:** the stationary-hold test — the evolved state stays on
the elliptic solution to solver tolerance for many dynamical times. The
oracle is the elliptic solve itself (rbs-elliptic), not the C code (which
does not evolve).

**Status:** formulation study only (BSSN vs CCZ4 vs generalized-harmonic
in 1D); no code paths yet.
