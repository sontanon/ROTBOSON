# rbs-viz

The visualization pipeline — first-class per ADR 0001 (placeholder).

**Role:** consume `rbs-io` snapshots (VTR/VTH + HDF5) and build the
animation/streaming pipeline: field movies along a continuation campaign
(ψ₀ or ω stepping), fold-approach visualizations, and collapse-threshold
sequences for the evolution track.

**Parity target:** none against C (the oracle has no viz pipeline) —
the target is ParaView-native output loadable directly, with VTH series
collection for campaign trajectories.

**Status:** placeholder; the VTR writer it will consume is real
(`rbs-io::vtr`).
