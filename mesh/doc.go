// Package mesh implements a GPU-friendly fixed-point vertex-attribute codec for
// mesh geometry and skinning data: positions (per-axis bounding-box
// fixed-point), normals (octahedral + signed-normalized), joint indices
// (u8/u16), and bone weights (u8 normalized).
//
// This codec is distinct from the root turboquant embedding vector quantizer.
// It is stateless: there are no codebooks and no learned rotation. Each
// attribute's decode is a small, branch-light formula — a multiply-add for
// positions and weights, the octahedral reconstruction for normals — that maps
// directly onto inline shader math. The Decode functions here are the reference
// implementation of that shader math, consumed by elio in Phase 3 of the Kiln
// GPU-skinning pipeline.
//
// The portable DequantSpec is a comparable value struct (no slices) that
// round-trips losslessly through JSON, so the renderer can reconstruct the
// dequant parameters from transport without depending on this package.
package mesh
