//go:build goexperiment.simd && amd64

package turboquant

import "simd/archsimd"

// signMask8Table[signs] holds +1/-1 per bit of signs: lane k is +1 when bit
// k of signs is set, -1 otherwise. signedProjectedSum8 uses it to replace 8
// data-dependent branches with one table load, one vector multiply, and a
// horizontal sum. This is NOT the signLUT gather pattern in ip.go (see the
// SIMD spike report): this table is indexed once per call by a plain Go
// slice index (signs, 0-255), not walked per-output-lane by a variable
// index, so it needs no gather/scatter support from the simd package.
var signMask8Table [256][8]float32

func init() {
	for v := 0; v < 256; v++ {
		for k := 0; k < 8; k++ {
			if v&(1<<uint(k)) != 0 {
				signMask8Table[v][k] = 1
			} else {
				signMask8Table[v][k] = -1
			}
		}
	}
}

// signedProjectedSum8 folds one packed sign byte against its 8 matching
// per-row dot products. The default (non-experiment) build uses a branchy
// scalar accumulation instead; see qjl_sign_generic.go.
func signedProjectedSum8(signs byte, dots [8]float32) float32 {
	mask := archsimd.LoadFloat32x8(&signMask8Table[signs])
	d := archsimd.LoadFloat32x8(&dots)
	return hsumFloat32x8(mask.Mul(d))
}
