//go:build !(goexperiment.simd && amd64)

package turboquant

// signedProjectedSum8 folds one packed sign byte against its 8 matching
// per-row dot products: bit k selects +dots[k] when set, -dots[k] when
// clear. A goexperiment.simd amd64 build uses a branchless masked-multiply
// variant instead; see qjl_sign_simd_amd64.go.
func signedProjectedSum8(signs byte, dots [8]float32) float32 {
	var sum float32
	if signs&(1<<0) != 0 {
		sum += dots[0]
	} else {
		sum -= dots[0]
	}
	if signs&(1<<1) != 0 {
		sum += dots[1]
	} else {
		sum -= dots[1]
	}
	if signs&(1<<2) != 0 {
		sum += dots[2]
	} else {
		sum -= dots[2]
	}
	if signs&(1<<3) != 0 {
		sum += dots[3]
	} else {
		sum -= dots[3]
	}
	if signs&(1<<4) != 0 {
		sum += dots[4]
	} else {
		sum -= dots[4]
	}
	if signs&(1<<5) != 0 {
		sum += dots[5]
	} else {
		sum -= dots[5]
	}
	if signs&(1<<6) != 0 {
		sum += dots[6]
	} else {
		sum -= dots[6]
	}
	if signs&(1<<7) != 0 {
		sum += dots[7]
	} else {
		sum -= dots[7]
	}
	return sum
}
