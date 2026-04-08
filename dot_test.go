package turboquant

import (
	"math"
	"math/rand"
	"testing"
)

func dotFloat32sScalar(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func TestDotFloat32sMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	lengths := []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 384}
	for _, n := range lengths {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := 0; i < n; i++ {
			a[i] = float32(rng.NormFloat64())
			b[i] = float32(rng.NormFloat64())
		}
		got := DotFloat32s(a, b)
		want := dotFloat32sScalar(a, b)
		diff := math.Abs(float64(got - want))
		limit := 1e-4 * math.Max(1, math.Abs(float64(want)))
		if diff > limit {
			t.Fatalf("n=%d: got %.8f want %.8f diff %.8f", n, got, want, diff)
		}
	}
}

func TestDotFloat32Rows4MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(456))
	lengths := []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 384}
	for _, n := range lengths {
		rows := make([]float32, 4*n)
		vec := make([]float32, n)
		for i := range rows {
			rows[i] = float32(rng.NormFloat64())
		}
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		var got [4]float32
		dotFloat32Rows4(&got, rows, vec)
		for row := 0; row < 4; row++ {
			want := dotFloat32sScalar(rows[row*n:(row+1)*n], vec)
			diff := math.Abs(float64(got[row] - want))
			limit := 1e-4 * math.Max(1, math.Abs(float64(want)))
			if diff > limit {
				t.Fatalf("n=%d row=%d: got %.8f want %.8f diff %.8f", n, row, got[row], want, diff)
			}
		}
	}
}

func TestDotFloat32Rows8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(789))
	lengths := []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 384}
	for _, n := range lengths {
		rows := make([]float32, 8*n)
		vec := make([]float32, n)
		for i := range rows {
			rows[i] = float32(rng.NormFloat64())
		}
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		var got [8]float32
		dotFloat32Rows8(&got, rows, vec)
		for row := 0; row < 8; row++ {
			want := dotFloat32sScalar(rows[row*n:(row+1)*n], vec)
			diff := math.Abs(float64(got[row] - want))
			limit := 1e-4 * math.Max(1, math.Abs(float64(want)))
			if diff > limit {
				t.Fatalf("n=%d row=%d: got %.8f want %.8f diff %.8f", n, row, got[row], want, diff)
			}
		}
	}
}

func TestDotFloat32Rows8BlockedMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(987))
	for _, n := range []int{8, 16, 32, 64, 128, 384} {
		rows := make([]float32, 8*n)
		vec := make([]float32, n)
		for i := range rows {
			rows[i] = float32(rng.NormFloat64())
		}
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		blocked := make([]float32, len(rows))
		blockProjectionRowGroup8(blocked, rows, n)
		var got [8]float32
		dotFloat32Rows8Blocked(&got, blocked, vec)
		for row := 0; row < 8; row++ {
			want := dotFloat32sScalar(rows[row*n:(row+1)*n], vec)
			diff := math.Abs(float64(got[row] - want))
			limit := 1e-4 * math.Max(1, math.Abs(float64(want)))
			if diff > limit {
				t.Fatalf("n=%d row=%d: got %.8f want %.8f diff %.8f", n, row, got[row], want, diff)
			}
		}
	}
}
