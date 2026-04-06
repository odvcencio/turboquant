package turboquant

import "math/rand"

// raceEnabled is set to true in race_test.go when the race detector is active.
var raceEnabled bool

func newTestRNG() *rand.Rand {
	return rand.New(rand.NewSource(42))
}
