package turboquant

import (
	"math/rand"
	"testing"
)

// raceEnabled is set to true in race_test.go when the race detector is active.
var raceEnabled bool

func newTestRNG() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

func skipAllocsUnderRace(t *testing.T) {
	t.Helper()
	if raceEnabled {
		t.Skip("skipping allocation assertion under race detector")
	}
}

func expectPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	fn()
}
