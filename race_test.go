//go:build race

package turboquant

func init() {
	raceEnabled = true
}
