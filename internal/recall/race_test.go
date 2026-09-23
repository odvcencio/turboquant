//go:build race

package recall

func init() {
	raceEnabled = true
}
