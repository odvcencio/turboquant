package turboquant

import (
	"os"
	"strings"
	"testing"
)

// TestReadmeLicenceMatchesFile asserts that README.md names the same licence
// family that LICENSE declares. This guards against the MIT/Apache-2.0
// conflict that the repository shipped with.
func TestReadmeLicenceMatchesFile(t *testing.T) {
	licenseBytes, err := os.ReadFile("LICENSE")
	if err != nil {
		t.Fatalf("reading LICENSE: %v", err)
	}
	readmeBytes, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatalf("reading README.md: %v", err)
	}
	license := string(licenseBytes)
	readme := string(readmeBytes)

	var family string
	switch {
	case strings.Contains(license, "Apache License") && strings.Contains(license, "Version 2.0"):
		family = "Apache-2.0"
	case strings.Contains(license, "MIT License"):
		family = "MIT"
	default:
		t.Fatalf("unrecognized LICENSE family, add a case to this test")
	}

	idx := strings.Index(readme, "## License")
	if idx < 0 {
		t.Fatalf("README.md has no ## License section")
	}
	section := readme[idx:]
	if !strings.Contains(section, family) {
		t.Fatalf("README.md License section does not name %q", family)
	}
}
