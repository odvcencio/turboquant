package turboquant

import (
	"strings"
	"testing"
)

func TestWalkTransformerLayerCapturesJSONSingleCapture(t *testing.T) {
	input := `{
 "name": "single",
  "token_position": "middle",
  "sequence_length": 9,
  "heads": 1,
  "head_dim": 2,
  "query": [1, 0],
  "keys": [1, 0],
  "values": [2, 0]
}`
	var samples []TransformerLayerCapture
	err := WalkTransformerLayerCapturesJSON(strings.NewReader(input), func(sample TransformerLayerCapture) error {
		samples = append(samples, sample)
		return nil
	})
	if err != nil {
		t.Fatalf("WalkTransformerLayerCapturesJSON: %v", err)
	}
	if len(samples) != 1 {
		t.Fatalf("len(samples) = %d want 1", len(samples))
	}
	if samples[0].Name != "single" {
		t.Fatalf("samples[0].Name = %q want %q", samples[0].Name, "single")
	}
	if samples[0].TokenPosition != "middle" || samples[0].SequenceLength != 9 {
		t.Fatalf("samples[0] token metadata = (%q,%d) want (%q,%d)", samples[0].TokenPosition, samples[0].SequenceLength, "middle", 9)
	}
}

func TestWalkTransformerLayerCapturesJSONWrappedSamples(t *testing.T) {
	input := `{
  "model": "tiny",
  "prompts": ["one", "two"],
  "layers": [0, 1],
  "token_index_spec": "last",
  "samples": [
    {
      "name": "first",
      "heads": 1,
      "head_dim": 2,
      "query": [1, 0],
      "keys": [1, 0],
      "values": [2, 0]
    },
    {
      "name": "second",
      "heads": 1,
      "head_dim": 2,
      "query": [0, 1],
      "keys": [0, 1],
      "values": [0, 2]
    }
  ]
}`
	var names []string
	err := WalkTransformerLayerCapturesJSON(strings.NewReader(input), func(sample TransformerLayerCapture) error {
		names = append(names, sample.Name)
		return nil
	})
	if err != nil {
		t.Fatalf("WalkTransformerLayerCapturesJSON: %v", err)
	}
	if len(names) != 2 {
		t.Fatalf("len(names) = %d want 2", len(names))
	}
	if names[0] != "first" || names[1] != "second" {
		t.Fatalf("names = %v want [first second]", names)
	}
}
