package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	turboquant "github.com/odvcencio/turboquant"
)

func TestRunCLIGroupedKVHeadsFallbackFromCapture(t *testing.T) {
	dir := t.TempDir()
	capturePath := filepath.Join(dir, "capture.json")
	profilePath := filepath.Join(dir, "profile.json")
	outputPath := filepath.Join(dir, "bench.json")

	capture := `{
  "samples": [
    {
      "model": "demo",
      "prompt_index": 0,
      "layer": 0,
      "token_index": 1,
      "token_position": "last",
      "sequence_length": 2,
      "heads": 4,
      "kv_heads": 2,
      "head_dim": 2,
      "tokens": 2,
      "query": [1,0, 0,1, 1,0, 0,1],
      "keys": [1,0, 0,1, 0,1, 1,0],
      "values": [10,0, 0,20, 0,30, 40,0]
    },
    {
      "model": "demo",
      "prompt_index": 0,
      "layer": 1,
      "token_index": 1,
      "token_position": "last",
      "sequence_length": 2,
      "heads": 4,
      "kv_heads": 2,
      "head_dim": 2,
      "tokens": 2,
      "query": [0,1, 1,0, 0,1, 1,0],
      "keys": [0,1, 1,0, 1,0, 0,1],
      "values": [0,11, 12,0, 13,0, 0,14]
    }
  ]
}`
	profile := `{
  "profile_set": "last",
  "token_positions": ["last"],
  "layers": [
    {
      "layer": 0,
      "heads": 4,
      "head_dim": 2,
      "key_bits": 2,
      "value_bits": 2,
      "top_k": 2,
      "capacity": 2
    },
    {
      "layer": 1,
      "heads": 4,
      "head_dim": 2,
      "key_bits": 2,
      "value_bits": 2,
      "top_k": 2,
      "capacity": 2
    }
  ],
  "profiles": [
    {
      "layer": 0,
      "heads": 4,
      "head_dim": 2,
      "key_bits": 2,
      "value_bits": 2,
      "capacity": 2,
      "seed": 1
    },
    {
      "layer": 1,
      "heads": 4,
      "head_dim": 2,
      "key_bits": 2,
      "value_bits": 2,
      "capacity": 2,
      "seed": 2
    }
  ]
}`
	if err := os.WriteFile(capturePath, []byte(capture), 0o644); err != nil {
		t.Fatalf("WriteFile(capture): %v", err)
	}
	if err := os.WriteFile(profilePath, []byte(profile), 0o644); err != nil {
		t.Fatalf("WriteFile(profile): %v", err)
	}
	if err := runCLI([]string{
		"--capture", capturePath,
		"--profile", profilePath,
		"--out", outputPath,
		"--warmup", "0",
		"--iterations", "1",
	}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got report
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if len(got.ProfileSets) != 1 {
		t.Fatalf("len(ProfileSets) = %d want 1", len(got.ProfileSets))
	}
	item := got.ProfileSets[0]
	if item.ProfileSet != "last" {
		t.Fatalf("ProfileSet = %q want %q", item.ProfileSet, "last")
	}
	if item.BenchmarkedGroups != 1 {
		t.Fatalf("BenchmarkedGroups = %d want 1", item.BenchmarkedGroups)
	}
	if item.Summary.Runs != 1 {
		t.Fatalf("Summary.Runs = %d want 1", item.Summary.Runs)
	}
	if len(item.LayerSummaries) != 2 {
		t.Fatalf("len(LayerSummaries) = %d want 2", len(item.LayerSummaries))
	}
	if item.LayerSummaries[0].KVHeads != 2 || item.LayerSummaries[1].KVHeads != 2 {
		t.Fatalf("LayerSummaries kv_heads = (%d,%d) want (2,2)", item.LayerSummaries[0].KVHeads, item.LayerSummaries[1].KVHeads)
	}
	if item.Summary.CompressionVsFP32 <= 0 {
		t.Fatalf("CompressionVsFP32 = %v want > 0", item.Summary.CompressionVsFP32)
	}
}

func TestRunCLIProfileSetFilter(t *testing.T) {
	dir := t.TempDir()
	capturePath := filepath.Join(dir, "capture.json")
	profilePath := filepath.Join(dir, "profile.json")
	outputPath := filepath.Join(dir, "bench.json")

	capture := `{
  "samples": [
    {
      "model": "demo",
      "prompt_index": 0,
      "layer": 0,
      "token_index": 1,
      "token_position": "last",
      "sequence_length": 2,
      "heads": 1,
      "head_dim": 2,
      "tokens": 2,
      "query": [1,0],
      "keys": [1,0, 0,1],
      "values": [2,0, 0,2]
    }
  ]
}`
	profile := `{
  "reports": [
    {
      "profile_set": "all",
      "layers": [
        {"layer": 0, "heads": 1, "head_dim": 2, "key_bits": 2, "value_bits": 2, "top_k": 2, "capacity": 2}
      ],
      "profiles": [
        {"layer": 0, "heads": 1, "head_dim": 2, "key_bits": 2, "value_bits": 2, "capacity": 2, "seed": 1}
      ]
    },
    {
      "profile_set": "last",
      "token_positions": ["last"],
      "layers": [
        {"layer": 0, "heads": 1, "head_dim": 2, "key_bits": 3, "value_bits": 2, "top_k": 2, "capacity": 2}
      ],
      "profiles": [
        {"layer": 0, "heads": 1, "head_dim": 2, "key_bits": 3, "value_bits": 2, "capacity": 2, "seed": 2}
      ]
    }
  ]
}`
	if err := os.WriteFile(capturePath, []byte(capture), 0o644); err != nil {
		t.Fatalf("WriteFile(capture): %v", err)
	}
	if err := os.WriteFile(profilePath, []byte(profile), 0o644); err != nil {
		t.Fatalf("WriteFile(profile): %v", err)
	}
	if err := runCLI([]string{
		"--capture", capturePath,
		"--profile", profilePath,
		"--profile-set", "last",
		"--out", outputPath,
		"--warmup", "0",
		"--iterations", "1",
	}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got report
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if len(got.ProfileSets) != 1 {
		t.Fatalf("len(ProfileSets) = %d want 1", len(got.ProfileSets))
	}
	if got.ProfileSets[0].ProfileSet != "last" {
		t.Fatalf("ProfileSet = %q want %q", got.ProfileSets[0].ProfileSet, "last")
	}
}

func TestRunCLIExpandProfile(t *testing.T) {
	dir := t.TempDir()
	capturePath := filepath.Join(dir, "capture.json")
	profilePath := filepath.Join(dir, "profile.json")
	outputPath := filepath.Join(dir, "bench.json")

	// Capture has 4 layers (0,1,2,3) but profile only covers layers 0 and 3.
	capture := `{
  "samples": [
    {"model":"demo","prompt_index":0,"layer":0,"token_index":1,"token_position":"last","sequence_length":2,"heads":2,"head_dim":2,"tokens":2,"query":[1,0,0,1],"keys":[1,0,0,1,0,1,1,0],"values":[10,0,0,20,0,30,40,0]},
    {"model":"demo","prompt_index":0,"layer":1,"token_index":1,"token_position":"last","sequence_length":2,"heads":2,"head_dim":2,"tokens":2,"query":[0,1,1,0],"keys":[0,1,1,0,1,0,0,1],"values":[0,11,12,0,13,0,0,14]},
    {"model":"demo","prompt_index":0,"layer":2,"token_index":1,"token_position":"last","sequence_length":2,"heads":2,"head_dim":2,"tokens":2,"query":[1,1,0,1],"keys":[1,1,0,1,0,1,1,1],"values":[5,5,5,5,5,5,5,5]},
    {"model":"demo","prompt_index":0,"layer":3,"token_index":1,"token_position":"last","sequence_length":2,"heads":2,"head_dim":2,"tokens":2,"query":[0,1,1,1],"keys":[0,1,1,1,1,1,0,1],"values":[1,2,3,4,5,6,7,8]}
  ]
}`
	profile := `{
  "profile_set": "last",
  "token_positions": ["last"],
  "layers": [
    {"layer": 0, "heads": 2, "head_dim": 2, "key_bits": 2, "value_bits": 2, "top_k": 2, "capacity": 2},
    {"layer": 3, "heads": 2, "head_dim": 2, "key_bits": 3, "value_bits": 2, "top_k": 2, "capacity": 2}
  ],
  "profiles": [
    {"layer": 0, "heads": 2, "head_dim": 2, "key_bits": 2, "value_bits": 2, "capacity": 2, "seed": 1},
    {"layer": 3, "heads": 2, "head_dim": 2, "key_bits": 3, "value_bits": 2, "capacity": 2, "seed": 3}
  ]
}`
	if err := os.WriteFile(capturePath, []byte(capture), 0o644); err != nil {
		t.Fatalf("WriteFile(capture): %v", err)
	}
	if err := os.WriteFile(profilePath, []byte(profile), 0o644); err != nil {
		t.Fatalf("WriteFile(profile): %v", err)
	}

	// Without --expand-profile this would fail (no complete groups).
	if err := runCLI([]string{
		"--capture", capturePath,
		"--profile", profilePath,
		"--expand-profile",
		"--out", outputPath,
		"--warmup", "0",
		"--iterations", "1",
	}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got report
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if len(got.ProfileSets) != 1 {
		t.Fatalf("len(ProfileSets) = %d want 1", len(got.ProfileSets))
	}
	item := got.ProfileSets[0]
	if len(item.LayerSummaries) != 4 {
		t.Fatalf("len(LayerSummaries) = %d want 4", len(item.LayerSummaries))
	}
	// Layers 0,1 should use key_bits=2 (donor 0). Layers 2,3 should use key_bits=3 (donor 3).
	for _, ls := range item.LayerSummaries {
		switch ls.Layer {
		case 0, 1:
			if ls.KeyBits != 2 {
				t.Fatalf("layer %d key_bits = %d want 2", ls.Layer, ls.KeyBits)
			}
		case 2, 3:
			if ls.KeyBits != 3 {
				t.Fatalf("layer %d key_bits = %d want 3", ls.Layer, ls.KeyBits)
			}
		}
	}
	if item.Summary.CompressionVsFP32 <= 0 {
		t.Fatalf("CompressionVsFP32 = %v want > 0", item.Summary.CompressionVsFP32)
	}
}

func TestExpandProfileToCapture(t *testing.T) {
	// Profile covers layers 0 and 6 (simulating spread:2 of a 10-layer model).
	// Capture covers layers 0-9. Expansion should fill layers 1-5 and 7-9.
	profile := profileReportInput{
		ProfileSet:     "last",
		TokenPositions: []string{"last"},
		Layers: []profileLayer{
			{Layer: 0, Heads: 4, KVHeads: 2, HeadDim: 2, KeyBits: 2, ValueBits: 2, TopK: 2, Capacity: 8},
			{Layer: 6, Heads: 4, KVHeads: 2, HeadDim: 2, KeyBits: 3, ValueBits: 3, TopK: 4, Capacity: 8},
		},
		Profiles: []turboquant.TransformerLayerKVProfile{
			{Layer: 0, Heads: 4, KVHeads: 2, HeadDim: 2, KeyBits: 2, ValueBits: 2, Capacity: 8, Seed: 10},
			{Layer: 6, Heads: 4, KVHeads: 2, HeadDim: 2, KeyBits: 3, ValueBits: 3, Capacity: 8, Seed: 60},
		},
	}
	captureLayers := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

	got := expandProfileToCapture(profile, captureLayers)

	if len(got.Layers) != 10 {
		t.Fatalf("len(Layers) = %d want 10", len(got.Layers))
	}
	if len(got.Profiles) != 10 {
		t.Fatalf("len(Profiles) = %d want 10", len(got.Profiles))
	}

	// Layer 0: original donor
	if got.Layers[0].KeyBits != 2 || got.Profiles[0].KeyBits != 2 {
		t.Fatalf("layer 0 should keep key_bits=2")
	}
	// Layer 3: equidistant from 0 (dist 3) and 6 (dist 3) — prefer lower donor (0)
	if got.Layers[3].KeyBits != 2 {
		t.Fatalf("layer 3 key_bits = %d want 2 (nearest-neighbor tie -> lower donor)", got.Layers[3].KeyBits)
	}
	if got.Profiles[3].Layer != 3 {
		t.Fatalf("expanded profile[3].Layer = %d want 3", got.Profiles[3].Layer)
	}
	if got.Layers[3].Layer != 3 {
		t.Fatalf("expanded layer[3].Layer = %d want 3", got.Layers[3].Layer)
	}
	// Layer 4: closer to 6 (dist 2) than 0 (dist 4) — donor is 6
	if got.Layers[4].KeyBits != 3 {
		t.Fatalf("layer 4 key_bits = %d want 3 (donor layer 6)", got.Layers[4].KeyBits)
	}
	if got.Profiles[4].Seed != 60 {
		t.Fatalf("layer 4 seed = %d want 60 (cloned from donor layer 6)", got.Profiles[4].Seed)
	}
	// Layer 6: original donor
	if got.Layers[6].KeyBits != 3 || got.Profiles[6].Seed != 60 {
		t.Fatalf("layer 6 should keep original values")
	}
	// Layer 9: closest to 6 (dist 3) — donor is 6
	if got.Layers[9].KeyBits != 3 {
		t.Fatalf("layer 9 key_bits = %d want 3 (donor layer 6)", got.Layers[9].KeyBits)
	}
	if got.Profiles[9].Layer != 9 {
		t.Fatalf("expanded profile[9].Layer = %d want 9", got.Profiles[9].Layer)
	}
	// TopK should also be cloned
	if got.Layers[1].TopK != 2 {
		t.Fatalf("layer 1 top_k = %d want 2 (donor layer 0)", got.Layers[1].TopK)
	}
	if got.Layers[7].TopK != 4 {
		t.Fatalf("layer 7 top_k = %d want 4 (donor layer 6)", got.Layers[7].TopK)
	}
	// ProfileSet and TokenPositions should be preserved
	if got.ProfileSet != "last" {
		t.Fatalf("ProfileSet = %q want %q", got.ProfileSet, "last")
	}
}
