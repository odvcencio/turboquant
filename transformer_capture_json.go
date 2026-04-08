package turboquant

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// WalkTransformerLayerCapturesJSON streams transformer captures from a JSON
// reader, supporting either a single capture object or a wrapper object with a
// "samples" array.
func WalkTransformerLayerCapturesJSON(r io.Reader, fn func(TransformerLayerCapture) error) error {
	if fn == nil {
		return fmt.Errorf("turboquant: nil transformer capture visitor")
	}

	dec := json.NewDecoder(r)
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	delim, ok := tok.(json.Delim)
	if !ok || delim != '{' {
		return fmt.Errorf("turboquant: expected top-level JSON object")
	}

	var (
		capture         TransformerLayerCapture
		sawCaptureField bool
		sawSamples      bool
	)
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return err
		}
		key, ok := keyTok.(string)
		if !ok {
			return fmt.Errorf("turboquant: expected object key, got %T", keyTok)
		}
		switch key {
		case "samples":
			sawSamples = true
			tok, err := dec.Token()
			if err != nil {
				return err
			}
			delim, ok := tok.(json.Delim)
			if !ok || delim != '[' {
				return fmt.Errorf("turboquant: expected samples array")
			}
			for dec.More() {
				var sample TransformerLayerCapture
				if err := dec.Decode(&sample); err != nil {
					return err
				}
				if err := fn(sample); err != nil {
					return err
				}
			}
			tok, err = dec.Token()
			if err != nil {
				return err
			}
			delim, ok = tok.(json.Delim)
			if !ok || delim != ']' {
				return fmt.Errorf("turboquant: expected samples array terminator")
			}
		case "name":
			sawCaptureField = true
			if err := dec.Decode(&capture.Name); err != nil {
				return err
			}
		case "model":
			sawCaptureField = true
			if err := dec.Decode(&capture.Model); err != nil {
				return err
			}
		case "prompt":
			sawCaptureField = true
			if err := dec.Decode(&capture.Prompt); err != nil {
				return err
			}
		case "prompt_index":
			sawCaptureField = true
			if err := dec.Decode(&capture.PromptIndex); err != nil {
				return err
			}
		case "layer":
			sawCaptureField = true
			if err := dec.Decode(&capture.Layer); err != nil {
				return err
			}
		case "token_index":
			sawCaptureField = true
			if err := dec.Decode(&capture.TokenIndex); err != nil {
				return err
			}
		case "token_position":
			sawCaptureField = true
			if err := dec.Decode(&capture.TokenPosition); err != nil {
				return err
			}
		case "sequence_length":
			sawCaptureField = true
			if err := dec.Decode(&capture.SequenceLength); err != nil {
				return err
			}
		case "heads":
			sawCaptureField = true
			if err := dec.Decode(&capture.Heads); err != nil {
				return err
			}
		case "kv_heads":
			sawCaptureField = true
			if err := dec.Decode(&capture.KVHeads); err != nil {
				return err
			}
		case "head_dim":
			sawCaptureField = true
			if err := dec.Decode(&capture.HeadDim); err != nil {
				return err
			}
		case "tokens":
			sawCaptureField = true
			if err := dec.Decode(&capture.Tokens); err != nil {
				return err
			}
		case "query_scale":
			sawCaptureField = true
			if err := dec.Decode(&capture.QueryScale); err != nil {
				return err
			}
		case "query":
			sawCaptureField = true
			if err := dec.Decode(&capture.Query); err != nil {
				return err
			}
		case "keys":
			sawCaptureField = true
			if err := dec.Decode(&capture.Keys); err != nil {
				return err
			}
		case "values":
			sawCaptureField = true
			if err := dec.Decode(&capture.Values); err != nil {
				return err
			}
		default:
			if err := skipJSONValue(dec); err != nil {
				return err
			}
		}
	}
	tok, err = dec.Token()
	if err != nil {
		return err
	}
	delim, ok = tok.(json.Delim)
	if !ok || delim != '}' {
		return fmt.Errorf("turboquant: expected top-level object terminator")
	}

	if sawSamples {
		return nil
	}
	if !sawCaptureField {
		return fmt.Errorf("turboquant: no transformer captures found")
	}
	return fn(capture)
}

// WalkTransformerLayerCapturesFile streams transformer captures from a JSON
// file on disk.
func WalkTransformerLayerCapturesFile(path string, fn func(TransformerLayerCapture) error) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return WalkTransformerLayerCapturesJSON(f, fn)
}

func skipJSONValue(dec *json.Decoder) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		for dec.More() {
			if _, err := dec.Token(); err != nil {
				return err
			}
			if err := skipJSONValue(dec); err != nil {
				return err
			}
		}
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		end, ok := tok.(json.Delim)
		if !ok || end != '}' {
			return fmt.Errorf("turboquant: expected object terminator while skipping JSON")
		}
	case '[':
		for dec.More() {
			if err := skipJSONValue(dec); err != nil {
				return err
			}
		}
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		end, ok := tok.(json.Delim)
		if !ok || end != ']' {
			return fmt.Errorf("turboquant: expected array terminator while skipping JSON")
		}
	default:
		return fmt.Errorf("turboquant: unexpected JSON delimiter %q", delim)
	}
	return nil
}
