package turboquant

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"sync"
)

//go:embed codebook_table.bin
var codebookTableBytes []byte

const (
	codebookTableMagic      = "TQCB"
	codebookTableVersion    = 1
	codebookTableHeaderSize = 12 // magic(4) + version(1) + pad(3) + count(4)
	codebookTableEntrySize  = 12 // dim(4) + bitWidth(4) + offset(4)
)

var (
	codebookTable     map[codebookKey]codebook
	codebookTableErr  error
	codebookTableOnce sync.Once
)

// loadCodebookTable parses the embedded codebook table once.
func loadCodebookTable() {
	codebookTableOnce.Do(func() {
		codebookTable, codebookTableErr = parseCodebookTable(codebookTableBytes)
	})
}

func parseCodebookTable(data []byte) (map[codebookKey]codebook, error) {
	if len(data) < codebookTableHeaderSize {
		return nil, fmt.Errorf("turboquant: codebook table too short (%d bytes)", len(data))
	}
	if string(data[0:4]) != codebookTableMagic {
		return nil, fmt.Errorf("turboquant: invalid codebook table magic %q", data[0:4])
	}
	if data[4] != codebookTableVersion {
		return nil, fmt.Errorf("turboquant: unsupported codebook table version %d", data[4])
	}
	count := int(binary.LittleEndian.Uint32(data[8:12]))

	entriesEnd := codebookTableHeaderSize + count*codebookTableEntrySize
	if count < 0 || len(data) < entriesEnd {
		return nil, fmt.Errorf("turboquant: codebook table truncated entry list")
	}

	type rawEntry struct {
		dim, bitWidth, offset int
	}
	raws := make([]rawEntry, count)
	for i := 0; i < count; i++ {
		base := codebookTableHeaderSize + i*codebookTableEntrySize
		raws[i] = rawEntry{
			dim:      int(binary.LittleEndian.Uint32(data[base : base+4])),
			bitWidth: int(binary.LittleEndian.Uint32(data[base+4 : base+8])),
			offset:   int(binary.LittleEndian.Uint32(data[base+8 : base+12])),
		}
	}

	payload := data[entriesEnd:]
	if len(payload)%4 != 0 {
		return nil, fmt.Errorf("turboquant: codebook table payload length %d not a multiple of 4", len(payload))
	}
	values := make([]float32, len(payload)/4)
	for i := range values {
		off := i * 4
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[off : off+4]))
	}

	table := make(map[codebookKey]codebook, count)
	for _, r := range raws {
		if r.bitWidth < 1 || r.bitWidth > 8 {
			return nil, fmt.Errorf("turboquant: codebook table entry dim=%d has invalid bitWidth %d", r.dim, r.bitWidth)
		}
		levels := 1 << uint(r.bitWidth)
		total := levels + (levels - 1)
		if r.offset < 0 || r.offset+total > len(values) {
			return nil, fmt.Errorf("turboquant: codebook table entry dim=%d bitWidth=%d out of range", r.dim, r.bitWidth)
		}
		centroids := append([]float32(nil), values[r.offset:r.offset+levels]...)
		boundaries := append([]float32(nil), values[r.offset+levels:r.offset+total]...)
		table[codebookKey{dim: r.dim, bitWidth: r.bitWidth}] = codebook{
			centroids:  centroids,
			boundaries: boundaries,
		}
	}
	return table, nil
}

// codebookFromTable returns the embedded codebook for (dim, bitWidth) if the
// build-time table carries it.
func codebookFromTable(dim, bitWidth int) (codebook, bool) {
	loadCodebookTable()
	if codebookTableErr != nil {
		return codebook{}, false
	}
	cb, ok := codebookTable[codebookKey{dim: dim, bitWidth: bitWidth}]
	return cb, ok
}

// PrecomputeCodebook builds and caches the codebook for a dimension and bit
// width. Call it during startup to move construction cost off the hot path.
// It is a no-op (aside from the cache warm-up) when the embedded table
// already carries the (dim, bitWidth) pair.
func PrecomputeCodebook(dim, bitWidth int) error {
	if err := validateDim(dim); err != nil {
		return err
	}
	if err := validateBitWidth(bitWidth); err != nil {
		return err
	}
	cachedCodebook(dim, bitWidth)
	return nil
}

// CodebookSource reports where the codebook for a dimension and bit width
// comes from. It returns "table" when the embedded build-time table carries
// the pair, and "computed" when it falls back to the Lloyd-Max solver.
func CodebookSource(dim, bitWidth int) string {
	if _, ok := codebookFromTable(dim, bitWidth); ok {
		return "table"
	}
	return "computed"
}

// TabulatedDims returns the dimensions carried by the embedded codebook
// table, in ascending order.
func TabulatedDims() []int {
	loadCodebookTable()
	if codebookTableErr != nil {
		return nil
	}
	seen := make(map[int]bool)
	dims := make([]int, 0, len(codebookTable))
	for key := range codebookTable {
		if !seen[key.dim] {
			seen[key.dim] = true
			dims = append(dims, key.dim)
		}
	}
	sort.Ints(dims)
	return dims
}
