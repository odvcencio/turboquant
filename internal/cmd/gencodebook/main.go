// Command gencodebook writes codebook_table.bin: precomputed Lloyd-Max
// centroids and boundaries for a curated set of dimensions and every bit
// width from 1 to 8. turboquant embeds the output and loads it instead of
// running the solver for any tabulated (dim, bitWidth) pair.
//
// Layout of codebook_table.bin:
//
//	magic   "TQCB"          4 bytes
//	version 1               1 byte
//	pad                     3 bytes
//	count   uint32          number of (dim, bitWidth) entries
//	entries[count]:
//	    dim       uint32
//	    bitWidth  uint32
//	    offset    uint32    float32 offset into the payload
//	payload: little-endian float32 values
//	         centroids (2^b values) then boundaries (2^b - 1 values)
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
)

// curatedDims mirrors the hardening spec's curated dimension list.
var curatedDims = []int{64, 128, 200, 256, 384, 512, 768, 1024, 1152, 1536, 2048, 3072, 4096}

const maxBitWidth = 8

func main() {
	out := flag.String("out", "codebook_table.bin", "output path for the codebook table")
	flag.Parse()

	if err := run(*out); err != nil {
		fmt.Fprintln(os.Stderr, "gencodebook:", err)
		os.Exit(1)
	}
}

type tableEntry struct {
	dim, bitWidth, offset int
}

func run(out string) error {
	var entries []tableEntry
	var payload []float32

	for _, dim := range curatedDims {
		grid := buildCodebookGrid(dim)
		for bw := 1; bw <= maxBitWidth; bw++ {
			centroids, boundaries := computeCodebook(dim, grid, bw)
			offset := len(payload)
			payload = append(payload, centroids...)
			payload = append(payload, boundaries...)
			entries = append(entries, tableEntry{dim: dim, bitWidth: bw, offset: offset})
		}
	}

	sort.Slice(entries, func(i, j int) bool {
		if entries[i].dim != entries[j].dim {
			return entries[i].dim < entries[j].dim
		}
		return entries[i].bitWidth < entries[j].bitWidth
	})

	buf := encodeTable(entries, payload)
	return os.WriteFile(out, buf, 0o644)
}

func encodeTable(entries []tableEntry, payload []float32) []byte {
	buf := make([]byte, 0, 12+len(entries)*12+len(payload)*4)
	buf = append(buf, 'T', 'Q', 'C', 'B')
	buf = append(buf, 1, 0, 0, 0) // version, then 3 pad bytes

	var countBuf [4]byte
	binary.LittleEndian.PutUint32(countBuf[:], uint32(len(entries)))
	buf = append(buf, countBuf[:]...)

	for _, e := range entries {
		var eb [12]byte
		binary.LittleEndian.PutUint32(eb[0:4], uint32(e.dim))
		binary.LittleEndian.PutUint32(eb[4:8], uint32(e.bitWidth))
		binary.LittleEndian.PutUint32(eb[8:12], uint32(e.offset))
		buf = append(buf, eb[:]...)
	}

	for _, v := range payload {
		var fb [4]byte
		binary.LittleEndian.PutUint32(fb[:], math.Float32bits(v))
		buf = append(buf, fb[:]...)
	}

	return buf
}
