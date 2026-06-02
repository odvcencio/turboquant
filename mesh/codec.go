package mesh

// Default bit widths applied when an Options field is left zero.
const (
	defaultPositionBits = 16
	defaultNormalBits   = 16
	defaultJointBits    = 8
	weightBits          = 8 // weights are always u8 normalized
)

// DequantSpec is the portable, self-describing recipe for dequantizing a mesh
// on the GPU. It is a comparable value struct (no slices) so it can be compared
// with == and round-tripped losslessly through JSON for transport to the
// renderer.
type DequantSpec struct {
	VertexCount int            `json:"vertexCount"`
	Position    PositionParams `json:"position"`
	NormalBits  int            `json:"normalBits"`
	JointBits   int            `json:"jointBits"`
	WeightBits  int            `json:"weightBits"`
}

// Quantized is the encoded mesh: the dequant Spec plus the packed byte streams.
// Optional streams (normals, joints, weights) are nil/omitted when absent.
type Quantized struct {
	Spec      DequantSpec `json:"spec"`
	Positions []byte      `json:"positions"`
	Normals   []byte      `json:"normals,omitempty"`
	Joints    []byte      `json:"joints,omitempty"`
	Weights   []byte      `json:"weights,omitempty"`
}

// Options selects the bit widths for the quantized streams. A zero field uses
// the default (positions 16, normals 16, joints 8). Weights are always u8.
type Options struct {
	PositionBits int
	NormalBits   int
	JointBits    int
}

// Encode quantizes a vertex stream into a Quantized mesh with a self-describing
// DequantSpec. Vertex count is derived from the position stream. The normals,
// joints, and weights streams are optional: when a stream is nil or empty it is
// omitted and its corresponding Spec bit width reflects absence (NormalBits = 0
// with nil Normals).
func Encode(pos, nrm []float32, joints []uint32, weights []float32, opt Options) Quantized {
	if opt.PositionBits == 0 {
		opt.PositionBits = defaultPositionBits
	}
	if opt.NormalBits == 0 {
		opt.NormalBits = defaultNormalBits
	}
	if opt.JointBits == 0 {
		opt.JointBits = defaultJointBits
	}

	n := len(pos) / 3

	packedPos, pp := EncodePositions(pos, opt.PositionBits)
	q := Quantized{
		Spec: DequantSpec{
			VertexCount: n,
			Position:    pp,
		},
		Positions: packedPos,
	}

	if len(nrm) > 0 {
		q.Normals = EncodeNormals(nrm, opt.NormalBits)
		q.Spec.NormalBits = opt.NormalBits
	}
	if len(joints) > 0 {
		q.Joints = EncodeJoints(joints, opt.JointBits)
		q.Spec.JointBits = opt.JointBits
	}
	if len(weights) > 0 {
		q.Weights = EncodeWeights(weights)
		q.Spec.WeightBits = weightBits
	}

	return q
}

// Decode reverses Encode, reconstructing the float/uint streams per the Spec.
// Absent streams come back nil (e.g. normals are nil when Spec.NormalBits == 0).
func Decode(q Quantized) (pos, nrm []float32, joints []uint32, weights []float32) {
	n := q.Spec.VertexCount

	pos = DecodePositions(q.Positions, q.Spec.Position, n)

	if q.Spec.NormalBits != 0 {
		nrm = DecodeNormals(q.Normals, q.Spec.NormalBits, n)
	}
	if q.Spec.JointBits != 0 && len(q.Joints) > 0 {
		joints = DecodeJoints(q.Joints, q.Spec.JointBits, n)
	}
	if q.Spec.WeightBits != 0 && len(q.Weights) > 0 {
		weights = DecodeWeights(q.Weights, n)
	}

	return pos, nrm, joints, weights
}
