//go:build linux && amd64 && cgo && cuda

package turboquant

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/odvcencio/turboquant/internal/cudaruntime"
)

type GPUPreparedScorer struct {
	mu sync.Mutex

	runtime *cudaruntime.Runtime

	dim         int
	bitWidth    int
	mseBitWidth int
	count       int
	mseBytes    int
	signBytes   int
	qjlScale    float32
	ranks       []uint32
	identityTie bool

	dMSE       cudaruntime.DevicePtr
	dSigns     cudaruntime.DevicePtr
	dResNorms  cudaruntime.DevicePtr
	dRanks     cudaruntime.DevicePtr
	dQueryMSE  cudaruntime.DevicePtr
	dQuerySign cudaruntime.DevicePtr
	dScores    cudaruntime.DevicePtr
	dTopKIdx   cudaruntime.DevicePtr
	dTopKScore cudaruntime.DevicePtr

	queryMSECap  int
	querySignCap int
	scoreCap     int
	topKCap      int

	queryMSEBuf  []float32
	querySignBuf []float32
	hostScores   []float32

	queryMSEPtr  unsafe.Pointer
	querySignPtr unsafe.Pointer
	hostScorePtr unsafe.Pointer

	closed bool
}

type GPUPreparedQueryBatch struct {
	scorer *GPUPreparedScorer
	count  int

	dMSE   cudaruntime.DevicePtr
	dSigns cudaruntime.DevicePtr

	closed bool
}

func newGPUPreparedScorer(q *IPQuantizer, data GPUPreparedData) (*GPUPreparedScorer, error) {
	count, err := ValidateGPUPreparedData(q.dim, q.bitWidth, data)
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, fmt.Errorf("turboquant: GPU prepared scorer requires at least one vector")
	}
	rt, err := cudaruntime.New()
	if err != nil {
		return nil, fmt.Errorf("turboquant: %w", err)
	}
	s := &GPUPreparedScorer{
		runtime:     rt,
		dim:         q.dim,
		bitWidth:    q.bitWidth,
		mseBitWidth: q.mse.bitWidth,
		count:       count,
		mseBytes:    PackedSize(q.dim, q.bitWidth-1),
		signBytes:   (q.dim + 7) / 8,
		qjlScale:    float32(math.Sqrt(math.Pi/2.0)) / float32(q.dim),
	}
	if hasIdentityTieBreakRanks(data.TieBreakRanks) {
		s.identityTie = true
	} else {
		s.ranks = append([]uint32(nil), data.TieBreakRanks...)
	}
	if err := s.uploadCorpus(data); err != nil {
		_ = s.Close()
		return nil, err
	}
	return s, nil
}

func (s *GPUPreparedScorer) uploadCorpus(data GPUPreparedData) error {
	if err := s.allocAndCopy(&s.dMSE, data.MSE); err != nil {
		return err
	}
	if err := s.allocAndCopy(&s.dSigns, data.Signs); err != nil {
		return err
	}
	resNormBytes := encodeFloat32s(nil, data.ResNorms)
	if err := s.allocAndCopy(&s.dResNorms, resNormBytes); err != nil {
		return err
	}
	if len(data.TieBreakRanks) != 0 && !s.identityTie {
		rankBytes := make([]byte, len(data.TieBreakRanks)*4)
		encodeUint32sBytes(rankBytes, data.TieBreakRanks)
		if err := s.allocAndCopy(&s.dRanks, rankBytes); err != nil {
			return err
		}
	}
	return nil
}

func (s *GPUPreparedScorer) Len() int {
	if s == nil {
		return 0
	}
	return s.count
}

func (s *GPUPreparedScorer) ScorePreparedQuery(pq PreparedQuery) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	dst := make([]float32, s.count)
	if err := s.ScorePreparedQueryTo(dst, pq); err != nil {
		return nil, err
	}
	return dst, nil
}

func (s *GPUPreparedScorer) ScorePreparedQueryTo(dst []float32, pq PreparedQuery) error {
	if err := s.validatePreparedQuery(pq); err != nil {
		return err
	}
	return s.ScorePreparedQueryToTrusted(dst, pq)
}

func (s *GPUPreparedScorer) ScorePreparedQueryToTrusted(dst []float32, pq PreparedQuery) error {
	if s == nil {
		return fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("turboquant: GPU prepared scorer is closed")
	}
	if len(dst) != s.count {
		return fmt.Errorf("turboquant: expected GPU score destination length %d, got %d", s.count, len(dst))
	}
	if err := s.launchPreparedQueryLocked(pq, s.count); err != nil {
		return err
	}
	if err := s.runtime.DtoHFloat32(dst, s.dScores); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	return nil
}

func (s *GPUPreparedScorer) ScorePreparedQueryTopK(pq PreparedQuery, k int) ([]uint32, []float32, error) {
	if err := s.validatePreparedQuery(pq); err != nil {
		return nil, nil, err
	}
	indices := make([]uint32, k)
	scores := make([]float32, k)
	if err := s.ScorePreparedQueryTopKToTrusted(indices, scores, pq); err != nil {
		return nil, nil, err
	}
	return indices, scores, nil
}

func (s *GPUPreparedScorer) ScorePreparedQueryTopKTo(indices []uint32, scores []float32, pq PreparedQuery) error {
	if err := s.validatePreparedQuery(pq); err != nil {
		return err
	}
	return s.ScorePreparedQueryTopKToTrusted(indices, scores, pq)
}

func (s *GPUPreparedScorer) ScorePreparedQueryTopKToTrusted(indices []uint32, scores []float32, pq PreparedQuery) error {
	if s == nil {
		return fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("turboquant: GPU prepared scorer is closed")
	}
	k := len(indices)
	if len(scores) != k {
		return fmt.Errorf("turboquant: GPU top-k destination length mismatch: %d indices vs %d scores", len(indices), len(scores))
	}
	if err := s.validateTopK(k); err != nil {
		return err
	}
	if k == 0 {
		return nil
	}
	host := s.ensureHostScoreCapacity(s.count)
	if err := s.launchPreparedQueryLocked(pq, s.count); err != nil {
		return err
	}
	if err := s.runtime.DtoHFloat32(host, s.dScores); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	selectTopK(indices, scores, host, s.tieBreakRanks(), k)
	return nil
}

func (s *GPUPreparedScorer) launchPreparedQueryLocked(pq PreparedQuery, scoreCount int) error {
	mseBytes := len(pq.mseLUT) * 4
	signBytes := len(pq.signLUT) * 4
	if err := s.ensureQueryCapacity(mseBytes, signBytes); err != nil {
		return err
	}
	copy(s.queryMSEBuf[:len(pq.mseLUT)], pq.mseLUT)
	copy(s.querySignBuf[:len(pq.signLUT)], pq.signLUT)
	if err := s.runtime.HtoDFloat32(s.dQueryMSE, s.queryMSEBuf[:len(pq.mseLUT)]); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	if err := s.runtime.HtoDFloat32(s.dQuerySign, s.querySignBuf[:len(pq.signLUT)]); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	if err := s.ensureScoreDeviceCapacity(scoreCount); err != nil {
		return err
	}
	if err := s.runtime.LaunchScore(s.dMSE, s.dSigns, s.dResNorms, s.dQueryMSE, s.dQuerySign, s.dScores, s.count, s.mseBytes, s.signBytes, 1, s.qjlScale); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	return nil
}

func (s *GPUPreparedScorer) ScorePreparedQueriesTopK(pqs []PreparedQuery, k int) ([]uint32, []float32, error) {
	indices := make([]uint32, len(pqs)*k)
	scores := make([]float32, len(indices))
	if err := s.ScorePreparedQueriesTopKTo(indices, scores, pqs, k); err != nil {
		return nil, nil, err
	}
	return indices, scores, nil
}

func (s *GPUPreparedScorer) ScorePreparedQueriesTopKTo(indices []uint32, scores []float32, pqs []PreparedQuery, k int) error {
	for i := range pqs {
		if err := s.validatePreparedQuery(pqs[i]); err != nil {
			return err
		}
	}
	return s.ScorePreparedQueriesTopKToTrusted(indices, scores, pqs, k)
}

func (s *GPUPreparedScorer) ScorePreparedQueriesTopKToTrusted(indices []uint32, scores []float32, pqs []PreparedQuery, k int) error {
	if s == nil {
		return fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	if len(indices) != len(pqs)*k {
		return fmt.Errorf("turboquant: expected GPU batch top-k index length %d, got %d", len(pqs)*k, len(indices))
	}
	if len(scores) != len(indices) {
		return fmt.Errorf("turboquant: GPU batch top-k destination length mismatch: %d indices vs %d scores", len(indices), len(scores))
	}
	if len(pqs) == 0 || k == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("turboquant: GPU prepared scorer is closed")
	}
	if err := s.validateTopK(k); err != nil {
		return err
	}
	scoreCount := len(pqs) * s.count
	if err := s.uploadPreparedQueriesLocked(pqs, scoreCount); err != nil {
		return err
	}
	host := s.ensureHostScoreCapacity(len(pqs) * s.count)
	if err := s.runtime.LaunchScoreToHost(host, s.dMSE, s.dSigns, s.dResNorms, s.dQueryMSE, s.dQuerySign, s.dScores, s.count, s.mseBytes, s.signBytes, len(pqs), s.qjlScale); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	for queryIdx := range pqs {
		base := queryIdx * s.count
		outBase := queryIdx * k
		selectTopK(indices[outBase:outBase+k], scores[outBase:outBase+k], host[base:base+s.count], s.tieBreakRanks(), k)
	}
	return nil
}

func (s *GPUPreparedScorer) UploadPreparedQueries(pqs []PreparedQuery) (*GPUPreparedQueryBatch, error) {
	for i := range pqs {
		if err := s.validatePreparedQuery(pqs[i]); err != nil {
			return nil, err
		}
	}
	return s.UploadPreparedQueriesTrusted(pqs)
}

func (s *GPUPreparedScorer) UploadPreparedQueriesTrusted(pqs []PreparedQuery) (*GPUPreparedQueryBatch, error) {
	if s == nil {
		return nil, fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	if len(pqs) == 0 {
		return nil, fmt.Errorf("turboquant: cannot upload an empty GPU prepared-query batch")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("turboquant: GPU prepared scorer is closed")
	}
	if !s.identityTie && len(s.ranks) != s.count {
		return nil, fmt.Errorf("turboquant: GPU top-k requires tie-break ranks in GPU prepared data")
	}
	mseBytes, signBytes := encodePreparedQueryBatch(pqs)
	batch := &GPUPreparedQueryBatch{
		scorer: s,
		count:  len(pqs),
	}
	if err := s.allocAndCopy(&batch.dMSE, mseBytes); err != nil {
		batch.Close()
		return nil, err
	}
	if err := s.allocAndCopy(&batch.dSigns, signBytes); err != nil {
		batch.Close()
		return nil, err
	}
	return batch, nil
}

func (s *GPUPreparedScorer) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	_ = s.freeDevice(s.dMSE)
	_ = s.freeDevice(s.dSigns)
	_ = s.freeDevice(s.dResNorms)
	_ = s.freeDevice(s.dRanks)
	_ = s.freeDevice(s.dQueryMSE)
	_ = s.freeDevice(s.dQuerySign)
	_ = s.freeDevice(s.dScores)
	_ = s.freeDevice(s.dTopKIdx)
	_ = s.freeDevice(s.dTopKScore)
	_ = s.freeHost(s.queryMSEPtr)
	_ = s.freeHost(s.querySignPtr)
	_ = s.freeHost(s.hostScorePtr)
	s.dMSE = 0
	s.dSigns = 0
	s.dResNorms = 0
	s.dRanks = 0
	s.dQueryMSE = 0
	s.dQuerySign = 0
	s.dScores = 0
	s.dTopKIdx = 0
	s.dTopKScore = 0
	s.queryMSEPtr = nil
	s.querySignPtr = nil
	s.hostScorePtr = nil
	if s.runtime != nil {
		s.runtime.Close()
		s.runtime = nil
	}
	return nil
}

func (s *GPUPreparedScorer) validatePreparedQuery(pq PreparedQuery) error {
	if s == nil {
		return fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	if s.runtime == nil {
		return ErrGPUBackendUnavailable
	}
	if err := ValidatePreparedQuery(s.dim, pq); err != nil {
		return err
	}
	if int(pq.mseBitWidth) != s.mseBitWidth {
		return fmt.Errorf("turboquant: expected prepared query MSE bit width %d, got %d", s.mseBitWidth, pq.mseBitWidth)
	}
	return nil
}

func (s *GPUPreparedScorer) validateTopK(k int) error {
	if !s.identityTie && len(s.ranks) != s.count {
		return fmt.Errorf("turboquant: GPU top-k requires tie-break ranks in GPU prepared data")
	}
	if k < 0 {
		return fmt.Errorf("turboquant: GPU top-k must be >= 0")
	}
	if k > GPUPreparedTopKMax {
		return fmt.Errorf("turboquant: GPU top-k %d exceeds max %d", k, GPUPreparedTopKMax)
	}
	if k > s.count {
		return fmt.Errorf("turboquant: GPU top-k %d exceeds corpus size %d", k, s.count)
	}
	return nil
}

func (s *GPUPreparedScorer) scorePreparedQueriesLocked(dst []float32, pqs []PreparedQuery) error {
	if len(pqs) == 0 {
		return nil
	}
	if len(dst) != len(pqs)*s.count {
		return fmt.Errorf("turboquant: expected GPU score destination length %d, got %d", len(pqs)*s.count, len(dst))
	}
	if err := s.launchPreparedQueriesLocked(pqs, len(dst)); err != nil {
		return err
	}
	if err := s.runtime.DtoHFloat32(dst, s.dScores); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	return nil
}

func (s *GPUPreparedScorer) uploadPreparedQueriesLocked(pqs []PreparedQuery, scoreCount int) error {
	if len(pqs) == 0 {
		return nil
	}
	mseBytesPerQuery := len(pqs[0].mseLUT) * 4
	signBytesPerQuery := len(pqs[0].signLUT) * 4
	mseBytes := len(pqs) * mseBytesPerQuery
	signBytes := len(pqs) * signBytesPerQuery
	if err := s.ensureQueryCapacity(mseBytes, signBytes); err != nil {
		return err
	}
	writePreparedQueryBatchFloats(s.queryMSEBuf[:mseBytes/4], s.querySignBuf[:signBytes/4], pqs)
	if err := s.runtime.HtoDFloat32(s.dQueryMSE, s.queryMSEBuf[:mseBytes/4]); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	if err := s.runtime.HtoDFloat32(s.dQuerySign, s.querySignBuf[:signBytes/4]); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	if err := s.ensureScoreDeviceCapacity(scoreCount); err != nil {
		return err
	}
	return nil
}

func (s *GPUPreparedScorer) launchPreparedQueriesLocked(pqs []PreparedQuery, scoreCount int) error {
	if err := s.uploadPreparedQueriesLocked(pqs, scoreCount); err != nil {
		return err
	}
	if err := s.runtime.LaunchScore(s.dMSE, s.dSigns, s.dResNorms, s.dQueryMSE, s.dQuerySign, s.dScores, s.count, s.mseBytes, s.signBytes, len(pqs), s.qjlScale); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	return nil
}

func (s *GPUPreparedScorer) ensureQueryCapacity(mseBytes, signBytes int) error {
	if mseBytes > s.queryMSECap {
		if err := s.freeDevice(s.dQueryMSE); err != nil {
			return err
		}
		if err := s.freeHost(s.queryMSEPtr); err != nil {
			return err
		}
		ptr, err := s.runtime.Alloc(mseBytes)
		if err != nil {
			return fmt.Errorf("turboquant: %w", err)
		}
		s.dQueryMSE = ptr
		hostPtr, err := s.runtime.AllocHost(mseBytes)
		if err != nil {
			return fmt.Errorf("turboquant: %w", err)
		}
		s.queryMSEPtr = hostPtr
		s.queryMSECap = mseBytes
		s.queryMSEBuf = unsafe.Slice((*float32)(hostPtr), mseBytes/4)
	} else {
		s.queryMSEBuf = s.queryMSEBuf[:mseBytes/4]
	}
	if signBytes > s.querySignCap {
		if err := s.freeDevice(s.dQuerySign); err != nil {
			return err
		}
		if err := s.freeHost(s.querySignPtr); err != nil {
			return err
		}
		ptr, err := s.runtime.Alloc(signBytes)
		if err != nil {
			return fmt.Errorf("turboquant: %w", err)
		}
		s.dQuerySign = ptr
		hostPtr, err := s.runtime.AllocHost(signBytes)
		if err != nil {
			return fmt.Errorf("turboquant: %w", err)
		}
		s.querySignPtr = hostPtr
		s.querySignCap = signBytes
		s.querySignBuf = unsafe.Slice((*float32)(hostPtr), signBytes/4)
	} else {
		s.querySignBuf = s.querySignBuf[:signBytes/4]
	}
	return nil
}

func (s *GPUPreparedScorer) ensureScoreDeviceCapacity(count int) error {
	wantBytes := count * 4
	if wantBytes <= s.scoreCap {
		return nil
	}
	if err := s.freeDevice(s.dScores); err != nil {
		return err
	}
	ptr, err := s.runtime.Alloc(wantBytes)
	if err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	s.dScores = ptr
	s.scoreCap = wantBytes
	return nil
}

func (s *GPUPreparedScorer) ensureTopKDeviceCapacity(count int) error {
	wantBytes := count * 4
	if wantBytes <= s.topKCap {
		return nil
	}
	if err := s.freeDevice(s.dTopKIdx); err != nil {
		return err
	}
	if err := s.freeDevice(s.dTopKScore); err != nil {
		return err
	}
	ptr, err := s.runtime.Alloc(wantBytes)
	if err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	s.dTopKIdx = ptr
	ptr, err = s.runtime.Alloc(wantBytes)
	if err != nil {
		_ = s.freeDevice(s.dTopKIdx)
		s.dTopKIdx = 0
		return fmt.Errorf("turboquant: %w", err)
	}
	s.dTopKScore = ptr
	s.topKCap = wantBytes
	return nil
}

func (s *GPUPreparedScorer) ensureHostScoreCapacity(count int) []float32 {
	if cap(s.hostScores) < count {
		_ = s.freeHost(s.hostScorePtr)
		hostPtr, err := s.runtime.AllocHost(count * 4)
		if err != nil {
			panic(fmt.Sprintf("turboquant: %v", err))
		}
		s.hostScorePtr = hostPtr
		s.hostScores = unsafe.Slice((*float32)(hostPtr), count)
	} else {
		s.hostScores = s.hostScores[:count]
	}
	return s.hostScores
}

func (s *GPUPreparedScorer) allocAndCopy(dst *cudaruntime.DevicePtr, data []byte) error {
	ptr, err := s.runtime.Alloc(len(data))
	if err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	if len(data) != 0 {
		if err := s.runtime.HtoD(ptr, data); err != nil {
			_ = s.runtime.Free(ptr)
			return fmt.Errorf("turboquant: %w", err)
		}
	}
	*dst = ptr
	return nil
}

func (s *GPUPreparedScorer) freeDevice(ptr cudaruntime.DevicePtr) error {
	if s.runtime == nil || ptr == 0 {
		return nil
	}
	if err := s.runtime.Free(ptr); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	return nil
}

func (s *GPUPreparedScorer) freeHost(ptr unsafe.Pointer) error {
	if s.runtime == nil || ptr == nil {
		return nil
	}
	if err := s.runtime.FreeHost(ptr); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	return nil
}

func encodePreparedQueryBatch(pqs []PreparedQuery) ([]byte, []byte) {
	if len(pqs) == 0 {
		return nil, nil
	}
	mseBytesPerQuery := len(pqs[0].mseLUT) * 4
	signBytesPerQuery := len(pqs[0].signLUT) * 4
	mseBytes := make([]byte, len(pqs)*mseBytesPerQuery)
	signBytes := make([]byte, len(pqs)*signBytesPerQuery)
	writePreparedQueryBatchBytes(mseBytes, signBytes, pqs)
	return mseBytes, signBytes
}

func writePreparedQueryBatchBytes(mseBytes, signBytes []byte, pqs []PreparedQuery) {
	if len(pqs) == 0 {
		return
	}
	mseBytesPerQuery := len(pqs[0].mseLUT) * 4
	signBytesPerQuery := len(pqs[0].signLUT) * 4
	for i := range pqs {
		encodeFloat32sBytes(mseBytes[i*mseBytesPerQuery:(i+1)*mseBytesPerQuery], pqs[i].mseLUT)
		encodeFloat32sBytes(signBytes[i*signBytesPerQuery:(i+1)*signBytesPerQuery], pqs[i].signLUT)
	}
}

func writePreparedQueryBatchFloats(mseFloats, signFloats []float32, pqs []PreparedQuery) {
	if len(pqs) == 0 {
		return
	}
	mseFloatsPerQuery := len(pqs[0].mseLUT)
	signFloatsPerQuery := len(pqs[0].signLUT)
	for i := range pqs {
		copy(mseFloats[i*mseFloatsPerQuery:(i+1)*mseFloatsPerQuery], pqs[i].mseLUT)
		copy(signFloats[i*signFloatsPerQuery:(i+1)*signFloatsPerQuery], pqs[i].signLUT)
	}
}

func encodeFloat32s(dst []byte, src []float32) []byte {
	if cap(dst) < len(src)*4 {
		dst = make([]byte, len(src)*4)
	} else {
		dst = dst[:len(src)*4]
	}
	encodeFloat32sBytes(dst, src)
	return dst
}

func encodeFloat32sBytes(dst []byte, src []float32) {
	for i, value := range src {
		binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(value))
	}
}

func encodeUint32sBytes(dst []byte, src []uint32) {
	for i, value := range src {
		dst[i*4+0] = byte(value)
		dst[i*4+1] = byte(value >> 8)
		dst[i*4+2] = byte(value >> 16)
		dst[i*4+3] = byte(value >> 24)
	}
}

func selectTopK(dstIndices []uint32, dstScores []float32, scores []float32, ranks []uint32, k int) {
	if k == 0 {
		return
	}
	var bestIdx [GPUPreparedTopKMax]uint32
	var bestRanks [GPUPreparedTopKMax]uint32
	var bestScores [GPUPreparedTopKMax]float32
	filled := 0
	if len(ranks) == 0 {
		for i, score := range scores {
			candidateIdx := uint32(i)
			candidateRank := candidateIdx
			if filled < k {
				bestIdx[filled] = candidateIdx
				bestRanks[filled] = candidateRank
				bestScores[filled] = score
				siftTopKWorstUp(bestIdx[:], bestRanks[:], bestScores[:], filled)
				filled++
				continue
			}
			if !betterTopK(score, candidateRank, candidateIdx, bestScores[0], bestRanks[0], bestIdx[0]) {
				continue
			}
			bestIdx[0] = candidateIdx
			bestRanks[0] = candidateRank
			bestScores[0] = score
			siftTopKWorstDown(bestIdx[:], bestRanks[:], bestScores[:], filled, 0)
		}
	} else {
		for i, score := range scores {
			candidateIdx := uint32(i)
			candidateRank := ranks[i]
			if filled < k {
				bestIdx[filled] = candidateIdx
				bestRanks[filled] = candidateRank
				bestScores[filled] = score
				siftTopKWorstUp(bestIdx[:], bestRanks[:], bestScores[:], filled)
				filled++
				continue
			}
			if !betterTopK(score, candidateRank, candidateIdx, bestScores[0], bestRanks[0], bestIdx[0]) {
				continue
			}
			bestIdx[0] = candidateIdx
			bestRanks[0] = candidateRank
			bestScores[0] = score
			siftTopKWorstDown(bestIdx[:], bestRanks[:], bestScores[:], filled, 0)
		}
	}
	for i := 1; i < filled; i++ {
		curIdx := bestIdx[i]
		curRank := bestRanks[i]
		curScore := bestScores[i]
		j := i - 1
		for ; j >= 0; j-- {
			if betterTopK(bestScores[j], bestRanks[j], bestIdx[j], curScore, curRank, curIdx) {
				break
			}
			bestIdx[j+1] = bestIdx[j]
			bestRanks[j+1] = bestRanks[j]
			bestScores[j+1] = bestScores[j]
		}
		bestIdx[j+1] = curIdx
		bestRanks[j+1] = curRank
		bestScores[j+1] = curScore
	}
	for i := 0; i < k; i++ {
		if i < filled {
			dstIndices[i] = bestIdx[i]
			dstScores[i] = bestScores[i]
		} else {
			dstIndices[i] = 0
			dstScores[i] = 0
		}
	}
}

func tieRank(ranks []uint32, idx int) uint32 {
	if len(ranks) > idx {
		return ranks[idx]
	}
	return uint32(idx)
}

func betterTopK(score float32, rank, idx uint32, bestScore float32, bestRank, bestIdx uint32) bool {
	if score != bestScore {
		return score > bestScore
	}
	if rank != bestRank {
		return rank < bestRank
	}
	return idx < bestIdx
}

func worseTopK(score float32, rank, idx uint32, otherScore float32, otherRank, otherIdx uint32) bool {
	if score != otherScore {
		return score < otherScore
	}
	if rank != otherRank {
		return rank > otherRank
	}
	return idx > otherIdx
}

func siftTopKWorstUp(indices []uint32, ranks []uint32, scores []float32, idx int) {
	for idx > 0 {
		parent := (idx - 1) / 2
		if !worseTopK(scores[idx], ranks[idx], indices[idx], scores[parent], ranks[parent], indices[parent]) {
			return
		}
		indices[idx], indices[parent] = indices[parent], indices[idx]
		ranks[idx], ranks[parent] = ranks[parent], ranks[idx]
		scores[idx], scores[parent] = scores[parent], scores[idx]
		idx = parent
	}
}

func siftTopKWorstDown(indices []uint32, ranks []uint32, scores []float32, size int, idx int) {
	for {
		left := idx*2 + 1
		if left >= size {
			return
		}
		worst := left
		right := left + 1
		if right < size && worseTopK(scores[right], ranks[right], indices[right], scores[left], ranks[left], indices[left]) {
			worst = right
		}
		if !worseTopK(scores[worst], ranks[worst], indices[worst], scores[idx], ranks[idx], indices[idx]) {
			return
		}
		indices[idx], indices[worst] = indices[worst], indices[idx]
		ranks[idx], ranks[worst] = ranks[worst], ranks[idx]
		scores[idx], scores[worst] = scores[worst], scores[idx]
		idx = worst
	}
}

func (b *GPUPreparedQueryBatch) Len() int {
	if b == nil {
		return 0
	}
	return b.count
}

func (b *GPUPreparedQueryBatch) ScoreTopK(k int) ([]uint32, []float32, error) {
	indices := make([]uint32, b.count*k)
	scores := make([]float32, len(indices))
	if err := b.ScoreTopKTo(indices, scores, k); err != nil {
		return nil, nil, err
	}
	return indices, scores, nil
}

func (b *GPUPreparedQueryBatch) ScoreTopKTo(indices []uint32, scores []float32, k int) error {
	if b == nil {
		return fmt.Errorf("turboquant: nil GPU prepared-query batch")
	}
	if len(indices) != b.count*k {
		return fmt.Errorf("turboquant: expected uploaded GPU batch top-k index length %d, got %d", b.count*k, len(indices))
	}
	if len(scores) != len(indices) {
		return fmt.Errorf("turboquant: uploaded GPU batch destination length mismatch: %d indices vs %d scores", len(indices), len(scores))
	}
	if k == 0 || b.count == 0 {
		return nil
	}
	s := b.scorer
	if s == nil {
		return fmt.Errorf("turboquant: uploaded GPU prepared-query batch has no scorer")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("turboquant: GPU prepared scorer is closed")
	}
	if b.closed {
		return fmt.Errorf("turboquant: uploaded GPU prepared-query batch is closed")
	}
	if err := s.validateTopK(k); err != nil {
		return err
	}
	if err := s.ensureScoreDeviceCapacity(b.count * s.count); err != nil {
		return err
	}
	host := s.ensureHostScoreCapacity(b.count * s.count)
	if err := s.runtime.LaunchScoreToHost(host, s.dMSE, s.dSigns, s.dResNorms, b.dMSE, b.dSigns, s.dScores, s.count, s.mseBytes, s.signBytes, b.count, s.qjlScale); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	for queryIdx := 0; queryIdx < b.count; queryIdx++ {
		base := queryIdx * s.count
		outBase := queryIdx * k
		selectTopK(indices[outBase:outBase+k], scores[outBase:outBase+k], host[base:base+s.count], s.tieBreakRanks(), k)
	}
	return nil
}

func hasIdentityTieBreakRanks(ranks []uint32) bool {
	if len(ranks) == 0 {
		return true
	}
	for i, rank := range ranks {
		if rank != uint32(i) {
			return false
		}
	}
	return true
}

func (b *GPUPreparedQueryBatch) Close() error {
	if b == nil || b.closed {
		return nil
	}
	b.closed = true
	if b.scorer == nil || b.scorer.closed {
		return nil
	}
	b.scorer.mu.Lock()
	defer b.scorer.mu.Unlock()
	_ = b.scorer.freeDevice(b.dMSE)
	_ = b.scorer.freeDevice(b.dSigns)
	b.dMSE = 0
	b.dSigns = 0
	return nil
}
func (s *GPUPreparedScorer) tieBreakRanks() []uint32 {
	if s == nil || s.identityTie {
		return nil
	}
	return s.ranks
}
