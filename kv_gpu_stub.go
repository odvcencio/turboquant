//go:build !(linux && amd64 && cgo && cuda)

package turboquant

func newKVGPUValueBackend(p *KVCachePage) (kvGPUValueBackend, error) {
	return nil, nil
}
