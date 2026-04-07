package tqserve

import (
	"context"
	"reflect"
)

type CapacitySnapshot struct {
	Accelerator      string `json:"accelerator,omitempty"`
	Device           string `json:"device,omitempty"`
	DeviceCount      int    `json:"device_count,omitempty"`
	TotalMemoryBytes uint64 `json:"total_memory_bytes,omitempty"`
	UsedMemoryBytes  uint64 `json:"used_memory_bytes,omitempty"`
	FreeMemoryBytes  uint64 `json:"free_memory_bytes,omitempty"`
	WeightsBytes     uint64 `json:"weights_bytes,omitempty"`
	KVCacheBytes     uint64 `json:"kv_cache_bytes,omitempty"`
	KVHeadroomBytes  uint64 `json:"kv_headroom_bytes,omitempty"`
	MaxSessions      int    `json:"max_sessions,omitempty"`
	ActiveSessions   int    `json:"active_sessions,omitempty"`
	Notes            string `json:"notes,omitempty"`
}

type CapacityProvider interface {
	Capacity(ctx context.Context) CapacitySnapshot
}

func (c CapacitySnapshot) Empty() bool {
	return reflect.ValueOf(c).IsZero()
}

func mergeCapacity(primary, fallback CapacitySnapshot) CapacitySnapshot {
	if primary.Empty() {
		return fallback
	}
	if fallback.Empty() {
		return primary
	}
	if primary.Accelerator == "" {
		primary.Accelerator = fallback.Accelerator
	}
	if primary.Device == "" {
		primary.Device = fallback.Device
	}
	if primary.DeviceCount == 0 {
		primary.DeviceCount = fallback.DeviceCount
	}
	if primary.TotalMemoryBytes == 0 {
		primary.TotalMemoryBytes = fallback.TotalMemoryBytes
	}
	if primary.UsedMemoryBytes == 0 {
		primary.UsedMemoryBytes = fallback.UsedMemoryBytes
	}
	if primary.FreeMemoryBytes == 0 {
		primary.FreeMemoryBytes = fallback.FreeMemoryBytes
	}
	if primary.WeightsBytes == 0 {
		primary.WeightsBytes = fallback.WeightsBytes
	}
	if primary.KVCacheBytes == 0 {
		primary.KVCacheBytes = fallback.KVCacheBytes
	}
	if primary.KVHeadroomBytes == 0 {
		primary.KVHeadroomBytes = fallback.KVHeadroomBytes
	}
	if primary.MaxSessions == 0 {
		primary.MaxSessions = fallback.MaxSessions
	}
	if primary.ActiveSessions == 0 {
		primary.ActiveSessions = fallback.ActiveSessions
	}
	if primary.Notes == "" {
		primary.Notes = fallback.Notes
	}
	return primary
}
