package tqserve

import (
	"io"
	"net/http"
)

type RequestEnvelope struct {
	Model     string `json:"model"`
	Stream    bool   `json:"stream,omitempty"`
	SessionID string `json:"-"`
}

type ModelRoute struct {
	PublicModel  string
	BackendName  string
	BackendModel string
	OwnedBy      string
}

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object,omitempty"`
	OwnedBy string `json:"owned_by,omitempty"`
}

type ModelList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

type BackendResponse struct {
	StatusCode int
	Header     http.Header
	Body       io.ReadCloser
}

type Backend interface {
	Models(ctx Context) (ModelList, error)
	ChatCompletions(ctx Context, req RequestEnvelope, body []byte) (*BackendResponse, error)
	Responses(ctx Context, req RequestEnvelope, body []byte) (*BackendResponse, error)
}
