// Package moe is the Go surface of the C MoE inference engine.
//
// The C side owns the model, the KV cache and the expert streamer. Go owns
// scheduling, prompt handling and output. The two meet through a flat C ABI
// (moe_bridge.h); no Go pointers are ever retained by C.
package moe

/*
#cgo CFLAGS: -I${SRCDIR}/../src/moe_stream
#include <stdlib.h>
#include "moe_bridge.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"runtime/cgo"
	"unsafe"
)

// Options mirrors moe_session_opts_t.
type Options struct {
	CacheMB int  // expert LRU budget
	Threads int  // I/O workers
	MaxSeq  int  // KV cache capacity
	ODirect bool // bypass page cache
}

// Session is a loaded model plus tokenizer plus KV cache.
type Session struct {
	p *C.moe_session_t
}

// Open loads the GGUF and runs the tokenizer init. This allocates ~1 GB of
// resident weights plus CacheMB, so it takes a second or two.
func Open(path string, o Options) (*Session, error) {
	if o.Threads <= 0 {
		o.Threads = 4
	}
	if o.MaxSeq <= 0 {
		o.MaxSeq = 256
	}
	if o.CacheMB <= 0 {
		o.CacheMB = 512
	}

	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var opts C.moe_session_opts_t
	opts.gguf_path = cpath
	opts.cache_bytes = C.size_t(o.CacheMB) * 1024 * 1024
	opts.threads = C.int(o.Threads)
	opts.max_seq = C.int(o.MaxSeq)
	if o.ODirect {
		opts.odirect = 1
	}

	p := C.moe_session_create(&opts)
	if p == nil {
		return nil, errors.New(C.GoString(C.moe_bridge_last_error()))
	}

	s := &Session{p: p}
	runtime.SetFinalizer(s, (*Session).Close)
	return s, nil
}

// Close releases the C session. Safe to call more than once.
func (s *Session) Close() {
	if s == nil || s.p == nil {
		return
	}
	C.moe_session_destroy(s.p)
	s.p = nil
	runtime.SetFinalizer(s, nil)
}

func (s *Session) Vocab() int   { return int(C.moe_session_vocab_size(s.p)) }
func (s *Session) Context() int { return int(C.moe_session_context(s.p)) }
func (s *Session) Pos() int     { return int(C.moe_session_pos(s.p)) }
func (s *Session) Backend() string {
	return C.GoString(C.moe_session_backend(s.p))
}

// UseVulkan swaps in the Vulkan math backend. Currently a stub that falls
// back to scalar; the call site is stable.
func (s *Session) UseVulkan() error {
	if C.moe_session_use_vulkan(s.p) != 0 {
		return fmt.Errorf("vulkan backend unavailable")
	}
	return nil
}

// Reset clears the KV cache and rewinds to position 0.
func (s *Session) Reset() error {
	if C.moe_session_reset(s.p) != 0 {
		return fmt.Errorf("reset failed")
	}
	return nil
}

// Encode tokenizes with the approximate greedy matcher. Use ids directly if
// you need exact HF tokenization.
func (s *Session) Encode(text string) ([]uint32, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	const maxIDs = 4096
	buf := make([]C.uint32_t, maxIDs)

	n := C.moe_session_encode(s.p, ctext,
		(*C.uint32_t)(unsafe.Pointer(&buf[0])), C.int(maxIDs))
	if n < 0 {
		return nil, errors.New("tokenize failed")
	}

	out := make([]uint32, int(n))
	for i := range out {
		out[i] = uint32(buf[i])
	}
	return out, nil
}

// Decode turns one token id into raw UTF-8 bytes.
func (s *Session) Decode(id uint32) string {
	buf := make([]C.char, 256)
	n := C.moe_session_decode(s.p, C.uint32_t(id),
		(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)))
	if n <= 0 {
		return ""
	}
	return C.GoStringN((*C.char)(unsafe.Pointer(&buf[0])), n)
}

// Prefill runs the layer sweep over the whole prompt, filling the KV cache.
// After it returns, logits are ready for the first generated token.
func (s *Session) Prefill(tokens []uint32) error {
	if len(tokens) == 0 {
		return nil
	}
	rc := C.moe_session_prefill(s.p,
		(*C.uint32_t)(unsafe.Pointer(&tokens[0])), C.int(len(tokens)))
	if rc != 0 {
		return fmt.Errorf("prefill failed: %d", int(rc))
	}
	return nil
}

// AutoConfig calls calculate_optimal_cache() in C. Returns the recommended
// expert cache (MiB), context length, and the MemAvailable figure used.
func AutoConfig() (cacheMB, ctxLen, availMB int) {
	var c, x, a C.int
	C.moe_auto_config(&c, &x, &a)
	return int(c), int(x), int(a)
}

// Generate produces up to maxNew tokens. out_ids is filled in C and copied
// back, so no Go pointer is retained across the call.
//
// If onToken is non-nil it is invoked once per token with the decoded text,
// which is the streaming path (a cgo.Handle carries the closure). With nil it
// is a single batch call and only the returned ids are available.
func (s *Session) Generate(maxNew int, onToken func(id uint32, text string)) ([]uint32, error) {
	if maxNew <= 0 {
		return nil, nil
	}

	ids := make([]C.uint32_t, maxNew)
	var n C.int

	if onToken == nil {
		n = C.moe_session_generate(s.p, C.int(maxNew),
			(*C.uint32_t)(unsafe.Pointer(&ids[0])), C.int(maxNew), nil, nil)
	} else {
		h := cgo.NewHandle(onToken)
		defer h.Delete()
		n = C.moe_session_generate_go(s.p, C.int(maxNew),
			(*C.uint32_t)(unsafe.Pointer(&ids[0])), C.int(maxNew), C.uintptr_t(h))
	}

	if n < 0 {
		return nil, fmt.Errorf("generate failed: %d", int(n))
	}

	out := make([]uint32, int(n))
	for i := range out {
		out[i] = uint32(ids[i])
	}
	return out, nil
}

// GenerateStream is Generate with a stop signal: returning true from onToken
// ends generation immediately (the C loop stops before forwarding that token).
func (s *Session) GenerateStream(maxNew int, onToken func(id uint32, text string) bool) ([]uint32, error) {
	if maxNew <= 0 {
		return nil, nil
	}

	ids := make([]C.uint32_t, maxNew)
	h := cgo.NewHandle(onToken)
	defer h.Delete()

	n := C.moe_session_generate_go(s.p, C.int(maxNew),
		(*C.uint32_t)(unsafe.Pointer(&ids[0])), C.int(maxNew), C.uintptr_t(h))
	if n < 0 {
		return nil, fmt.Errorf("generate failed: %d", int(n))
	}

	out := make([]uint32, int(n))
	for i := range out {
		out[i] = uint32(ids[i])
	}
	return out, nil
}

// Raw token-id streaming (no detokenisation). The handler receives vocabulary
// indices, not characters. Kept for the requested raw callback path.
var rawTokenHandler func(int)

// SetRawTokenHandler installs the handler used by GenerateRaw.
func SetRawTokenHandler(f func(int)) { rawTokenHandler = f }

// GenerateRaw streams token ids through go_token_callback. Prefer GenerateStream.
func (s *Session) GenerateRaw(maxNew int) int {
	return int(C.moe_session_generate_raw(s.p, C.int(maxNew)))
}

// Forward runs one decode step at the current position.
func (s *Session) Forward(id uint32) error {
	if C.moe_session_forward(s.p, C.uint32_t(id)) != 0 {
		return errors.New("forward failed")
	}
	return nil
}

// Argmax returns the greedy next token without advancing.
func (s *Session) Argmax() int { return int(C.moe_session_argmax(s.p)) }

//export moe_go_token_callback
func moe_go_token_callback(user unsafe.Pointer, id C.uint32_t, text *C.char, n C.int) C.int {
	h := cgo.Handle(uintptr(user))
	s := C.GoStringN(text, n)

	switch fn := h.Value().(type) {
	case func(uint32, string) bool:
		if fn(uint32(id), s) {
			return 1
		}
	case func(uint32, string):
		fn(uint32(id), s)
	}
	return 0
}

//export go_token_callback
func go_token_callback(id C.int) {
	if rawTokenHandler != nil {
		rawTokenHandler(int(id))
	}
}
