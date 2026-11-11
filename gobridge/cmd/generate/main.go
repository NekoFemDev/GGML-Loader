// Command generate is a minimal CGO smoke test: load, tokenize, prefill with
// the layer sweep, then stream generated tokens to stdout.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"ggml-loader/moe"
)

func main() {
	model := flag.String("model", "../../Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf", "gguf path")
	prompt := flag.String("prompt", "The capital of France is", "prompt text")
	ngen := flag.Int("ngen", 8, "tokens to generate")
	cache := flag.Int("cache", 512, "expert cache MiB")
	maxSeq := flag.Int("ctx", 256, "context length")
	direct := flag.Bool("direct", true, "O_DIRECT reads")
	flag.Parse()

	s, err := moe.Open(*model, moe.Options{
		CacheMB: *cache,
		Threads: 4,
		MaxSeq:  *maxSeq,
		ODirect: *direct,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer s.Close()

	fmt.Printf("backend=%s vocab=%d ctx=%d\n", s.Backend(), s.Vocab(), s.Context())

	ids, err := s.Encode(*prompt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode:", err)
		os.Exit(1)
	}
	fmt.Printf("prompt %q -> %d tokens\n", *prompt, len(ids))

	t0 := time.Now()
	if err := s.Prefill(ids); err != nil {
		fmt.Fprintln(os.Stderr, "prefill:", err)
		os.Exit(1)
	}
	fmt.Printf("prefill: %v (%.0f ms/token)\n\n", time.Since(t0),
		float64(time.Since(t0).Milliseconds())/float64(len(ids)))

	var sb strings.Builder
	t1 := time.Now()
	got, err := s.Generate(*ngen, func(id uint32, text string) {
		sb.WriteString(text)
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate:", err)
		os.Exit(1)
	}
	el := time.Since(t1)

	fmt.Printf("generated:\n%s\n\n", sb.String())
	fmt.Printf("%d tokens in %v (%.3f tok/s), ids=%v\n",
		len(got), el, float64(len(got))/el.Seconds(), got)
}
