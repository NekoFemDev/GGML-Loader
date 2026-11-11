// Command ggml-loader is an interactive chat front-end for the C MoE engine.
//
//	./ggml-loader chat <model.gguf> --backend vulkan
//
// Context length and expert cache default to calculate_optimal_cache() and can
// be overridden with --ctx / --cache-mb.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"ggml-loader/moe"
)

const (
	chatMLSystem = "<|im_start|>system\n%s<|im_end|>\n"
	chatMLUser   = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"

	bold  = "\033[1m"
	dim   = "\033[2m"
	reset = "\033[0m"
)

func usage() {
	fmt.Fprintf(os.Stderr, `usage: ggml-loader chat <model.gguf> [flags]

flags:
  --backend vulkan|cpu   math backend (default vulkan)
  --cache-mb N           expert cache MiB (default: auto)
  --ctx N                context length   (default: auto)
  --threads N            I/O workers      (default 4)
  --max-new N            max tokens per reply (default 512)
  --system "..."         system prompt
  --no-direct            disable O_DIRECT
  --debug                tokenizer diagnostics
`)
}

func main() {
	args := os.Args[1:]
	if len(args) > 0 && args[0] == "chat" {
		args = args[1:]
	}
	if len(args) == 0 {
		usage()
		os.Exit(2)
	}

	model := ""
	if !strings.HasPrefix(args[0], "-") {
		model = args[0]
		args = args[1:]
	}

	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	backend := fs.String("backend", "vulkan", "vulkan|cpu")
	cacheMB := fs.Int("cache-mb", 0, "expert cache MiB (0 = auto)")
	ctxLen := fs.Int("ctx", 0, "context length (0 = auto)")
	threads := fs.Int("threads", 4, "I/O workers")
	maxNew := fs.Int("max-new", 512, "max tokens per reply")
	system := fs.String("system", "", "system prompt")
	noDirect := fs.Bool("no-direct", false, "disable O_DIRECT")
	debug := fs.Bool("debug", false, "tokenizer diagnostics")
	modelFlag := fs.String("model", "", "model path (alternative to positional)")
	fs.Parse(args)

	if model == "" {
		model = *modelFlag
	}
	if model == "" {
		usage()
		os.Exit(2)
	}

	// Auto configuration comes from the C side, which reads MemAvailable and
	// applies the 2 GiB OS reserve and the 4.5 GiB process cap.
	autoCache, autoCtx, availMB := moe.AutoConfig()
	if *cacheMB == 0 {
		*cacheMB = autoCache
	}
	if *ctxLen == 0 {
		*ctxLen = autoCtx
	}
	fmt.Printf("[Auto-Cache] Using %d MB for experts, %d context length based on %d MB available RAM\n",
		*cacheMB, *ctxLen, availMB)

	s, err := moe.Open(model, moe.Options{
		CacheMB: *cacheMB,
		Threads: *threads,
		MaxSeq:  *ctxLen,
		ODirect: !*noDirect,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer s.Close()

	if *backend == "vulkan" {
		if err := s.UseVulkan(); err != nil {
			fmt.Fprintln(os.Stderr, "vulkan:", err)
			os.Exit(1)
		}
	}
	fmt.Printf("[backend] %s, vocab %d, ctx %d\n", s.Backend(), s.Vocab(), s.Context())

	// The stop token must be resolved before the loop. It is normally a single
	// vocabulary entry; the text fallback below covers a split encoding.
	endIDs, err := s.Encode("<|im_end|>")
	if err != nil || len(endIDs) == 0 {
		fmt.Fprintln(os.Stderr, "cannot encode <|im_end|>")
		os.Exit(1)
	}
	imEnd := int(endIDs[0])
	if *debug {
		fmt.Printf("[debug] <|im_end|> -> id %d (%d token(s))\n", imEnd, len(endIDs))
	}

	reader := bufio.NewReader(os.Stdin)
	first := true

	for {
		fmt.Printf("\n%suser>%s ", bold, reset)
		line, rerr := reader.ReadString('\n')
		if len(line) == 0 && rerr != nil {
			fmt.Println()
			break
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			continue
		}
		switch line {
		case "/exit", "/quit":
			return
		case "/reset":
			_ = s.Reset()
			first = true
			fmt.Printf("%s[context reset]%s\n", dim, reset)
			continue
		}

		var prompt strings.Builder
		if first {
			if *system != "" {
				fmt.Fprintf(&prompt, chatMLSystem, *system)
			}
			first = false
		} else {
			// The previous turn ended with <|im_end|> already in the KV cache.
			prompt.WriteString("\n")
		}
		fmt.Fprintf(&prompt, chatMLUser, line)

		ids, err := s.Encode(prompt.String())
		if err != nil {
			fmt.Fprintln(os.Stderr, "tokenize:", err)
			continue
		}
		if s.Pos()+len(ids)+*maxNew >= s.Context() {
			fmt.Fprintf(os.Stderr,
				"\n%s[context full: %d/%d tokens; use /reset]%s\n",
				dim, s.Pos(), s.Context(), reset)
			continue
		}
		if *debug {
			fmt.Printf("%s[debug] turn %d tokens, pos %d%s\n", dim, len(ids), s.Pos(), reset)
			for _, id := range ids {
				fmt.Printf("  %6d %q\n", id, s.Decode(id))
			}
		}

		if err := s.Prefill(ids); err != nil {
			fmt.Fprintln(os.Stderr, "prefill:", err)
			continue
		}

		fmt.Printf("%sassistant>%s ", bold, reset)

		stopped := false
		out, gerr := s.GenerateStream(*maxNew, func(id uint32, text string) bool {
			if int(id) == imEnd || strings.Contains(text, "<|im_end|>") {
				stopped = true
				return true
			}
			fmt.Print(text)
			return false
		})
		if gerr != nil {
			fmt.Fprintf(os.Stderr, "\n%serror: %v%s\n", dim, gerr, reset)
		}
		fmt.Println()

		// The C loop never forwards the final emitted token. Push it, then
		// close the assistant turn so the next prompt is contiguous ChatML.
		if len(out) > 0 {
			last := out[len(out)-1]
			_ = s.Forward(last)
			if int(last) != imEnd && s.Pos() < s.Context() {
				_ = s.Forward(uint32(imEnd))
			}
		} else if s.Pos() < s.Context() {
			_ = s.Forward(uint32(imEnd))
		}
		_ = stopped
	}
}
