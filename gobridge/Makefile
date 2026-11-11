CC      ?= gcc
GO      ?= go

.PHONY: all cpu vulkan lib clean

all: cpu

# CPU-only build: scalar math, no -lvulkan needed.
cpu:
	$(MAKE) -C ../src/moe_stream lib
	CGO_ENABLED=1 $(GO) build -o ggml-loader ./cmd

# Vulkan build: rebuild the C library with the GPU backend, then link -lvulkan.
vulkan:
	$(MAKE) -C ../src/moe_stream clean
	$(MAKE) -C ../src/moe_stream VULKAN=1 lib
	CGO_ENABLED=1 $(GO) build -tags vulkan -o ggml-loader ./cmd

lib:
	$(MAKE) -C ../src/moe_stream lib

clean:
	rm -f ggml-loader
