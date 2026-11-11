CC      ?= gcc
AR      ?= ar
VULKAN  ?= 0

CFLAGS  ?= -std=c11 -O3 -g -Wall -Wextra -Wshadow -pthread -ffast-math
CFLAGS  += -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64
LDFLAGS += -pthread -lm

ifeq ($(VULKAN),1)
CFLAGS  += -DMOE_ENABLE_VULKAN
LDFLAGS += -lvulkan
endif

BUILD   := build
CORE    := expert_loader.c lru_cache.c gguf_index.c moe_stream.c memtrack.c
MODEL   := dequant.c tokenizer.c qwen_model.c autotune.c
BACKEND := compute_backend.c compute_vulkan.c
BRIDGE  := moe_bridge.c

CORE_OBJ    := $(addprefix $(BUILD)/,$(CORE:.c=.o))
MODEL_OBJ   := $(addprefix $(BUILD)/,$(MODEL:.c=.o))
BACKEND_OBJ := $(addprefix $(BUILD)/,$(BACKEND:.c=.o))
BRIDGE_OBJ  := $(addprefix $(BUILD)/,$(BRIDGE:.c=.o))

.PHONY: all clean test generate lib vulkan

all: $(BUILD)/moe_stream_test $(BUILD)/moe_generate $(BUILD)/moe_chat $(BUILD)/libmoe_bridge.a

lib: $(BUILD)/libmoe_bridge.a

vulkan:
	$(MAKE) clean
	$(MAKE) VULKAN=1 all $(BUILD)/test_vulkan

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: %.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

# embedded SPIR-V: shader regen must force a rebuild of the backend
$(BUILD)/compute_vulkan.o: vulkan_shaders.h
$(BUILD)/test_vulkan.o: vulkan_shaders.h

$(BUILD)/moe_stream_test: $(CORE_OBJ) $(BUILD)/test_moe_stream.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BUILD)/moe_generate: $(CORE_OBJ) $(MODEL_OBJ) $(BACKEND_OBJ) $(BUILD)/test_generate.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BUILD)/moe_chat: $(CORE_OBJ) $(MODEL_OBJ) $(BACKEND_OBJ) $(BUILD)/chat.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BUILD)/test_vulkan: $(CORE_OBJ) $(MODEL_OBJ) $(BACKEND_OBJ) $(BUILD)/test_vulkan.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BUILD)/libmoe_bridge.a: $(CORE_OBJ) $(MODEL_OBJ) $(BACKEND_OBJ) $(BRIDGE_OBJ)
	$(AR) rcs $@ $^

test: $(BUILD)/moe_stream_test
	./$(BUILD)/moe_stream_test ../../Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf --cache-mb 1024 --threads 4 --direct

clean:
	rm -rf $(BUILD)
