// `ggml-loader inspect <model.gguf>`: dump GGUF metadata and tensor list.
#include "commands.h"
#include "ollama_model.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <string>

static std::string format_kv_value(const gguf_context * ctx, int64_t key_id) {
    switch (gguf_get_kv_type(ctx, key_id)) {
        case GGUF_TYPE_UINT8:   return std::to_string(gguf_get_val_u8(ctx, key_id));
        case GGUF_TYPE_INT8:    return std::to_string(gguf_get_val_i8(ctx, key_id));
        case GGUF_TYPE_UINT16:  return std::to_string(gguf_get_val_u16(ctx, key_id));
        case GGUF_TYPE_INT16:   return std::to_string(gguf_get_val_i16(ctx, key_id));
        case GGUF_TYPE_UINT32:  return std::to_string(gguf_get_val_u32(ctx, key_id));
        case GGUF_TYPE_INT32:   return std::to_string(gguf_get_val_i32(ctx, key_id));
        case GGUF_TYPE_FLOAT32: return std::to_string(gguf_get_val_f32(ctx, key_id));
        case GGUF_TYPE_UINT64:  return std::to_string(gguf_get_val_u64(ctx, key_id));
        case GGUF_TYPE_INT64:   return std::to_string(gguf_get_val_i64(ctx, key_id));
        case GGUF_TYPE_FLOAT64: return std::to_string(gguf_get_val_f64(ctx, key_id));
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(ctx, key_id) ? "true" : "false";
        case GGUF_TYPE_STRING: {
            std::string s = gguf_get_val_str(ctx, key_id);
            if (s.size() > 80) {
                s = s.substr(0, 77) + "...";
            }
            return s;
        }
        case GGUF_TYPE_ARRAY: {
            enum gguf_type arr_type = gguf_get_arr_type(ctx, key_id);
            size_t n = gguf_get_arr_n(ctx, key_id);
            return "array<" + std::string(gguf_type_name(arr_type)) + ">[" + std::to_string(n) + "]";
        }
        default:
            return "?";
    }
}

int cmd_inspect(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    std::string path = resolve_model_ref(argv[1]);

    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "error: failed to load '%s' as GGUF\n", path.c_str());
        return 1;
    }

    printf("file:      %s\n", path.c_str());
    printf("version:   %u\n", gguf_get_version(ctx));
    printf("alignment: %zu\n", gguf_get_alignment(ctx));

    int64_t n_kv = gguf_get_n_kv(ctx);
    printf("\n-- metadata (%lld keys) --\n", (long long) n_kv);
    for (int64_t i = 0; i < n_kv; i++) {
        const char * key = gguf_get_key(ctx, i);
        printf("  %-40s = %s\n", key, format_kv_value(ctx, i).c_str());
    }

    int64_t n_tensors = gguf_get_n_tensors(ctx);
    size_t total_bytes = 0;
    printf("\n-- tensors (%lld) --\n", (long long) n_tensors);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        const int64_t * ne = gguf_get_tensor_ne(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        size_t size = gguf_get_tensor_size(ctx, i);
        total_bytes += size;
        printf("  %-40s %6s  [%5lld, %5lld, %5lld, %5lld]  %10zu bytes\n",
               name, ggml_type_name(type),
               (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3],
               size);
    }

    printf("\ntotal tensor data: %.2f MiB\n", total_bytes / 1024.0 / 1024.0);

    gguf_free(ctx);
    return 0;
}
