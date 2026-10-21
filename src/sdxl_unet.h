// SDXL's U-Net (the actual diffusion denoiser): ResBlock + SpatialTransformer (cross-attention to
// the CLIP context) stacked across 3 resolution levels, down/mid/up with skip connections. Ported
// from reference/stable-diffusion.cpp's UnetModelBlock (src/model/diffusion/unet.hpp) and its
// ResBlock/SpatialTransformer/BasicTransformerBlock/CrossAttention (src/model/common/block.hpp),
// cross-checked against our own gguf's tensor names via `inspect`.
//
// SDXL-fixed config (this checkpoint, confirmed via inspect -- see load_sdxl_unet's .cpp): in/out
// channels=4, model_channels=320, num_res_blocks=2, channel_mult={1,2,4}, attention_resolutions=
// {4,2} (so the highest-res level has NO attention -- notably different from SD1.x), transformer_
// depth={1,2,10} indexed by channel_mult level, context_dim=2048, num_head_channels=64 (so
// n_head=channels/64 at each level), adm_in_channels=2816, time_embed_dim=1280,
// use_linear_projection=true in config but this checkpoint's proj_in/proj_out are actually stored
// as 1x1-conv-shaped tensors -- mathematically identical to a linear layer applied per spatial
// position (no spatial mixing in a 1x1 kernel), so sdxl_unet.cpp always treats them as linear via
// sdxl_linear on the [C,seq,N]-reshaped tensor regardless of stored shape; see that file for the
// n_dims-based reshape.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

#include <vector>

struct unet_resblock {
    struct ggml_tensor * in_norm_w = nullptr, * in_norm_b = nullptr;
    struct ggml_tensor * in_conv_w = nullptr, * in_conv_b = nullptr;
    struct ggml_tensor * emb_w = nullptr, * emb_b = nullptr;
    struct ggml_tensor * out_norm_w = nullptr, * out_norm_b = nullptr;
    struct ggml_tensor * out_conv_w = nullptr, * out_conv_b = nullptr;
    struct ggml_tensor * skip_w = nullptr, * skip_b = nullptr; // only when in_ch != out_ch
};

// one BasicTransformerBlock: self-attn (attn1) + cross-attn to CLIP context (attn2) + GEGLU FFN
struct unet_xattn_layer {
    struct ggml_tensor * norm1_w = nullptr, * norm1_b = nullptr;
    struct ggml_tensor * attn1_q_w = nullptr, * attn1_k_w = nullptr, * attn1_v_w = nullptr;
    struct ggml_tensor * attn1_out_w = nullptr, * attn1_out_b = nullptr;
    struct ggml_tensor * norm2_w = nullptr, * norm2_b = nullptr;
    struct ggml_tensor * attn2_q_w = nullptr, * attn2_k_w = nullptr, * attn2_v_w = nullptr;
    struct ggml_tensor * attn2_out_w = nullptr, * attn2_out_b = nullptr;
    struct ggml_tensor * norm3_w = nullptr, * norm3_b = nullptr;
    struct ggml_tensor * ff_proj_w = nullptr, * ff_proj_b = nullptr; // GEGLU: dim -> 2*inner_dim
    struct ggml_tensor * ff_out_w = nullptr, * ff_out_b = nullptr;   // inner_dim -> dim
};

struct unet_spatial_transformer {
    struct ggml_tensor * norm_w = nullptr, * norm_b = nullptr;
    struct ggml_tensor * proj_in_w = nullptr, * proj_in_b = nullptr;
    struct ggml_tensor * proj_out_w = nullptr, * proj_out_b = nullptr;
    std::vector<unet_xattn_layer> layers;
    int n_head = 0;
};

// One "input_blocks.N" slot: either a plain stride-2 downsample conv, or a ResBlock optionally
// followed by a SpatialTransformer -- matches the reference's flat input_blocks numbering directly
// (mirrors its index bookkeeping) rather than a per-level nested structure, to minimize the chance
// of subtly misordering blocks relative to the actual checkpoint.
struct unet_down_slot {
    bool is_downsample = false;
    struct ggml_tensor * down_w = nullptr, * down_b = nullptr; // only when is_downsample
    unet_resblock res;                                          // only when !is_downsample
    bool has_attn = false;
    unet_spatial_transformer attn;
};

// One "output_blocks.N" slot: always a ResBlock, optionally followed by a SpatialTransformer,
// optionally followed by an UpSampleBlock (per reference: upsample sits inside the same numbered
// block as the last ResBlock of a level, not its own slot).
struct unet_up_slot {
    unet_resblock res;
    bool has_attn = false;
    unet_spatial_transformer attn;
    bool has_upsample = false;
    struct ggml_tensor * up_w = nullptr, * up_b = nullptr;
};

struct sdxl_unet {
    struct ggml_tensor * time_embed_0_w = nullptr, * time_embed_0_b = nullptr;
    struct ggml_tensor * time_embed_2_w = nullptr, * time_embed_2_b = nullptr;
    struct ggml_tensor * label_emb_0_w = nullptr, * label_emb_0_b = nullptr;
    struct ggml_tensor * label_emb_2_w = nullptr, * label_emb_2_b = nullptr;
    struct ggml_tensor * conv_in_w = nullptr, * conv_in_b = nullptr;
    std::vector<unet_down_slot> down; // input_blocks.1 .. input_blocks.8 (input_blocks.0 is conv_in)
    unet_resblock mid_res1;
    unet_spatial_transformer mid_attn;
    unet_resblock mid_res2;
    std::vector<unet_up_slot> up; // output_blocks.0 .. output_blocks.8
    struct ggml_tensor * out_norm_w = nullptr, * out_norm_b = nullptr;
    struct ggml_tensor * out_conv_w = nullptr, * out_conv_b = nullptr;
};

sdxl_unet load_sdxl_unet(struct ggml_context * wctx);

// latent: [latent_w * latent_h * 4] noisy latent, ggml-native [w,h,c,n] order (n=1). timestep: the
// scalar diffusion timestep (0..999). context: SDXL text conditioning's cross-attention context,
// [2048 * context_len] (from clip_encode). y: the 2816-dim micro-conditioning vector. Returns the
// U-Net's raw epsilon prediction, same shape/order as `latent`.
std::vector<float> unet_forward(const sdxl_unet & unet, ggml_backend_t backend,
                                 const std::vector<float> & latent, int64_t latent_w, int64_t latent_h,
                                 float timestep, const std::vector<float> & context, int context_len,
                                 const std::vector<float> & y);
