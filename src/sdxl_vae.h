// SDXL/SD VAE decoder: latent -> RGB image. Ported from stable-diffusion.cpp's AutoencoderKL
// Decoder (see reference/stable-diffusion.cpp/src/model/vae/auto_encoder_kl.hpp), which we read
// directly for the exact tensor layout/reshape conventions rather than guessing -- this is new
// territory for this project (first conv/image graph; everything else so far has been
// transformer-decoder text models).
//
// Unlike llama.cpp-style gguf, stable-diffusion.cpp's gguf conversion carries NO
// general.architecture-style metadata at all -- hparams below (ch=128, ch_mult={1,2,4,4},
// num_res_blocks=2, z_channels=4) are the fixed, standard SD1.x/SDXL VAE config (unchanged
// across those model families), not read from the file.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

#include <vector>

struct vae_resnet_block {
    struct ggml_tensor * norm1_w = nullptr, * norm1_b = nullptr;
    struct ggml_tensor * conv1_w = nullptr, * conv1_b = nullptr;
    struct ggml_tensor * norm2_w = nullptr, * norm2_b = nullptr;
    struct ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;
    struct ggml_tensor * nin_shortcut_w = nullptr, * nin_shortcut_b = nullptr; // only if in_ch != out_ch
};

struct vae_attn_block {
    struct ggml_tensor * norm_w = nullptr, * norm_b = nullptr;
    struct ggml_tensor * q_w = nullptr, * q_b = nullptr;
    struct ggml_tensor * k_w = nullptr, * k_b = nullptr;
    struct ggml_tensor * v_w = nullptr, * v_b = nullptr;
    struct ggml_tensor * proj_out_w = nullptr, * proj_out_b = nullptr;
};

struct vae_up_level {
    std::vector<vae_resnet_block> blocks; // num_res_blocks+1 (3) for the decoder
    struct ggml_tensor * upsample_conv_w = nullptr, * upsample_conv_b = nullptr; // absent at level 0
};

struct vae_decoder {
    struct ggml_tensor * conv_in_w = nullptr, * conv_in_b = nullptr;
    vae_resnet_block mid_block_1;
    vae_attn_block   mid_attn_1;
    vae_resnet_block mid_block_2;
    std::vector<vae_up_level> up; // indices 0..3, high-to-low channel count as index decreases
    struct ggml_tensor * norm_out_w = nullptr, * norm_out_b = nullptr;
    struct ggml_tensor * conv_out_w = nullptr, * conv_out_b = nullptr;
};

// Looks up every "first_stage_model.decoder.*" tensor by the standard SD/SDXL VAE names.
// Exits with an error naming the missing tensor if the file doesn't match this layout.
vae_decoder load_vae_decoder(struct ggml_context * wctx);

// latent_w/latent_h are in latent space (image space is 8x that, per the VAE's 3 upsample
// stages). latent_data must have latent_w*latent_h*4 elements in ggml-native [w,h,c,n] order
// (w fastest, c slowest, n=1). Returns RGB in that same ggml-native [w*8, h*8, 3, 1] order and
// the model's raw (untransformed, roughly [-1,1]) value range -- the caller is responsible for
// both the value-range rescale to [0,255] and the transpose to stb_image_write's row-major HWC
// layout (see cmd_sdxl.cpp). backend must be the one `wctx`'s tensors were allocated on (see
// load_weights() in model.h).
std::vector<float> vae_decode(const vae_decoder & vae, ggml_backend_t backend,
                               const std::vector<float> & latent_data,
                               int64_t latent_w, int64_t latent_h);
