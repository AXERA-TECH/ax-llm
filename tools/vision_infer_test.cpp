// Host unit test for the pure vision encoder-geometry / token inference helpers
// (src/runner/vision/vision_infer.hpp). No NPU / no model: runs on any x86 CI runner.
// Covers the logic extracted during the IVisionAdapter refactor (resolveInputGeometry /
// resolveOutputTokens), so regressions in that pure math are caught without hardware.

#include "vision_infer.hpp"

#include <cstdio>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (cond) {                                                        \
            ++g_pass;                                                      \
        } else {                                                           \
            ++g_fail;                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                  \
    } while (0)

int main() {
    using namespace vision;

    // ---- try_infer_qwen_hw_from_input_bytes ----------------------------------------
    // input bytes = H*W*temporal*channels. 448x448, temporal=2, ch=3, patch=14 -> square.
    {
        int h = 0, w = 0;
        std::string note;
        const bool ok = try_infer_qwen_hw_from_input_bytes(
            (size_t)448 * 448 * 2 * 3, 2, 3, 14, h, w, note);
        CHECK(ok);
        CHECK(h == 448 && w == 448);
    }
    // Non-square, resolved by keeping configured height (336) and solving width (448).
    {
        int h = 336, w = 0;  // io_h preset
        std::string note;
        const bool ok = try_infer_qwen_hw_from_input_bytes(
            (size_t)336 * 448 * 2 * 3, 2, 3, 14, h, w, note);
        CHECK(ok);
        CHECK(h == 336 && w == 448);
    }
    // Not divisible by temporal*channels -> fail.
    {
        int h = 0, w = 0;
        std::string note;
        CHECK(!try_infer_qwen_hw_from_input_bytes(100, 2, 3, 14, h, w, note));
    }
    // Perfect square but side not divisible by patch_size -> fail (450 % 14 != 0).
    {
        int h = 0, w = 0;
        std::string note;
        CHECK(!try_infer_qwen_hw_from_input_bytes(
            (size_t)450 * 450 * 2 * 3, 2, 3, 14, h, w, note));
    }

    // ---- try_infer_hw_from_4d_shape_with_c3 ----------------------------------------
    {
        std::vector<unsigned int> nchw{1, 3, 448, 448};
        int h = 0, w = 0, is_nchw = -1;
        CHECK(try_infer_hw_from_4d_shape_with_c3(nchw, h, w, &is_nchw));
        CHECK(h == 448 && w == 448 && is_nchw == 1);
    }
    {
        std::vector<unsigned int> nhwc{1, 224, 224, 3};
        int h = 0, w = 0, is_nchw = -1;
        CHECK(try_infer_hw_from_4d_shape_with_c3(nhwc, h, w, &is_nchw));
        CHECK(h == 224 && w == 224 && is_nchw == 0);
    }
    {
        std::vector<unsigned int> not4d{1, 3, 448};  // size 3 -> reject
        int h = 0, w = 0;
        CHECK(!try_infer_hw_from_4d_shape_with_c3(not4d, h, w));
    }
    {
        std::vector<unsigned int> no_c3{1, 5, 448, 448};  // no channel==3 dim -> reject
        int h = 0, w = 0;
        CHECK(!try_infer_hw_from_4d_shape_with_c3(no_c3, h, w));
    }

    // ---- parse_gemma4_profile_from_path --------------------------------------------
    {
        int h = 0, w = 0, t = 0;
        CHECK(parse_gemma4_profile_from_path("gemma4_vit_h256_w256_t64.axmodel", h, w, t));
        CHECK(h == 256 && w == 256 && t == 64);
    }
    {
        int h = 0, w = 0, t = 0;
        CHECK(!parse_gemma4_profile_from_path("gemma4_vit_noprofile.axmodel", h, w, t));
    }

    // ---- pick_tokens_by_bytes ------------------------------------------------------
    // tokens_embed=1536, tokens_per_block=64. bf16 (2 bytes/elem).
    {
        int is_bf16 = -1, tpb = 0;
        CHECK(pick_tokens_by_bytes((size_t)64 * 1536 * 2, 1536, 2, is_bf16, tpb));
        CHECK(is_bf16 == 1 && tpb == 64);
    }
    // fp32 (4 bytes/elem).
    {
        int is_bf16 = -1, tpb = 0;
        CHECK(pick_tokens_by_bytes((size_t)64 * 1536 * 4, 1536, 4, is_bf16, tpb));
        CHECK(is_bf16 == 0 && tpb == 64);
    }
    // elem count not divisible by tokens_embed -> fail.
    {
        int is_bf16 = -1, tpb = 0;
        CHECK(!pick_tokens_by_bytes(100, 1536, 2, is_bf16, tpb));
    }
    // nSize not divisible by bytes_per_elem -> fail.
    {
        int is_bf16 = -1, tpb = 0;
        CHECK(!pick_tokens_by_bytes(7, 1536, 2, is_bf16, tpb));
    }

    std::printf("vision_infer_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
