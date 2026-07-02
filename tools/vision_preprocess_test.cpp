// Regression guard for the Qwen-VL image patchifier layout.
//
// Background: a bug once shipped where the channel loop (d8) was hoisted out of
// the innermost position in Qwen2VideoProcessor, producing a channel-PLANAR
// tensor (R..R G..G B..B per block) instead of the channel-INTERLEAVED
// (HWC-per-patch, R G B R G B ...) layout the vision encoder is calibrated for.
// That scrambled every image and the model described real photos as
// "distorted / pixelated / grayscale / digital corruption".
//
// This test locks the patchify layout so that regression cannot recur silently.
// It is host-runnable and needs no hardware (pure CPU preprocessing).

#include <cstdint>
#include <cstdio>
#include <vector>

#include "ax_cv.hpp"
#include "image_processor.hpp"

// Build a synthetic BGR image. Each channel occupies a distinct value range so a
// channel-planar regression is directly detectable; a small position-dependent
// term also makes any spatial-order regression change the bytes.
static axcv::Mat make_synthetic_image(int H, int W)
{
#if defined(AXLLM_USE_OPENCV)
    axcv::Mat m(H, W, CV_8UC3);
#else
    axcv::Mat m(H, W, 3);
#endif
    for (int y = 0; y < H; ++y) {
        uint8_t* row = axcv::row_ptr(m, y);
        for (int x = 0; x < W; ++x) {
            const int p = (y * W + x) % 40;
            row[3 * x + 0] = (uint8_t)(0 + p);    // B in [0,39]
            row[3 * x + 1] = (uint8_t)(100 + p);  // G in [100,139]
            row[3 * x + 2] = (uint8_t)(200 + p);  // R in [200,239]
        }
    }
    return m;
}

int main()
{
    const int patch = 16, merge = 2, tp = 2;
    const int H = 64, W = 64;  // grid 4x4 patches -> 2x2 merge blocks
    const int C = 3;

    axcv::Mat img = make_synthetic_image(H, W);
    std::vector<axcv::Mat> src{img};
    std::vector<std::vector<unsigned char>> out;

    // tgt == src size => identity resize (BGR->RGB only), so bytes are exact.
    Qwen2VideoProcessor(src, out, H, W, tp, merge, patch);

    const int grid_h = H / patch, grid_w = W / patch;
    const size_t expect_len =
        (size_t)(grid_h / merge) * (grid_w / merge) * merge * merge * tp * patch * patch * C;

    int fails = 0;

    if (out.size() != 1) {
        printf("FAIL grid_t: out.size()=%zu expected 1\n", out.size());
        return 1;
    }
    if (out[0].size() != expect_len) {
        printf("FAIL len: out[0].size()=%zu expected %zu\n", out[0].size(), expect_len);
        return 1;
    }

    // Reference layout (the SPEC): channel d8 is the INNERMOST loop, matching the
    // numpy reshape+transpose(0,2,5,3,6,1,4,7,8). Both temporal frames == RGB(img).
    std::vector<unsigned char> rgb((size_t)H * W * C);
    for (int y = 0; y < H; ++y) {
        const uint8_t* r = axcv::row_ptr(img, y);
        for (int x = 0; x < W; ++x) {
            rgb[((size_t)y * W + x) * 3 + 0] = r[3 * x + 2];  // R
            rgb[((size_t)y * W + x) * 3 + 1] = r[3 * x + 1];  // G
            rgb[((size_t)y * W + x) * 3 + 2] = r[3 * x + 0];  // B
        }
    }
    std::vector<unsigned char> expected;
    expected.reserve(expect_len);
    for (int d2 = 0; d2 < grid_h / merge; ++d2)
        for (int d5 = 0; d5 < grid_w / merge; ++d5)
            for (int d3 = 0; d3 < merge; ++d3)
                for (int d6 = 0; d6 < merge; ++d6)
                    for (int d1 = 0; d1 < tp; ++d1)
                        for (int d4 = 0; d4 < patch; ++d4)
                            for (int d7 = 0; d7 < patch; ++d7)
                                for (int d8 = 0; d8 < C; ++d8) {
                                    (void)d1;  // both temporal frames identical here
                                    const int y = (d2 * merge + d3) * patch + d4;
                                    const int x = (d5 * merge + d6) * patch + d7;
                                    expected.push_back(rgb[((size_t)y * W + x) * 3 + d8]);
                                }
    if (expected != out[0]) {
        printf("FAIL layout: patchify output != channel-interleaved reference layout\n");
        ++fails;
    }

    // Independent invariant (does not reuse the transpose): with the value ranges
    // above, every consecutive output triple MUST be (R>=200, 100<=G<150, B<50).
    // A channel-planar regression yields runs like (R,R,R)/(G,G,G) and trips this.
    int bad = 0;
    for (size_t k = 0; k + 2 < out[0].size(); k += 3) {
        const unsigned char R = out[0][k], G = out[0][k + 1], B = out[0][k + 2];
        if (!(R >= 200 && G >= 100 && G < 150 && B < 50)) {
            if (++bad <= 3) printf("  interleave break @%zu: (%u,%u,%u)\n", k, R, G, B);
        }
    }
    if (bad) {
        printf("FAIL interleave: %d triples not (R,G,B) — channel not innermost (planar regression)\n", bad);
        ++fails;
    }

    if (fails) {
        printf("vision_preprocess_test: FAILED (%d check(s))\n", fails);
        return 1;
    }
    printf("vision_preprocess_test: PASS (Qwen2VideoProcessor layout is channel-interleaved HWC)\n");
    return 0;
}
