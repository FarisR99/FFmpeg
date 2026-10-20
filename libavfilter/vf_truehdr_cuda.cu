/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

extern "C" {

/* Must match TrueHDRConvParams in vf_truehdr_cuda.c. */
struct RGB {
    float r, g, b;
};

struct TrueHDRConvParams {
    float m[3][3];  /* input primaries to BT.2020, linear light */
    float scale;    /* input linear light to PQ's 0..1 (10000 nits) */
    int width, height;
};

static __device__ float half_to_float(unsigned short h)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

static __device__ float pq_oetf(float y)
{
    const float m1 = 2610.0f / 16384.0f;
    const float m2 = 2523.0f / 4096.0f * 128.0f;
    const float c1 = 3424.0f / 4096.0f;
    const float c2 = 2413.0f / 4096.0f * 32.0f;
    const float c3 = 2392.0f / 4096.0f * 32.0f;
    float ym;

    y  = min(max(y, 0.0f), 1.0f);
    ym = __powf(y, m1);
    return __powf((c1 + c2 * ym) / (1.0f + c3 * ym), m2);
}

/* Load one linear light pixel and return PQ coded BT.2020 R'G'B'. */
static __device__ RGB load_rgb(const unsigned char *src, int pitch,
                                  int x, int y, const TrueHDRConvParams &p)
{
    const unsigned short *px;
    float r, g, b;
    RGB out;

    x  = min(max(x, 0), p.width - 1);
    px = (const unsigned short *)(src + y * pitch) + 4 * x;
    r  = half_to_float(px[0]);
    g  = half_to_float(px[1]);
    b  = half_to_float(px[2]);

    out.r = pq_oetf(p.scale * (p.m[0][0] * r + p.m[0][1] * g + p.m[0][2] * b));
    out.g = pq_oetf(p.scale * (p.m[1][0] * r + p.m[1][1] * g + p.m[1][2] * b));
    out.b = pq_oetf(p.scale * (p.m[2][0] * r + p.m[2][1] * g + p.m[2][2] * b));
    return out;
}

static __device__ float luma(RGB c)
{
    return 0.2627f * c.r + 0.6780f * c.g + 0.0593f * c.b;
}

static __device__ unsigned short to_p010(float v, int lo, int hi)
{
    int i = (int)(v + 0.5f);
    return (unsigned short)(min(max(i, lo), hi) << 6);
}

/*
 * One thread per 2x2 block: write four luma samples and one left-sited
 * chroma sample (horizontal taps 1/4, 1/2, 1/4, averaged over both rows),
 * as limited range BT.2020 non-constant luminance P010.
 */
__global__ void truehdr_to_p010(const unsigned char *src, int src_pitch,
                                unsigned char *dst_y, int pitch_y,
                                unsigned char *dst_uv, int pitch_uv,
                                TrueHDRConvParams p)
{
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int x = 2 * bx, y = 2 * by;
    float cb = 0.0f, cr = 0.0f;

    if (x >= p.width || y >= p.height)
        return;

    for (int row = 0; row < 2; row++) {
        unsigned short *out_y = (unsigned short *)(dst_y + (y + row) * pitch_y);
        RGB c[3];
        float l[3];

        for (int i = 0; i < 3; i++) {
            c[i] = load_rgb(src, src_pitch, x - 1 + i, y + row, p);
            l[i] = luma(c[i]);
        }
        out_y[x] = to_p010(64.0f + 876.0f * l[1], 4, 1019);
        if (x + 1 < p.width) {
            RGB c2 = load_rgb(src, src_pitch, x + 1, y + row, p);
            out_y[x + 1] = to_p010(64.0f + 876.0f * luma(c2), 4, 1019);
        }

        for (int i = 0; i < 3; i++) {
            float w = i == 1 ? 0.5f : 0.25f;
            cb += w * (c[i].b - l[i]) / 1.8814f;
            cr += w * (c[i].r - l[i]) / 1.4746f;
        }
    }

    unsigned short *out_uv = (unsigned short *)(dst_uv + by * pitch_uv) + 2 * bx;
    out_uv[0] = to_p010(512.0f + 896.0f * 0.5f * cb, 4, 1019);
    out_uv[1] = to_p010(512.0f + 896.0f * 0.5f * cr, 4, 1019);
}

}
