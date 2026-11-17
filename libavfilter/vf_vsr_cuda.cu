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

/*
 * Conversions between the formats vsr_cuda accepts and the 8-bit RGBA that
 * RTX Video Super Resolution works on.
 */

extern "C" {

/* Must match VSRConvParams in vf_vsr_cuda.c. */
struct VSRConvParams {
    float m[3][3];       /* YUV->RGB (normalised Y, Cb, Cr) or RGB->YUV */
    float y_off, y_scale; /* Y code = y_off + y_scale * Y; and the inverse */
    float c_off, c_scale; /* the same for Cb and Cr */
    float cx, cy;        /* chroma sample position within a 2x2 block */
    int width, height;   /* luma size */
    int high_bit;        /* P010 instead of NV12 */
};

/* Only what FFmpeg's compat/cuda/cuda_runtime.h provides, for cuda_llvm. */
static __device__ int clampi(int v, int lo, int hi)
{
    return min(max(v, lo), hi);
}

static __device__ int round_int(float v)
{
    return (int)rintf(v);
}

static __device__ float load_y(const unsigned char *plane, int pitch,
                               int x, int y, int high_bit)
{
    const unsigned char *row = plane + y * pitch;
    return high_bit ? (float)(((const unsigned short *)row)[x] >> 6) : (float)row[x];
}

static __device__ float2 load_uv(const unsigned char *plane, int pitch,
                                 int x, int y, int high_bit)
{
    const unsigned char *row = plane + y * pitch;
    if (high_bit) {
        const unsigned short *p = (const unsigned short *)row + 2 * x;
        return make_float2((float)(p[0] >> 6), (float)(p[1] >> 6));
    }
    return make_float2((float)row[2 * x], (float)row[2 * x + 1]);
}

static __device__ unsigned char to_u8(float v)
{
    return (unsigned char)clampi(round_int(v * 255.0f), 0, 255);
}

/* NV12/P010 to RGBA, one thread per pixel, bilinear chroma. */
__global__ void vsr_yuv_to_rgba(const unsigned char *src_y, int pitch_y,
                                const unsigned char *src_uv, int pitch_uv,
                                unsigned char *dst, int dst_pitch,
                                VSRConvParams p)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int cw = (p.width + 1) >> 1, ch = (p.height + 1) >> 1;

    if (x >= p.width || y >= p.height)
        return;

    /* Chroma sample i sits at luma position 2 * i + cx. */
    float fx = (x - p.cx) * 0.5f, fy = (y - p.cy) * 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float wx = fx - x0, wy = fy - y0;
    int xa = clampi(x0, 0, cw - 1), xb = clampi(x0 + 1, 0, cw - 1);
    int ya = clampi(y0, 0, ch - 1), yb = clampi(y0 + 1, 0, ch - 1);
    float2 c00 = load_uv(src_uv, pitch_uv, xa, ya, p.high_bit);
    float2 c10 = load_uv(src_uv, pitch_uv, xb, ya, p.high_bit);
    float2 c01 = load_uv(src_uv, pitch_uv, xa, yb, p.high_bit);
    float2 c11 = load_uv(src_uv, pitch_uv, xb, yb, p.high_bit);
    float u = (1.0f - wy) * ((1.0f - wx) * c00.x + wx * c10.x) +
                      wy  * ((1.0f - wx) * c01.x + wx * c11.x);
    float v = (1.0f - wy) * ((1.0f - wx) * c00.y + wx * c10.y) +
                      wy  * ((1.0f - wx) * c01.y + wx * c11.y);

    float Y  = (load_y(src_y, pitch_y, x, y, p.high_bit) - p.y_off) * p.y_scale;
    float cb = (u - p.c_off) * p.c_scale;
    float cr = (v - p.c_off) * p.c_scale;

    unsigned char *out = dst + y * dst_pitch + 4 * x;
    out[0] = to_u8(p.m[0][0] * Y + p.m[0][1] * cb + p.m[0][2] * cr);
    out[1] = to_u8(p.m[1][0] * Y + p.m[1][1] * cb + p.m[1][2] * cr);
    out[2] = to_u8(p.m[2][0] * Y + p.m[2][1] * cb + p.m[2][2] * cr);
    out[3] = 255;
}

static __device__ float4 load_rgb(const unsigned char *src, int pitch,
                                  int x, int y, const VSRConvParams &p)
{
    const unsigned char *px = src + clampi(y, 0, p.height - 1) * pitch +
                              4 * clampi(x, 0, p.width - 1);
    return make_float4(px[0] / 255.0f, px[1] / 255.0f, px[2] / 255.0f, 1.0f);
}

static __device__ void store_code(unsigned char *row, int i, float code, int high_bit)
{
    if (high_bit)
        ((unsigned short *)row)[i] = (unsigned short)(clampi(round_int(code), 0, 1023) << 6);
    else
        row[i] = (unsigned char)clampi(round_int(code), 0, 255);
}

/* Tent weight of a luma sample dx away from a chroma sample (2:1). */
static __device__ float tent(float dx)
{
    return max(0.0f, 1.0f - fabsf(dx) * 0.5f);
}

/* RGBA to NV12/P010, one thread per 2x2 block. */
__global__ void vsr_rgba_to_yuv(const unsigned char *src, int src_pitch,
                                unsigned char *dst_y, int pitch_y,
                                unsigned char *dst_uv, int pitch_uv,
                                VSRConvParams p)
{
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int x = 2 * bx, y = 2 * by;

    if (x >= p.width || y >= p.height)
        return;

    for (int j = 0; j < 2 && y + j < p.height; j++) {
        unsigned char *row = dst_y + (y + j) * pitch_y;
        for (int i = 0; i < 2 && x + i < p.width; i++) {
            float4 c = load_rgb(src, src_pitch, x + i, y + j, p);
            float Y = p.m[0][0] * c.x + p.m[0][1] * c.y + p.m[0][2] * c.z;
            store_code(row, x + i, p.y_off + p.y_scale * Y, p.high_bit);
        }
    }

    /* Chroma at (x + cx, y + cy), filtered from the 4x4 luma neighbourhood. */
    float cb = 0.0f, cr = 0.0f, wsum = 0.0f;
    for (int j = -1; j <= 2; j++) {
        float wy = tent(j - p.cy);
        if (wy == 0.0f)
            continue;
        for (int i = -1; i <= 2; i++) {
            float w = wy * tent(i - p.cx);
            if (w == 0.0f)
                continue;
            float4 c = load_rgb(src, src_pitch, x + i, y + j, p);
            cb   += w * (p.m[1][0] * c.x + p.m[1][1] * c.y + p.m[1][2] * c.z);
            cr   += w * (p.m[2][0] * c.x + p.m[2][1] * c.y + p.m[2][2] * c.z);
            wsum += w;
        }
    }
    unsigned char *row = dst_uv + by * pitch_uv;
    store_code(row, 2 * bx,     p.c_off + p.c_scale * cb / wsum, p.high_bit);
    store_code(row, 2 * bx + 1, p.c_off + p.c_scale * cr / wsum, p.high_bit);
}

/* RGBA <-> BGRA (and the 0 variants): swap R and B, one thread per pixel. */
__global__ void vsr_swap_rb(const unsigned char *src, int src_pitch,
                            unsigned char *dst, int dst_pitch,
                            VSRConvParams p)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= p.width || y >= p.height)
        return;

    uchar4 c = *(const uchar4 *)(src + y * src_pitch + 4 * x);
    *(uchar4 *)(dst + y * dst_pitch + 4 * x) = make_uchar4(c.z, c.y, c.x, c.w);
}

}
