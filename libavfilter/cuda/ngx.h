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

/**
 * @file
 * The C API of NVEncNVSDKNGX.dll, which wraps the RTX Video SDK (VSR and
 * TrueHDR) with a CUDA device pointer API. It is loaded at runtime.
 */

#ifndef AVFILTER_CUDA_NGX_H
#define AVFILTER_CUDA_NGX_H

#include <stdint.h>

#include "compat/w32dlfcn.h"

#define FF_NGX_DLL_NAME "NVEncNVSDKNGX.dll"

enum FFNGXFeature {
    FF_NGX_FEATURE_VSR     = 1,
    FF_NGX_FEATURE_TRUEHDR = 2,
};

typedef struct FFNGXRect {
    int left, top, right, bottom;
} FFNGXRect;

/* The params argument of proc_frame, per feature. */
typedef struct FFNGXVSRParams {
    int quality;         /* NVSDK_NGX_VSR_QualityLevel */
    float strength;
} FFNGXVSRParams;

typedef struct FFNGXTrueHDRParams {
    uint32_t contrast;
    uint32_t saturation;
    uint32_t middle_gray;
    uint32_t max_luminance;
} FFNGXTrueHDRParams;

/* The functions return 0 on success. */
typedef int  (__stdcall *ff_ngx_create_fn)(void **handle, int feature);
typedef int  (__stdcall *ff_ngx_init_fn)(void *handle, int cuda_device_ordinal,
                                         void *cu_context, void *cu_stream);
typedef void (__stdcall *ff_ngx_delete_fn)(void *handle);
typedef int  (__stdcall *ff_ngx_proc_frame_fn)(void *handle,
                                               const FFNGXRect *rect_dst,
                                               const FFNGXRect *rect_src,
                                               const void *params,
                                               const void *src, int src_pitch,
                                               void *dst, int dst_pitch,
                                               int src_bytes_per_pixel,
                                               int dst_bytes_per_pixel);

typedef struct FFNGX {
    HMODULE dll;
    void *handle;
    ff_ngx_create_fn     create;
    ff_ngx_init_fn       init;
    ff_ngx_delete_fn     destroy;
    ff_ngx_proc_frame_fn proc_frame;
} FFNGX;

/**
 * Load the DLL, then create and initialize a feature on a CUDA context,
 * which must be current. The DLL works on the context's legacy default
 * stream. name ("VSR") and feature_dll ("nvngx_vsr.dll") are for messages.
 * On failure, ff_ngx_close() must still be called.
 */
int ff_ngx_open(void *log_ctx, FFNGX *ngx, enum FFNGXFeature feature,
                const char *name, const char *feature_dll, void *cu_context);

/**
 * Delete the feature, with its CUDA context current, and unload the DLL.
 * Unloading also resets the DLL's once-per-process NGX initialization.
 * Safe to call on a zeroed or closed FFNGX.
 */
void ff_ngx_close(FFNGX *ngx);

#endif /* AVFILTER_CUDA_NGX_H */
