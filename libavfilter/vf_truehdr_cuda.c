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
 * NVIDIA RTX Video TrueHDR (SDR to HDR10) on CUDA frames.
 *
 * The RTX Video SDK is not linked. NVEncNVSDKNGX.dll, which wraps it with a
 * CUDA device pointer API, is loaded at runtime; it needs nvngx_truehdr.dll
 * and cudart64_12.dll next to it. TrueHDR writes half float linear light
 * RGBA, which a CUDA kernel converts to BT.2020 PQ P010.
 */

#include "config.h"

#include "compat/w32dlfcn.h"

#include "libavutil/common.h"
#include "libavutil/cuda_check.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/mastering_display_metadata.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"

#include "cuda/load_helper.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#define BLOCKX 32
#define BLOCKY 8

/* TrueHDR's linear output has 1.0 at 100 nits. */
#define TRUEHDR_LINEAR_NITS 100.0f

/* The C API of NVEncNVSDKNGX.dll. The functions return 0 on success. */
#define NGX_DLL_NAME "NVEncNVSDKNGX.dll"
#define NGX_FEATURE_TRUEHDR 2

typedef struct NGXRect {
    int left, top, right, bottom;
} NGXRect;

typedef struct NGXTrueHDRParams {
    uint32_t contrast;
    uint32_t saturation;
    uint32_t middle_gray;
    uint32_t max_luminance;
} NGXTrueHDRParams;

typedef int  (__stdcall *ngx_create_fn)(void **handle, int feature);
typedef int  (__stdcall *ngx_init_fn)(void *handle, int cuda_device_ordinal,
                                      void *cu_context, void *cu_stream);
typedef void (__stdcall *ngx_delete_fn)(void *handle);
typedef int  (__stdcall *ngx_proc_frame_fn)(void *handle,
                                            const NGXRect *rect_dst,
                                            const NGXRect *rect_src,
                                            const void *params,
                                            const void *src, int src_pitch,
                                            void *dst, int dst_pitch,
                                            int src_bytes_per_pixel,
                                            int dst_bytes_per_pixel);

/* Must match TrueHDRConvParams in vf_truehdr_cuda.cu. */
typedef struct TrueHDRConvParams {
    float m[3][3];
    float scale;
    int width, height;
} TrueHDRConvParams;

typedef struct TrueHDRCUDAContext {
    const AVClass *class;

    int contrast;
    int saturation;
    int middle_gray;
    int max_luma;
    int metadata;

    AVCUDADeviceContext *hwctx;
    AVBufferRef *frames_ctx;
    CUstream cu_stream;
    CUmodule cu_module;
    CUfunction cu_convert;
    CUevent ev_in, ev_out;

    CUdeviceptr hdr_buf;
    size_t hdr_pitch;
    TrueHDRConvParams conv;

    HMODULE dll;
    ngx_create_fn ngx_create;
    ngx_init_fn ngx_init;
    ngx_delete_fn ngx_delete;
    ngx_proc_frame_fn ngx_proc_frame;
    void *ngx;
} TrueHDRCUDAContext;

static av_cold void truehdr_uninit(AVFilterContext *ctx)
{
    TrueHDRCUDAContext *s = ctx->priv;

    if (s->hwctx) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        CHECK_CU(cu->cuCtxSynchronize());
        if (s->ngx)
            s->ngx_delete(s->ngx);
        if (s->hdr_buf)
            CHECK_CU(cu->cuMemFree(s->hdr_buf));
        if (s->ev_in)
            CHECK_CU(cu->cuEventDestroy(s->ev_in));
        if (s->ev_out)
            CHECK_CU(cu->cuEventDestroy(s->ev_out));
        if (s->cu_module)
            CHECK_CU(cu->cuModuleUnload(s->cu_module));
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }
    s->ngx = NULL;
    s->hdr_buf = 0;
    s->ev_in = s->ev_out = NULL;
    s->cu_module = NULL;

    /* Unloading also resets the DLL's once-per-process NGX initialization. */
    if (s->dll)
        dlclose(s->dll);
    s->dll = NULL;

    av_buffer_unref(&s->frames_ctx);
}

static int truehdr_query_formats(const AVFilterContext *ctx,
                                 AVFilterFormatsConfig **cfg_in,
                                 AVFilterFormatsConfig **cfg_out)
{
    static const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE };
    int ret;

    if ((ret = ff_set_common_formats_from_list2(ctx, cfg_in, cfg_out, pix_fmts)) < 0)
        return ret;

    /* The input is RGB; the output is always limited range BT.2020 YUV. */
    if ((ret = ff_formats_ref(ff_all_color_spaces(), &cfg_in[0]->color_spaces)) < 0 ||
        (ret = ff_formats_ref(ff_all_color_ranges(), &cfg_in[0]->color_ranges)) < 0)
        return ret;
    if ((ret = ff_formats_ref(ff_make_formats_list_singleton(AVCOL_SPC_BT2020_NCL),
                              &cfg_out[0]->color_spaces)) < 0 ||
        (ret = ff_formats_ref(ff_make_formats_list_singleton(AVCOL_RANGE_MPEG),
                              &cfg_out[0]->color_ranges)) < 0)
        return ret;
    return 0;
}

static av_cold int load_ngx(AVFilterContext *ctx)
{
    TrueHDRCUDAContext *s = ctx->priv;
    int err;

    s->dll = dlopen(NGX_DLL_NAME, 0);
    if (!s->dll) {
        av_log(ctx, AV_LOG_ERROR, "Could not load " NGX_DLL_NAME ". Place it, "
               "nvngx_truehdr.dll and cudart64_12.dll next to ffmpeg.\n");
        return AVERROR_EXTERNAL;
    }

#define LOAD(field, type, name) do {                                        \
        s->field = (type)dlsym(s->dll, name);                               \
        if (!s->field) {                                                    \
            av_log(ctx, AV_LOG_ERROR, NGX_DLL_NAME " has no %s.\n", name); \
            return AVERROR_EXTERNAL;                                        \
        }                                                                   \
    } while (0)
    LOAD(ngx_create,     ngx_create_fn,     "NVEncNVSDKNGXCreate");
    LOAD(ngx_init,       ngx_init_fn,       "NVEncNVSDKNGXInit");
    LOAD(ngx_delete,     ngx_delete_fn,     "NVEncNVSDKNGXDelete");
    LOAD(ngx_proc_frame, ngx_proc_frame_fn, "NVEncNVSDKNGXProcFrame");
#undef LOAD

    err = s->ngx_create(&s->ngx, NGX_FEATURE_TRUEHDR);
    if (err || !s->ngx) {
        av_log(ctx, AV_LOG_ERROR, "Could not create TrueHDR (error %d).\n", err);
        s->ngx = NULL;
        return AVERROR_EXTERNAL;
    }

    /* The DLL copies frames on the context's legacy default stream, so its
     * evaluation goes there too. The caller has the context current. */
    err = s->ngx_init(s->ngx, -1, s->hwctx->cuda_ctx, NULL);
    if (err) {
        av_log(ctx, AV_LOG_ERROR, "Could not initialize TrueHDR (error %d). "
               "It requires nvngx_truehdr.dll and an RTX GPU.\n", err);
        return AVERROR_EXTERNAL;
    }
    return 0;
}

/* Map linear light in the input primaries to BT.2020 (ITU-R BT.2087). */
static void set_conv_params(TrueHDRCUDAContext *s, enum AVColorPrimaries prim,
                            int width, int height)
{
    static const float bt709_to_bt2020[3][3] = {
        { 0.627404f, 0.329283f, 0.043313f },
        { 0.069097f, 0.919540f, 0.011362f },
        { 0.016391f, 0.088013f, 0.895595f },
    };
    static const float identity[3][3] = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f },
    };

    memcpy(s->conv.m, prim == AVCOL_PRI_BT2020 ? identity : bt709_to_bt2020,
           sizeof(s->conv.m));
    s->conv.scale  = TRUEHDR_LINEAR_NITS / 10000.0f;
    s->conv.width  = width;
    s->conv.height = height;
}

static int add_hdr_metadata(AVFilterContext *ctx, AVFrameSideData ***sd, int *nb_sd)
{
    TrueHDRCUDAContext *s = ctx->priv;
    AVFrameSideData *data;
    AVMasteringDisplayMetadata *mdm;
    AVContentLightMetadata *cll;
    static const double prim[3][2] = { { 0.708, 0.292 }, { 0.170, 0.797 },
                                       { 0.131, 0.046 } };

    data = av_frame_side_data_new(sd, nb_sd, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA,
                                  sizeof(*mdm), AV_FRAME_SIDE_DATA_FLAG_REPLACE);
    if (!data)
        return AVERROR(ENOMEM);
    mdm = (AVMasteringDisplayMetadata *)data->data;
    memset(mdm, 0, sizeof(*mdm));
    for (int i = 0; i < 3; i++) {
        mdm->display_primaries[i][0] = av_d2q(prim[i][0], 50000);
        mdm->display_primaries[i][1] = av_d2q(prim[i][1], 50000);
    }
    mdm->white_point[0] = av_make_q(15635, 50000);
    mdm->white_point[1] = av_make_q(16450, 50000);
    mdm->max_luminance  = av_make_q(s->max_luma, 1);
    mdm->min_luminance  = av_make_q(1, 10000);
    mdm->has_primaries  = 1;
    mdm->has_luminance  = 1;

    data = av_frame_side_data_new(sd, nb_sd, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL,
                                  sizeof(*cll), AV_FRAME_SIDE_DATA_FLAG_REPLACE);
    if (!data)
        return AVERROR(ENOMEM);
    cll = (AVContentLightMetadata *)data->data;
    cll->MaxCLL  = s->max_luma;
    cll->MaxFALL = 0;
    return 0;
}

static av_cold int truehdr_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    FilterLink *inl  = ff_filter_link(inlink);
    FilterLink *outl = ff_filter_link(outlink);
    TrueHDRCUDAContext *s = ctx->priv;
    AVHWFramesContext *in_frames;
    AVHWFramesContext *out_frames;
    CudaFunctions *cu;
    CUcontext dummy;
    int ret;

    extern const unsigned char ff_vf_truehdr_cuda_ptx_data[];
    extern const unsigned int ff_vf_truehdr_cuda_ptx_len;

    if (!inl->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames = (AVHWFramesContext *)inl->hw_frames_ctx->data;
    if (in_frames->sw_format != AV_PIX_FMT_RGBA &&
        in_frames->sw_format != AV_PIX_FMT_RGB0) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format %s; TrueHDR needs "
               "rgba or rgb0.\n", av_get_pix_fmt_name(in_frames->sw_format));
        return AVERROR(EINVAL);
    }
    if ((inlink->w | inlink->h) & 1) {
        av_log(ctx, AV_LOG_ERROR, "The video size must be even.\n");
        return AVERROR(EINVAL);
    }

    truehdr_uninit(ctx);
    s->hwctx     = in_frames->device_ctx->hwctx;
    s->cu_stream = s->hwctx->stream;
    cu = s->hwctx->internal->cuda_dl;

    outlink->w = inlink->w;
    outlink->h = inlink->h;
    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    s->frames_ctx = av_hwframe_ctx_alloc(in_frames->device_ref);
    if (!s->frames_ctx)
        return AVERROR(ENOMEM);
    out_frames = (AVHWFramesContext *)s->frames_ctx->data;
    out_frames->format    = AV_PIX_FMT_CUDA;
    out_frames->sw_format = AV_PIX_FMT_P010;
    out_frames->width     = outlink->w;
    out_frames->height    = outlink->h;
    ret = av_hwframe_ctx_init(s->frames_ctx);
    if (ret < 0)
        return ret;
    av_buffer_unref(&outl->hw_frames_ctx);
    outl->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!outl->hw_frames_ctx)
        return AVERROR(ENOMEM);

    if (s->metadata) {
        ret = add_hdr_metadata(ctx, &outlink->side_data, &outlink->nb_side_data);
        if (ret < 0)
            return ret;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;

    ret = ff_cuda_load_module(ctx, s->hwctx, &s->cu_module,
                              ff_vf_truehdr_cuda_ptx_data,
                              ff_vf_truehdr_cuda_ptx_len);
    if (ret >= 0)
        ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_convert, s->cu_module,
                                               "truehdr_to_p010"));
    if (ret >= 0)
        ret = CHECK_CU(cu->cuMemAllocPitch(&s->hdr_buf, &s->hdr_pitch,
                                           outlink->w * 8, outlink->h, 16));
    if (ret >= 0 && s->cu_stream) {
        ret = CHECK_CU(cu->cuEventCreate(&s->ev_in, CU_EVENT_DISABLE_TIMING));
        if (ret >= 0)
            ret = CHECK_CU(cu->cuEventCreate(&s->ev_out, CU_EVENT_DISABLE_TIMING));
    }
    if (ret >= 0)
        ret = load_ngx(ctx);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
}

static int truehdr_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    TrueHDRCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    const NGXRect rect = { 0, 0, inlink->w, inlink->h };
    const NGXTrueHDRParams params = {
        .contrast      = s->contrast,
        .saturation    = s->saturation,
        .middle_gray   = s->middle_gray,
        .max_luminance = s->max_luma,
    };
    AVFrame *out = NULL;
    CUcontext dummy;
    int ret, err;

    if (in->color_trc == AVCOL_TRC_SMPTE2084 || in->color_trc == AVCOL_TRC_ARIB_STD_B67) {
        av_log(ctx, AV_LOG_ERROR, "TrueHDR requires SDR video.\n");
        av_frame_free(&in);
        return AVERROR(EINVAL);
    }

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    ret = av_hwframe_get_buffer(s->frames_ctx, out, 0);
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    /* The DLL works on the legacy default stream; order it after the
     * filter chain's stream and back again. */
    if (s->cu_stream) {
        ret = CHECK_CU(cu->cuEventRecord(s->ev_in, s->cu_stream));
        if (ret >= 0)
            ret = CHECK_CU(cu->cuStreamWaitEvent(NULL, s->ev_in, 0));
    }

    if (ret >= 0) {
        err = s->ngx_proc_frame(s->ngx, &rect, &rect, &params,
                                in->data[0], in->linesize[0],
                                (void *)(uintptr_t)s->hdr_buf, s->hdr_pitch,
                                4, 8);
        if (err) {
            av_log(ctx, AV_LOG_ERROR, "TrueHDR failed (error %d).\n", err);
            ret = AVERROR_EXTERNAL;
        }
    }

    if (ret >= 0 && s->cu_stream) {
        ret = CHECK_CU(cu->cuEventRecord(s->ev_out, NULL));
        if (ret >= 0)
            ret = CHECK_CU(cu->cuStreamWaitEvent(s->cu_stream, s->ev_out, 0));
    }

    if (ret >= 0) {
        int src_pitch = s->hdr_pitch, pitch_y = out->linesize[0],
            pitch_uv = out->linesize[1];
        void *args[] = { &s->hdr_buf, &src_pitch, &out->data[0], &pitch_y,
                         &out->data[1], &pitch_uv, &s->conv };

        set_conv_params(s, in->color_primaries, inlink->w, inlink->h);
        ret = CHECK_CU(cu->cuLaunchKernel(s->cu_convert,
                                          DIV_UP(inlink->w / 2, BLOCKX),
                                          DIV_UP(inlink->h / 2, BLOCKY), 1,
                                          BLOCKX, BLOCKY, 1, 0, s->cu_stream,
                                          args, NULL));
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        goto fail;
    out->color_primaries = AVCOL_PRI_BT2020;
    out->color_trc       = AVCOL_TRC_SMPTE2084;
    out->colorspace      = AVCOL_SPC_BT2020_NCL;
    out->color_range     = AVCOL_RANGE_MPEG;
    out->chroma_location = AVCHROMA_LOC_LEFT;
    av_frame_remove_side_data(out, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    av_frame_remove_side_data(out, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    if (s->metadata) {
        ret = add_hdr_metadata(ctx, &out->side_data, &out->nb_side_data);
        if (ret < 0)
            goto fail;
    }

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);

fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

#define OFFSET(x) offsetof(TrueHDRCUDAContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption truehdr_cuda_options[] = {
    { "contrast",    "difference between lights and darks", OFFSET(contrast),    AV_OPT_TYPE_INT,  { .i64 = 125 },  0,  200, FLAGS },
    { "saturation",  "color intensity",                     OFFSET(saturation),  AV_OPT_TYPE_INT,  { .i64 = 75 },   0,  200, FLAGS },
    { "middle_gray", "average brightness",                  OFFSET(middle_gray), AV_OPT_TYPE_INT,  { .i64 = 44 },  10,  100, FLAGS },
    { "max_luma",    "peak brightness in nits",             OFFSET(max_luma),    AV_OPT_TYPE_INT,  { .i64 = 1000 }, 400, 2000, FLAGS },
    { "metadata",    "attach HDR10 mastering display and light level metadata",
                                                            OFFSET(metadata),    AV_OPT_TYPE_BOOL, { .i64 = 1 },    0,    1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(truehdr_cuda);

static const AVFilterPad truehdr_cuda_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = truehdr_filter_frame,
    },
};

static const AVFilterPad truehdr_cuda_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = truehdr_config_props,
    },
};

const FFFilter ff_vf_truehdr_cuda = {
    .p.name         = "truehdr_cuda",
    .p.description  = NULL_IF_CONFIG_SMALL("NVIDIA RTX Video TrueHDR: SDR to HDR10"),
    .p.priv_class   = &truehdr_cuda_class,
    .priv_size      = sizeof(TrueHDRCUDAContext),
    .uninit         = truehdr_uninit,
    FILTER_INPUTS(truehdr_cuda_inputs),
    FILTER_OUTPUTS(truehdr_cuda_outputs),
    FILTER_QUERY_FUNC2(truehdr_query_formats),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
