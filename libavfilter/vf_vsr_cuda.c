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
 * NVIDIA RTX Video Super Resolution on CUDA frames.
 *
 * The RTX Video SDK is not linked. NVEncNVSDKNGX.dll, which wraps it with a
 * CUDA device pointer API, is loaded at runtime; it needs nvngx_vsr.dll and
 * cudart64_12.dll next to it. VSR reads and writes 8-bit RGBA.
 */

#include "config.h"

#include "compat/w32dlfcn.h"

#include "libavutil/common.h"
#include "libavutil/cuda_check.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "scale_eval.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

/* The C API of NVEncNVSDKNGX.dll. The functions return 0 on success. */
#define NGX_DLL_NAME "NVEncNVSDKNGX.dll"
#define NGX_FEATURE_VSR 1

typedef struct NGXRect {
    int left, top, right, bottom;
} NGXRect;

typedef struct NGXVSRParams {
    int quality;
    float strength;
} NGXVSRParams;

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

/* NVSDK_NGX_VSR_QualityLevel values. */
enum VSRQuality {
    VSR_BICUBIC          = 0,
    VSR_LOW              = 1,
    VSR_MEDIUM           = 2,
    VSR_HIGH             = 3,
    VSR_ULTRA            = 4,
    VSR_DENOISE_LOW      = 8,
    VSR_DENOISE_ULTRA    = 11,
    VSR_DEBLUR_LOW       = 12,
    VSR_DEBLUR_ULTRA     = 15,
    VSR_HIGHBITRATE_LOW  = 16,
    VSR_HIGHBITRATE_ULTRA = 19,
    VSR_STREAMING_MEDIUM = 21,
    VSR_STREAMING_ULTRA  = 23,
};

typedef struct VSRCUDAContext {
    const AVClass *class;

    char *w_expr, *h_expr;
    int force_original_aspect_ratio;
    int force_divisible_by;
    int quality;
    float strength;

    AVCUDADeviceContext *hwctx;
    AVBufferRef *frames_ctx;
    CUstream cu_stream;
    CUevent ev_in, ev_out;

    HMODULE dll;
    ngx_create_fn ngx_create;
    ngx_init_fn ngx_init;
    ngx_delete_fn ngx_delete;
    ngx_proc_frame_fn ngx_proc_frame;
    void *ngx;
} VSRCUDAContext;

static av_cold int vsr_init(AVFilterContext *ctx)
{
    VSRCUDAContext *s = ctx->priv;

    if ((s->quality >= 5 && s->quality <= 7) || s->quality == 20 ||
        s->quality == 22) {
        av_log(ctx, AV_LOG_ERROR, "quality %d is reserved by NVIDIA.\n",
               s->quality);
        return AVERROR(EINVAL);
    }
    return 0;
}

static av_cold void vsr_uninit(AVFilterContext *ctx)
{
    VSRCUDAContext *s = ctx->priv;

    if (s->hwctx) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        CHECK_CU(cu->cuCtxSynchronize());
        if (s->ngx)
            s->ngx_delete(s->ngx);
        if (s->ev_in)
            CHECK_CU(cu->cuEventDestroy(s->ev_in));
        if (s->ev_out)
            CHECK_CU(cu->cuEventDestroy(s->ev_out));
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }
    s->ngx = NULL;
    s->ev_in = s->ev_out = NULL;

    /* Unloading also resets the DLL's once-per-process NGX initialization. */
    if (s->dll)
        dlclose(s->dll);
    s->dll = NULL;

    av_buffer_unref(&s->frames_ctx);
}

static av_cold int load_ngx(AVFilterContext *ctx)
{
    VSRCUDAContext *s = ctx->priv;
    int err;

    s->dll = dlopen(NGX_DLL_NAME, 0);
    if (!s->dll) {
        av_log(ctx, AV_LOG_ERROR, "Could not load " NGX_DLL_NAME ". Place it, "
               "nvngx_vsr.dll and cudart64_12.dll next to ffmpeg.\n");
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

    err = s->ngx_create(&s->ngx, NGX_FEATURE_VSR);
    if (err || !s->ngx) {
        av_log(ctx, AV_LOG_ERROR, "Could not create VSR (error %d).\n", err);
        s->ngx = NULL;
        return AVERROR_EXTERNAL;
    }

    /* The DLL copies frames on the context's legacy default stream, so its
     * evaluation goes there too. The caller has the context current. */
    err = s->ngx_init(s->ngx, -1, s->hwctx->cuda_ctx, NULL);
    if (err) {
        av_log(ctx, AV_LOG_ERROR, "Could not initialize VSR (error %d). "
               "It requires nvngx_vsr.dll and an RTX GPU.\n", err);
        return AVERROR_EXTERNAL;
    }
    return 0;
}

static av_cold int vsr_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    FilterLink *inl  = ff_filter_link(inlink);
    FilterLink *outl = ff_filter_link(outlink);
    VSRCUDAContext *s = ctx->priv;
    AVHWFramesContext *in_frames;
    AVHWFramesContext *out_frames;
    CudaFunctions *cu;
    CUcontext dummy;
    int w, h, ret;

    if (!inl->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames = (AVHWFramesContext *)inl->hw_frames_ctx->data;
    if (in_frames->sw_format != AV_PIX_FMT_RGBA &&
        in_frames->sw_format != AV_PIX_FMT_RGB0) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format %s; VSR needs "
               "rgba or rgb0.\n", av_get_pix_fmt_name(in_frames->sw_format));
        return AVERROR(EINVAL);
    }

    if ((ret = ff_scale_eval_dimensions(s, s->w_expr, s->h_expr,
                                        inlink, outlink, &w, &h)) < 0)
        return ret;
    if ((ret = ff_scale_adjust_dimensions(inlink, &w, &h,
                                          s->force_original_aspect_ratio,
                                          s->force_divisible_by, 1.0)) < 0)
        return ret;

    if (s->quality >= VSR_DENOISE_LOW && s->quality <= VSR_DEBLUR_ULTRA &&
        (w != inlink->w || h != inlink->h)) {
        av_log(ctx, AV_LOG_ERROR, "quality %d (denoise or deblur) does not "
               "scale; the output size must equal the input size (%dx%d).\n",
               s->quality, inlink->w, inlink->h);
        return AVERROR(EINVAL);
    }

    vsr_uninit(ctx);
    s->hwctx     = in_frames->device_ctx->hwctx;
    s->cu_stream = s->hwctx->stream;
    cu = s->hwctx->internal->cuda_dl;

    outlink->w = w;
    outlink->h = h;
    if (inlink->sample_aspect_ratio.num)
        outlink->sample_aspect_ratio = av_mul_q((AVRational){ h * inlink->w,
                                                              w * inlink->h },
                                                inlink->sample_aspect_ratio);
    else
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    s->frames_ctx = av_hwframe_ctx_alloc(in_frames->device_ref);
    if (!s->frames_ctx)
        return AVERROR(ENOMEM);
    out_frames = (AVHWFramesContext *)s->frames_ctx->data;
    out_frames->format    = AV_PIX_FMT_CUDA;
    out_frames->sw_format = in_frames->sw_format;
    out_frames->width     = w;
    out_frames->height    = h;
    ret = av_hwframe_ctx_init(s->frames_ctx);
    if (ret < 0)
        return ret;
    av_buffer_unref(&outl->hw_frames_ctx);
    outl->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!outl->hw_frames_ctx)
        return AVERROR(ENOMEM);

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;

    if (s->cu_stream) {
        ret = CHECK_CU(cu->cuEventCreate(&s->ev_in, CU_EVENT_DISABLE_TIMING));
        if (ret >= 0)
            ret = CHECK_CU(cu->cuEventCreate(&s->ev_out, CU_EVENT_DISABLE_TIMING));
    }
    if (ret >= 0)
        ret = load_ngx(ctx);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        return ret;

    av_log(ctx, AV_LOG_VERBOSE, "%dx%d -> %dx%d %s, quality %d, strength %.2f\n",
           inlink->w, inlink->h, w, h, av_get_pix_fmt_name(in_frames->sw_format),
           s->quality, s->strength);
    return 0;
}

static int vsr_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    VSRCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    const NGXRect rect_src = { 0, 0, inlink->w, inlink->h };
    const NGXRect rect_dst = { 0, 0, outlink->w, outlink->h };
    const NGXVSRParams params = {
        .quality  = s->quality,
        .strength = s->strength,
    };
    AVFrame *out = NULL;
    CUcontext dummy;
    int ret, err;

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
        err = s->ngx_proc_frame(s->ngx, &rect_dst, &rect_src, &params,
                                in->data[0], in->linesize[0],
                                out->data[0], out->linesize[0], 4, 4);
        if (err) {
            av_log(ctx, AV_LOG_ERROR, "VSR failed (error %d).\n", err);
            ret = AVERROR_EXTERNAL;
        }
    }

    if (ret >= 0 && s->cu_stream) {
        ret = CHECK_CU(cu->cuEventRecord(s->ev_out, NULL));
        if (ret >= 0)
            ret = CHECK_CU(cu->cuStreamWaitEvent(s->cu_stream, s->ev_out, 0));
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        goto fail;
    out->sample_aspect_ratio = outlink->sample_aspect_ratio;

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);

fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

#define OFFSET(x) offsetof(VSRCUDAContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption vsr_cuda_options[] = {
    { "w", "Output video width",  OFFSET(w_expr), AV_OPT_TYPE_STRING, { .str = "iw" }, .flags = FLAGS },
    { "h", "Output video height", OFFSET(h_expr), AV_OPT_TYPE_STRING, { .str = "ih" }, .flags = FLAGS },
    { "force_original_aspect_ratio", "decrease or increase w/h if necessary to keep the original AR", OFFSET(force_original_aspect_ratio), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, SCALE_FORCE_OAR_NB-1, FLAGS, .unit = "force_oar" },
        { "disable",  NULL, 0, AV_OPT_TYPE_CONST, { .i64 = SCALE_FORCE_OAR_DISABLE  }, 0, 0, FLAGS, .unit = "force_oar" },
        { "decrease", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = SCALE_FORCE_OAR_DECREASE }, 0, 0, FLAGS, .unit = "force_oar" },
        { "increase", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = SCALE_FORCE_OAR_INCREASE }, 0, 0, FLAGS, .unit = "force_oar" },
    { "force_divisible_by", "enforce that the output resolution is divisible by a defined integer when force_original_aspect_ratio is used", OFFSET(force_divisible_by), AV_OPT_TYPE_INT, { .i64 = 1 }, 1, 256, FLAGS },
    { "quality", "VSR mode (NVIDIA's QualityLevel)", OFFSET(quality), AV_OPT_TYPE_INT, { .i64 = VSR_LOW }, 0, VSR_STREAMING_ULTRA, FLAGS, .unit = "quality" },
        { "bicubic",           "bicubic, not AI",                   0, AV_OPT_TYPE_CONST, { .i64 = VSR_BICUBIC           }, 0, 0, FLAGS, .unit = "quality" },
        { "low",               NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_LOW               }, 0, 0, FLAGS, .unit = "quality" },
        { "medium",            NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_MEDIUM            }, 0, 0, FLAGS, .unit = "quality" },
        { "high",              NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_HIGH              }, 0, 0, FLAGS, .unit = "quality" },
        { "ultra",             NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_ULTRA             }, 0, 0, FLAGS, .unit = "quality" },
        { "denoise_low",       "denoise without scaling",           0, AV_OPT_TYPE_CONST, { .i64 = VSR_DENOISE_LOW       }, 0, 0, FLAGS, .unit = "quality" },
        { "denoise_medium",    NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_DENOISE_LOW + 1   }, 0, 0, FLAGS, .unit = "quality" },
        { "denoise_high",      NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_DENOISE_LOW + 2   }, 0, 0, FLAGS, .unit = "quality" },
        { "denoise_ultra",     NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_DENOISE_ULTRA     }, 0, 0, FLAGS, .unit = "quality" },
        { "deblur_low",        "deblur without scaling",            0, AV_OPT_TYPE_CONST, { .i64 = VSR_DEBLUR_LOW        }, 0, 0, FLAGS, .unit = "quality" },
        { "deblur_medium",     NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_DEBLUR_LOW + 1    }, 0, 0, FLAGS, .unit = "quality" },
        { "deblur_high",       NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_DEBLUR_LOW + 2    }, 0, 0, FLAGS, .unit = "quality" },
        { "deblur_ultra",      NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_DEBLUR_ULTRA      }, 0, 0, FLAGS, .unit = "quality" },
        { "highbitrate_low",   "for high bitrate sources",          0, AV_OPT_TYPE_CONST, { .i64 = VSR_HIGHBITRATE_LOW   }, 0, 0, FLAGS, .unit = "quality" },
        { "highbitrate_medium", NULL,                               0, AV_OPT_TYPE_CONST, { .i64 = VSR_HIGHBITRATE_LOW + 1 }, 0, 0, FLAGS, .unit = "quality" },
        { "highbitrate_high",  NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_HIGHBITRATE_LOW + 2 }, 0, 0, FLAGS, .unit = "quality" },
        { "highbitrate_ultra", NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_HIGHBITRATE_ULTRA }, 0, 0, FLAGS, .unit = "quality" },
        { "streaming_medium",  "for streamed sources",              0, AV_OPT_TYPE_CONST, { .i64 = VSR_STREAMING_MEDIUM  }, 0, 0, FLAGS, .unit = "quality" },
        { "streaming_ultra",   NULL,                                0, AV_OPT_TYPE_CONST, { .i64 = VSR_STREAMING_ULTRA   }, 0, 0, FLAGS, .unit = "quality" },
    { "strength", "strength of the effect", OFFSET(strength), AV_OPT_TYPE_FLOAT, { .dbl = 1.0 }, 0.0, 1.0, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(vsr_cuda);

static const AVFilterPad vsr_cuda_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = vsr_filter_frame,
    },
};

static const AVFilterPad vsr_cuda_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = vsr_config_props,
    },
};

const FFFilter ff_vf_vsr_cuda = {
    .p.name         = "vsr_cuda",
    .p.description  = NULL_IF_CONFIG_SMALL("NVIDIA RTX Video Super Resolution"),
    .p.priv_class   = &vsr_cuda_class,
    .priv_size      = sizeof(VSRCUDAContext),
    .init           = vsr_init,
    .uninit         = vsr_uninit,
    FILTER_INPUTS(vsr_cuda_inputs),
    FILTER_OUTPUTS(vsr_cuda_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
