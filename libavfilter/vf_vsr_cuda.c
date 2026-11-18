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
 * cudart64_12.dll next to it. VSR reads and writes 8-bit RGBA; other formats
 * are converted to and from it by CUDA kernels.
 */

#include "config.h"

#include "libavutil/common.h"
#include "libavutil/csp.h"
#include "libavutil/cuda_check.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "scale_eval.h"

#include "cuda/load_helper.h"
#include "cuda/ngx.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#define BLOCKX 32
#define BLOCKY 8

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

/* How a format gets to and from VSR's RGBA. */
enum VSRConv {
    CONV_NONE,  /* rgba, rgb0: used as is */
    CONV_SWAP,  /* bgra, bgr0: R and B swapped */
    CONV_YUV,   /* nv12, p010 */
};

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_RGBA, AV_PIX_FMT_RGB0, AV_PIX_FMT_BGRA, AV_PIX_FMT_BGR0,
    AV_PIX_FMT_NV12, AV_PIX_FMT_P010,
};

/* Must match VSRConvParams in vf_vsr_cuda.cu. */
typedef struct VSRConvParams {
    float m[3][3];
    float y_off, y_scale;
    float c_off, c_scale;
    float cx, cy;
    int width, height;
    int high_bit;
} VSRConvParams;

typedef struct VSRCUDAContext {
    const AVClass *class;

    char *w_expr, *h_expr;
    int force_original_aspect_ratio;
    int force_divisible_by;
    int quality;
    float strength;
    enum AVPixelFormat format;

    AVCUDADeviceContext *hwctx;
    AVBufferRef *frames_ctx;
    CUstream cu_stream;
    CUevent ev_in, ev_out;
    CUmodule cu_module;
    CUfunction cu_yuv_to_rgba, cu_rgba_to_yuv, cu_swap_rb;

    enum AVPixelFormat in_fmt, out_fmt;
    enum VSRConv in_conv, out_conv;
    VSRConvParams in_params, out_params;
    enum AVColorSpace out_csp;
    enum AVColorRange out_range;
    enum AVChromaLocation out_loc;

    /* RGBA buffers for VSR when the input or output is not RGBA. */
    CUdeviceptr rgba_in, rgba_out;
    size_t rgba_in_pitch, rgba_out_pitch;

    FFNGX ngx;
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
        ff_ngx_close(&s->ngx);
        if (s->rgba_in)
            CHECK_CU(cu->cuMemFree(s->rgba_in));
        if (s->rgba_out)
            CHECK_CU(cu->cuMemFree(s->rgba_out));
        if (s->ev_in)
            CHECK_CU(cu->cuEventDestroy(s->ev_in));
        if (s->ev_out)
            CHECK_CU(cu->cuEventDestroy(s->ev_out));
        if (s->cu_module)
            CHECK_CU(cu->cuModuleUnload(s->cu_module));
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }
    s->rgba_in = s->rgba_out = 0;
    s->ev_in = s->ev_out = NULL;
    s->cu_module = NULL;

    av_buffer_unref(&s->frames_ctx);
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    for (int i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static enum VSRConv format_conv(enum AVPixelFormat fmt)
{
    switch (fmt) {
    case AV_PIX_FMT_BGRA:
    case AV_PIX_FMT_BGR0: return CONV_SWAP;
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_P010: return CONV_YUV;
    default:              return CONV_NONE;
    }
}

static AVFilterFormats *yuv_color_spaces(void)
{
    AVFilterFormats *ret = NULL;

    for (int csp = 0; csp < AVCOL_SPC_NB; csp++) {
        if (csp == AVCOL_SPC_RGB || csp == AVCOL_SPC_RESERVED)
            continue;
        if (ff_add_format(&ret, csp) < 0)
            return NULL;
    }
    return ret;
}

/*
 * As in scale_cuda: the software format of CUDA frames is unknown during
 * negotiation, so the input and output colour properties are negotiated
 * independently, and configuration works out what the conversion does.
 */
static int vsr_query_formats(const AVFilterContext *ctx,
                             AVFilterFormatsConfig **cfg_in,
                             AVFilterFormatsConfig **cfg_out)
{
    const VSRCUDAContext *s = ctx->priv;
    static const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE };
    static const int rgb_ranges[] = { AVCOL_RANGE_JPEG, AVCOL_RANGE_UNSPECIFIED, -1 };
    static const int yuv_ranges[] = { AVCOL_RANGE_MPEG, AVCOL_RANGE_JPEG, -1 };
    /* -1: the output format follows the input's, unknown here. */
    const int out_conv = s->format != AV_PIX_FMT_NONE ? (int)format_conv(s->format) : -1;
    AVFilterFormats *formats;
    int ret;

    if ((ret = ff_set_common_formats_from_list2(ctx, cfg_in, cfg_out, pix_fmts)) < 0)
        return ret;

    if ((ret = ff_formats_ref(ff_all_color_spaces(), &cfg_in[0]->color_spaces)) < 0 ||
        (ret = ff_formats_ref(ff_all_color_ranges(), &cfg_in[0]->color_ranges)) < 0 ||
        (ret = ff_formats_ref(ff_all_chroma_locations(), &cfg_in[0]->chroma_locations)) < 0)
        return ret;

    if (out_conv == CONV_YUV)
        formats = yuv_color_spaces();
    else if (out_conv >= 0)
        formats = ff_make_formats_list_singleton(AVCOL_SPC_RGB);
    else
        formats = ff_all_color_spaces();
    if ((ret = ff_formats_ref(formats, &cfg_out[0]->color_spaces)) < 0)
        return ret;

    if (out_conv == CONV_YUV)
        formats = ff_make_format_list(yuv_ranges);
    else if (out_conv >= 0)
        formats = ff_make_format_list(rgb_ranges);
    else
        formats = ff_all_color_ranges();
    if ((ret = ff_formats_ref(formats, &cfg_out[0]->color_ranges)) < 0)
        return ret;

    formats = out_conv == CONV_YUV || out_conv < 0 ?
              ff_all_chroma_locations() :
              ff_make_formats_list_singleton(AVCHROMA_LOC_UNSPECIFIED);
    return ff_formats_ref(formats, &cfg_out[0]->chroma_locations);
}

/* Chroma sample position within a 2x2 luma block. */
static void chroma_position(enum AVChromaLocation loc, float *cx, float *cy)
{
    switch (loc) {
    case AVCHROMA_LOC_CENTER:     *cx = 0.5f; *cy = 0.5f; break;
    case AVCHROMA_LOC_TOPLEFT:    *cx = 0.0f; *cy = 0.0f; break;
    case AVCHROMA_LOC_TOP:        *cx = 0.5f; *cy = 0.0f; break;
    case AVCHROMA_LOC_BOTTOMLEFT: *cx = 0.0f; *cy = 1.0f; break;
    case AVCHROMA_LOC_BOTTOM:     *cx = 0.5f; *cy = 1.0f; break;
    default:                      *cx = 0.0f; *cy = 0.5f; break; /* left */
    }
}

/*
 * Fill p for a YUV format: to_rgb selects YUV->RGB (input) or RGB->YUV
 * (output). Like scale_cuda and swscale, unspecified means BT.601 limited.
 */
static int set_yuv_params(AVFilterContext *ctx, VSRConvParams *p, int to_rgb,
                          enum AVPixelFormat fmt, enum AVColorSpace csp,
                          enum AVColorRange range, enum AVChromaLocation loc,
                          int w, int h)
{
    const AVLumaCoefficients *coeffs;
    const int high = fmt == AV_PIX_FMT_P010;
    const int full = range == AVCOL_RANGE_JPEG;
    const float y_off   = full ? 0.0f : (high ? 64.0f : 16.0f);
    const float y_span  = full ? (high ? 1023.0f : 255.0f) : (high ? 876.0f : 219.0f);
    const float c_off   = high ? 512.0f : 128.0f;
    const float c_span  = full ? (high ? 1023.0f : 255.0f) : (high ? 896.0f : 224.0f);
    double kr, kg, kb;

    if (csp == AVCOL_SPC_UNSPECIFIED)
        csp = AVCOL_SPC_SMPTE170M;
    coeffs = av_csp_luma_coeffs_from_avcsp(csp);
    if (!coeffs || csp == AVCOL_SPC_RGB || csp == AVCOL_SPC_BT2020_CL ||
        csp == AVCOL_SPC_YCGCO || csp == AVCOL_SPC_YCGCO_RE ||
        csp == AVCOL_SPC_YCGCO_RO || csp == AVCOL_SPC_ICTCP ||
        csp == AVCOL_SPC_IPT_C2) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported YCbCr matrix %s.\n",
               av_color_space_name(csp));
        return AVERROR(ENOSYS);
    }
    kr = av_q2d(coeffs->cr);
    kg = av_q2d(coeffs->cg);
    kb = av_q2d(coeffs->cb);

    memset(p, 0, sizeof(*p));
    if (to_rgb) {
        p->m[0][0] = 1.0f; p->m[0][2] = 2.0 * (1.0 - kr);
        p->m[1][0] = 1.0f; p->m[1][1] = -2.0 * kb * (1.0 - kb) / kg;
                           p->m[1][2] = -2.0 * kr * (1.0 - kr) / kg;
        p->m[2][0] = 1.0f; p->m[2][1] = 2.0 * (1.0 - kb);
        p->y_scale = 1.0f / y_span;
        p->c_scale = 1.0f / c_span;
    } else {
        p->m[0][0] = kr; p->m[0][1] = kg; p->m[0][2] = kb;
        p->m[1][0] = -kr / (2.0 * (1.0 - kb));
        p->m[1][1] = -kg / (2.0 * (1.0 - kb));
        p->m[1][2] = 0.5f;
        p->m[2][0] = 0.5f;
        p->m[2][1] = -kg / (2.0 * (1.0 - kr));
        p->m[2][2] = -kb / (2.0 * (1.0 - kr));
        p->y_scale = y_span;
        p->c_scale = c_span;
    }
    p->y_off    = y_off;
    p->c_off    = c_off;
    p->width    = w;
    p->height   = h;
    p->high_bit = high;
    chroma_position(loc, &p->cx, &p->cy);

    av_log(ctx, AV_LOG_VERBOSE, "%s %s: %s, %s range, %s chroma\n",
           to_rgb ? "Input" : "Output", av_get_pix_fmt_name(fmt),
           av_color_space_name(csp), full ? "full" : "limited",
           av_chroma_location_name(loc == AVCHROMA_LOC_UNSPECIFIED ?
                                   AVCHROMA_LOC_LEFT : loc));
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

    extern const unsigned char ff_vf_vsr_cuda_ptx_data[];
    extern const unsigned int ff_vf_vsr_cuda_ptx_len;

    if (!inl->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames = (AVHWFramesContext *)inl->hw_frames_ctx->data;
    s->in_fmt  = in_frames->sw_format;
    s->out_fmt = s->format != AV_PIX_FMT_NONE ? s->format : s->in_fmt;
    if (!format_is_supported(s->in_fmt) || !format_is_supported(s->out_fmt)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported %s format %s; vsr_cuda takes "
               "rgba, rgb0, bgra, bgr0, nv12 and p010.\n",
               format_is_supported(s->in_fmt) ? "output" : "input",
               av_get_pix_fmt_name(format_is_supported(s->in_fmt) ? s->out_fmt
                                                                  : s->in_fmt));
        return AVERROR(ENOSYS);
    }
    s->in_conv  = format_conv(s->in_fmt);
    s->out_conv = format_conv(s->out_fmt);

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

    /* Conversion parameters and the output's colour description. */
    s->out_csp   = AVCOL_SPC_RGB;
    s->out_range = AVCOL_RANGE_JPEG;
    s->out_loc   = AVCHROMA_LOC_UNSPECIFIED;
    memset(&s->in_params, 0, sizeof(s->in_params));
    memset(&s->out_params, 0, sizeof(s->out_params));
    s->in_params.width   = inlink->w;
    s->in_params.height  = inlink->h;
    s->out_params.width  = w;
    s->out_params.height = h;
    if (s->in_conv == CONV_YUV) {
        ret = set_yuv_params(ctx, &s->in_params, 1, s->in_fmt, inlink->colorspace,
                             inlink->color_range, inlink->chroma_location,
                             inlink->w, inlink->h);
        if (ret < 0)
            return ret;
    }
    if (s->out_conv == CONV_YUV) {
        /* Specified output properties win; otherwise YUV input keeps its own. */
        const int in_yuv = s->in_conv == CONV_YUV;
        s->out_csp   = outlink->colorspace != AVCOL_SPC_UNSPECIFIED ? outlink->colorspace :
                       in_yuv ? inlink->colorspace : AVCOL_SPC_UNSPECIFIED;
        s->out_range = outlink->color_range != AVCOL_RANGE_UNSPECIFIED ? outlink->color_range :
                       in_yuv && inlink->color_range != AVCOL_RANGE_UNSPECIFIED ?
                       inlink->color_range : AVCOL_RANGE_MPEG;
        s->out_loc   = outlink->chroma_location != AVCHROMA_LOC_UNSPECIFIED ?
                       outlink->chroma_location :
                       in_yuv && inlink->chroma_location != AVCHROMA_LOC_UNSPECIFIED ?
                       inlink->chroma_location : AVCHROMA_LOC_LEFT;
        if (s->out_csp == AVCOL_SPC_UNSPECIFIED)
            s->out_csp = AVCOL_SPC_SMPTE170M;
        ret = set_yuv_params(ctx, &s->out_params, 0, s->out_fmt, s->out_csp,
                             s->out_range, s->out_loc, w, h);
        if (ret < 0)
            return ret;
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
    out_frames->sw_format = s->out_fmt;
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

    if (s->in_conv != CONV_NONE || s->out_conv != CONV_NONE) {
        ret = ff_cuda_load_module(ctx, s->hwctx, &s->cu_module,
                                  ff_vf_vsr_cuda_ptx_data, ff_vf_vsr_cuda_ptx_len);
        if (ret >= 0)
            ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_yuv_to_rgba, s->cu_module,
                                                   "vsr_yuv_to_rgba"));
        if (ret >= 0)
            ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_rgba_to_yuv, s->cu_module,
                                                   "vsr_rgba_to_yuv"));
        if (ret >= 0)
            ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_swap_rb, s->cu_module,
                                                   "vsr_swap_rb"));
    }
    if (ret >= 0 && s->in_conv != CONV_NONE)
        ret = CHECK_CU(cu->cuMemAllocPitch(&s->rgba_in, &s->rgba_in_pitch,
                                           inlink->w * 4, inlink->h, 16));
    if (ret >= 0 && s->out_conv != CONV_NONE)
        ret = CHECK_CU(cu->cuMemAllocPitch(&s->rgba_out, &s->rgba_out_pitch,
                                           w * 4, h, 16));
    if (ret >= 0 && s->cu_stream) {
        ret = CHECK_CU(cu->cuEventCreate(&s->ev_in, CU_EVENT_DISABLE_TIMING));
        if (ret >= 0)
            ret = CHECK_CU(cu->cuEventCreate(&s->ev_out, CU_EVENT_DISABLE_TIMING));
    }
    if (ret >= 0)
        ret = ff_ngx_open(ctx, &s->ngx, FF_NGX_FEATURE_VSR, "VSR",
                          "nvngx_vsr.dll", s->hwctx->cuda_ctx);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        return ret;

    av_log(ctx, AV_LOG_VERBOSE, "%dx%d %s -> %dx%d %s, quality %d, strength %.2f\n",
           inlink->w, inlink->h, av_get_pix_fmt_name(s->in_fmt), w, h,
           av_get_pix_fmt_name(s->out_fmt), s->quality, s->strength);
    return 0;
}

/* Convert the input frame to RGBA in s->rgba_in, on the filter's stream. */
static int convert_input(AVFilterContext *ctx, const AVFrame *in)
{
    VSRCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    const VSRConvParams *p = &s->in_params;
    int dst_pitch = s->rgba_in_pitch;

    if (s->in_conv == CONV_YUV) {
        int pitch_y = in->linesize[0], pitch_uv = in->linesize[1];
        void *args[] = { (void *)&in->data[0], &pitch_y, (void *)&in->data[1], &pitch_uv,
                         &s->rgba_in, &dst_pitch, (void *)p };
        return CHECK_CU(cu->cuLaunchKernel(s->cu_yuv_to_rgba,
                                           DIV_UP(p->width, BLOCKX), DIV_UP(p->height, BLOCKY), 1,
                                           BLOCKX, BLOCKY, 1, 0, s->cu_stream, args, NULL));
    } else {
        int src_pitch = in->linesize[0];
        void *args[] = { (void *)&in->data[0], &src_pitch, &s->rgba_in, &dst_pitch, (void *)p };
        return CHECK_CU(cu->cuLaunchKernel(s->cu_swap_rb,
                                           DIV_UP(p->width, BLOCKX), DIV_UP(p->height, BLOCKY), 1,
                                           BLOCKX, BLOCKY, 1, 0, s->cu_stream, args, NULL));
    }
}

/* Convert s->rgba_out into the output frame, on the filter's stream. */
static int convert_output(AVFilterContext *ctx, AVFrame *out)
{
    VSRCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    const VSRConvParams *p = &s->out_params;
    int src_pitch = s->rgba_out_pitch;

    if (s->out_conv == CONV_YUV) {
        int pitch_y = out->linesize[0], pitch_uv = out->linesize[1];
        void *args[] = { &s->rgba_out, &src_pitch, &out->data[0], &pitch_y,
                         &out->data[1], &pitch_uv, (void *)p };
        return CHECK_CU(cu->cuLaunchKernel(s->cu_rgba_to_yuv,
                                           DIV_UP(DIV_UP(p->width, 2), BLOCKX),
                                           DIV_UP(DIV_UP(p->height, 2), BLOCKY), 1,
                                           BLOCKX, BLOCKY, 1, 0, s->cu_stream, args, NULL));
    } else {
        int dst_pitch = out->linesize[0];
        void *args[] = { &s->rgba_out, &src_pitch, &out->data[0], &dst_pitch, (void *)p };
        return CHECK_CU(cu->cuLaunchKernel(s->cu_swap_rb,
                                           DIV_UP(p->width, BLOCKX), DIV_UP(p->height, BLOCKY), 1,
                                           BLOCKX, BLOCKY, 1, 0, s->cu_stream, args, NULL));
    }
}

static int vsr_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    VSRCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    const FFNGXRect rect_src = { 0, 0, inlink->w, inlink->h };
    const FFNGXRect rect_dst = { 0, 0, outlink->w, outlink->h };
    const FFNGXVSRParams params = {
        .quality  = s->quality,
        .strength = s->strength,
    };
    AVFrame *out = NULL;
    CUcontext dummy;
    const void *vsr_src;
    void *vsr_dst;
    int vsr_src_pitch, vsr_dst_pitch;
    int ret, err;

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    ret = av_hwframe_get_buffer(s->frames_ctx, out, 0);
    if (ret < 0)
        goto fail;

    if (s->in_conv == CONV_NONE) {
        vsr_src       = in->data[0];
        vsr_src_pitch = in->linesize[0];
    } else {
        vsr_src       = (const void *)(uintptr_t)s->rgba_in;
        vsr_src_pitch = s->rgba_in_pitch;
    }
    if (s->out_conv == CONV_NONE) {
        vsr_dst       = out->data[0];
        vsr_dst_pitch = out->linesize[0];
    } else {
        vsr_dst       = (void *)(uintptr_t)s->rgba_out;
        vsr_dst_pitch = s->rgba_out_pitch;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    if (s->in_conv != CONV_NONE)
        ret = convert_input(ctx, in);

    /* The DLL works on the legacy default stream; order it after the
     * filter chain's stream and back again. */
    if (ret >= 0 && s->cu_stream) {
        ret = CHECK_CU(cu->cuEventRecord(s->ev_in, s->cu_stream));
        if (ret >= 0)
            ret = CHECK_CU(cu->cuStreamWaitEvent(NULL, s->ev_in, 0));
    }

    if (ret >= 0) {
        err = s->ngx.proc_frame(s->ngx.handle, &rect_dst, &rect_src, &params,
                                vsr_src, vsr_src_pitch, vsr_dst, vsr_dst_pitch, 4, 4);
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

    if (ret >= 0 && s->out_conv != CONV_NONE)
        ret = convert_output(ctx, out);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        goto fail;
    out->sample_aspect_ratio = outlink->sample_aspect_ratio;
    if (s->in_conv == CONV_YUV || s->out_conv == CONV_YUV) {
        out->colorspace      = s->out_csp;
        out->color_range     = s->out_range;
        out->chroma_location = s->out_loc;
    }

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
    { "format", "Output pixel format (default: same as input)", OFFSET(format), AV_OPT_TYPE_PIXEL_FMT, { .i64 = AV_PIX_FMT_NONE }, -1, INT_MAX, FLAGS },
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
    FILTER_QUERY_FUNC2(vsr_query_formats),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
