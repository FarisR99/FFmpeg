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
 * Frame rate up-conversion on CUDA frames with NVIDIA Optical Flow FRUC.
 *
 * The Optical Flow SDK is not linked. NVEncNVOFFRUC.dll, which wraps
 * NvOFFRUC.dll with a CUDA device pointer API, is loaded at runtime; it needs
 * NvOFFRUC.dll and cudart64_110.dll next to it. Output timing follows
 * vf_fruc_vulkan.
 *
 * NvOFFRUC interpolates between the last two frames it was given, so one FRUC
 * instance yields one frame per input pair. Rates above twice the input use
 * one instance per output position within a pair; an instance that skipped
 * a pair is restarted from the pair's first frame.
 *
 * NvOFFRUC's process call blocks until the frame is done, so it runs on a
 * worker thread: meanwhile the filter fetches the next input, which lets the
 * filters before it (and mpv's renderer) prepare that frame.
 */

#include <stdbool.h>

#include "config.h"

#include "compat/w32dlfcn.h"

#include "libavutil/common.h"
#include "libavutil/cuda_check.h"
#include "libavutil/eval.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/thread.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "video.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

/* The C API of NVEncNVOFFRUC.dll. The functions return 0 on success. Frame
 * pointers are the addresses of CUdeviceptr variables, not the device
 * pointers themselves; buffers must be contiguous with pitch == width. */
#define FRUC_DLL_NAME "NVEncNVOFFRUC.dll"

typedef struct FRUCProcParams {
    void   *frame_in;
    int64_t timestamp_in;
    void   *frame_out;
    int64_t timestamp_out;
} FRUCProcParams;

typedef int  (__stdcall *fruc_create_fn)(void **handle);
typedef int  (__stdcall *fruc_load_fn)(void *handle);
typedef void (__stdcall *fruc_delete_fn)(void *handle);
typedef int  (__stdcall *fruc_create_session_fn)(void *handle, int width, int height,
                                                 bool nv12);
typedef int  (__stdcall *fruc_register_fn)(void *handle, void *buf0, void *buf1,
                                           void *buf2);
typedef int  (__stdcall *fruc_proc_fn)(void *handle, FRUCProcParams *params);

typedef struct FRUCInstance {
    void   *handle;
    int64_t last_pts;   ///< pts of the last frame it was given, or AV_NOPTS_VALUE
} FRUCInstance;

static const char *const var_names[] = {
    "source_fps",
    NULL
};

enum var_name {
    VAR_SOURCE_FPS,
    VARS_NB
};

typedef struct FRUCCUDAContext {
    const AVClass *class;

    char       *requested_frame_rate;   ///< output fps as an expression
    AVRational dest_frame_rate;
    AVRational srce_time_base;
    AVRational dest_time_base;

    AVCUDADeviceContext *hwctx;
    CUstream cu_stream;
    int nv12;
    int width, height;
    size_t frame_size;

    /* [0] and [1] hold the input pair, alternating; [2] is the output. */
    CUdeviceptr buf[3];
    int buf_f0, buf_f1;
    int64_t nb_in;

    HMODULE dll;
    fruc_create_fn         fruc_create;
    fruc_load_fn           fruc_load;
    fruc_delete_fn         fruc_delete;
    fruc_create_session_fn fruc_create_session;
    fruc_register_fn       fruc_register;
    fruc_proc_fn           fruc_proc;

    FRUCInstance *inst;
    int nb_inst;
    int pair_idx;       ///< interpolated frames already made from this pair

    /* Worker thread running one interpolation (a job) at a time. */
    pthread_t worker;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int worker_started;
    int quit;
    int job_running;    ///< set by the filter, cleared by the worker
    int job_err;
    FRUCInstance *job_inst;
    int job_restart;    ///< restart the instance from buf[job_buf_f0] first
    int job_buf_f0;
    int64_t job_pts0;
    FRUCProcParams job_params;
    AVFrame *pending;   ///< output frame the running or finished job fills

    AVFrame *next;      ///< input fetched while a job runs, not yet used
    int eof;
    int eof_status;
    int64_t eof_pts;

    AVFrame *work;
    AVFrame *f0;        ///< last frame
    AVFrame *f1;        ///< current frame
    int64_t pts0;       ///< last frame pts in dest_time_base
    int64_t pts1;       ///< current frame pts in dest_time_base
    int64_t delta;      ///< pts1 to pts0 delta
    int flush;
    int64_t start_pts;  ///< pts of the first output frame
    int64_t n;          ///< output frame counter
} FRUCCUDAContext;

#define OFFSET(x) offsetof(FRUCCUDAContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption fruc_cuda_options[] = {
    { "fps", "A string describing the desired output frame rate",
      OFFSET(requested_frame_rate), AV_OPT_TYPE_STRING, { .str = "source_fps*2" }, 0, 0, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(fruc_cuda);

static av_cold void fruc_free_instances(AVFilterContext *ctx)
{
    FRUCCUDAContext *s = ctx->priv;

    for (int i = 0; i < s->nb_inst; i++)
        if (s->inst[i].handle)
            s->fruc_delete(s->inst[i].handle);
    av_freep(&s->inst);
    s->nb_inst = 0;
}

static av_cold void fruc_uninit(AVFilterContext *ctx)
{
    FRUCCUDAContext *s = ctx->priv;

    if (s->worker_started) {
        pthread_mutex_lock(&s->lock);
        s->quit = 1;
        pthread_cond_broadcast(&s->cond);
        pthread_mutex_unlock(&s->lock);
        pthread_join(s->worker, NULL);
        pthread_cond_destroy(&s->cond);
        pthread_mutex_destroy(&s->lock);
        s->worker_started = 0;
        s->quit = 0;
        s->job_running = 0;
    }
    av_frame_free(&s->pending);
    av_frame_free(&s->next);

    if (s->hwctx) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        CHECK_CU(cu->cuCtxSynchronize());
        fruc_free_instances(ctx);
        for (int i = 0; i < FF_ARRAY_ELEMS(s->buf); i++) {
            if (s->buf[i])
                CHECK_CU(cu->cuMemFree(s->buf[i]));
            s->buf[i] = 0;
        }
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    if (s->dll)
        dlclose(s->dll);
    s->dll = NULL;

    av_frame_free(&s->f0);
    av_frame_free(&s->f1);
}

static av_cold int fruc_init(AVFilterContext *ctx)
{
    FRUCCUDAContext *s = ctx->priv;

    s->start_pts = AV_NOPTS_VALUE;
    return 0;
}

static av_cold int load_fruc(AVFilterContext *ctx)
{
    FRUCCUDAContext *s = ctx->priv;

    s->dll = dlopen(FRUC_DLL_NAME, 0);
    if (!s->dll) {
        av_log(ctx, AV_LOG_ERROR, "Could not load " FRUC_DLL_NAME ". Place it, "
               "NvOFFRUC.dll and cudart64_110.dll next to ffmpeg.\n");
        return AVERROR_EXTERNAL;
    }

#define LOAD(field, type, name) do {                                         \
        s->field = (type)dlsym(s->dll, name);                                \
        if (!s->field) {                                                     \
            av_log(ctx, AV_LOG_ERROR, FRUC_DLL_NAME " has no %s.\n", name); \
            return AVERROR_EXTERNAL;                                         \
        }                                                                    \
    } while (0)
    LOAD(fruc_create,         fruc_create_fn,         "NVEncNVOFFRUCCreate");
    LOAD(fruc_load,           fruc_load_fn,           "NVEncNVOFFRUCLoad");
    LOAD(fruc_delete,         fruc_delete_fn,         "NVEncNVOFFRUCDelete");
    LOAD(fruc_create_session, fruc_create_session_fn, "NVEncNVOFFRUCCreateFURCHandle");
    LOAD(fruc_register,       fruc_register_fn,       "NVEncNVOFFRUCRegisterResource");
    LOAD(fruc_proc,           fruc_proc_fn,           "NVEncNVOFFRUCProc");
#undef LOAD
    return 0;
}

/* Add FRUC instance s->nb_inst. The caller has the CUDA context current. */
static int add_instance(AVFilterContext *ctx)
{
    FRUCCUDAContext *s = ctx->priv;
    FRUCInstance *inst;
    int err;

    inst = av_realloc_array(s->inst, s->nb_inst + 1, sizeof(*s->inst));
    if (!inst)
        return AVERROR(ENOMEM);
    s->inst = inst;
    inst = &s->inst[s->nb_inst];
    inst->handle   = NULL;
    inst->last_pts = AV_NOPTS_VALUE;

    err = s->fruc_create(&inst->handle);
    if (err || !inst->handle) {
        av_log(ctx, AV_LOG_ERROR, "Could not create FRUC (error %d).\n", err);
        return AVERROR_EXTERNAL;
    }
    s->nb_inst++;

    err = s->fruc_load(inst->handle);
    if (err) {
        av_log(ctx, AV_LOG_ERROR, "Could not load NvOFFRUC.dll (error %d). "
               "It requires cudart64_110.dll and a Turing or newer GPU.\n", err);
        return AVERROR_EXTERNAL;
    }
    return 0;
}

/* (Re)start an instance with buf[buf_idx] (at pts) as its previous frame. */
static int start_instance(AVFilterContext *ctx, FRUCInstance *inst, int buf_idx,
                          int64_t pts)
{
    FRUCCUDAContext *s = ctx->priv;
    FRUCProcParams params = {
        .frame_in      = &s->buf[buf_idx],
        .timestamp_in  = pts,
        .frame_out     = &s->buf[2],
        .timestamp_out = pts,
    };
    int err;

    av_log(ctx, AV_LOG_DEBUG, "Starting FRUC instance %d at pts %"PRId64
           " (last %"PRId64")\n", (int)(inst - s->inst), pts, inst->last_pts);
    err = s->fruc_create_session(inst->handle, s->width, s->height, s->nv12);
    if (!err)
        err = s->fruc_register(inst->handle, &s->buf[0], &s->buf[1], &s->buf[2]);
    if (!err)
        err = s->fruc_proc(inst->handle, &params);
    if (err) {
        av_log(ctx, AV_LOG_ERROR, "Could not start FRUC (error %d).\n", err);
        return AVERROR_EXTERNAL;
    }
    inst->last_pts = pts;
    return 0;
}

static int copy_planes(AVFilterContext *ctx, CUdeviceptr dst_buf, AVFrame *frame,
                       int to_frame)
{
    FRUCCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    int planes = s->nv12 ? 2 : 1;
    size_t offset = 0;
    int ret;

    for (int i = 0; i < planes; i++) {
        int rows = i ? s->height / 2 : s->height;
        size_t row_bytes = s->nv12 ? s->width : s->width * 4;
        CUDA_MEMCPY2D cpy = {
            .srcMemoryType = CU_MEMORYTYPE_DEVICE,
            .dstMemoryType = CU_MEMORYTYPE_DEVICE,
            .WidthInBytes  = row_bytes,
            .Height        = rows,
        };

        if (to_frame) {
            cpy.srcDevice = dst_buf + offset;
            cpy.srcPitch  = row_bytes;
            cpy.dstDevice = (CUdeviceptr)frame->data[i];
            cpy.dstPitch  = frame->linesize[i];
        } else {
            cpy.srcDevice = (CUdeviceptr)frame->data[i];
            cpy.srcPitch  = frame->linesize[i];
            cpy.dstDevice = dst_buf + offset;
            cpy.dstPitch  = row_bytes;
        }
        ret = CHECK_CU(cu->cuMemcpy2DAsync(&cpy, s->cu_stream));
        if (ret < 0)
            return ret;
        offset += row_bytes * rows;
    }
    return 0;
}

/* Run the queued job on the worker thread. */
static int run_job(AVFilterContext *ctx)
{
    FRUCCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    FRUCInstance *inst = s->job_inst;
    CUcontext dummy;
    int ret, err;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
    /* NvOFFRUC does its own stream handling: finish the input copies (and
     * the copy of the previous result out of buf[2]) first. Its process call
     * returns once the output is written. */
    ret = CHECK_CU(cu->cuStreamSynchronize(s->cu_stream));
    if (ret >= 0 && s->job_restart)
        ret = start_instance(ctx, inst, s->job_buf_f0, s->job_pts0);
    if (ret >= 0) {
        err = s->fruc_proc(inst->handle, &s->job_params);
        if (err) {
            av_log(ctx, AV_LOG_ERROR, "FRUC failed (error %d).\n", err);
            ret = AVERROR_EXTERNAL;
        }
        inst->last_pts = s->job_params.timestamp_in;
    }
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
}

static void *fruc_worker(void *arg)
{
    AVFilterContext *ctx = arg;
    FRUCCUDAContext *s = ctx->priv;
    int ret;

    pthread_mutex_lock(&s->lock);
    for (;;) {
        while (!s->job_running && !s->quit)
            pthread_cond_wait(&s->cond, &s->lock);
        if (!s->job_running)
            break;
        pthread_mutex_unlock(&s->lock);
        ret = run_job(ctx);
        pthread_mutex_lock(&s->lock);
        s->job_err = ret;
        s->job_running = 0;
        pthread_cond_broadcast(&s->cond);
    }
    pthread_mutex_unlock(&s->lock);
    return NULL;
}

/* Queue the interpolation of the current pair at pts into s->pending. */
static int start_job(AVFilterContext *ctx, int64_t pts)
{
    FRUCCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;
    int ret = 0;

    if (s->pair_idx == s->nb_inst) {
        ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        if (ret < 0)
            return ret;
        ret = add_instance(ctx);
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        if (ret < 0)
            return ret;
    }

    pthread_mutex_lock(&s->lock);
    s->job_inst    = &s->inst[s->pair_idx++];
    s->job_restart = s->job_inst->last_pts != s->pts0;
    s->job_buf_f0  = s->buf_f0;
    s->job_pts0    = s->pts0;
    s->job_params  = (FRUCProcParams) {
        .frame_in      = &s->buf[s->buf_f1],
        .timestamp_in  = s->pts1,
        .frame_out     = &s->buf[2],
        .timestamp_out = pts,
    };
    s->job_err     = 0;
    s->job_running = 1;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->lock);
    return 0;
}

static int job_running(FRUCCUDAContext *s)
{
    int running;

    pthread_mutex_lock(&s->lock);
    running = s->job_running;
    pthread_mutex_unlock(&s->lock);
    return running;
}

/* Wait for the job; return its result. */
static int wait_job(FRUCCUDAContext *s)
{
    int ret;

    pthread_mutex_lock(&s->lock);
    while (s->job_running)
        pthread_cond_wait(&s->cond, &s->lock);
    ret = s->job_err;
    pthread_mutex_unlock(&s->lock);
    return ret;
}

/* Wait for the job and copy its result into s->pending. */
static int finish_job(AVFilterContext *ctx)
{
    FRUCCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;
    int ret;

    ret = wait_job(s);
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
    ret = copy_planes(ctx, s->buf[2], s->pending, 1);
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
}

static int store_input(AVFilterContext *ctx, AVFrame *in)
{
    FRUCCUDAContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;
    int ret;

    s->buf_f0 = s->buf_f1;
    s->buf_f1 = s->nb_in++ & 1;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
    ret = copy_planes(ctx, s->buf[s->buf_f1], in, 0);
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
}

static int passthrough_frame(AVFrame **work, AVFrame *src)
{
    *work = av_frame_clone(src);
    return *work ? 0 : AVERROR(ENOMEM);
}

static int process_work_frame(AVFilterContext *ctx)
{
    FRUCCUDAContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    int64_t work_pts, snap;
    int ret;

    /* An interpolated frame precedes everything after it. */
    if (s->pending) {
        if (job_running(s))
            return 2;
        ret = finish_job(ctx);
        if (ret < 0) {
            av_frame_free(&s->pending);
            return ret;
        }
        s->work = s->pending;
        s->pending = NULL;
        s->n++;
        return 1;
    }

    if (!s->f1)
        return 0;
    if (!s->f0 && !s->flush)
        return 0;

    work_pts = s->start_pts + av_rescale_q(s->n, av_inv_q(s->dest_frame_rate),
                                           s->dest_time_base);
    /* An output within 1/8 of an output interval of an input frame is that
     * frame, so timestamp jitter (e.g. millisecond Matroska timestamps) does
     * not produce near-duplicate interpolations. */
    snap = av_rescale_q(1, av_inv_q(s->dest_frame_rate), s->dest_time_base) / 8;

    if (work_pts > s->pts1 && !s->flush)
        return 0;

    if (!s->f0) {
        ret = passthrough_frame(&s->work, s->f1);
        if (ret < 0)
            return ret;
        /* Flushing: f1 is not needed once passed through. */
        av_frame_free(&s->f1);
    } else {
        if (work_pts >= s->pts1 + s->delta && s->flush)
            return 0;

        if (work_pts - s->pts0 <= snap) {
            ret = passthrough_frame(&s->work, s->f0);
        } else if (s->pts1 - work_pts <= snap) {
            ret = passthrough_frame(&s->work, s->f1);
        } else {
            s->pending = ff_get_video_buffer(outlink, outlink->w, outlink->h);
            if (!s->pending)
                return AVERROR(ENOMEM);
            ret = av_frame_copy_props(s->pending, s->f0);
            if (ret >= 0)
                ret = start_job(ctx, work_pts);
            if (ret < 0) {
                av_frame_free(&s->pending);
                return ret;
            }
            s->pending->pts = work_pts;
            return 2;
        }
        if (ret < 0)
            return ret;
    }

    s->work->pts = work_pts;
    s->n++;

    return 1;
}

/* Make in (from the input) the current frame, f1. Takes ownership. */
static int take_input(AVFilterContext *ctx, AVFrame *in)
{
    FRUCCUDAContext *s = ctx->priv;
    int64_t pts;
    int ret;

    if (in->flags & AV_FRAME_FLAG_INTERLACED)
        av_log(ctx, AV_LOG_WARNING, "Interlaced frame found - the output will not be correct.\n");

    if (in->pts == AV_NOPTS_VALUE) {
        av_log(ctx, AV_LOG_WARNING, "Ignoring frame without PTS.\n");
        av_frame_free(&in);
        return 0;
    }

    pts = av_rescale_q(in->pts, s->srce_time_base, s->dest_time_base);
    if (s->f1 && pts == s->pts1) {
        av_log(ctx, AV_LOG_WARNING, "Ignoring frame with same PTS.\n");
        av_frame_free(&in);
        return 0;
    }

    ret = store_input(ctx, in);
    if (ret < 0) {
        av_frame_free(&in);
        return ret;
    }

    av_frame_free(&s->f0);
    s->f0 = s->f1;
    s->pts0 = s->pts1;
    s->f1 = in;
    s->pts1 = pts;
    s->delta = s->pts1 - s->pts0;
    s->pair_idx = 0;

    if (s->delta < 0) {
        av_log(ctx, AV_LOG_WARNING, "PTS discontinuity.\n");
        s->start_pts = s->pts1;
        s->n = 0;
        av_frame_free(&s->f0);
        for (int i = 0; i < s->nb_inst; i++)
            s->inst[i].last_pts = AV_NOPTS_VALUE;
    }

    if (s->start_pts == AV_NOPTS_VALUE)
        s->start_pts = s->pts1;
    return 0;
}

static int fruc_activate(AVFilterContext *ctx)
{
    int ret, status;
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    FRUCCUDAContext *s = ctx->priv;
    AVFrame *in;
    int64_t pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

retry:
    ret = process_work_frame(ctx);
    if (ret < 0)
        return ret;
    else if (ret == 1)
        return ff_filter_frame(outlink, s->work);

    if (ret == 2) {
        /* A job is running. Fetch the next input meanwhile (without using it
         * yet), so that the filters before this one prepare it; only then,
         * or at EOF, wait for the job. */
        if (!s->next && !s->eof) {
            ret = ff_inlink_consume_frame(inlink, &s->next);
            if (ret < 0)
                return ret;
            if (!s->next) {
                if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
                    s->eof        = 1;
                    s->eof_status = status;
                    s->eof_pts    = pts;
                    goto retry;
                }
                ff_inlink_request_frame(inlink);
                return FFERROR_NOT_READY;
            }
        }
        /* process_work_frame() then outputs s->pending. */
        wait_job(s);
        goto retry;
    }

    /* The current pair is done: move on to the next input. */
    in = s->next;
    s->next = NULL;
    if (!in && !s->eof) {
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
    }
    if (in) {
        ret = take_input(ctx, in);
        if (ret < 0)
            return ret;
        goto retry;
    }

    if (!s->eof && ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        s->eof        = 1;
        s->eof_status = status;
        s->eof_pts    = pts;
    }
    if (s->eof) {
        if (!s->flush) {
            s->flush = 1;
            goto retry;
        }
        ff_outlink_set_status(outlink, s->eof_status, s->eof_pts);
        return 0;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

static int fruc_config_input(AVFilterLink *inlink)
{
    FRUCCUDAContext *s = inlink->dst->priv;

    s->srce_time_base = inlink->time_base;
    return 0;
}

static av_cold int fruc_config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    FilterLink *il = ff_filter_link(inlink);
    FilterLink *ol = ff_filter_link(outlink);
    FRUCCUDAContext *s = ctx->priv;
    AVHWFramesContext *in_frames;
    CudaFunctions *cu;
    double var_values[VARS_NB], res;
    CUcontext dummy;
    int ret, exact;

    if (!il->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames = (AVHWFramesContext *)il->hw_frames_ctx->data;
    switch (in_frames->sw_format) {
    case AV_PIX_FMT_NV12:
        s->nv12 = 1;
        break;
    case AV_PIX_FMT_RGBA: case AV_PIX_FMT_BGRA:
    case AV_PIX_FMT_RGB0: case AV_PIX_FMT_BGR0:
        s->nv12 = 0;
        break;
    default:
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format %s; FRUC needs nv12 "
               "or 8-bit packed RGB.\n", av_get_pix_fmt_name(in_frames->sw_format));
        return AVERROR(EINVAL);
    }
    if (s->nv12 && ((inlink->w | inlink->h) & 1)) {
        av_log(ctx, AV_LOG_ERROR, "The video size must be even for nv12.\n");
        return AVERROR(EINVAL);
    }

    var_values[VAR_SOURCE_FPS] = av_q2d(il->frame_rate);
    ret = av_expr_parse_and_eval(&res, s->requested_frame_rate,
                                 var_names, var_values,
                                 NULL, NULL, NULL, NULL, NULL, 0, ctx);
    if (ret < 0)
        return ret;
    s->dest_frame_rate = av_d2q(res, INT_MAX);
    if (s->dest_frame_rate.num <= 0 || s->dest_frame_rate.den <= 0) {
        av_log(ctx, AV_LOG_ERROR,
               "Invalid output frame rate '%s' (must evaluate to a positive value)\n",
               s->requested_frame_rate);
        return AVERROR(EINVAL);
    }

    /* Make sure the timebase is small enough to hold the frame rate. */
    exact = av_reduce(&s->dest_time_base.num, &s->dest_time_base.den,
                      av_gcd((int64_t)s->srce_time_base.num * s->dest_frame_rate.num,
                             (int64_t)s->srce_time_base.den * s->dest_frame_rate.den),
                      (int64_t)s->srce_time_base.den * s->dest_frame_rate.num, INT_MAX);
    if (!s->dest_time_base.num || !s->dest_time_base.den)
        exact = av_reduce(&s->dest_time_base.num, &s->dest_time_base.den,
                          s->dest_frame_rate.den, s->dest_frame_rate.num, INT_MAX);
    av_log(ctx, AV_LOG_VERBOSE, "time base:%u/%u -> %u/%u exact:%d\n",
           s->srce_time_base.num, s->srce_time_base.den,
           s->dest_time_base.num, s->dest_time_base.den, exact);
    if (!exact)
        av_log(ctx, AV_LOG_WARNING, "Timebase conversion is not exact\n");

    outlink->w = inlink->w;
    outlink->h = inlink->h;
    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    outlink->time_base = s->dest_time_base;
    ol->frame_rate = s->dest_frame_rate;
    av_buffer_unref(&ol->hw_frames_ctx);
    ol->hw_frames_ctx = av_buffer_ref(il->hw_frames_ctx);
    if (!ol->hw_frames_ctx)
        return AVERROR(ENOMEM);

    fruc_uninit(ctx);
    s->hwctx      = in_frames->device_ctx->hwctx;
    s->cu_stream  = s->hwctx->stream;
    s->width      = inlink->w;
    s->height     = inlink->h;
    s->frame_size = s->nv12 ? (size_t)s->width * s->height * 3 / 2
                            : (size_t)s->width * s->height * 4;
    cu = s->hwctx->internal->cuda_dl;

    ret = load_fruc(ctx);
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
    for (int i = 0; i < FF_ARRAY_ELEMS(s->buf) && ret >= 0; i++)
        ret = CHECK_CU(cu->cuMemAlloc(&s->buf[i], s->frame_size));
    /* Fail early if NvOFFRUC.dll can't be loaded. */
    if (ret >= 0)
        ret = add_instance(ctx);
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        return ret;

    ret = AVERROR(pthread_mutex_init(&s->lock, NULL));
    if (ret < 0)
        return ret;
    ret = AVERROR(pthread_cond_init(&s->cond, NULL));
    if (ret < 0) {
        pthread_mutex_destroy(&s->lock);
        return ret;
    }
    ret = AVERROR(pthread_create(&s->worker, NULL, fruc_worker, ctx));
    if (ret < 0) {
        pthread_cond_destroy(&s->cond);
        pthread_mutex_destroy(&s->lock);
        return ret;
    }
    s->worker_started = 1;
    return 0;
}

static const AVFilterPad fruc_cuda_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = fruc_config_input,
    },
};

static const AVFilterPad fruc_cuda_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = fruc_config_output,
    },
};

const FFFilter ff_vf_fruc_cuda = {
    .p.name         = "fruc_cuda",
    .p.description  = NULL_IF_CONFIG_SMALL("Frame rate up-conversion using NVIDIA Optical Flow FRUC"),
    .p.priv_class   = &fruc_cuda_class,
    .priv_size      = sizeof(FRUCCUDAContext),
    .init           = fruc_init,
    .uninit         = fruc_uninit,
    .activate       = fruc_activate,
    FILTER_INPUTS(fruc_cuda_inputs),
    FILTER_OUTPUTS(fruc_cuda_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
