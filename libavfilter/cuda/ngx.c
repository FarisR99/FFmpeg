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

#include "libavutil/error.h"
#include "libavutil/log.h"

#include "ngx.h"

int ff_ngx_open(void *log_ctx, FFNGX *ngx, enum FFNGXFeature feature,
                const char *name, const char *feature_dll, void *cu_context)
{
    int err;

    ngx->dll = dlopen(FF_NGX_DLL_NAME, 0);
    if (!ngx->dll) {
        av_log(log_ctx, AV_LOG_ERROR, "Could not load " FF_NGX_DLL_NAME ". "
               "Place it, %s and cudart64_12.dll next to ffmpeg.\n", feature_dll);
        return AVERROR_EXTERNAL;
    }

#define LOAD(field, type, sym) do {                                             \
        ngx->field = (type)dlsym(ngx->dll, sym);                                \
        if (!ngx->field) {                                                      \
            av_log(log_ctx, AV_LOG_ERROR, FF_NGX_DLL_NAME " has no " sym ".\n"); \
            return AVERROR_EXTERNAL;                                            \
        }                                                                       \
    } while (0)
    LOAD(create,     ff_ngx_create_fn,     "NVEncNVSDKNGXCreate");
    LOAD(init,       ff_ngx_init_fn,       "NVEncNVSDKNGXInit");
    LOAD(destroy,    ff_ngx_delete_fn,     "NVEncNVSDKNGXDelete");
    LOAD(proc_frame, ff_ngx_proc_frame_fn, "NVEncNVSDKNGXProcFrame");
#undef LOAD

    err = ngx->create(&ngx->handle, feature);
    if (err || !ngx->handle) {
        av_log(log_ctx, AV_LOG_ERROR, "Could not create %s (error %d).\n",
               name, err);
        ngx->handle = NULL;
        return AVERROR_EXTERNAL;
    }

    err = ngx->init(ngx->handle, -1, cu_context, NULL);
    if (err) {
        av_log(log_ctx, AV_LOG_ERROR, "Could not initialize %s (error %d). "
               "It requires %s and an RTX GPU.\n", name, err, feature_dll);
        return AVERROR_EXTERNAL;
    }
    return 0;
}

void ff_ngx_close(FFNGX *ngx)
{
    if (ngx->handle)
        ngx->destroy(ngx->handle);
    ngx->handle = NULL;
    if (ngx->dll)
        dlclose(ngx->dll);
    ngx->dll = NULL;
}
