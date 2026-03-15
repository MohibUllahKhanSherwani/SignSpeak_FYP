package com.example.kotlinfrontend

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.roundToLong

data class VideoFrame(
    val bitmap: Bitmap,
    val timestampMs: Long
)

class VideoFrameExtractor(private val context: Context) {
    companion object {
        const val DEFAULT_FRAME_COUNT = 30
        private const val MAX_FRAME_SIDE_PX = 640
    }

    fun sampleFrames(videoUri: Uri, frameCount: Int = DEFAULT_FRAME_COUNT): List<VideoFrame> {
        require(frameCount > 0) { "frameCount must be greater than zero." }

        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, videoUri)
            val durationMs = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?.coerceAtLeast(1L)
                ?: throw IllegalStateException("Unable to read video duration.")

            val rotationDegrees = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                ?.toIntOrNull()
                ?.mod(360)
                ?: 0

            val timestampsMs = buildEvenTimestamps(durationMs, frameCount)
            val frames = ArrayList<VideoFrame>(frameCount)
            var previousFrame: Bitmap? = null

            timestampsMs.forEach { timestampMs ->
                val rawBitmap = retriever.getFrameAtTime(
                    timestampMs * 1_000L,
                    MediaMetadataRetriever.OPTION_CLOSEST
                )

                val preparedBitmap = when {
                    rawBitmap != null -> {
                        val rotated = rotateIfRequired(rawBitmap, rotationDegrees)
                        val scaled = downscaleIfRequired(rotated)
                        previousFrame = scaled
                        scaled
                    }
                    previousFrame != null -> {
                        val cachedFrame = previousFrame!!
                        cachedFrame.copy(Bitmap.Config.ARGB_8888, false)
                            ?: Bitmap.createBitmap(
                                cachedFrame.width,
                                cachedFrame.height,
                                Bitmap.Config.ARGB_8888
                            )
                    }
                    else -> Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
                }

                frames.add(VideoFrame(preparedBitmap, timestampMs))
            }

            return frames
        } finally {
            retriever.release()
        }
    }

    private fun buildEvenTimestamps(durationMs: Long, frameCount: Int): List<Long> {
        if (frameCount == 1) {
            return listOf((durationMs / 2L).coerceAtLeast(0L))
        }
        val endMs = (durationMs - 1L).coerceAtLeast(0L)
        return List(frameCount) { index ->
            val ratio = index.toDouble() / (frameCount - 1).toDouble()
            (ratio * endMs.toDouble()).roundToLong()
        }
    }

    private fun rotateIfRequired(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) {
            return ensureArgb8888(bitmap)
        }

        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
        if (rotated !== bitmap && !bitmap.isRecycled) {
            bitmap.recycle()
        }
        return ensureArgb8888(rotated)
    }

    private fun downscaleIfRequired(bitmap: Bitmap): Bitmap {
        val longestSide = max(bitmap.width, bitmap.height)
        if (longestSide <= MAX_FRAME_SIDE_PX) {
            return ensureArgb8888(bitmap)
        }

        val scale = MAX_FRAME_SIDE_PX.toFloat() / longestSide.toFloat()
        val targetWidth = (bitmap.width * scale).roundToInt().coerceAtLeast(1)
        val targetHeight = (bitmap.height * scale).roundToInt().coerceAtLeast(1)
        val scaled = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        if (scaled !== bitmap && !bitmap.isRecycled) {
            bitmap.recycle()
        }
        return ensureArgb8888(scaled)
    }

    private fun ensureArgb8888(bitmap: Bitmap): Bitmap {
        if (bitmap.config == Bitmap.Config.ARGB_8888) {
            return bitmap
        }
        val converted = bitmap.copy(Bitmap.Config.ARGB_8888, false)
            ?: throw IllegalStateException("Failed to convert bitmap to ARGB_8888.")
        if (converted !== bitmap && !bitmap.isRecycled) {
            bitmap.recycle()
        }
        return converted
    }
}
