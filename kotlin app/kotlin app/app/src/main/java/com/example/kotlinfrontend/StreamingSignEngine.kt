package com.example.kotlinfrontend

import androidx.camera.core.ImageProxy
import java.io.Closeable

interface StreamingSignEngine : Closeable {
    fun warmUp()

    fun resetCapture(clearTranscript: Boolean = false)

    fun process(
        imageProxy: ImageProxy,
        includeOverlay: Boolean = true,
        includeFace: Boolean = true
    ): LiveInferenceState

    fun clearTranscript()
}
