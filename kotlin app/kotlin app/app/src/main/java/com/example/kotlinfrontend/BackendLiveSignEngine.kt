package com.example.kotlinfrontend

import android.content.Context
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class BackendLiveSignEngine(
    context: Context,
    private val baseUrl: String,
    private val model: SignModel,
    private val backendClient: SignBackendClient = SignBackendClient(),
    private val minInferenceGapMs: Long = 250L,
    smoothingWindowSize: Int = 8,
    private val sequenceLength: Int = 60
) : StreamingSignEngine {
    companion object {
        private const val FEATURE_SIZE = 126
    }

    private val handExtractor = HandExtractor(context)
    private val frameBuffer = ArrayDeque<FloatArray>(sequenceLength)
    private val accumulator = PredictionAccumulator(smoothingWindowSize)
    private val requestExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val requestInFlight = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private val requestGeneration = AtomicInteger(0)

    private var lastDispatchTimestampMs = -1L
    private var lastFrameTimestampNs = -1L
    private var smoothedFps = 0f
    private var latestPrediction: PredictionResult? = null
    private var latestRoundTripMs: Long? = null
    private var latestErrorMessage: String? = null
    private var frameCount = 0

    override fun warmUp() {
        // No model to warm up on device in backend mode.
    }

    @Synchronized
    override fun resetCapture(clearTranscript: Boolean) {
        frameBuffer.clear()
        latestPrediction = null
        latestRoundTripMs = null
        latestErrorMessage = null
        lastDispatchTimestampMs = -1L
        frameCount = 0
        requestGeneration.incrementAndGet()
        accumulator.reset(clearTranscript = clearTranscript)
    }

    override fun process(
        imageProxy: ImageProxy,
        includeOverlay: Boolean,
        includeFace: Boolean
    ): LiveInferenceState {
        val timestampNs = imageProxy.imageInfo.timestamp
        val timestampMs = (timestampNs / 1_000_000L).coerceAtLeast(0L)

        val extraction = handExtractor.extractFromImageProxyWithOverlay(
            imageProxy = imageProxy,
            timestampMs = timestampMs,
            includeOverlay = includeOverlay,
            includeFace = includeFace
        )
        val frameVector = extraction.vector

        val requestSnapshot: List<FloatArray>?
        synchronized(this) {
            updateFps(timestampNs)
            frameCount++
            if (frameBuffer.size == sequenceLength) {
                frameBuffer.removeFirst()
            }
            frameBuffer.addLast(frameVector.copyOf())

            // Strict interval: Every 3 frames when the buffer is full
            requestSnapshot = if (
                frameBuffer.size == sequenceLength &&
                frameCount % 3 == 0 &&
                !requestInFlight.get() &&
                (lastDispatchTimestampMs < 0L || (timestampMs - lastDispatchTimestampMs) >= minInferenceGapMs)
            ) {
                lastDispatchTimestampMs = timestampMs
                latestErrorMessage = null
                frameBuffer.map { it.copyOf() }
            } else {
                null
            }
        }

        if (requestSnapshot != null && requestInFlight.compareAndSet(false, true)) {
            submitPrediction(
                sequenceSnapshot = requestSnapshot,
                generation = requestGeneration.get()
            )
        }

        val displayState: PredictionDisplayState
        val latestPredictionSnapshot: PredictionResult?
        val latestRoundTripSnapshot: Long?
        val latestErrorSnapshot: String?
        synchronized(this) {
            displayState = accumulator.snapshot()
            latestPredictionSnapshot = latestPrediction
            latestRoundTripSnapshot = latestRoundTripMs
            latestErrorSnapshot = latestErrorMessage
        }

        return LiveInferenceState(
            rawPrediction = latestPredictionSnapshot,
            displayedLabel = displayState.label,
            displayedConfidence = displayState.confidence,
            stability = displayState.stability,
            transcript = displayState.transcript,
            overlayHands = extraction.hands,
            overlayFace = extraction.face,
            bufferCount = synchronized(this) { frameBuffer.size },
            sequenceLength = sequenceLength,
            handsInFrame = countHandsInFrame(frameVector),
            fps = synchronized(this) { smoothedFps },
            roundTripMs = latestRoundTripSnapshot,
            requestInFlight = requestInFlight.get(),
            errorMessage = latestErrorSnapshot
        )
    }

    @Synchronized
    override fun clearTranscript() {
        accumulator.clearTranscript()
    }

    private fun submitPrediction(sequenceSnapshot: List<FloatArray>, generation: Int) {
        requestExecutor.execute {
            val startedAt = SystemClock.elapsedRealtime()
            try {
                val prediction = backendClient.predictLandmarks(
                    baseUrl = baseUrl,
                    model = model,
                    sequence = sequenceSnapshot
                )
                val roundTripMs = SystemClock.elapsedRealtime() - startedAt
                if (closed.get() || requestGeneration.get() != generation) {
                    return@execute
                }

                synchronized(this) {
                    latestPrediction = prediction
                    latestRoundTripMs = roundTripMs
                    latestErrorMessage = null
                    accumulator.applyPrediction(prediction, SystemClock.elapsedRealtime())
                }
            } catch (error: Exception) {
                if (closed.get() || requestGeneration.get() != generation) {
                    return@execute
                }
                synchronized(this) {
                    latestErrorMessage = error.message ?: "Backend prediction failed."
                }
            } finally {
                requestInFlight.set(false)
            }
        }
    }

    @Synchronized
    private fun updateFps(timestampNs: Long) {
        if (lastFrameTimestampNs > 0L && timestampNs > lastFrameTimestampNs) {
            val deltaSec = (timestampNs - lastFrameTimestampNs) / 1_000_000_000f
            if (deltaSec > 0f) {
                val instantaneousFps = 1f / deltaSec
                smoothedFps = if (smoothedFps == 0f) {
                    instantaneousFps
                } else {
                    (smoothedFps * 0.85f) + (instantaneousFps * 0.15f)
                }
            }
        }
        lastFrameTimestampNs = timestampNs
    }

    private fun countHandsInFrame(frame: FloatArray): Int {
        var leftPresent = false
        var rightPresent = false

        for (i in 0 until 63) {
            if (frame[i] != 0f) {
                leftPresent = true
                break
            }
        }
        for (i in 63 until FEATURE_SIZE) {
            if (frame[i] != 0f) {
                rightPresent = true
                break
            }
        }

        return when {
            leftPresent && rightPresent -> 2
            leftPresent || rightPresent -> 1
            else -> 0
        }
    }

    override fun close() {
        closed.set(true)
        requestExecutor.shutdownNow()
        handExtractor.close()
    }
}
