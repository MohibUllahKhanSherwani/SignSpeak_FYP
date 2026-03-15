package com.example.kotlinfrontend

import androidx.camera.core.ImageProxy
import java.io.Closeable
import kotlin.math.max

data class LiveInferenceState(
    val rawPrediction: PredictionResult?,
    val displayedLabel: String,
    val displayedConfidence: Float,
    val stability: Float,
    val overlayHands: List<HandWireframe>,
    val overlayFace: FaceWireframe?,
    val bufferCount: Int,
    val sequenceLength: Int,
    val handsInFrame: Int,
    val fps: Float,
    val roundTripMs: Long?,
    val requestInFlight: Boolean,
    val errorMessage: String?
)

class LiveSignEngine(
    context: android.content.Context,
    private val model: SignModel,
    private val useQuantizedModel: Boolean = false,
    private val minInferenceGapMs: Long = 250L,
    private val smoothingWindowSize: Int = 8
) : StreamingSignEngine {
    private val handExtractor = HandExtractor(context)
    private val classifier = TfliteSignClassifier(
        context = context,
        model = model,
        useQuantizedModel = useQuantizedModel
    )
    private val frameBuffer = ArrayDeque<FloatArray>(classifier.sequenceLength)
    private val accumulator = PredictionAccumulator(smoothingWindowSize)

    private var lastInferenceTimestampMs = -1L
    private var lastFrameTimestampNs = -1L
    private var smoothedFps = 0f
    private var latestPrediction: PredictionResult? = null

    @Synchronized
    override fun warmUp() {
        val zeros = List(classifier.sequenceLength) { FloatArray(126) }
        classifier.predict(zeros)
    }

    @Synchronized
    override fun resetCapture(clearTranscript: Boolean) {
        frameBuffer.clear()
        latestPrediction = null
        lastInferenceTimestampMs = -1L
        accumulator.reset(clearTranscript = clearTranscript)
    }

    @Synchronized
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
        updateFps(timestampNs)

        if (frameBuffer.size == classifier.sequenceLength) {
            frameBuffer.removeFirst()
        }
        frameBuffer.addLast(frameVector)

        val shouldInfer = frameBuffer.size == classifier.sequenceLength &&
            (lastInferenceTimestampMs < 0L || (timestampMs - lastInferenceTimestampMs) >= minInferenceGapMs)

        if (shouldInfer) {
            val snapshot = frameBuffer.toList()
            latestPrediction = classifier.predict(snapshot)
            lastInferenceTimestampMs = timestampMs
            accumulator.applyPrediction(latestPrediction!!, timestampMs)
        }

        val displayState = accumulator.snapshot()

        return LiveInferenceState(
            rawPrediction = latestPrediction,
            displayedLabel = displayState.label,
            displayedConfidence = displayState.confidence,
            stability = displayState.stability,
            overlayHands = extraction.hands,
            overlayFace = extraction.face,
            bufferCount = frameBuffer.size,
            sequenceLength = classifier.sequenceLength,
            handsInFrame = countHandsInFrame(frameVector),
            fps = smoothedFps,
            roundTripMs = null,
            requestInFlight = false,
            errorMessage = null
        )
    }

    @Synchronized
    override fun clearTranscript() {
        accumulator.clearTranscript()
    }

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
        for (i in 63 until 126) {
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
        handExtractor.close()
        classifier.close()
    }
}
