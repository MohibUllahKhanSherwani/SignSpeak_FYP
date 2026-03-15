package com.example.kotlinfrontend

import kotlin.math.max

data class PredictionDisplayState(
    val label: String,
    val confidence: Float,
    val stability: Float
)

class PredictionAccumulator(
    // Strict buffer size matching Python's `if len(predictions) > 10: predictions.pop(0)`
    private val smoothingWindowSize: Int = 10
) {
    private val predictionHistory = ArrayDeque<PredictionResult>(smoothingWindowSize)
    
    private var displayedLabel: String = "--"
    private var displayedConfidence = 0f
    private var displayedStability = 0f

    @Synchronized
    fun reset(clearTranscript: Boolean = false) {
        predictionHistory.clear()
        displayedLabel = "--"
        displayedConfidence = 0f
        displayedStability = 0f
    }

    @Synchronized
    fun clearTranscript() {
        // No-op. Left for interface compatibility with MainActivity,
        // but sentence building has been removed.
    }

    @Synchronized
    fun applyPrediction(prediction: PredictionResult, timestampMs: Long): PredictionDisplayState {
        // 1. Maintain strict buffer length
        if (predictionHistory.size == smoothingWindowSize) {
            predictionHistory.removeFirst()
        }
        predictionHistory.addLast(prediction)

        val historyCount = predictionHistory.size.toFloat()
        
        // Count identical labels in recent history
        val occurrenceCount = predictionHistory.count { it.label == prediction.label }
        val stabilityRatio = occurrenceCount / historyCount

        // 2. Strict Threshold Logic (matches Python)
        // Python: if confidence > PREDICTION_THRESHOLD (0.6) and predictions.count(prediction_idx) > 8:
        val meetsConfidenceThreshold = prediction.confidence > 0.6f
        val meetsHistoryThreshold = occurrenceCount >= 8

        if (meetsConfidenceThreshold && meetsHistoryThreshold) {
            displayedLabel = prediction.label
            displayedConfidence = prediction.confidence
            displayedStability = stabilityRatio
        } else {
            // If the current prediction doesn't strictly match, we hold the *last* valid displayed label naturally,
            // but we linearly decay the "displayed" confidence down a bit so the UI knows we are between confident signs
            displayedConfidence = (displayedConfidence * 0.9f).coerceIn(0f, 1f)
            displayedStability = (displayedStability * 0.9f).coerceIn(0f, 1f)
        }

        return snapshot()
    }

    @Synchronized
    fun snapshot(): PredictionDisplayState {
        return PredictionDisplayState(
            label = displayedLabel,
            confidence = displayedConfidence,
            stability = displayedStability
        )
    }
}
