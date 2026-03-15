package com.example.kotlinfrontend

import android.content.Context
import android.os.SystemClock
import org.json.JSONArray
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.min

enum class SignModel(
    val displayName: String,
    val backendKey: String,
    val modelAssetPath: String,
    val quantizedModelAssetPath: String,
    val labelsAssetPath: String
) {
    BASELINE(
        displayName = "Baseline",
        backendKey = "baseline",
        modelAssetPath = "action_model_baseline.tflite",
        quantizedModelAssetPath = "action_model_baseline_quantized.tflite",
        labelsAssetPath = "labels_baseline.json"
    ),
    AUGMENTED(
        displayName = "Augmented",
        backendKey = "augmented",
        modelAssetPath = "action_model_augmented.tflite",
        quantizedModelAssetPath = "action_model_augmented_quantized.tflite",
        labelsAssetPath = "labels_augmented.json"
    )
}

data class PredictionScore(
    val label: String,
    val confidence: Float
)

data class PredictionResult(
    val label: String,
    val confidence: Float,
    val topScores: List<PredictionScore>,
    val processingTimeMs: Long,
    val framesProcessed: Int,
    val handsDetected: Int,
    val model: SignModel
)

class TfliteSignClassifier(
    context: Context,
    val model: SignModel,
    val useQuantizedModel: Boolean = false
) : Closeable {
    companion object {
        private const val EXPECTED_FEATURE_SIZE = 126
    }

    private val interpreter: Interpreter
    private val labels: List<String>
    val sequenceLength: Int
    private val featureSize: Int
    private val outputClassCount: Int
    private lateinit var reusableInput: Array<Array<FloatArray>>
    private lateinit var reusableOutput: Array<FloatArray>
    private val selectedModelAssetPath: String

    init {
        selectedModelAssetPath = if (useQuantizedModel) {
            model.quantizedModelAssetPath
        } else {
            model.modelAssetPath
        }
        val modelBuffer = loadModelFile(context, selectedModelAssetPath)
        interpreter = Interpreter(
            modelBuffer,
            Interpreter.Options().apply { setNumThreads(4) }
        )

        val inputShape = interpreter.getInputTensor(0).shape()
        require(inputShape.size == 3 && inputShape[0] == 1) {
            "Unsupported input shape: ${inputShape.joinToString(prefix = "[", postfix = "]")}"
        }
        sequenceLength = inputShape[1]
        featureSize = inputShape[2]
        require(featureSize == EXPECTED_FEATURE_SIZE) {
            "Expected feature size $EXPECTED_FEATURE_SIZE but model has $featureSize."
        }

        val loadedLabels = loadLabels(context, model.labelsAssetPath)
        require(loadedLabels.isNotEmpty()) { "Label file '${model.labelsAssetPath}' is empty." }
        labels = loadedLabels

        val outputShape = interpreter.getOutputTensor(0).shape()
        outputClassCount = when (outputShape.size) {
            1 -> outputShape[0]
            2 -> outputShape[1]
            else -> throw IllegalStateException(
                "Unsupported output shape: ${outputShape.joinToString(prefix = "[", postfix = "]")}"
            )
        }
        require(outputClassCount > 0) {
            "Model output classes must be greater than zero."
        }

        reusableInput = Array(1) { Array(sequenceLength) { FloatArray(featureSize) } }
        reusableOutput = Array(1) { FloatArray(outputClassCount) }
    }

    fun predict(sequence: List<FloatArray>): PredictionResult {
        require(sequence.size == sequenceLength) {
            "Invalid sequence length ${sequence.size}; expected $sequenceLength."
        }
        sequence.forEachIndexed { index, frame ->
            require(frame.size == featureSize) {
                "Frame $index has ${frame.size} values; expected $featureSize."
            }
        }

        val input = reusableInput
        for (t in 0 until sequenceLength) {
            val frame = sequence[t]
            for (i in 0 until featureSize) {
                input[0][t][i] = frame[i]
            }
        }

        val output = reusableOutput
        java.util.Arrays.fill(output[0], 0f)
        val startedAt = SystemClock.elapsedRealtime()
        interpreter.run(input, output)
        val processingTimeMs = SystemClock.elapsedRealtime() - startedAt

        val probabilities = output[0]
        val labelCount = min(labels.size, probabilities.size)
        if (labelCount == 0) {
            return PredictionResult(
                label = "unknown",
                confidence = 0f,
                topScores = emptyList(),
                processingTimeMs = processingTimeMs,
                framesProcessed = sequenceLength,
                handsDetected = countDetectedFrames(sequence),
                model = model
            )
        }

        var bestIndex = 0
        for (i in 1 until labelCount) {
            if (probabilities[i] > probabilities[bestIndex]) {
                bestIndex = i
            }
        }

        val sortedScores = (0 until labelCount)
            .map { index ->
                PredictionScore(
                    label = labels[index],
                    confidence = probabilities[index]
                )
            }
            .sortedByDescending { it.confidence }

        return PredictionResult(
            label = labels[bestIndex],
            confidence = probabilities[bestIndex],
            topScores = sortedScores,
            processingTimeMs = processingTimeMs,
            framesProcessed = sequenceLength,
            handsDetected = countDetectedFrames(sequence),
            model = model
        )
    }

    private fun countDetectedFrames(sequence: List<FloatArray>): Int {
        var count = 0
        sequence.forEach { frame ->
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

            if (leftPresent || rightPresent) {
                count += 1
            }
        }
        return count
    }

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        return try {
            val fileDescriptor = context.assets.openFd(assetPath)
            FileInputStream(fileDescriptor.fileDescriptor).channel.use { channel ->
                channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.startOffset,
                    fileDescriptor.declaredLength
                )
            }
        } catch (error: Exception) {
            throw IllegalStateException(
                "Unable to load model asset '$assetPath'. Place it in app/src/main/assets/.",
                error
            )
        }
    }

    private fun loadLabels(context: Context, assetPath: String): List<String> {
        val rawJson = try {
            context.assets.open(assetPath).bufferedReader().use { it.readText() }
        } catch (error: Exception) {
            throw IllegalStateException(
                "Unable to load labels asset '$assetPath'. Place it in app/src/main/assets/.",
                error
            )
        }
        val jsonArray = JSONArray(rawJson)
        return List(jsonArray.length()) { index ->
            jsonArray.optString(index, "").trim()
        }.filter { it.isNotEmpty() }
    }

    override fun close() {
        interpreter.close()
    }
}
