package com.example.kotlinfrontend

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class SignBackendClient {
    companion object {
        private const val EXPECTED_SEQUENCE_LENGTH = 60
        private const val EXPECTED_FEATURE_SIZE = 126
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }

    private val healthClient = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(3, TimeUnit.SECONDS)
        .writeTimeout(3, TimeUnit.SECONDS)
        .build()

    private val predictionClient = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(8, TimeUnit.SECONDS)
        .writeTimeout(8, TimeUnit.SECONDS)
        .build()

    fun checkHealth(baseUrl: String): Boolean {
        val url = buildBaseUrl(baseUrl)
            .newBuilder()
            .addPathSegment("health")
            .build()
        val request = Request.Builder()
            .url(url)
            .get()
            .build()

        return try {
            healthClient.newCall(request).execute().use { response ->
                response.isSuccessful
            }
        } catch (_: Exception) {
            false
        }
    }

    fun predictLandmarks(
        baseUrl: String,
        model: SignModel,
        sequence: List<FloatArray>
    ): PredictionResult {
        validateSequence(sequence)

        val url = buildBaseUrl(baseUrl)
            .newBuilder()
            .addPathSegment("predict")
            .addQueryParameter("model", model.backendKey)
            .build()

        val request = Request.Builder()
            .url(url)
            .post(buildPayload(sequence).toRequestBody(JSON_MEDIA_TYPE))
            .build()

        return predictionClient.newCall(request).execute().use { response ->
            val bodyText = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                val detail = bodyText.ifBlank { "HTTP ${response.code}" }
                throw IOException("Backend prediction failed: $detail")
            }

            val json = JSONObject(bodyText)
            val action = json.optString("action", "unknown").ifBlank { "unknown" }
            val confidence = json.optDouble("confidence", 0.0).toFloat().coerceIn(0f, 1f)
            val processingTimeMs = json.optLong("processing_time_ms", 0L)
            val framesProcessed = json.optInt("frames_processed", sequence.size)
            val handsDetected = json.optInt("hands_detected", countDetectedFrames(sequence))
            val probabilities = json.optJSONObject("all_probabilities")

            val topScores = mutableListOf<PredictionScore>()
            if (probabilities != null) {
                val keys = probabilities.keys()
                while (keys.hasNext()) {
                    val label = keys.next()
                    topScores.add(
                        PredictionScore(
                            label = label,
                            confidence = probabilities.optDouble(label, 0.0).toFloat()
                        )
                    )
                }
            }

            if (topScores.none { it.label == action }) {
                topScores.add(PredictionScore(label = action, confidence = confidence))
            }

            topScores.sortByDescending { it.confidence }

            return PredictionResult(
                label = action,
                confidence = confidence,
                topScores = topScores,
                processingTimeMs = processingTimeMs,
                framesProcessed = framesProcessed,
                handsDetected = handsDetected,
                model = model
            )
        }
    }

    private fun buildBaseUrl(baseUrl: String) = baseUrl
        .trim()
        .toHttpUrlOrNull()
        ?: throw IllegalArgumentException("Invalid backend URL: $baseUrl")

    private fun buildPayload(sequence: List<FloatArray>): String {
        val landmarksArray = JSONArray()
        sequence.forEach { frame ->
            val frameArray = JSONArray()
            frame.forEach { value ->
                frameArray.put(value.toDouble())
            }
            landmarksArray.put(frameArray)
        }

        return JSONObject()
            .put("landmarks", landmarksArray)
            .toString()
    }

    private fun validateSequence(sequence: List<FloatArray>) {
        require(sequence.size == EXPECTED_SEQUENCE_LENGTH) {
            "Invalid sequence length ${sequence.size}; expected $EXPECTED_SEQUENCE_LENGTH."
        }
        sequence.forEachIndexed { index, frame ->
            require(frame.size == EXPECTED_FEATURE_SIZE) {
                "Frame $index has ${frame.size} values; expected $EXPECTED_FEATURE_SIZE."
            }
        }
    }

    private fun countDetectedFrames(sequence: List<FloatArray>): Int {
        var count = 0
        sequence.forEach { frame ->
            val leftPresent = frame.slice(0 until 63).any { it != 0f }
            val rightPresent = frame.slice(63 until EXPECTED_FEATURE_SIZE).any { it != 0f }
            if (leftPresent || rightPresent) {
                count += 1
            }
        }
        return count
    }
}
