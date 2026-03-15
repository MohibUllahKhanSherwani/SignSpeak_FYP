package com.example.kotlinfrontend

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class JsonSaveResult(
    val uri: Uri,
    val fileName: String
)

class JsonWriter(private val context: Context) {
    companion object {
        const val SEQUENCE_LENGTH = 30
        const val FRAME_DIM = 126
    }

    fun writeToDownloads(
        baseName: String,
        frameVectors: List<FloatArray>,
        sequenceLength: Int = SEQUENCE_LENGTH,
        dim: Int = FRAME_DIM
    ): JsonSaveResult {
        validateShape(frameVectors, sequenceLength, dim)

        val payload = buildJsonPayload(
            sequenceLength = sequenceLength,
            dim = dim,
            frameVectors = frameVectors
        )
        val fileName = buildFileName(baseName)

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            writeWithMediaStore(fileName, payload)
        } else {
            writeLegacy(fileName, payload)
        }
    }

    private fun validateShape(frameVectors: List<FloatArray>, sequenceLength: Int, dim: Int) {
        require(frameVectors.size == sequenceLength) {
            "Expected $sequenceLength frames, received ${frameVectors.size}."
        }
        frameVectors.forEachIndexed { index, frame ->
            require(frame.size == dim) {
                "Frame $index has ${frame.size} values, expected $dim."
            }
        }
    }

    private fun buildJsonPayload(
        sequenceLength: Int,
        dim: Int,
        frameVectors: List<FloatArray>
    ): String {
        val framesArray = JSONArray()
        frameVectors.forEach { frame ->
            val frameArray = JSONArray()
            frame.forEach { value ->
                frameArray.put(value.toDouble())
            }
            framesArray.put(frameArray)
        }

        return JSONObject().apply {
            put("sequence_length", sequenceLength)
            put("dim", dim)
            put("frames", framesArray)
        }.toString(2)
    }

    private fun buildFileName(baseName: String): String {
        val safeName = baseName
            .trim()
            .ifBlank { "landmarks_sequence" }
            .replace(Regex("[^A-Za-z0-9._-]"), "_")
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return "${safeName}_${timestamp}.json"
    }

    private fun writeWithMediaStore(fileName: String, payload: String): JsonSaveResult {
        val resolver = context.contentResolver
        val collection = MediaStore.Downloads.EXTERNAL_CONTENT_URI
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "application/json")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }

        val itemUri = resolver.insert(collection, values)
            ?: throw IOException("Failed to create Downloads entry.")

        try {
            resolver.openOutputStream(itemUri)?.use { stream ->
                stream.write(payload.toByteArray(Charsets.UTF_8))
            } ?: throw IOException("Failed to open output stream for $itemUri")

            val completed = ContentValues().apply {
                put(MediaStore.MediaColumns.IS_PENDING, 0)
            }
            resolver.update(itemUri, completed, null, null)

            return JsonSaveResult(uri = itemUri, fileName = fileName)
        } catch (error: Exception) {
            resolver.delete(itemUri, null, null)
            throw error
        }
    }

    @Suppress("DEPRECATION")
    private fun writeLegacy(fileName: String, payload: String): JsonSaveResult {
        val downloadsDir = Environment.getExternalStoragePublicDirectory(
            Environment.DIRECTORY_DOWNLOADS
        )
        if (!downloadsDir.exists() && !downloadsDir.mkdirs()) {
            throw IOException("Unable to create Downloads directory.")
        }

        val outputFile = File(downloadsDir, fileName)
        FileOutputStream(outputFile).use { stream ->
            stream.write(payload.toByteArray(Charsets.UTF_8))
        }
        return JsonSaveResult(uri = Uri.fromFile(outputFile), fileName = fileName)
    }
}
