package com.example.kotlinfrontend

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.kotlinfrontend.ui.theme.KotlinFrontendTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.net.URI
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            KotlinFrontendTheme {
                Surface(color = MaterialTheme.colorScheme.background) {
                    LiveCameraScreen()
                }
            }
        }
    }
}

private val HAND_CONNECTIONS: List<Pair<Int, Int>> = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4,
    0 to 5, 5 to 6, 6 to 7, 7 to 8,
    5 to 9, 9 to 10, 10 to 11, 11 to 12,
    9 to 13, 13 to 14, 14 to 15, 15 to 16,
    13 to 17, 17 to 18, 18 to 19, 19 to 20,
    0 to 17
)

private fun validateAndNormalizeBackendUrl(rawValue: String): String {
    val normalized = rawValue.trim().trimEnd('/')
    require(normalized.isNotBlank()) { "Backend URL is required." }

    val uri = try {
        URI(normalized)
    } catch (_: Exception) {
        throw IllegalArgumentException("Enter a valid http:// or https:// URL.")
    }

    val scheme = uri.scheme?.lowercase().orEmpty()
    require(scheme == "http" || scheme == "https") {
        "Enter a valid http:// or https:// URL."
    }
    require(!uri.host.isNullOrBlank()) {
        "Enter a valid backend host."
    }

    return normalized
}

@Composable
private fun LiveCameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainExecutor = remember { ContextCompat.getMainExecutor(context) }
    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
    }
    val analysisExecutor: ExecutorService = remember { Executors.newSingleThreadExecutor() }
    val modelLoadExecutor: ExecutorService = remember { Executors.newSingleThreadExecutor() }
    val recordingGate = remember { AtomicBoolean(false) }
    val modelReadyGate = remember { AtomicBoolean(false) }
    val sessionCounter = remember { AtomicInteger(0) }
    val liveEngineRef = remember { AtomicReference<StreamingSignEngine?>(null) }
    val targetFpsGate = remember { AtomicInteger(50) }
    val lastProcessedFrameNs = remember { AtomicLong(0L) }
    val overlayEnabledGate = remember { AtomicBoolean(true) }
    val faceEnabledGate = remember { AtomicBoolean(true) }
    val backendSettingsRepository = remember { BackendSettingsRepository(context.applicationContext) }
    val backendClient = remember { SignBackendClient() }
    val coroutineScope = rememberCoroutineScope()

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    var inferenceMode by remember { mutableStateOf(InferenceMode.ON_DEVICE) }
    var selectedModel by remember { mutableStateOf(SignModel.AUGMENTED) }
    var useQuantizedModel by remember { mutableStateOf(false) }
    var performanceMode by remember { mutableStateOf(true) }
    var showHandOverlay by remember { mutableStateOf(false) }
    var showFaceOverlay by remember { mutableStateOf(false) }
    var controlsExpanded by remember { mutableStateOf(false) }
    var isRecording by remember { mutableStateOf(false) }
    var isModelLoading by remember { mutableStateOf(false) }
    var isModelReady by remember { mutableStateOf(false) }
    var targetCaptureFps by remember { mutableIntStateOf(50) }
    var statusText by remember { mutableStateOf("Initializing live camera...") }
    var predictedText by remember { mutableStateOf("--") }
    var confidence by remember { mutableFloatStateOf(0f) }
    var stability by remember { mutableFloatStateOf(0f) }
    var overlayHands by remember { mutableStateOf<List<HandWireframe>>(emptyList()) }
    var overlayFace by remember { mutableStateOf<FaceWireframe?>(null) }
    var fps by remember { mutableFloatStateOf(0f) }
    var handsInFrame by remember { mutableIntStateOf(0) }
    var bufferCount by remember { mutableIntStateOf(0) }
    var sequenceLength by remember { mutableIntStateOf(60) }
    var capturedFrameCount by remember { mutableIntStateOf(0) }
    var inferenceMs by remember { mutableLongStateOf(0L) }
    var roundTripMs by remember { mutableStateOf<Long?>(null) }
    var requestInFlight by remember { mutableStateOf(false) }
    var cameraProvider by remember { mutableStateOf<ProcessCameraProvider?>(null) }
    var backendUrlInput by remember { mutableStateOf("") }
    var activeBackendUrl by remember { mutableStateOf("") }
    var backendUrlError by remember { mutableStateOf<String?>(null) }
    var backendConnectionMessage by remember { mutableStateOf<String?>(null) }
    var isBackendReachable by remember { mutableStateOf(false) }
    var isBackendChecking by remember { mutableStateOf(false) }
    var didLoadSavedBackendUrl by remember { mutableStateOf(false) }

    overlayEnabledGate.set(showHandOverlay && !performanceMode)
    faceEnabledGate.set(showFaceOverlay && !performanceMode)

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasPermission = granted
        statusText = if (granted) {
            "Camera permission granted. Initializing translator..."
        } else {
            "Camera permission denied."
        }
    }

    suspend fun refreshBackendHealth(updateStatusText: Boolean): Boolean {
        if (activeBackendUrl.isBlank()) {
            isBackendReachable = false
            isBackendChecking = false
            backendConnectionMessage = "Set and save a backend URL first."
            if (updateStatusText && inferenceMode == InferenceMode.BACKEND_LANDMARKS) {
                statusText = "Backend mode needs a reachable server before recording."
            }
            return false
        }

        isBackendChecking = true
        backendConnectionMessage = "Checking backend health..."
        val connected = withContext(Dispatchers.IO) {
            backendClient.checkHealth(activeBackendUrl)
        }
        isBackendChecking = false
        isBackendReachable = connected
        backendConnectionMessage = if (connected) {
            "Backend reachable at $activeBackendUrl"
        } else {
            "Backend unreachable from phone"
        }

        if (updateStatusText && inferenceMode == InferenceMode.BACKEND_LANDMARKS) {
            statusText = if (connected) {
                "Backend connected. Press Record."
            } else {
                "Backend health check failed. Verify the URL and server."
            }
        }
        return connected
    }

    suspend fun applyBackendUrl() {
        val normalized = try {
            validateAndNormalizeBackendUrl(backendUrlInput)
        } catch (error: IllegalArgumentException) {
            backendUrlError = error.message ?: "Enter a valid backend URL."
            isBackendReachable = false
            backendConnectionMessage = null
            return
        }

        backendUrlError = null
        backendUrlInput = normalized
        activeBackendUrl = normalized
        withContext(Dispatchers.IO) {
            backendSettingsRepository.saveBackendBaseUrl(normalized)
        }
        refreshBackendHealth(updateStatusText = inferenceMode == InferenceMode.BACKEND_LANDMARKS)
    }

    fun stopRecording() {
        isRecording = false
        recordingGate.set(false)
        lastProcessedFrameNs.set(0L)
        overlayHands = emptyList()
        overlayFace = null
        roundTripMs = null
        requestInFlight = false
    }

    suspend fun startRecording() {
        if (!isModelReady) {
            statusText = "Pipeline not ready yet. Please wait..."
            return
        }

        if (inferenceMode == InferenceMode.BACKEND_LANDMARKS) {
            val connected = refreshBackendHealth(updateStatusText = true)
            if (!connected) {
                return
            }
        }

        liveEngineRef.get()?.resetCapture(clearTranscript = false)
        predictedText = "--"
        confidence = 0f
        stability = 0f
        inferenceMs = 0L
        roundTripMs = null
        requestInFlight = false
        capturedFrameCount = 0
        overlayHands = emptyList()
        overlayFace = null
        lastProcessedFrameNs.set(0L)
        isRecording = true
        recordingGate.set(true)
        statusText = when (inferenceMode) {
            InferenceMode.ON_DEVICE -> {
                if (performanceMode) {
                    "Recording started at $targetCaptureFps FPS (speed mode)."
                } else {
                    "Recording started at $targetCaptureFps FPS."
                }
            }
            InferenceMode.BACKEND_LANDMARKS -> {
                "Recording started at $targetCaptureFps FPS. Sending landmarks to backend."
            }
        }
    }

    fun shutdownSession() {
        stopRecording()
        modelReadyGate.set(false)
        isModelReady = false
        isModelLoading = false
        cameraProvider?.unbindAll()
        liveEngineRef.getAndSet(null)?.close()
    }

    fun bindLivePipeline(
        mode: InferenceMode,
        model: SignModel,
        performanceModeEnabled: Boolean,
        quantizedEnabled: Boolean,
        backendBaseUrl: String
    ) {
        val sessionId = sessionCounter.incrementAndGet()
        stopRecording()
        modelReadyGate.set(false)
        isModelReady = false
        isModelLoading = true
        val modelVariantText = if (quantizedEnabled && mode == InferenceMode.ON_DEVICE) {
            "quantized"
        } else {
            "standard"
        }
        statusText = when (mode) {
            InferenceMode.ON_DEVICE -> "Loading ${model.displayName} ($modelVariantText) model..."
            InferenceMode.BACKEND_LANDMARKS -> "Initializing backend landmark pipeline..."
        }
        predictedText = "--"
        confidence = 0f
        stability = 0f
        inferenceMs = 0L
        roundTripMs = null
        requestInFlight = false
        bufferCount = 0
        overlayHands = emptyList()
        overlayFace = null
        capturedFrameCount = 0
        lastProcessedFrameNs.set(0L)

        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener(
            {
                try {
                    val provider = providerFuture.get()
                    cameraProvider = provider
                    provider.unbindAll()
                    liveEngineRef.getAndSet(null)?.close()

                    val previewUseCase = Preview.Builder().build().also { preview ->
                        preview.surfaceProvider = previewView.surfaceProvider
                    }
                    val analysisResolution = if (performanceModeEnabled) {
                        if (mode == InferenceMode.BACKEND_LANDMARKS) {
                            Size(256, 192)
                        } else {
                            Size(320, 240)
                        }
                    } else {
                        Size(640, 480)
                    }
                    val analysisUseCase = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setTargetResolution(analysisResolution)
                        .build()

                    analysisUseCase.setAnalyzer(analysisExecutor) { imageProxy ->
                        if (!recordingGate.get() || !modelReadyGate.get()) {
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        val targetFps = targetFpsGate.get().coerceAtLeast(1)
                        val frameTimestampNs = imageProxy.imageInfo.timestamp
                        val minFrameIntervalNs = 1_000_000_000L / targetFps.toLong()
                        val previousTimestampNs = lastProcessedFrameNs.get()
                        if (previousTimestampNs > 0L &&
                            frameTimestampNs > previousTimestampNs &&
                            frameTimestampNs - previousTimestampNs < minFrameIntervalNs
                        ) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        lastProcessedFrameNs.set(frameTimestampNs)

                        try {
                            val engine = liveEngineRef.get() ?: return@setAnalyzer
                            val liveState = engine.process(
                                imageProxy = imageProxy,
                                includeOverlay = overlayEnabledGate.get(),
                                includeFace = faceEnabledGate.get()
                            )
                            mainExecutor.execute {
                                if (sessionCounter.get() != sessionId) {
                                    return@execute
                                }
                                capturedFrameCount += 1
                                bufferCount = liveState.bufferCount
                                sequenceLength = liveState.sequenceLength
                                handsInFrame = liveState.handsInFrame
                                fps = liveState.fps
                                predictedText = liveState.displayedLabel
                                confidence = liveState.displayedConfidence.coerceIn(0f, 1f)
                                stability = liveState.stability.coerceIn(0f, 1f)
                                overlayHands = liveState.overlayHands
                                overlayFace = liveState.overlayFace
                                roundTripMs = liveState.roundTripMs
                                requestInFlight = liveState.requestInFlight

                                liveState.rawPrediction?.let { prediction ->
                                    inferenceMs = prediction.processingTimeMs
                                }

                                statusText = when {
                                    !liveState.errorMessage.isNullOrBlank() -> liveState.errorMessage
                                    liveState.rawPrediction != null &&
                                        mode == InferenceMode.BACKEND_LANDMARKS &&
                                        liveState.requestInFlight ->
                                        "Recording at ${targetFpsGate.get()} FPS. Backend request in flight..."
                                    liveState.rawPrediction != null &&
                                        mode == InferenceMode.BACKEND_LANDMARKS ->
                                        "Recording at ${targetFpsGate.get()} FPS and predicting on backend..."
                                    liveState.rawPrediction != null ->
                                        "Recording at ${targetFpsGate.get()} FPS and predicting..."
                                    liveState.requestInFlight &&
                                        mode == InferenceMode.BACKEND_LANDMARKS ->
                                        "Sending landmarks to backend..."
                                    else ->
                                        "Collecting landmarks (${liveState.bufferCount}/${liveState.sequenceLength})"
                                }
                            }
                        } catch (error: Exception) {
                            mainExecutor.execute {
                                if (sessionCounter.get() != sessionId) {
                                    return@execute
                                }
                                statusText = error.message ?: "Live inference error."
                            }
                        } finally {
                            imageProxy.close()
                        }
                    }

                    provider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_FRONT_CAMERA,
                        previewUseCase,
                        analysisUseCase
                    )

                    modelLoadExecutor.execute {
                        try {
                            val engine: StreamingSignEngine = when (mode) {
                                InferenceMode.ON_DEVICE -> LiveSignEngine(
                                    context = context,
                                    model = model,
                                    useQuantizedModel = quantizedEnabled
                                )
                                InferenceMode.BACKEND_LANDMARKS -> BackendLiveSignEngine(
                                    context = context,
                                    baseUrl = backendBaseUrl,
                                    model = model
                                )
                            }
                            engine.warmUp()
                            mainExecutor.execute {
                                if (sessionCounter.get() != sessionId) {
                                    engine.close()
                                    return@execute
                                }
                                liveEngineRef.getAndSet(engine)?.close()
                                modelReadyGate.set(true)
                                isModelLoading = false
                                isModelReady = true
                                statusText = when (mode) {
                                    InferenceMode.ON_DEVICE -> {
                                        val readyVariantText = if (quantizedEnabled) {
                                            "Quantized"
                                        } else {
                                            "Standard"
                                        }
                                        if (performanceModeEnabled) {
                                            "$readyVariantText model ready in speed mode (${analysisResolution.width}x${analysisResolution.height}). Press Record."
                                        } else {
                                            "$readyVariantText model ready. Set FPS and press Record."
                                        }
                                    }
                                    InferenceMode.BACKEND_LANDMARKS -> {
                                        if (backendBaseUrl.isBlank()) {
                                            "Backend pipeline ready. Set backend URL and check health."
                                        } else {
                                            "Backend pipeline ready. Check health before recording."
                                        }
                                    }
                                }
                            }
                        } catch (error: Exception) {
                            mainExecutor.execute {
                                if (sessionCounter.get() != sessionId) {
                                    return@execute
                                }
                                modelReadyGate.set(false)
                                isModelLoading = false
                                isModelReady = false
                                statusText = error.message ?: "Failed to load pipeline."
                            }
                        }
                    }
                } catch (error: Exception) {
                    isModelLoading = false
                    isModelReady = false
                    modelReadyGate.set(false)
                    statusText = error.message ?: "Failed to start camera."
                }
            },
            mainExecutor
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            shutdownSession()
            analysisExecutor.shutdown()
            modelLoadExecutor.shutdown()
        }
    }

    LaunchedEffect(Unit) {
        val savedBackendUrl = backendSettingsRepository.backendBaseUrlFlow.first()
        backendUrlInput = savedBackendUrl
        activeBackendUrl = savedBackendUrl
        didLoadSavedBackendUrl = true
        if (savedBackendUrl.isNotBlank()) {
            refreshBackendHealth(updateStatusText = false)
        }
    }

    LaunchedEffect(hasPermission, inferenceMode, selectedModel, performanceMode, useQuantizedModel, activeBackendUrl) {
        if (hasPermission) {
            bindLivePipeline(
                mode = inferenceMode,
                model = selectedModel,
                performanceModeEnabled = performanceMode,
                quantizedEnabled = inferenceMode == InferenceMode.ON_DEVICE && useQuantizedModel,
                backendBaseUrl = activeBackendUrl
            )
        }
    }

    LaunchedEffect(inferenceMode, activeBackendUrl, didLoadSavedBackendUrl) {
        if (!didLoadSavedBackendUrl) {
            return@LaunchedEffect
        }
        if (inferenceMode == InferenceMode.BACKEND_LANDMARKS) {
            if (activeBackendUrl.isBlank()) {
                isBackendReachable = false
                backendConnectionMessage = "Enter a backend URL to enable server inference."
            } else {
                refreshBackendHealth(updateStatusText = !isRecording)
            }
        }
    }

    val gradient = Brush.linearGradient(
        colors = listOf(Color(0xFF0A1628), Color(0xFF12355B), Color(0xFF1F7A8C)),
        start = Offset.Zero,
        end = Offset(1600f, 2400f)
    )
    val readyIndicatorColor = when {
        isModelLoading || isBackendChecking -> Color(0xFFF59E0B)
        inferenceMode == InferenceMode.BACKEND_LANDMARKS && isModelReady && isBackendReachable -> Color(0xFF22C55E)
        inferenceMode == InferenceMode.BACKEND_LANDMARKS && isModelReady -> Color(0xFF0EA5E9)
        isModelReady -> Color(0xFF22C55E)
        else -> Color(0xFF64748B)
    }
    val readyIndicatorLabel = when {
        isModelLoading -> "Loading"
        isBackendChecking -> "Checking"
        inferenceMode == InferenceMode.BACKEND_LANDMARKS && isModelReady && isBackendReachable -> "Ready"
        inferenceMode == InferenceMode.BACKEND_LANDMARKS && isModelReady -> "Offline"
        isModelReady -> "Ready"
        else -> "Idle"
    }
    val canStartRecording = if (isRecording) {
        true
    } else {
        hasPermission &&
            isModelReady &&
            !isModelLoading &&
            (
                inferenceMode == InferenceMode.ON_DEVICE ||
                    (isBackendReachable && !isBackendChecking)
                )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(gradient)
            .padding(14.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xCCFFFFFF)),
                shape = RoundedCornerShape(20.dp)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "SignSpeak Live Translator",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF0B2239)
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = statusText,
                            style = MaterialTheme.typography.bodyMedium,
                            color = Color(0xFF355070),
                            modifier = Modifier.weight(1f)
                        )
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .width(10.dp)
                                    .height(10.dp)
                                    .clip(CircleShape)
                                    .background(readyIndicatorColor)
                            )
                            Text(
                                text = readyIndicatorLabel,
                                style = MaterialTheme.typography.labelMedium,
                                color = Color(0xFF0F172A)
                            )
                        }
                    }

                    if (isModelLoading || isBackendChecking) {
                        LinearProgressIndicator(
                            progress = {
                                if (isModelLoading) 0.4f else 0.8f
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(6.dp)
                                .clip(RoundedCornerShape(10.dp)),
                            color = Color(0xFF0EA5E9),
                            trackColor = Color(0xFFE2E8F0)
                        )
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Quick Controls",
                            style = MaterialTheme.typography.titleSmall,
                            color = Color(0xFF0F172A),
                            fontWeight = FontWeight.SemiBold
                        )
                        TextButton(
                            onClick = { controlsExpanded = !controlsExpanded },
                            enabled = !isModelLoading
                        ) {
                            Text(if (controlsExpanded) "Hide" else "Show")
                        }
                    }

                    AnimatedVisibility(
                        visible = controlsExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                ModelChip(
                                    text = InferenceMode.ON_DEVICE.displayName,
                                    selected = inferenceMode == InferenceMode.ON_DEVICE,
                                    onClick = {
                                        if (inferenceMode != InferenceMode.ON_DEVICE) {
                                            inferenceMode = InferenceMode.ON_DEVICE
                                        }
                                    }
                                )
                                ModelChip(
                                    text = InferenceMode.BACKEND_LANDMARKS.displayName,
                                    selected = inferenceMode == InferenceMode.BACKEND_LANDMARKS,
                                    onClick = {
                                        if (inferenceMode != InferenceMode.BACKEND_LANDMARKS) {
                                            inferenceMode = InferenceMode.BACKEND_LANDMARKS
                                        }
                                    }
                                )
                            }

                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                ModelChip(
                                    text = "Augmented",
                                    selected = selectedModel == SignModel.AUGMENTED,
                                    onClick = {
                                        if (selectedModel != SignModel.AUGMENTED) {
                                            selectedModel = SignModel.AUGMENTED
                                        }
                                    }
                                )
                                ModelChip(
                                    text = "Baseline",
                                    selected = selectedModel == SignModel.BASELINE,
                                    onClick = {
                                        if (selectedModel != SignModel.BASELINE) {
                                            selectedModel = SignModel.BASELINE
                                        }
                                    }
                                )
                            }

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC)),
                                shape = RoundedCornerShape(16.dp)
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(horizontal = 14.dp, vertical = 10.dp),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(
                                            text = "Model Variant",
                                            style = MaterialTheme.typography.titleSmall,
                                            color = Color(0xFF0F172A),
                                            fontWeight = FontWeight.SemiBold
                                        )
                                        Text(
                                            text = when {
                                                inferenceMode == InferenceMode.BACKEND_LANDMARKS ->
                                                    "Backend mode uses the server's baseline or augmented model."
                                                useQuantizedModel ->
                                                    "Quantized (faster, smaller)"
                                                else ->
                                                    "Standard (full precision)"
                                            },
                                            style = MaterialTheme.typography.bodySmall,
                                            color = Color(0xFF475569)
                                        )
                                    }
                                    Switch(
                                        checked = useQuantizedModel,
                                        onCheckedChange = { enabled ->
                                            useQuantizedModel = enabled
                                            statusText = if (enabled) {
                                                "Switching to quantized model..."
                                            } else {
                                                "Switching to standard model..."
                                            }
                                        },
                                        enabled = !isModelLoading && inferenceMode == InferenceMode.ON_DEVICE
                                    )
                                }
                            }

                            if (inferenceMode == InferenceMode.BACKEND_LANDMARKS) {
                                Card(
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC)),
                                    shape = RoundedCornerShape(16.dp)
                                ) {
                                    Column(
                                        modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                                        verticalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        Text(
                                            text = "Backend Server",
                                            style = MaterialTheme.typography.titleSmall,
                                            color = Color(0xFF0F172A),
                                            fontWeight = FontWeight.SemiBold
                                        )
                                        OutlinedTextField(
                                            value = backendUrlInput,
                                            onValueChange = {
                                                backendUrlInput = it
                                                backendUrlError = null
                                            },
                                            modifier = Modifier.fillMaxWidth(),
                                            singleLine = true,
                                            enabled = !isRecording,
                                            label = { Text("Base URL") },
                                            placeholder = { Text("http://192.168.x.x:8000") },
                                            isError = backendUrlError != null
                                        )
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                                        ) {
                                            Button(
                                                onClick = {
                                                    coroutineScope.launch {
                                                        applyBackendUrl()
                                                    }
                                                },
                                                enabled = !isRecording && !isBackendChecking,
                                                modifier = Modifier.weight(1f),
                                                colors = ButtonDefaults.buttonColors(
                                                    containerColor = Color(0xFF0F766E),
                                                    contentColor = Color.White
                                                )
                                            ) {
                                                Text("Apply & Save")
                                            }
                                            Button(
                                                onClick = {
                                                    coroutineScope.launch {
                                                        refreshBackendHealth(updateStatusText = true)
                                                    }
                                                },
                                                enabled = activeBackendUrl.isNotBlank() && !isBackendChecking,
                                                modifier = Modifier.weight(1f),
                                                colors = ButtonDefaults.buttonColors(
                                                    containerColor = Color(0xFF155E75),
                                                    contentColor = Color.White
                                                )
                                            ) {
                                                Text(if (isBackendChecking) "Checking..." else "Check Health")
                                            }
                                        }
                                        Text(
                                            text = if (activeBackendUrl.isBlank()) {
                                                "Active URL: none"
                                            } else {
                                                "Active URL: $activeBackendUrl"
                                            },
                                            style = MaterialTheme.typography.bodySmall,
                                            color = Color(0xFF475569)
                                        )
                                        backendConnectionMessage?.let { message ->
                                            Text(
                                                text = message,
                                                style = MaterialTheme.typography.bodySmall,
                                                color = if (isBackendReachable) Color(0xFF0F766E) else Color(0xFF92400E)
                                            )
                                        }
                                        backendUrlError?.let { message ->
                                            Text(
                                                text = message,
                                                style = MaterialTheme.typography.bodySmall,
                                                color = Color(0xFFB91C1C)
                                            )
                                        }
                                    }
                                }
                            }

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFF1F5F9)),
                                shape = RoundedCornerShape(16.dp)
                            ) {
                                Column(
                                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                                    verticalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            text = "Capture Speed",
                                            style = MaterialTheme.typography.titleSmall,
                                            color = Color(0xFF0F172A),
                                            fontWeight = FontWeight.SemiBold
                                        )
                                        Text(
                                            text = "$targetCaptureFps FPS",
                                            style = MaterialTheme.typography.titleSmall,
                                            color = Color(0xFF0E7490),
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                    Slider(
                                        value = targetCaptureFps.toFloat(),
                                        onValueChange = { rawValue ->
                                            val adjusted = rawValue.roundToInt().coerceIn(5, 60)
                                            targetCaptureFps = adjusted
                                            targetFpsGate.set(adjusted)
                                        },
                                        valueRange = 5f..60f,
                                        steps = 54,
                                        enabled = !isModelLoading,
                                        colors = SliderDefaults.colors(
                                            thumbColor = Color(0xFF0E7490),
                                            activeTrackColor = Color(0xFF0891B2),
                                            inactiveTrackColor = Color(0xFFCBD5E1)
                                        )
                                    )
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween
                                    ) {
                                        Text(
                                            text = "5 FPS",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = Color(0xFF64748B)
                                        )
                                        Text(
                                            text = "Balanced: 18-30",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = Color(0xFF64748B)
                                        )
                                        Text(
                                            text = "60 FPS",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = Color(0xFF64748B)
                                        )
                                    }
                                }
                            }

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC)),
                                shape = RoundedCornerShape(16.dp)
                            ) {
                                Column(
                                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                                    verticalArrangement = Arrangement.spacedBy(2.dp)
                                ) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            text = "Performance Mode",
                                            style = MaterialTheme.typography.titleSmall,
                                            color = Color(0xFF0F172A),
                                            fontWeight = FontWeight.SemiBold
                                        )
                                        Switch(
                                            checked = performanceMode,
                                            onCheckedChange = { enabled ->
                                                performanceMode = enabled
                                                statusText = if (enabled) {
                                                    "Speed mode enabled. Rebuilding live pipeline..."
                                                } else {
                                                    "Speed mode disabled. Rebuilding live pipeline..."
                                                }
                                            },
                                            enabled = !isModelLoading
                                        )
                                    }
                                    Text(
                                        text = if (performanceMode) {
                                            "Lower analysis resolution + optional overlay skip for higher throughput."
                                        } else {
                                            "Normal mode keeps max visual detail."
                                        },
                                        style = MaterialTheme.typography.bodySmall,
                                        color = Color(0xFF475569)
                                    )

                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            text = "Hand Wireframe",
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = Color(0xFF1E293B)
                                        )
                                        Switch(
                                            checked = showHandOverlay,
                                            onCheckedChange = { showHandOverlay = it },
                                            enabled = !isModelLoading && !performanceMode
                                        )
                                    }

                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            text = "Face Points",
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = Color(0xFF1E293B)
                                        )
                                        Switch(
                                            checked = showFaceOverlay,
                                            onCheckedChange = { showFaceOverlay = it },
                                            enabled = !isModelLoading && !performanceMode
                                        )
                                    }
                                }
                            }
                        }
                    }

                    Button(
                        onClick = {
                            coroutineScope.launch {
                                if (isRecording) {
                                    stopRecording()
                                    statusText = "Recording stopped."
                                } else {
                                    startRecording()
                                }
                            }
                        },
                        enabled = canStartRecording,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isRecording) Color(0xFFB91C1C) else Color(0xFF0F766E),
                            contentColor = Color.White
                        ),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(if (isRecording) "Stop Recording" else "Start Recording")
                    }
                }
            }

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF08111F))
            ) {
                Box(modifier = Modifier.fillMaxSize()) {
                    if (hasPermission) {
                        AndroidView(
                            factory = { previewView },
                            modifier = Modifier.fillMaxSize()
                        )
                        if (isRecording && (overlayHands.isNotEmpty() || overlayFace != null)) {
                            Canvas(modifier = Modifier.fillMaxSize()) {
                                overlayHands.forEach { hand ->
                                    val color = when (hand.side) {
                                        HandSide.LEFT -> Color(0xFF34D399)
                                        HandSide.RIGHT -> Color(0xFFF59E0B)
                                        HandSide.UNKNOWN -> Color(0xFF60A5FA)
                                    }

                                    val mappedPoints = hand.points.map { point ->
                                        val drawX = ((1f - point.x).coerceIn(0f, 1f)) * size.width
                                        val drawY = point.y.coerceIn(0f, 1f) * size.height
                                        Offset(drawX, drawY)
                                    }

                                    HAND_CONNECTIONS.forEach { (start, end) ->
                                        if (start < mappedPoints.size && end < mappedPoints.size) {
                                            drawLine(
                                                color = color.copy(alpha = 0.9f),
                                                start = mappedPoints[start],
                                                end = mappedPoints[end],
                                                strokeWidth = 4f
                                            )
                                        }
                                    }

                                    mappedPoints.forEach { point ->
                                        drawCircle(
                                            color = Color.White,
                                            radius = 5f,
                                            center = point
                                        )
                                        drawCircle(
                                            color = color,
                                            radius = 3f,
                                            center = point
                                        )
                                    }
                                }

                                overlayFace?.let { face ->
                                    val nose = face.nose?.let { point ->
                                        Offset(
                                            ((1f - point.x).coerceIn(0f, 1f)) * size.width,
                                            point.y.coerceIn(0f, 1f) * size.height
                                        )
                                    }
                                    val leftEar = face.leftEar?.let { point ->
                                        Offset(
                                            ((1f - point.x).coerceIn(0f, 1f)) * size.width,
                                            point.y.coerceIn(0f, 1f) * size.height
                                        )
                                    }
                                    val rightEar = face.rightEar?.let { point ->
                                        Offset(
                                            ((1f - point.x).coerceIn(0f, 1f)) * size.width,
                                            point.y.coerceIn(0f, 1f) * size.height
                                        )
                                    }

                                    nose?.let {
                                        drawCircle(color = Color(0xFF38BDF8), radius = 8f, center = it)
                                    }
                                    leftEar?.let {
                                        drawCircle(color = Color(0xFFFB7185), radius = 8f, center = it)
                                    }
                                    rightEar?.let {
                                        drawCircle(color = Color(0xFFFB7185), radius = 8f, center = it)
                                    }
                                }
                            }
                        }
                    } else {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                                Text("Grant Camera Permission")
                            }
                        }
                    }

                    Column(
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .fillMaxWidth()
                            .padding(8.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = Color(0xD90E1B2E)),
                            shape = RoundedCornerShape(14.dp)
                        ) {
                            Column(
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                                verticalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = "Detected: $predictedText",
                                        style = MaterialTheme.typography.titleMedium,
                                        color = Color(0xFFF1FAFF),
                                        fontWeight = FontWeight.Bold,
                                        modifier = Modifier.weight(1f),
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                    Text(
                                        text = "${(confidence * 100f).roundToInt()}%",
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = Color(0xFFD5F7FF),
                                        fontWeight = FontWeight.SemiBold
                                    )
                                }
                                LinearProgressIndicator(
                                    progress = { confidence.coerceIn(0f, 1f) },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(5.dp)
                                        .clip(RoundedCornerShape(10.dp)),
                                    color = Color(0xFF00D1B2),
                                    trackColor = Color(0x3349A9A4)
                                )
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = if (transcript.isBlank()) {
                                            "Sentence: ..."
                                        } else {
                                            "Sentence: $transcript"
                                        },
                                        style = MaterialTheme.typography.labelMedium,
                                        color = Color(0xFFD8F3FF),
                                        modifier = Modifier.weight(1f),
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                    TextButton(
                                        onClick = {
                                            liveEngineRef.get()?.clearTranscript()
                                            transcript = ""
                                            statusText = "Sentence cleared"
                                        },
                                        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 0.dp)
                                    ) {
                                        Text("Clr")
                                    }
                                }

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    CompactMetric(
                                        label = "Cap",
                                        value = capturedFrameCount.toString(),
                                        modifier = Modifier.weight(1f)
                                    )
                                    CompactMetric(
                                        label = "Buf",
                                        value = "$bufferCount/$sequenceLength",
                                        modifier = Modifier.weight(1f)
                                    )
                                    CompactMetric(
                                        label = "FPS",
                                        value = fps.roundToInt().toString(),
                                        modifier = Modifier.weight(1f)
                                    )
                                    CompactMetric(
                                        label = "Inf",
                                        value = "${inferenceMs}ms",
                                        modifier = Modifier.weight(1f)
                                    )
                                }

                                Text(
                                    text = buildString {
                                        append("Hands ")
                                        append(handsInFrame)
                                        append(" | Stable ")
                                        append((stability * 100f).roundToInt())
                                        append("% | Target ")
                                        append(targetCaptureFps)
                                        append("fps")
                                        if (inferenceMode == InferenceMode.BACKEND_LANDMARKS) {
                                            append(" | RTT ")
                                            append(roundTripMs?.let { "${it}ms" } ?: "--")
                                            append(" | ")
                                            append(if (requestInFlight) "Requesting" else "Idle")
                                        }
                                    },
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color(0xFF9CC6D9),
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ModelChip(text: String, selected: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        shape = RoundedCornerShape(50),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) Color(0xFF0E7490) else Color(0xFFE2E8F0),
            contentColor = if (selected) Color.White else Color(0xFF0F172A)
        )
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .width(8.dp)
                    .height(8.dp)
                    .clip(CircleShape)
                    .background(if (selected) Color(0xFF22C55E) else Color(0xFF64748B))
            )
            Text(text)
        }
    }
}

@Composable
private fun CompactMetric(label: String, value: String, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = Color(0xA6192C44)),
        shape = RoundedCornerShape(10.dp)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = Color(0xFF9CC6D9)
            )
            Text(
                text = value,
                style = MaterialTheme.typography.labelMedium,
                color = Color(0xFFF5FBFF),
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}
