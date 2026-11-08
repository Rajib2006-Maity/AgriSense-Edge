package com.agrisensemobile

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

/**
 * React Native bridge around the ExecuTorch Android runtime (org.pytorch:executorch-android).
 *
 * Loads the CNN classifier (.pte) and UNet segmentation (.pte) models bundled
 * in android/app/src/main/assets/models/, and exposes:
 *   - classifyLeaf(imageBase64): { label, confidence, allScores[] }
 *   - segmentLesion(imageBase64): { maskBase64, affectedAreaPct }
 *
 * Models are loaded lazily and cached; call preloadModels() at app start
 * (from JS, on splash screen) to avoid a cold-start delay on first inference.
 */
class ExecuTorchModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        private const val IMG_SIZE = 64
        private val CLASSES = arrayOf("healthy", "leaf_blight", "leaf_rust", "powdery_mildew")
    }

    private var classifierModule: Module? = null
    private var segmentationModule: Module? = null
    private var loadedQuantized: Boolean? = null

    override fun getName(): String = "ExecuTorchModule"

    /** Copies a .pte from app assets into internal storage because
     * Module.load expects a filesystem path rather than an asset stream. */
    private fun assetToFile(assetName: String): String {
        val outFile = File(reactApplicationContext.filesDir, assetName)
        // Replace the cached copy so updated model assets are used after an app update.
        if (outFile.exists() && !outFile.delete()) {
            throw IllegalStateException("Could not replace cached model: $assetName")
        }
        reactApplicationContext.assets.open("models/$assetName").use { input ->
            FileOutputStream(outFile, false).use { output -> input.copyTo(output) }
        }
        return outFile.absolutePath
    }

    @ReactMethod
    fun preloadModels(useQuantized: Boolean, promise: Promise) {
        try {
            val classifierAsset = if (useQuantized) "cnn_classifier_int8.pte" else "cnn_classifier_fp32.pte"
            val segAsset = if (useQuantized) "unet_segmentation_int8.pte" else "unet_segmentation_fp32.pte"

            if (loadedQuantized != useQuantized) {
                classifierModule?.destroy()
                segmentationModule?.destroy()
                classifierModule = null
                segmentationModule = null
                loadedQuantized = null
            }

            if (classifierModule == null) {
                classifierModule = Module.load(assetToFile(classifierAsset))
            }
            if (segmentationModule == null) {
                segmentationModule = Module.load(assetToFile(segAsset))
            }
            loadedQuantized = useQuantized
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("MODEL_LOAD_FAILED", "Failed to load ExecuTorch models: ${e.message}", e)
        }
    }

    /** Decodes base64 JPEG/PNG -> resized 64x64 RGB float tensor, CHW, [0,1]. */
    private fun bitmapToInputTensor(bitmap: Bitmap): Tensor {
        val resized = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true)
        val floatBuffer = FloatBuffer.allocate(3 * IMG_SIZE * IMG_SIZE)

        // CHW layout: all R, then all G, then all B — matches PyTorch's
        // ToTensor() channel ordering used during training.
        val pixels = IntArray(IMG_SIZE * IMG_SIZE)
        resized.getPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)

        val rBuf = FloatArray(IMG_SIZE * IMG_SIZE)
        val gBuf = FloatArray(IMG_SIZE * IMG_SIZE)
        val bBuf = FloatArray(IMG_SIZE * IMG_SIZE)
        for (i in pixels.indices) {
            val p = pixels[i]
            rBuf[i] = ((p shr 16) and 0xFF) / 255.0f
            gBuf[i] = ((p shr 8) and 0xFF) / 255.0f
            bBuf[i] = (p and 0xFF) / 255.0f
        }
        floatBuffer.put(rBuf); floatBuffer.put(gBuf); floatBuffer.put(bBuf)
        floatBuffer.rewind()

        return Tensor.fromBlob(floatBuffer, longArrayOf(1, 3, IMG_SIZE.toLong(), IMG_SIZE.toLong()))
    }

    private fun decodeBase64Image(base64: String): Bitmap {
        val clean = base64.substringAfter(",", base64) // strip data:image/...;base64, prefix if present
        val bytes = Base64.decode(clean, Base64.DEFAULT)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: throw IllegalArgumentException("Could not decode image bytes")
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exps = logits.map { Math.exp((it - max).toDouble()).toFloat() }
        val sum = exps.sum()
        return exps.map { it / sum }.toFloatArray()
    }

    @ReactMethod
    fun classifyLeaf(imageBase64: String, useQuantized: Boolean, promise: Promise) {
        try {
            if (loadedQuantized != useQuantized) {
                return promise.reject("MODEL_MODE_MISMATCH", "Load the requested FP32/INT8 model first")
            }
            val module = classifierModule
                ?: return promise.reject("MODEL_NOT_LOADED", "Call preloadModels() first")

            val bitmap = decodeBase64Image(imageBase64)
            val inputTensor = bitmapToInputTensor(bitmap)

            val t0 = System.nanoTime()
            val outputs = module.forward(EValue.from(inputTensor))
            val latencyMs = (System.nanoTime() - t0) / 1_000_000.0

            val logits = outputs[0].toTensor().dataAsFloatArray
            val scores = softmax(logits)

            var bestIdx = 0
            for (i in scores.indices) if (scores[i] > scores[bestIdx]) bestIdx = i

            val allScores: WritableArray = Arguments.createArray()
            for (i in CLASSES.indices) {
                val entry: WritableMap = Arguments.createMap()
                entry.putString("label", CLASSES[i])
                entry.putDouble("score", scores[i].toDouble())
                allScores.pushMap(entry)
            }

            val result: WritableMap = Arguments.createMap()
            result.putString("label", CLASSES[bestIdx])
            result.putDouble("confidence", scores[bestIdx].toDouble())
            result.putArray("allScores", allScores)
            result.putDouble("latencyMs", latencyMs)
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("INFERENCE_FAILED", "Classifier inference failed: ${e.message}", e)
        }
    }

    @ReactMethod
    fun segmentLesion(imageBase64: String, useQuantized: Boolean, promise: Promise) {
        try {
            if (loadedQuantized != useQuantized) {
                return promise.reject("MODEL_MODE_MISMATCH", "Load the requested FP32/INT8 model first")
            }
            val module = segmentationModule
                ?: return promise.reject("MODEL_NOT_LOADED", "Call preloadModels() first")

            val bitmap = decodeBase64Image(imageBase64)
            val inputTensor = bitmapToInputTensor(bitmap)

            val t0 = System.nanoTime()
            val outputs = module.forward(EValue.from(inputTensor))
            val latencyMs = (System.nanoTime() - t0) / 1_000_000.0

            // Model already applies sigmoid internally (see UNetWithSigmoid
            // wrapper in export/models.py) so this is a [0,1] probability map.
            val maskProbs = outputs[0].toTensor().dataAsFloatArray

            var affectedPixels = 0
            val maskBitmap = Bitmap.createBitmap(IMG_SIZE, IMG_SIZE, Bitmap.Config.ARGB_8888)
            for (y in 0 until IMG_SIZE) {
                for (x in 0 until IMG_SIZE) {
                    val prob = maskProbs[y * IMG_SIZE + x]
                    val isLesion = prob > 0.5f
                    if (isLesion) affectedPixels++
                    // Red overlay, alpha scaled by confidence, for a heatmap-style mask
                    val alpha = (prob * 255).toInt().coerceIn(0, 255)
                    maskBitmap.setPixel(x, y, (alpha shl 24) or 0x00FF3B30)
                }
            }

            val affectedAreaPct = (affectedPixels.toDouble() / (IMG_SIZE * IMG_SIZE)) * 100.0

            val stream = java.io.ByteArrayOutputStream()
            maskBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
            val maskBase64 = "data:image/png;base64," +
                Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)

            val result: WritableMap = Arguments.createMap()
            result.putString("maskBase64", maskBase64)
            result.putDouble("affectedAreaPct", affectedAreaPct)
            result.putDouble("latencyMs", latencyMs)
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("INFERENCE_FAILED", "Segmentation inference failed: ${e.message}", e)
        }
    }

    @ReactMethod
    fun unloadModels(promise: Promise) {
        classifierModule?.destroy()
        segmentationModule?.destroy()
        classifierModule = null
        segmentationModule = null
        loadedQuantized = null
        promise.resolve(true)
    }
}
