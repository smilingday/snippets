/*
 * Copyright 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.snippets.ai

// [START android_litert_migration_skill_imports]
import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.support.image.TensorImage
import com.google.ai.edge.litert.support.tensorbuffer.TensorBuffer as SupportTensorBuffer
import java.io.File
import kotlin.math.abs
import org.tensorflow.lite.DataType
// [END android_litert_migration_skill_imports]

// [START android_litert_migration_skill_gpu_delegate]
// Replaces GpuDelegate: the CPU runs the operators that the GPU doesn't
// support, as it did with the delegate. Keep the CPU fallback of Step 3 for a
// model that the GPU can't compile at all.
val gpuWithCpuFallback = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)
// [END android_litert_migration_skill_gpu_delegate]

// [START android_litert_migration_skill_num_threads]
// Replaces Interpreter.Options.setNumThreads. Create new options; never change
// the shared CompiledModel.Options.CPU.
fun cpuOptionsWithThreads(numThreads: Int): CompiledModel.Options =
    CompiledModel.Options(Accelerator.CPU).apply {
        cpuOptions = CompiledModel.CpuOptions(numThreads = numThreads)
    }
// [END android_litert_migration_skill_num_threads]

// [START android_litert_migration_skill_input_shape]
// Replaces interpreter.getInputTensor(0).shape(). Look the tensor up by name.
fun inputShape(model: CompiledModel, inputName: String): List<Int>? =
    model.getInputTensorType(inputName).layout?.dimensions
// [END android_litert_migration_skill_input_shape]

// [START android_litert_migration_skill_signature_run]
// Replaces interpreter.runSignature. Create the buffers once with
// model.createInputBuffers(signature) and model.createOutputBuffers(signature),
// and reuse them for every call.
fun runSignature(
    model: CompiledModel,
    signature: String,
    inputs: Map<String, TensorBuffer>,
    outputs: Map<String, TensorBuffer>
) {
    model.run(inputs, outputs, signature)
}
// [END android_litert_migration_skill_signature_run]

// [START android_litert_migration_skill_labels]
// Replaces FileUtil.loadLabels in an app that drops the Support library.
fun loadLabels(context: Context, name: String = "labels.txt"): List<String> =
    context.assets.open(name).bufferedReader().use { it.readLines() }
// [END android_litert_migration_skill_labels]

// [START android_litert_migration_skill_support_library]
// Replaces the Support library's Model class. ImageProcessor, TensorImage, and
// TensorLabel stay; the image goes into the CompiledModel buffers directly.
fun runWithSupportImage(
    model: CompiledModel,
    inputs: List<TensorBuffer>,
    outputs: List<TensorBuffer>,
    image: TensorImage,
    inputElementCount: Int,
    outputShape: IntArray
): SupportTensorBuffer {
    // A uint8 image (one that skipped NormalizeOp) in a float input yields NaN.
    check(image.dataType == DataType.FLOAT32) { "Normalize the image first" }
    val pixels = image.tensorBuffer.floatArray
    // writeFloat accepts a shorter array without an error.
    check(pixels.size == inputElementCount) { "Input size ${pixels.size}" }
    inputs[0].writeFloat(pixels)
    model.run(inputs, outputs)
    return SupportTensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        .apply { loadArray(outputs[0].readFloat()) }
}
// [END android_litert_migration_skill_support_library]

// [START android_litert_migration_skill_golden_output]
// Before the dependency swap: record the legacy app's output for one fixed
// input, one value per line.
fun writeGoldenOutput(file: File, output: FloatArray) {
    file.writeText(output.joinToString("\n"))
}

// After the migration: compare the migrated output with the recorded one.
// A difference below 1e-3 passes for outputs of order 1.
fun maxAbsDifference(golden: FloatArray, migrated: FloatArray): Float {
    check(golden.size == migrated.size) { "Size ${migrated.size} != ${golden.size}" }
    return golden.indices.maxOf { abs(golden[it] - migrated[it]) }
}

fun readGoldenOutput(context: Context, path: String = "golden/output.txt"): FloatArray =
    context.assets.open(path).bufferedReader().use { reader ->
        reader.readLines().filter { it.isNotBlank() }.map { it.toFloat() }.toFloatArray()
    }
// [END android_litert_migration_skill_golden_output]
