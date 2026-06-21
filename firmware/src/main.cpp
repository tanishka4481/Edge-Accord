#include <Arduino.h>
#include "model_data.h"

#ifndef SIMULATION_MODE
#define SIMULATION_MODE true
#endif

#if SIMULATION_MODE
#include "test_audio.h"
#else
#include "driver/i2s.h"
#endif

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
    constexpr int kTensorArenaSize = 80 * 1024;
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];

    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;

    tflite::MicroErrorReporter error_reporter;
    tflite::AllOpsResolver resolver;

    const char* kClassNames[] = { "yes", "no", "stop", "go", "unknown", "silence" };
    constexpr int kNumClasses = 6;

#if !SIMULATION_MODE
    // Statically allocate input buffer to prevent ESP32 stack overflow
    int16_t i2s_sample_buffer[16000];
#endif
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("=== EdgeAccord Firmware Starting ===");

    // Get the model
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("Model schema version ");
        Serial.print(model->version());
        Serial.print(" does not match supported schema version ");
        Serial.println(TFLITE_SCHEMA_VERSION);
        return;
    }

    // Instantiate interpreter dynamically on heap
    interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

    // Allocate Tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        return;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.print("Input tensor type: ");
    Serial.println(input_tensor->type);
    Serial.print("Input tensor bytes: ");
    Serial.println(input_tensor->bytes);

#if !SIMULATION_MODE
    Serial.println("Configuring I2S interface...");
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    if (i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL) != ESP_OK) {
        Serial.println("I2S driver install failed");
    }

    i2s_pin_config_t pin_config = {
        .bck_io_num = 14,
        .ws_io_num = 15,
        .data_out_num = -1,
        .data_in_num = 32
    };

    if (i2s_set_pin(I2S_NUM_0, &pin_config) != ESP_OK) {
        Serial.println("I2S pin configuration failed");
    }
    Serial.println("I2S Initialized successfully");
#endif
}

void loop() {
    if (interpreter == nullptr || input_tensor == nullptr || output_tensor == nullptr) {
        Serial.println("System not initialized correctly");
        delay(2000);
        return;
    }

    int num_input_elements = 1;
    for (int i = 0; i < input_tensor->dims->size; ++i) {
        num_input_elements *= input_tensor->dims->data[i];
    }

#if !SIMULATION_MODE
    size_t bytes_read = 0;
    // Read 1 second of audio (16000 samples, 32000 bytes)
    i2s_read(I2S_NUM_0, i2s_sample_buffer, sizeof(i2s_sample_buffer), &bytes_read, portMAX_DELAY);
#endif

    // Map/downsample the 1-second PCM buffer (16000 samples) to model input size
    float step = 16000.0f / num_input_elements;
    for (int i = 0; i < num_input_elements; ++i) {
        int pcm_idx = (int)(i * step);
        if (pcm_idx >= 16000) pcm_idx = 15999;

        int16_t sample = 0;
#if SIMULATION_MODE
        sample = G_TEST_AUDIO_DATA[pcm_idx];
#else
        sample = i2s_sample_buffer[pcm_idx];
#endif

        if (input_tensor->type == kTfLiteInt8) {
            input_tensor->data.int8[i] = (int8_t)(sample / 256);
        } else if (input_tensor->type == kTfLiteFloat32) {
            input_tensor->data.f[i] = (float)sample / 32768.0f;
        }
    }

    // Run TFLite inference
    unsigned long start_time = millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long duration = millis() - start_time;

    if (invoke_status != kTfLiteOk) {
        Serial.println("Interpreter Invoke() failed");
        delay(1000);
        return;
    }

    // Print inference info and prediction confidence scores
    Serial.print("Inference completed in ");
    Serial.print(duration);
    Serial.println(" ms");

    Serial.print("Predictions: ");
    int best_class_idx = 0;
    float best_class_score = -999.0f;

    for (int i = 0; i < kNumClasses; ++i) {
        float score = 0.0f;
        if (output_tensor->type == kTfLiteInt8) {
            float scale = output_tensor->params.scale;
            int32_t zero_point = output_tensor->params.zero_point;
            score = (output_tensor->data.int8[i] - zero_point) * scale;
        } else if (output_tensor->type == kTfLiteFloat32) {
            score = output_tensor->data.f[i];
        }

        Serial.print(kClassNames[i]);
        Serial.print(": ");
        Serial.print(score, 4);
        if (i < kNumClasses - 1) {
            Serial.print(", ");
        }

        if (score > best_class_score) {
            best_class_score = score;
            best_class_idx = i;
        }
    }
    Serial.println();
    Serial.print("Best class: ");
    Serial.print(kClassNames[best_class_idx]);
    Serial.print(" (");
    Serial.print(best_class_score, 4);
    Serial.println(")");

#if SIMULATION_MODE
    delay(1000);
#endif
}
