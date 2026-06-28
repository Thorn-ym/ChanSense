#include "nn_model.h"
#include "model_data.h"
#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <cmath>
#include <cstring>

#define TAG "nn_model"

// Global model instance kept in memory
static dl::Model *g_model = nullptr;
static dl::TensorBase *g_model_input = nullptr;
static dl::TensorBase *g_model_output = nullptr;

void nn_model_init(void)
{
    if (g_model != nullptr) {
        ESP_LOGW(TAG, "Model already initialized!");
        return;
    }

    ESP_LOGI(TAG, "Loading model from memory...");
    // Create the model instance from RODATA
    g_model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    
    if (!g_model) {
        ESP_LOGE(TAG, "Failed to load model!");
        return;
    }
    ESP_LOGI(TAG, "Model loaded successfully! Size: %u bytes", model_espdl_len);
    
    // Cache the input and output tensor pointers for fast real-time access
    std::map<std::string, dl::TensorBase *> model_inputs = g_model->get_inputs();
    g_model_input = model_inputs.begin()->second;
    
    std::map<std::string, dl::TensorBase *> model_outputs = g_model->get_outputs();
    g_model_output = model_outputs.begin()->second;
    
    ESP_LOGI(TAG, "Model cached. Input Exp: %d, Output Exp: %d", 
             g_model_input->get_exponent(), g_model_output->get_exponent());
}

int nn_model_predict(float x, float y)
{
    if (g_model == nullptr || g_model_input == nullptr || g_model_output == nullptr) {
        ESP_LOGE(TAG, "Model not initialized!");
        return -1;
    }
    
    int input_exponent = g_model_input->get_exponent();
    
    // Quantize inputs: float_val / (2^input_exponent)
    int8_t q_x = std::round(x / std::pow(2.0f, input_exponent));
    int8_t q_y = std::round(y / std::pow(2.0f, input_exponent));
    
    // Fill input tensor data
    int8_t *input_data = (int8_t *)g_model_input->data;
    input_data[0] = q_x;
    input_data[1] = q_y;
    
    // Run forward pass
    g_model->run();
    
    // Read quantized outputs
    int8_t *output_data = (int8_t *)g_model_output->data;
    
    // Find class with the maximum score (argmax)
    int max_class = 0;
    int8_t max_val = output_data[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (output_data[c] > max_val) {
            max_val = output_data[c];
            max_class = c;
        }
    }
    
    return max_class;
}

#define NUM_FRAMES 50
#define NUM_SUBCARRIERS 114
#define FUSION_CHANNELS 228
#define TOTAL_FEATURES 11400

static const float CONST_D = 123452.5f;

static float get_x_dev(int s) {
    return (float)s - 56.5f;
}

static void unwrap_phase_frame(const float *p_in, float *p_out) {
    p_out[0] = p_in[0];
    float cum_shift = 0.0f;
    const float inv_two_pi = 1.0f / (2.0f * (float)M_PI);
    const float two_pi = 2.0f * (float)M_PI;
    
    for (int i = 1; i < NUM_SUBCARRIERS; i++) {
        float diff = p_in[i] - p_in[i-1];
        float shift = 0.0f;
        if (diff > (float)M_PI) {
            shift = -two_pi * (int)((diff + (float)M_PI) * inv_two_pi);
        } else if (diff < -(float)M_PI) {
            shift = two_pi * (int)((-diff + (float)M_PI) * inv_two_pi);
        }
        cum_shift += shift;
        p_out[i] = p_in[i] + cum_shift;
    }
}

void preprocess_csi_fusion(const float *raw_csi, float *out_features)
{
    static float *amp = nullptr;
    static float *phase = nullptr;
    static float *unwrapped = nullptr;
    static float *phase_cal = nullptr;

    if (amp == nullptr) {
        amp = (float *)heap_caps_malloc(NUM_FRAMES * NUM_SUBCARRIERS * sizeof(float), MALLOC_CAP_SPIRAM);
        phase = (float *)heap_caps_malloc(NUM_FRAMES * NUM_SUBCARRIERS * sizeof(float), MALLOC_CAP_SPIRAM);
        unwrapped = (float *)heap_caps_malloc(NUM_FRAMES * NUM_SUBCARRIERS * sizeof(float), MALLOC_CAP_SPIRAM);
        phase_cal = (float *)heap_caps_malloc(NUM_FRAMES * NUM_SUBCARRIERS * sizeof(float), MALLOC_CAP_SPIRAM);
    }

    // 1. Extract Amplitude and Phase
    for (int t = 0; t < NUM_FRAMES; t++) {
        int t_offset = t * NUM_SUBCARRIERS;
        int t_fusion_offset = t * FUSION_CHANNELS;
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            float imag = raw_csi[t_fusion_offset + 2 * s];
            float real = raw_csi[t_fusion_offset + 2 * s + 1];
            
            amp[t_offset + s] = sqrtf(real * real + imag * imag);
            phase[t_offset + s] = atan2f(imag, real);
        }
    }

    // 2. Phase Unwrapping and Linear Calibration (Frame by frame)
    for (int t = 0; t < NUM_FRAMES; t++) {
        float *p_in = &phase[t * NUM_SUBCARRIERS];
        float *p_unwrapped = &unwrapped[t * NUM_SUBCARRIERS];
        float *p_cal = &phase_cal[t * NUM_SUBCARRIERS];
        
        unwrap_phase_frame(p_in, p_unwrapped);
        
        float y_sum = 0.0f;
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            y_sum += p_unwrapped[s];
        }
        float y_mean = y_sum * (1.0f / (float)NUM_SUBCARRIERS);
        
        float num_sum = 0.0f;
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            float y_dev = p_unwrapped[s] - y_mean;
            float x_dev = (float)s - 56.5f;
            num_sum += y_dev * x_dev;
        }
        float a = num_sum * (1.0f / CONST_D);
        
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            float y_dev = p_unwrapped[s] - y_mean;
            float x_dev = (float)s - 56.5f;
            p_cal[s] = y_dev - a * x_dev;
        }
    }

    // 3. Standardization (SR-Std) over time dimension and Feature Fusion
    const float eps = 2.0f;
    
    static float amp_mean[NUM_SUBCARRIERS];
    static float phase_mean[NUM_SUBCARRIERS];
    static float amp_var[NUM_SUBCARRIERS];
    static float phase_var[NUM_SUBCARRIERS];
    
    memset(amp_mean, 0, sizeof(amp_mean));
    memset(phase_mean, 0, sizeof(phase_mean));
    memset(amp_var, 0, sizeof(amp_var));
    memset(phase_var, 0, sizeof(phase_var));
    
    // Sum over time
    for (int t = 0; t < NUM_FRAMES; t++) {
        int t_offset = t * NUM_SUBCARRIERS;
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            amp_mean[s] += amp[t_offset + s];
            phase_mean[s] += phase_cal[t_offset + s];
        }
    }
    
    // Compute means
    const float inv_num_frames = 1.0f / (float)NUM_FRAMES;
    for (int s = 0; s < NUM_SUBCARRIERS; s++) {
        amp_mean[s] *= inv_num_frames;
        phase_mean[s] *= inv_num_frames;
    }
    
    // Sum variance
    for (int t = 0; t < NUM_FRAMES; t++) {
        int t_offset = t * NUM_SUBCARRIERS;
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            float amp_diff = amp[t_offset + s] - amp_mean[s];
            amp_var[s] += amp_diff * amp_diff;
            
            float phase_diff = phase_cal[t_offset + s] - phase_mean[s];
            phase_var[s] += phase_diff * phase_diff;
        }
    }
    
    // Compute std and scales (reciprocals)
    static float amp_scale[NUM_SUBCARRIERS];
    static float phase_scale[NUM_SUBCARRIERS];
    for (int s = 0; s < NUM_SUBCARRIERS; s++) {
        float amp_std = sqrtf(amp_var[s] * inv_num_frames);
        float phase_std = sqrtf(phase_var[s] * inv_num_frames);
        amp_scale[s] = 1.0f / (amp_std + eps);
        phase_scale[s] = 1.0f / (phase_std + eps);
    }
    
    // Normalize and fuse (sequential access, cache friendly, multiplication only)
    for (int t = 0; t < NUM_FRAMES; t++) {
        int t_offset_in = t * NUM_SUBCARRIERS;
        int t_offset_out = t * FUSION_CHANNELS;
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            float norm_amp = (amp[t_offset_in + s] - amp_mean[s]) * amp_scale[s];
            float norm_phase = (phase_cal[t_offset_in + s] - phase_mean[s]) * phase_scale[s];
            
            out_features[t_offset_out + s] = norm_amp;
            out_features[t_offset_out + s + NUM_SUBCARRIERS] = norm_phase;
        }
    }
}

int nn_model_predict_cnn(float *raw_csi)
{
    if (g_model == nullptr || g_model_input == nullptr || g_model_output == nullptr) {
        ESP_LOGE(TAG, "Model not initialized!");
        return -1;
    }
    
    static float *preprocessed_features = nullptr;
    if (preprocessed_features == nullptr) {
        preprocessed_features = (float *)heap_caps_malloc(TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
    }
    
    // Perform on-board preprocessing (combining amplitude/phase extraction, calibration, standardization)
    preprocess_csi_fusion(raw_csi, preprocessed_features);
    
    int input_exponent = g_model_input->get_exponent();
    float inv_scale = 1.0f / std::pow(2.0f, input_exponent); 
    
    int8_t *input_data = (int8_t *)g_model_input->data;
    
    // Quantize 11400 features into the input tensor using fast multiplication
    for (int i = 0; i < TOTAL_FEATURES; i++) {
        float q_val = std::round(preprocessed_features[i] * inv_scale);
        int32_t qi = (int32_t)q_val;
        if (qi > 127) qi = 127;
        else if (qi < -128) qi = -128;
        input_data[i] = (int8_t)qi;
    }
    
    // Run forward pass
    g_model->run();
    
    // Read output tensor data
    int8_t *output_data = (int8_t *)g_model_output->data;
    
    // Find class with the maximum score (argmax among classes)
    int max_class = 0;
    int8_t max_val = output_data[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (output_data[c] > max_val) {
            max_val = output_data[c];
            max_class = c;
        }
    }
    
    return max_class;
}

int nn_model_predict_cnn_with_probs(float *raw_csi, float *out_probs)
{
    if (g_model == nullptr || g_model_input == nullptr || g_model_output == nullptr) {
        ESP_LOGE(TAG, "Model not initialized!");
        return -1;
    }
    
    static float *preprocessed_features = nullptr;
    if (preprocessed_features == nullptr) {
        preprocessed_features = (float *)heap_caps_malloc(TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
    }
    
    // Perform on-board preprocessing
    preprocess_csi_fusion(raw_csi, preprocessed_features);
    
    int input_exponent = g_model_input->get_exponent();
    float inv_scale = 1.0f / std::pow(2.0f, input_exponent); 
    
    int8_t *input_data = (int8_t *)g_model_input->data;
    
    // Quantize features using fast multiplication
    for (int i = 0; i < TOTAL_FEATURES; i++) {
        float q_val = std::round(preprocessed_features[i] * inv_scale);
        int32_t qi = (int32_t)q_val;
        if (qi > 127) qi = 127;
        else if (qi < -128) qi = -128;
        input_data[i] = (int8_t)qi;
    }
    
    // Run forward pass
    g_model->run();
    
    // Read output tensor data
    int8_t *output_data = (int8_t *)g_model_output->data;
    int output_exponent = g_model_output->get_exponent();
    float output_scale = std::pow(2.0f, output_exponent);
    
    // Calculate Softmax probabilities
    float sum_exp = 0.0f;
    float float_logits[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) {
        float_logits[c] = (float)output_data[c] * output_scale;
        sum_exp += expf(float_logits[c]);
    }
    for (int c = 0; c < NUM_CLASSES; c++) {
        out_probs[c] = expf(float_logits[c]) / sum_exp;
    }
    
    // Find class with the maximum score
    int max_class = 0;
    float max_prob = out_probs[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (out_probs[c] > max_prob) {
            max_prob = out_probs[c];
            max_class = c;
        }
    }
    
    return max_class;
}

