#include "nn_model.h"
#include "model_data.h"
#include "model2_data.h"
#include "sys_config.h"
#include "csi_dsp.h"
#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <cmath>
#include <cstring>

#define TAG "nn_model"

typedef struct {
    const char *name;
    const uint8_t *data;
    unsigned int len;
    const char *class_names[NUM_CLASSES];
    dl::Model *model;
    dl::TensorBase *input;
    dl::TensorBase *output;
} model_slot_t;

static model_slot_t g_model_slots[NN_MODEL_SLOT_COUNT] = {
    {
        "Model 1",
        model_espdl,
        model_espdl_len,
        {"高抬腿", "展臂", "深蹲", "Unknown"},
        nullptr,
        nullptr,
        nullptr,
    },
    {
        "Model 2",
        model2_espdl,
        model2_espdl_len,
        {"Model2-0", "Model2-1", "Model2-2", "Unknown"},
        nullptr,
        nullptr,
        nullptr,
    },
};

// Active model pointers kept hot for real-time inference.
static int g_active_model_id = 0;
static dl::Model *g_model = nullptr;
static dl::TensorBase *g_model_input = nullptr;
static dl::TensorBase *g_model_output = nullptr;

static bool is_valid_model_id(int model_id)
{
    return model_id >= 0 && model_id < NN_MODEL_SLOT_COUNT;
}

static esp_err_t ensure_model_loaded(int model_id)
{
    if (!is_valid_model_id(model_id)) {
        return ESP_ERR_INVALID_ARG;
    }

    model_slot_t *slot = &g_model_slots[model_id];
    if (slot->len == 0 || slot->data == nullptr) {
        ESP_LOGW(TAG, "%s is not installed", slot->name);
        return ESP_ERR_NOT_FOUND;
    }

    if (slot->model != nullptr) {
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Loading %s from memory...", slot->name);
    slot->model = new dl::Model((const char *)slot->data, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    if (!slot->model) {
        ESP_LOGE(TAG, "Failed to load %s", slot->name);
        return ESP_FAIL;
    }

    std::map<std::string, dl::TensorBase *> model_inputs = slot->model->get_inputs();
    std::map<std::string, dl::TensorBase *> model_outputs = slot->model->get_outputs();
    if (model_inputs.empty() || model_outputs.empty()) {
        ESP_LOGE(TAG, "%s has invalid tensor metadata", slot->name);
        return ESP_FAIL;
    }

    slot->input = model_inputs.begin()->second;
    slot->output = model_outputs.begin()->second;
    ESP_LOGI(TAG, "%s loaded. Size: %u bytes, Input Exp: %d, Output Exp: %d",
             slot->name,
             slot->len,
             slot->input->get_exponent(),
             slot->output->get_exponent());
    return ESP_OK;
}

void nn_model_init(void)
{
    int requested_id = sys_config_get_active_model_id();
    if (nn_model_select(requested_id) != ESP_OK) {
        ESP_LOGW(TAG, "Falling back to Model 1");
        (void)nn_model_select(0);
    }
}

int nn_model_get_slot_count(void)
{
    return NN_MODEL_SLOT_COUNT;
}

bool nn_model_is_installed(int model_id)
{
    if (!is_valid_model_id(model_id)) {
        return false;
    }
    return g_model_slots[model_id].data != nullptr && g_model_slots[model_id].len > 0;
}

esp_err_t nn_model_select(int model_id)
{
    esp_err_t ret = ensure_model_loaded(model_id);
    if (ret != ESP_OK) {
        return ret;
    }

    model_slot_t *slot = &g_model_slots[model_id];
    g_active_model_id = model_id;
    g_model = slot->model;
    g_model_input = slot->input;
    g_model_output = slot->output;
    sys_config_set_active_model_id(model_id);
    ESP_LOGI(TAG, "Active model switched to %s", slot->name);
    return ESP_OK;
}

int nn_model_get_active_id(void)
{
    return g_active_model_id;
}

const char *nn_model_get_active_name(void)
{
    return nn_model_get_slot_name(g_active_model_id);
}

const char *nn_model_get_slot_name(int model_id)
{
    if (!is_valid_model_id(model_id)) {
        return "Invalid";
    }
    return g_model_slots[model_id].name;
}

const char *nn_model_get_class_name(int class_id)
{
    if (class_id < 0 || class_id >= NUM_CLASSES) {
        return "Unknown";
    }
    const char *name = g_model_slots[g_active_model_id].class_names[class_id];
    return name ? name : "Unknown";
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

int nn_model_predict_cnn(float *raw_csi)
{
    if (g_model == nullptr || g_model_input == nullptr || g_model_output == nullptr) {
        ESP_LOGE(TAG, "Model not initialized!");
        return -1;
    }
    
    static float *preprocessed_features = nullptr;
    if (preprocessed_features == nullptr) {
        preprocessed_features = (float *)heap_caps_malloc(CSI_DSP_TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
        if (preprocessed_features == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate preprocessed feature buffer in PSRAM");
            return -1;
        }
    }
    
    // Perform on-board preprocessing
    csi_dsp_preprocess_fusion(raw_csi, preprocessed_features);
    
    int input_exponent = g_model_input->get_exponent();
    float inv_scale = 1.0f / std::pow(2.0f, input_exponent); 
    
    int8_t *input_data = (int8_t *)g_model_input->data;
    
    // Quantize 11400 features into the input tensor using fast multiplication
    for (int i = 0; i < CSI_DSP_TOTAL_FEATURES; i++) {
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
    if (raw_csi == nullptr || out_probs == nullptr) {
        return -1;
    }
    
    static float *preprocessed_features = nullptr;
    if (preprocessed_features == nullptr) {
        preprocessed_features = (float *)heap_caps_malloc(CSI_DSP_TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
        if (preprocessed_features == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate preprocessed feature buffer in PSRAM");
            return -1;
        }
    }
    
    // Perform on-board preprocessing
    csi_dsp_preprocess_fusion(raw_csi, preprocessed_features);
    
    int input_exponent = g_model_input->get_exponent();
    float inv_scale = 1.0f / std::pow(2.0f, input_exponent); 
    
    int8_t *input_data = (int8_t *)g_model_input->data;
    
    // Quantize features using fast multiplication
    for (int i = 0; i < CSI_DSP_TOTAL_FEATURES; i++) {
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
