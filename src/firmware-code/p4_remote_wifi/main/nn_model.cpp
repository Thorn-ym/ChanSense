#include "nn_model.h"
#include "sys_config.h"
#include "csi_dsp.h"
#include "dl_model_base.hpp"
#include "driver/gpio.h"
#include "driver/sdmmc_host.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "sdmmc_cmd.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <map>
#include <strings.h>
#include <sys/stat.h>

#define TAG "nn_model"

#define SD_MOUNT_POINT "/sdcard"
#define SD_MODEL_DIR SD_MOUNT_POINT "/models"
#define SD_LEGACY_MODEL_DIR SD_MOUNT_POINT "/sdcard_models"
#define MODEL_NAME_MAX_LEN 24
#define MODEL_PATH_MAX_LEN 96

static constexpr gpio_num_t SD_PWR_EN_GPIO = GPIO_NUM_45;
static constexpr int SDMMC_IO_LDO_CHAN_ID = 4;
static constexpr int SD_CARD_SDMMC_SLOT = SDMMC_HOST_SLOT_0;

typedef struct {
    char name[MODEL_NAME_MAX_LEN];
    char path[MODEL_PATH_MAX_LEN];
    long size;
    const char *class_names[NUM_CLASSES];
    dl::Model *model;
    dl::TensorBase *input;
    dl::TensorBase *output;
} model_slot_t;

static model_slot_t g_model_slots[NN_MODEL_MAX_SLOT_COUNT];
static int g_model_slot_count = 0;
static bool g_sdcard_mounted = false;
static sdmmc_card_t *g_sdcard = nullptr;
static sd_pwr_ctrl_handle_t g_sd_pwr_ctrl = nullptr;

// Active model pointers kept hot for real-time inference.
static int g_active_model_id = 0;
static dl::Model *g_model = nullptr;
static dl::TensorBase *g_model_input = nullptr;
static dl::TensorBase *g_model_output = nullptr;

static const char *const DEFAULT_CLASS_NAMES[NUM_CLASSES] = {
    "高抬腿", "展臂", "深蹲", "Unknown"
};

static bool is_valid_model_id(int model_id)
{
    return model_id >= 0 && model_id < g_model_slot_count;
}

static bool has_espdl_extension(const char *name)
{
    if (!name) {
        return false;
    }

    const char *dot = strrchr(name, '.');
    return dot && strcasecmp(dot, ".espdl") == 0;
}

static const char *basename_from_path(const char *path)
{
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static void copy_truncated(char *dst, size_t dst_size, const char *src)
{
    if (!dst || dst_size == 0) {
        return;
    }

    const char *safe_src = src ? src : "";
    size_t len = strlen(safe_src);
    if (len >= dst_size) {
        len = dst_size - 1;
    }
    memcpy(dst, safe_src, len);
    dst[len] = '\0';
}

static void copy_model_display_name(char *dst, size_t dst_size, const char *filename)
{
    if (!dst || dst_size == 0) {
        return;
    }

    const char *base = basename_from_path(filename);
    copy_truncated(dst, dst_size, base ? base : "MODEL");

    char *dot = strrchr(dst, '.');
    if (dot && strcasecmp(dot, ".espdl") == 0) {
        *dot = '\0';
    }

    if (dst[0] == '\0') {
        copy_truncated(dst, dst_size, "MODEL");
    }
}

static esp_err_t enable_sdcard_power(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = 1ULL << SD_PWR_EN_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
#if SOC_GPIO_SUPPORT_PIN_HYS_FILTER
        .hys_ctrl_mode = GPIO_HYS_SOFT_DISABLE,
#endif
    };
    esp_err_t ret = gpio_config(&io_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to configure SD power gpio: %s", esp_err_to_name(ret));
        return ret;
    }

    // ESP32-P4-Function-EV-Board uses an active-low SD_PWRn enable on GPIO45.
    ret = gpio_set_level(SD_PWR_EN_GPIO, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to enable SD power: %s", esp_err_to_name(ret));
    }
    return ret;
}

static esp_err_t mount_sdcard_once(void)
{
    if (g_sdcard_mounted) {
        return ESP_OK;
    }

    esp_err_t ret = enable_sdcard_power();
    if (ret != ESP_OK) {
        return ret;
    }

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 6,
        .allocation_unit_size = 16 * 1024,
        .disk_status_check_enable = false,
        .use_one_fat = false,
    };

    sd_pwr_ctrl_ldo_config_t ldo_config = {
        .ldo_chan_id = SDMMC_IO_LDO_CHAN_ID,
    };
    ret = sd_pwr_ctrl_new_on_chip_ldo(&ldo_config, &g_sd_pwr_ctrl);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to create SDMMC IO LDO power control: %s", esp_err_to_name(ret));
        g_sd_pwr_ctrl = nullptr;
        return ret;
    }

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    // ESP32-P4 board SD-card socket is on slot 0; ESP-Hosted Wi-Fi uses slot 1.
    host.slot = SD_CARD_SDMMC_SLOT;
    host.pwr_ctrl_handle = g_sd_pwr_ctrl;
    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 4;
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    ESP_LOGI(TAG, "mounting SD card at %s via SDMMC slot %d", SD_MOUNT_POINT, host.slot);
    ret = esp_vfs_fat_sdmmc_mount(SD_MOUNT_POINT, &host, &slot_config, &mount_config, &g_sdcard);
    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "failed to mount FAT filesystem on SD card");
        } else {
            ESP_LOGE(TAG, "failed to initialize SD card: %s", esp_err_to_name(ret));
        }
        sd_pwr_ctrl_del_on_chip_ldo(g_sd_pwr_ctrl);
        g_sd_pwr_ctrl = nullptr;
        g_sdcard = nullptr;
        return ret;
    }

    sdmmc_card_print_info(stdout, g_sdcard);
    g_sdcard_mounted = true;
    return ESP_OK;
}

static bool add_model_file(const char *path)
{
    if (!path || g_model_slot_count >= NN_MODEL_MAX_SLOT_COUNT) {
        return false;
    }

    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0) {
        ESP_LOGW(TAG, "skip unreadable model file: %s", path);
        return false;
    }
    if (S_ISDIR(st.st_mode)) {
        return false;
    }

    model_slot_t *slot = &g_model_slots[g_model_slot_count];
    memset(slot, 0, sizeof(*slot));
    copy_model_display_name(slot->name, sizeof(slot->name), path);
    copy_truncated(slot->path, sizeof(slot->path), path);
    slot->size = (long)st.st_size;
    for (int i = 0; i < NUM_CLASSES; i++) {
        slot->class_names[i] = DEFAULT_CLASS_NAMES[i];
    }

    ESP_LOGI(TAG, "found SD model %d: %s (%ld bytes)",
             g_model_slot_count,
             slot->path,
             slot->size);
    g_model_slot_count++;
    return true;
}

static void scan_model_dir(const char *dir_path)
{
    DIR *dir = opendir(dir_path);
    if (!dir) {
        ESP_LOGW(TAG, "model directory not found or unreadable: %s", dir_path);
        return;
    }

    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (g_model_slot_count >= NN_MODEL_MAX_SLOT_COUNT) {
            ESP_LOGW(TAG, "model slot limit reached (%d)", NN_MODEL_MAX_SLOT_COUNT);
            break;
        }
        if (entry->d_name[0] == '.' || !has_espdl_extension(entry->d_name)) {
            continue;
        }

        char path[MODEL_PATH_MAX_LEN];
        int written = snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
        if (written < 0 || written >= (int)sizeof(path)) {
            ESP_LOGW(TAG, "skip model with too long path: %s/%s", dir_path, entry->d_name);
            continue;
        }
        add_model_file(path);
    }
    closedir(dir);
}

static void scan_sd_models(void)
{
    g_model_slot_count = 0;
    memset(g_model_slots, 0, sizeof(g_model_slots));

    if (mount_sdcard_once() != ESP_OK) {
        return;
    }

    scan_model_dir(SD_MOUNT_POINT);
    scan_model_dir(SD_MODEL_DIR);
    scan_model_dir(SD_LEGACY_MODEL_DIR);

    if (g_model_slot_count == 0) {
        ESP_LOGW(TAG,
                 "no .espdl models found in %s, %s or %s",
                 SD_MOUNT_POINT,
                 SD_MODEL_DIR,
                 SD_LEGACY_MODEL_DIR);
    }
}

static esp_err_t ensure_model_loaded(int model_id)
{
    if (!is_valid_model_id(model_id)) {
        return ESP_ERR_INVALID_ARG;
    }

    model_slot_t *slot = &g_model_slots[model_id];
    if (slot->path[0] == '\0' || slot->size <= 0) {
        ESP_LOGW(TAG, "%s is not installed", slot->name);
        return ESP_ERR_NOT_FOUND;
    }

    if (slot->model != nullptr) {
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Loading %s from SD card: %s", slot->name, slot->path);
    slot->model = new dl::Model(slot->path,
                                fbs::MODEL_LOCATION_IN_SDCARD,
                                16 * 1024,
                                dl::MEMORY_MANAGER_GREEDY,
                                nullptr,
                                false);
    if (!slot->model) {
        ESP_LOGE(TAG, "Failed to load %s", slot->name);
        return ESP_FAIL;
    }

    std::map<std::string, dl::TensorBase *> model_inputs = slot->model->get_inputs();
    std::map<std::string, dl::TensorBase *> model_outputs = slot->model->get_outputs();
    if (model_inputs.empty() || model_outputs.empty()) {
        ESP_LOGE(TAG, "%s has invalid tensor metadata", slot->name);
        delete slot->model;
        slot->model = nullptr;
        return ESP_FAIL;
    }

    slot->input = model_inputs.begin()->second;
    slot->output = model_outputs.begin()->second;
    ESP_LOGI(TAG, "%s loaded. Size: %ld bytes, Input Exp: %d, Output Exp: %d",
             slot->name,
             slot->size,
             slot->input->get_exponent(),
             slot->output->get_exponent());
    return ESP_OK;
}

void nn_model_init(void)
{
    scan_sd_models();

    if (g_model_slot_count <= 0) {
        ESP_LOGE(TAG, "No SD-card models available; inference will stay disabled.");
        return;
    }

    int requested_id = sys_config_get_active_model_id();
    if (!is_valid_model_id(requested_id)) {
        requested_id = 0;
    }

    if (nn_model_select(requested_id) != ESP_OK) {
        ESP_LOGW(TAG, "Falling back to first SD-card model");
        (void)nn_model_select(0);
    }
}

void nn_model_rescan(void)
{
    scan_sd_models();
}

int nn_model_get_slot_count(void)
{
    return g_model_slot_count;
}

bool nn_model_is_installed(int model_id)
{
    if (!is_valid_model_id(model_id)) {
        return false;
    }
    return g_model_slots[model_id].path[0] != '\0' && g_model_slots[model_id].size > 0;
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
        return "NO MODEL";
    }
    return g_model_slots[model_id].name;
}

const char *nn_model_get_class_name(int class_id)
{
    if (class_id < 0 || class_id >= NUM_CLASSES || !is_valid_model_id(g_active_model_id)) {
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
