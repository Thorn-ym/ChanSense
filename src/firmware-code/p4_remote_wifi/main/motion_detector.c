#include "motion_detector.h"
#include "sys_config.h"
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_heap_caps.h"

static const char *TAG = "motion_detector";

motion_detector_t *motion_detector_create(void)
{
    motion_detector_t *detector = (motion_detector_t *)malloc(sizeof(motion_detector_t));
    if (!detector) {
        ESP_LOGE(TAG, "Failed to allocate motion detector structure");
        return NULL;
    }

    // 在 PSRAM 动态分配滑动窗口缓冲区以节约内部 SRAM
    detector->window_buf = (float *)heap_caps_malloc(TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!detector->window_buf) {
        ESP_LOGE(TAG, "Failed to allocate sliding window buffer in PSRAM");
        free(detector);
        return NULL;
    }

    motion_detector_reset(detector);
    return detector;
}

void motion_detector_destroy(motion_detector_t *detector)
{
    if (detector) {
        if (detector->window_buf) {
            free(detector->window_buf);
        }
        free(detector);
    }
}

void motion_detector_reset(motion_detector_t *detector)
{
    if (detector) {
        memset(detector->window_buf, 0, TOTAL_FEATURES * sizeof(float));
        detector->window_count = 0;
        detector->is_moving = false;
        detector->motion_trigger_counter = 0;
        detector->idle_counter = 0;
    }
}

// CSI 子载波解析与对齐滤波
static void parse_and_filter_csi(const int8_t *raw_payload, size_t raw_len, float *out_frame_228)
{
    int num_in = raw_len / 2;
    int count = 0;
    for (int i = 0; i < num_in && count < 114; i++) {
        int8_t imag = raw_payload[2 * i];
        int8_t real = raw_payload[2 * i + 1];
        if (real != 0 || imag != 0) {
            out_frame_228[2 * count] = (float)imag;
            out_frame_228[2 * count + 1] = (float)real;
            count++;
        }
    }
    
    if (count < 114) {
        if (count > 0) {
            float last_imag = out_frame_228[2 * (count - 1)];
            float last_real = out_frame_228[2 * (count - 1) + 1];
            for (int i = count; i < 114; i++) {
                out_frame_228[2 * i] = last_imag;
                out_frame_228[2 * i + 1] = last_real;
            }
        } else {
            memset(out_frame_228, 0, 228 * sizeof(float));
        }
    }
}

// 根据当前滑动窗口中的 50 帧，计算 114 个子载波的平均幅度标准差
static float calculate_motion_level(const float *window_buf)
{
    float std_sum = 0.0f;
    for (int s = 0; s < NUM_SUBCARRIERS; s++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int t = 0; t < SLIDING_WINDOW_SIZE; t++) {
            float imag = window_buf[t * FUSION_CHANNELS + 2 * s];
            float real = window_buf[t * FUSION_CHANNELS + 2 * s + 1];
            float amp = sqrtf(real * real + imag * imag);
            sum += amp;
            sum_sq += amp * amp;
        }
        float mean = sum / (float)SLIDING_WINDOW_SIZE;
        float var = (sum_sq / (float)SLIDING_WINDOW_SIZE) - (mean * mean);
        if (var < 0.0f) {
            var = 0.0f;
        }
        float std = sqrtf(var);
        std_sum += std;
    }
    return std_sum / (float)NUM_SUBCARRIERS;
}

bool motion_detector_process_frame(motion_detector_t *detector,
                                   const int8_t *raw_payload,
                                   size_t raw_len,
                                   float *out_motion_val,
                                   float *out_window_buf,
                                   bool *out_is_started,
                                   bool *out_is_ended)
{
    if (!detector) return false;

    static float new_frame[FUSION_CHANNELS];
    *out_is_started = false;
    *out_is_ended = false;

    // 1. 解析子载波
    parse_and_filter_csi(raw_payload, raw_len, new_frame);

    // 2. 写入滑动窗口
    if (detector->window_count < SLIDING_WINDOW_SIZE) {
        memcpy(detector->window_buf + detector->window_count * FUSION_CHANNELS, new_frame, sizeof(new_frame));
        detector->window_count++;
    } else {
        // 左移 1 帧空间，并在尾部追加新帧
        memmove(detector->window_buf, detector->window_buf + FUSION_CHANNELS, (SLIDING_WINDOW_SIZE - 1) * FUSION_CHANNELS * sizeof(float));
        memcpy(detector->window_buf + (SLIDING_WINDOW_SIZE - 1) * FUSION_CHANNELS, new_frame, sizeof(new_frame));
    }

    // 3. 滑动窗口未满 50 帧时暂不进行检测
    if (detector->window_count < SLIDING_WINDOW_SIZE) {
        return false;
    }

    // 4. 计算当前运动强度
    float motion_val = calculate_motion_level(detector->window_buf);
    *out_motion_val = motion_val;

    // 从线程安全配置模块动态获取当前配置
    float motion_threshold = sys_config_get_motion_threshold();
    int debounce_frames = sys_config_get_debounce_frames();
    int required_trigger_frames = sys_config_get_required_trigger_frames();

    // 5. 状态机评估
    bool request_inference = false;

    if (motion_val >= motion_threshold) {
        detector->idle_counter = 0;
        if (!detector->is_moving) {
            detector->motion_trigger_counter++;
            if (detector->motion_trigger_counter >= required_trigger_frames) {
                detector->is_moving = true;
                *out_is_started = true;
                ESP_LOGI(TAG, "🔴 [MOTION DETECTED] Gesture started! (Motion Level: %.2f)", motion_val);
            }
        }

        if (detector->is_moving) {
            if (out_window_buf) {
                memcpy(out_window_buf, detector->window_buf, TOTAL_FEATURES * sizeof(float));
            }
            request_inference = true;
        }
    } else {
        if (detector->is_moving) {
            detector->motion_trigger_counter = 0;
            detector->idle_counter++;

            if (detector->idle_counter >= debounce_frames) {
                detector->is_moving = false;
                *out_is_ended = true;
                ESP_LOGI(TAG, "⚪ [MOTION ENDED] Returning to standby.");
            }
        } else {
            detector->motion_trigger_counter = 0;
        }
    }

    return request_inference;
}
