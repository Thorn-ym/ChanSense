#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "nn_model.h"

static const char *TAG = "csi_receiver";

// ==================== 1. 串口配置参数 ====================
#define CSI_UART_NUM            UART_NUM_1
#define CSI_UART_RX_GPIO        GPIO_NUM_21
#define CSI_UART_TX_GPIO        GPIO_NUM_22
#define CSI_UART_BAUD_RATE      921600
#define CSI_UART_RX_BUFFER_SIZE (16 * 1024)

// ==================== 2. 数据协议帧定义 ====================
static const uint8_t CSI_FRAME_MAGIC[4] = {'C', 'S', 'I', '1'};
#define CSI_FRAME_MAX_RAW_LEN   1024
#define CSI_FRAME_HEADER_SIZE   10
#define CSI_FRAME_CHECKSUM_SIZE 2

// ==================== 3. 辅助转换函数 ====================
static uint16_t read_le16(const uint8_t *src)
{
    return ((uint16_t)src[0]) | (((uint16_t)src[1]) << 8);
}

static uint32_t read_le32(const uint8_t *src)
{
    return ((uint32_t)src[0]) |
           (((uint32_t)src[1]) << 8) |
           (((uint32_t)src[2]) << 16) |
           (((uint32_t)src[3]) << 24);
}

// ==================== 4. 驱动与接收逻辑 ====================
static esp_err_t init_csi_uart(void)
{
    const uart_config_t uart_config = {
        .baud_rate = CSI_UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
    };

    esp_err_t ret = uart_driver_install(CSI_UART_NUM, CSI_UART_RX_BUFFER_SIZE, 0, 0, NULL, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to install CSI UART driver: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = uart_param_config(CSI_UART_NUM, &uart_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to configure CSI UART: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = uart_set_pin(CSI_UART_NUM,
                       CSI_UART_TX_GPIO,
                       CSI_UART_RX_GPIO,
                       UART_PIN_NO_CHANGE,
                       UART_PIN_NO_CHANGE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to set CSI UART pins: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG,
             "CSI UART%d initialized: RX GPIO%d, TX GPIO%d, baud=%d",
             CSI_UART_NUM,
             CSI_UART_RX_GPIO,
             CSI_UART_TX_GPIO,
             CSI_UART_BAUD_RATE);
    return ESP_OK;
}

static bool uart_read_exact(uint8_t *dst, size_t len)
{
    size_t received = 0;
    while (received < len) {
        const int ret = uart_read_bytes(CSI_UART_NUM,
                                        dst + received,
                                        len - received,
                                        pdMS_TO_TICKS(1000));
        if (ret < 0) {
            ESP_LOGE(TAG, "CSI UART read failed");
            return false;
        }
        received += (size_t)ret;
    }
    return true;
}

static bool read_csi_frame(uint32_t *seq, int8_t *raw_payload, size_t *raw_len)
{
    uint8_t frame_header[CSI_FRAME_HEADER_SIZE] = {0};
    size_t matched = 0;

    while (matched < sizeof(CSI_FRAME_MAGIC)) {
        uint8_t byte = 0;
        if (!uart_read_exact(&byte, 1)) {
            return false;
        }

        if (byte == CSI_FRAME_MAGIC[matched]) {
            frame_header[matched++] = byte;
        } else {
            matched = (byte == CSI_FRAME_MAGIC[0]) ? 1 : 0;
            if (matched == 1) {
                frame_header[0] = byte;
            }
        }
    }

    if (!uart_read_exact(frame_header + sizeof(CSI_FRAME_MAGIC),
                         CSI_FRAME_HEADER_SIZE - sizeof(CSI_FRAME_MAGIC))) {
        return false;
    }

    const uint16_t len = read_le16(&frame_header[8]);
    if (len == 0 || len > CSI_FRAME_MAX_RAW_LEN) {
        ESP_LOGW(TAG, "invalid raw_len=%u (max=%u)", len, CSI_FRAME_MAX_RAW_LEN);
        return false;
    }

    if (!uart_read_exact((uint8_t *)raw_payload, len)) {
        return false;
    }

    uint8_t checksum_bytes[CSI_FRAME_CHECKSUM_SIZE] = {0};
    if (!uart_read_exact(checksum_bytes, sizeof(checksum_bytes))) {
        return false;
    }

    uint32_t sum = 0;
    for (size_t i = 0; i < CSI_FRAME_HEADER_SIZE; ++i) {
        sum += frame_header[i];
    }
    for (size_t i = 0; i < len; ++i) {
        sum += (uint8_t)raw_payload[i];
    }
    const uint16_t expected = read_le16(checksum_bytes);
    const uint16_t actual = (uint16_t)(sum & 0xffff);
    if (expected != actual) {
        ESP_LOGW(TAG, "checksum failed: expected=0x%04x actual=0x%04x", expected, actual);
        return false;
    }

    *seq = read_le32(&frame_header[4]);
    *raw_len = len;
    return true;
}

// ==================== 5. 全局变量与核心逻辑 ====================
#define SLIDING_WINDOW_SIZE 50
#define NUM_SUBCARRIERS 114
#define FUSION_CHANNELS 228
#define TOTAL_FEATURES (SLIDING_WINDOW_SIZE * FUSION_CHANNELS)

#define MOTION_THRESHOLD 2.5f
#define DEBOUNCE_FRAMES 15
#define REQUIRED_TRIGGER_FRAMES 3
#define MIN_CONFIDENCE 0.80f

static float *csi_window = NULL;
static int csi_window_count = 0;

static float *inference_input_buf = NULL;
static float inference_motion_val = 0.0f;
static volatile bool inference_request_pending = false;
static volatile bool gesture_ended_pending = false;
static portMUX_TYPE sync_mux = portMUX_INITIALIZER_UNLOCKED;
static TaskHandle_t ai_task_handle = NULL;

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

// ==================== 6. 核心任务定义 ====================

/**
 * @brief Core 1 任务: AI模型推理
 */
static void ai_inference_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting AI Inference Task on Core %d...", xPortGetCoreID());
    
    // 初始化神经网络模型（从Flash/RODATA加载）
    nn_model_init();
    
    static float local_input[TOTAL_FEATURES];
    float local_motion_val = 0.0f;
    bool local_ended = false;
    
    // 动作周期内概率加权求和及统计变量
    float accumulated_probs[NUM_CLASSES] = {0.0f};
    float unweighted_probs[NUM_CLASSES] = {0.0f};
    float total_weight = 0.0f;
    int gesture_frames_count = 0;
    
    const char *gesture_names[NUM_CLASSES] = {
        "高抬腿",
        "展臂",
        "深蹲"
    };
    
    // Real-time smoothing history
    #define SMOOTHING_FRAMES 7
    int pred_history[SMOOTHING_FRAMES];
    int pred_history_count = 0;
    
    while (1) {
        // 无限期阻塞等待Core 0的通知
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        // 安全获取待推理数据
        portENTER_CRITICAL(&sync_mux);
        bool request = inference_request_pending;
        local_ended = gesture_ended_pending;
        if (request) {
            memcpy(local_input, inference_input_buf, TOTAL_FEATURES * sizeof(float));
            local_motion_val = inference_motion_val;
            inference_request_pending = false;
        }
        if (local_ended) {
            gesture_ended_pending = false;
        }
        portEXIT_CRITICAL(&sync_mux);
        
        if (request) {
            float probs[NUM_CLASSES] = {0.0f};
            int64_t start = esp_timer_get_time();
            int pred_class = nn_model_predict_cnn_with_probs(local_input, probs);
            int64_t end = esp_timer_get_time();
            
            float confidence = probs[pred_class];
            
            // 计算加权系数 (motion_val - threshold)^2
            float weight = (local_motion_val - MOTION_THRESHOLD) * (local_motion_val - MOTION_THRESHOLD);
            
            // 过滤置信度低于0.50的帧，参与最终的决策累加
            if (confidence >= 0.50f) {
                for (int c = 0; c < NUM_CLASSES; c++) {
                    accumulated_probs[c] += probs[c] * weight;
                    unweighted_probs[c] += probs[c];
                }
                total_weight += weight;
                gesture_frames_count++;
            }
            
            // 检查置信度是否大于等于阈值以推入平滑历史
            if (confidence >= MIN_CONFIDENCE) {
                if (pred_history_count < SMOOTHING_FRAMES) {
                    pred_history[pred_history_count++] = pred_class;
                } else {
                    for (int i = 0; i < SMOOTHING_FRAMES - 1; i++) {
                        pred_history[i] = pred_history[i + 1];
                    }
                    pred_history[SMOOTHING_FRAMES - 1] = pred_class;
                }
            }
            
            // 根据平滑历史投票决定当前预测输出
            int smoothed_class = -1;
            if (pred_history_count > 0) {
                int counts[NUM_CLASSES] = {0};
                for (int i = 0; i < pred_history_count; i++) {
                    counts[pred_history[i]]++;
                }
                int max_count = 0;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    if (counts[c] > max_count) {
                        max_count = counts[c];
                        smoothed_class = c;
                    }
                }
            }
            
            if (smoothed_class != -1) {
                const char *name = (smoothed_class < NUM_CLASSES && gesture_names[smoothed_class]) ? gesture_names[smoothed_class] : "Unknown";
                ESP_LOGI(TAG, "🔴 [ACTIVE] Real-time Pred: %s (Conf: %.1f%%, Motion: %.2f, Time: %lld us)", 
                         name, confidence * 100.0f, local_motion_val, (end - start));
            } else {
                ESP_LOGI(TAG, "🔴 [ACTIVE] Real-time Pred: UNCERTAIN (Motion: %.2f)", local_motion_val);
            }
        }
        
        if (local_ended) {
            // 动作结束，进行加权总决策
            if (gesture_frames_count > 0) {
                float final_probs[NUM_CLASSES];
                if (total_weight > 0.0f) {
                    for (int c = 0; c < NUM_CLASSES; c++) {
                        final_probs[c] = accumulated_probs[c] / total_weight;
                    }
                } else {
                    for (int c = 0; c < NUM_CLASSES; c++) {
                        final_probs[c] = unweighted_probs[c] / (float)gesture_frames_count;
                      }
                }
                // 在所有动作分类中寻找最大概率值 (argmax over classes)
                int final_class = 0;
                float max_prob = final_probs[0];
                for (int c = 1; c < NUM_CLASSES; c++) {
                    if (final_probs[c] > max_prob) {
                        max_prob = final_probs[c];
                        final_class = c;
                    }
                }
                
                ESP_LOGI(TAG, "================ GESTURE DETECTED ================");
                const char *final_name = (final_class < NUM_CLASSES && gesture_names[final_class]) ? gesture_names[final_class] : "Unknown";
                ESP_LOGI(TAG, "🏆🏆🏆 Final Result: %s (Score: %.1f%%)", final_name, max_prob * 100.0f);
                for (int c = 0; c < NUM_CLASSES; c++) {
                    const char *name = (c < NUM_CLASSES && gesture_names[c]) ? gesture_names[c] : "Unknown";
                    ESP_LOGI(TAG, "      - Class %d (%s) Prob: %.2f%%", c, name, final_probs[c] * 100.0f);
                }
                ESP_LOGI(TAG, "      - Accum Frames: %d, Total Weight: %.2f", gesture_frames_count, total_weight);
                ESP_LOGI(TAG, "==================================================");
            } else {
                ESP_LOGI(TAG, "================ GESTURE DETECTED ================");
                ESP_LOGI(TAG, "⚠️⚠️⚠️ Finished, but no high-confidence frames accumulated.");
                ESP_LOGI(TAG, "==================================================");
            }
            
            // 重置累加状态与平滑历史
            memset(accumulated_probs, 0, sizeof(accumulated_probs));
            memset(unweighted_probs, 0, sizeof(unweighted_probs));
            total_weight = 0.0f;
            gesture_frames_count = 0;
            pred_history_count = 0;
        }
    }
}

/**
 * @brief Core 0 任务: UART 接收和运动监测状态机
 */
static void csi_uart_rx_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting CSI UART RX Task on Core %d...", xPortGetCoreID());
    
    int8_t *csi_raw_payload = (int8_t *)malloc(CSI_FRAME_MAX_RAW_LEN);
    if (!csi_raw_payload) {
        ESP_LOGE(TAG, "Failed to allocate CSI raw payload buffer!");
        vTaskDelete(NULL);
        return;
    }
    
    static float new_frame[FUSION_CHANNELS];
    
    bool is_moving = false;
    int motion_trigger_counter = 0;
    int idle_counter = 0;
    
    uint32_t seq = 0;
    size_t raw_len = 0;
    
    while (true) {
        if (read_csi_frame(&seq, csi_raw_payload, &raw_len)) {
            // 解析并对齐子载波
            parse_and_filter_csi(csi_raw_payload, raw_len, new_frame);
            
            // 写入滑动窗口
            if (csi_window_count < SLIDING_WINDOW_SIZE) {
                memcpy(csi_window + csi_window_count * FUSION_CHANNELS, new_frame, sizeof(new_frame));
                csi_window_count++;
            } else {
                // 滑动左移1帧，并将新帧追加在尾部
                memmove(csi_window, csi_window + FUSION_CHANNELS, (SLIDING_WINDOW_SIZE - 1) * FUSION_CHANNELS * sizeof(float));
                memcpy(csi_window + (SLIDING_WINDOW_SIZE - 1) * FUSION_CHANNELS, new_frame, sizeof(new_frame));
            }
            
            // 当滑动窗口攒满50帧后，触发计算标准差与状态机
            if (csi_window_count == SLIDING_WINDOW_SIZE) {
                float motion_val = calculate_motion_level(csi_window);
                
                if (motion_val >= MOTION_THRESHOLD) {
                    idle_counter = 0;
                    if (!is_moving) {
                        motion_trigger_counter++;
                        if (motion_trigger_counter >= REQUIRED_TRIGGER_FRAMES) {
                            is_moving = true;
                            ESP_LOGI(TAG, "🔴 [MOTION DETECTED] Gesture started! (Motion Level: %.2f)", motion_val);
                        }
                    }
                    
                    if (is_moving) {
                        // 利用双缓冲区异步推送给Core 1进行AI推理，防止卡死Core 0的串口接收中断
                        portENTER_CRITICAL(&sync_mux);
                        if (!inference_request_pending) {
                            memcpy(inference_input_buf, csi_window, TOTAL_FEATURES * sizeof(float));
                            inference_motion_val = motion_val;
                            inference_request_pending = true;
                            if (ai_task_handle != NULL) {
                                xTaskNotifyGive(ai_task_handle);
                            }
                        }
                        portEXIT_CRITICAL(&sync_mux);
                    }
                } else {
                    if (is_moving) {
                        motion_trigger_counter = 0;
                        idle_counter++;
                        
                        if (idle_counter >= DEBOUNCE_FRAMES) {
                            is_moving = false;
                            
                            // 通知Core 1动作结束并计算最终决策
                            portENTER_CRITICAL(&sync_mux);
                            gesture_ended_pending = true;
                            if (ai_task_handle != NULL) {
                                xTaskNotifyGive(ai_task_handle);
                            }
                            portEXIT_CRITICAL(&sync_mux);
                            
                            ESP_LOGI(TAG, "⚪ [MOTION ENDED] Returning to standby.");
                        }
                    } else {
                        motion_trigger_counter = 0;
                    }
                }
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    
    free(csi_raw_payload);
    vTaskDelete(NULL);
}

// ==================== 7. 主函数入口 ====================
void app_main(void)
{
    ESP_LOGI(TAG, "======= ESP32-P4 WiFi CSI Inference Startup =======");

    // 1. 在 PSRAM 中动态分配滑动窗口
    csi_window = (float *)heap_caps_malloc(TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
    inference_input_buf = (float *)heap_caps_malloc(TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!csi_window || !inference_input_buf) {
        ESP_LOGE(TAG, "Failed to allocate CSI window buffers in PSRAM!");
        return;
    }
    memset(csi_window, 0, TOTAL_FEATURES * sizeof(float));
    memset(inference_input_buf, 0, TOTAL_FEATURES * sizeof(float));

    // 2. 初始化 UART 串口外设
    if (init_csi_uart() != ESP_OK) {
        ESP_LOGE(TAG, "CSI UART Init failed! System halted.");
        return;
    }
    
    // 3. 创建核心任务并绑定CPU
    // Pinned to APP_CPU (Core 1) for Neural Network inference (降低栈大小至 4096 words/16KB)
    xTaskCreatePinnedToCore(
        ai_inference_task,
        "ai_inference_task",
        4096,
        NULL,
        5,
        &ai_task_handle,
        1
    );

    // Pinned to PRO_CPU (Core 0) for UART IO & State Machine (降低栈大小至 3072 words/12KB)
    xTaskCreatePinnedToCore(
        csi_uart_rx_task,
        "csi_uart_rx_task",
        3072,
        NULL,
        6, // 略微调高数据接收任务的优先级，确保串口不阻塞丢帧
        NULL,
        0
    );
}
