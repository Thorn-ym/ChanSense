#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

// 引入重构后的各个子组件
#include "sys_config.h"
#include "csi_receiver.h"
#include "motion_detector.h"
#include "output_dispatcher.h"
#include "nn_model.h"

static const char *TAG = "csi_main";

// ==================== 1. 双核共享的推理双缓冲 ====================
static float *inference_input_buf = NULL;
static float inference_motion_val = 0.0f;
static volatile bool inference_request_pending = false;
static volatile bool gesture_ended_pending = false;
static portMUX_TYPE sync_mux = portMUX_INITIALIZER_UNLOCKED;
static TaskHandle_t ai_task_handle = NULL;

static const char *gesture_names[NUM_CLASSES] = {
    "高抬腿",
    "展臂",
    "深蹲"
};

// ==================== 2. 默认输出处理器（控制台输出） ====================
static void console_logger_cb(const inference_result_t *res)
{
    if (!res->is_final) {
        if (res->class_id != -1) {
            ESP_LOGI(TAG, "🔴 [ACTIVE] Real-time Pred: %s (Conf: %.1f%%, Motion: %.2f, Time: %" PRIu32 " us)", 
                     res->class_name, res->confidence * 100.0f, res->motion_val, res->execution_time_us);
        } else {
            ESP_LOGI(TAG, "🔴 [ACTIVE] Real-time Pred: UNCERTAIN (Motion: %.2f)", res->motion_val);
        }
    } else {
        ESP_LOGI(TAG, "================ GESTURE DETECTED ================");
        if (res->class_id != -1) {
            ESP_LOGI(TAG, "🏆🏆🏆 Final Result: %s (Score: %.1f%%)", res->class_name, res->confidence * 100.0f);
            for (int c = 0; c < NUM_CLASSES; c++) {
                const char *name = (c < NUM_CLASSES && gesture_names[c]) ? gesture_names[c] : "Unknown";
                ESP_LOGI(TAG, "      - Class %d (%s) Prob: %.2f%%", c, name, res->probs[c] * 100.0f);
            }
            ESP_LOGI(TAG, "      - Accum Frames: %d, Total Weight: %.2f", res->accum_frames, res->total_weight);
        } else {
            ESP_LOGI(TAG, "⚠️⚠️⚠️ Finished, but no high-confidence frames accumulated.");
        }
        ESP_LOGI(TAG, "==================================================");
    }
}

/*
// ==================== 3. 示例：如何添加其他输出通道 ====================
// 示例 1: OLED 屏幕异步更新（可能耗时 20-50ms，因此注册为异步，避免拖慢推理）
static void oled_output_cb(const inference_result_t *res)
{
    if (res->is_final) {
        // 在这里编写您的 OLED 驱动库显示代码，例如：
        // oled_draw_string(0, 0, "Gesture: %s", res->class_name);
        // oled_draw_string(0, 16, "Conf: %.1f%%", res->confidence * 100.0f);
        // oled_refresh();
    }
}

// 示例 2: HTTP 发送异步回调（发起网络连接通常耗时几百毫秒，注册为异步，保证时序稳定）
static void http_post_output_cb(const inference_result_t *res)
{
    if (res->is_final) {
        // 在这里发起您的 HTTP POST 请求上传推理结果：
        // esp_http_client_config_t config = { .url = "http://my-server.com/api/gesture", ... };
        // esp_http_client_handle_t client = esp_http_client_init(&config);
        // esp_http_client_set_post_field(client, post_data, strlen(post_data));
        // esp_http_client_perform(client);
        // esp_http_client_cleanup(client);
    }
}
*/

// ==================== 4. 双核执行任务 ====================

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
            uint32_t execution_time_us = (uint32_t)(end - start);
            
            float confidence = probs[pred_class];
            
            // 从配置模块动态获取当前配置
            float motion_threshold = sys_config_get_motion_threshold();
            float min_confidence = sys_config_get_min_confidence();
            
            // 计算加权系数 (motion_val - threshold)^2
            float weight = (local_motion_val - motion_threshold) * (local_motion_val - motion_threshold);
            
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
            if (confidence >= min_confidence) {
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
            
            // 构造推理结果对象
            inference_result_t active_res = {
                .class_id = smoothed_class,
                .class_name = (smoothed_class >= 0 && smoothed_class < NUM_CLASSES && gesture_names[smoothed_class]) ? gesture_names[smoothed_class] : "Unknown",
                .confidence = (smoothed_class != -1) ? probs[smoothed_class] : 0.0f,
                .motion_val = local_motion_val,
                .is_final = false,
                .execution_time_us = execution_time_us,
                .total_weight = 0.0f,
                .accum_frames = 0
            };
            memcpy(active_res.probs, probs, sizeof(probs));
            
            // 通过输出分发器广播给所有 IO 设备
            output_dispatcher_dispatch(&active_res);
        }
        
        if (local_ended) {
            // 动作结束，进行加权总决策
            inference_result_t final_res = {
                .class_id = -1,
                .class_name = "Unknown",
                .confidence = 0.0f,
                .motion_val = local_motion_val,
                .is_final = true,
                .execution_time_us = 0,
                .total_weight = total_weight,
                .accum_frames = gesture_frames_count
            };
            memset(final_res.probs, 0, sizeof(final_res.probs));

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
                
                final_res.class_id = final_class;
                final_res.class_name = (final_class < NUM_CLASSES && gesture_names[final_class]) ? gesture_names[final_class] : "Unknown";
                final_res.confidence = max_prob;
                memcpy(final_res.probs, final_probs, sizeof(final_probs));
            }
            
            // 分发最终决策结果
            output_dispatcher_dispatch(&final_res);
            
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
    
    // 创建运动检测器实例
    motion_detector_t *detector = motion_detector_create();
    if (!detector) {
        ESP_LOGE(TAG, "Failed to create motion detector!");
        free(csi_raw_payload);
        vTaskDelete(NULL);
        return;
    }
    
    // 在 PSRAM 分配临时缓冲区，用于在状态机动作期间将滑动窗口内容送至双缓冲共享区
    float *temp_window_buf = (float *)heap_caps_malloc(TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!temp_window_buf) {
        ESP_LOGE(TAG, "Failed to allocate temp window buffer!");
        motion_detector_destroy(detector);
        free(csi_raw_payload);
        vTaskDelete(NULL);
        return;
    }
    
    uint32_t seq = 0;
    size_t raw_len = 0;
    
    while (true) {
        // 读取并同步串口数据帧
        if (csi_receiver_read_frame(&seq, csi_raw_payload, &raw_len)) {
            float motion_val = 0.0f;
            bool is_started = false;
            bool is_ended = false;
            
            // 输入状态机处理
            bool req_inference = motion_detector_process_frame(
                detector, csi_raw_payload, raw_len, 
                &motion_val, temp_window_buf, &is_started, &is_ended
            );
            
            // 动作帧：双缓冲区安全推送给 Core 1 推理线程，避免直接在 Core 0 运行时造成串口丢帧
            if (req_inference) {
                portENTER_CRITICAL(&sync_mux);
                if (!inference_request_pending) {
                    memcpy(inference_input_buf, temp_window_buf, TOTAL_FEATURES * sizeof(float));
                    inference_motion_val = motion_val;
                    inference_request_pending = true;
                    if (ai_task_handle != NULL) {
                        xTaskNotifyGive(ai_task_handle);
                    }
                }
                portEXIT_CRITICAL(&sync_mux);
            }
            
            // 动作结束帧：推送结束标志并激活 Core 1 进行最终加权投票统计
            if (is_ended) {
                portENTER_CRITICAL(&sync_mux);
                gesture_ended_pending = true;
                if (ai_task_handle != NULL) {
                    xTaskNotifyGive(ai_task_handle);
                }
                portEXIT_CRITICAL(&sync_mux);
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    
    heap_caps_free(temp_window_buf);
    motion_detector_destroy(detector);
    free(csi_raw_payload);
    vTaskDelete(NULL);
}

// ==================== 5. 主函数入口 ====================
void app_main(void)
{
    ESP_LOGI(TAG, "======= ESP32-P4 WiFi CSI Inference Startup (Refactored) =======");

    // 1. 初始化动态配置中心
    sys_config_init();

    // 2. 初始化输出分发器，并注册默认控制台同步输出回调
    ESP_ERROR_CHECK(output_dispatcher_init());
    ESP_ERROR_CHECK(output_dispatcher_register_sync(console_logger_cb));

    /*
    // 在这里，您可以轻松注册更多的外设输出通道，例如：
    // 注册异步刷新 OLED 的回调 (以防止 OLED I2C 刷屏慢阻塞 AI 任务)
    ESP_ERROR_CHECK(output_dispatcher_register_async(oled_output_cb));
    
    // 注册异步发送 HTTP 的回调 (网络请求耗时长，必须置于后台异步执行)
    ESP_ERROR_CHECK(output_dispatcher_register_async(http_post_output_cb));
    */

    // 3. 在 PSRAM 中动态分配推理交互使用的双缓冲区
    inference_input_buf = (float *)heap_caps_malloc(TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!inference_input_buf) {
        ESP_LOGE(TAG, "Failed to allocate CSI window buffers in PSRAM!");
        return;
    }
    memset(inference_input_buf, 0, TOTAL_FEATURES * sizeof(float));

    // 4. 初始化 UART 接收物理通道
    if (csi_receiver_init() != ESP_OK) {
        ESP_LOGE(TAG, "CSI UART Init failed! System halted.");
        return;
    }
    
    // 5. 创建核心任务并绑定CPU
    // Pinned to APP_CPU (Core 1) for Neural Network inference
    xTaskCreatePinnedToCore(
        ai_inference_task,
        "ai_inference_task",
        4096,
        NULL,
        5,
        &ai_task_handle,
        1
    );

    // Pinned to PRO_CPU (Core 0) for UART IO & State Machine
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
