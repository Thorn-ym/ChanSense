#ifndef OUTPUT_DISPATCHER_H
#define OUTPUT_DISPATCHER_H

#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NUM_CLASSES
#define NUM_CLASSES 4
#endif

// 推理结果数据结构
typedef struct {
    int class_id;                 // 预测得到的类别 ID (-1 表示不确定)
    const char *class_name;        // 类别名称
    float confidence;             // 预测置信度 (0.0 ~ 1.0)
    float motion_val;             // 当前的运动强度值
    bool is_final;                // true 表示动作结束后的最终决策结果，false 表示过程中的逐帧预测结果
    float probs[NUM_CLASSES];     // 四个分类的完整概率分布
    uint32_t execution_time_us;   // 推理耗时 (单位：微秒)
    float total_weight;           // 决策累加的总权重
    int accum_frames;             // 参与决策的有效帧数
} inference_result_t;

// 结果回调函数指针定义
typedef void (*output_callback_t)(const inference_result_t *result);

/**
 * @brief 初始化输出分发器（创建异步处理队列与服务任务）
 * 
 * @return esp_err_t 初始化状态，ESP_OK 表示成功
 */
esp_err_t output_dispatcher_init(void);

/**
 * @brief 注册同步输出通道（适用于极低延迟的非阻塞外设，例如 GPIO 状态指示灯、快速控制台日志）
 * 
 * @param cb 回调函数指针
 * @return esp_err_t ESP_OK 表示成功
 */
esp_err_t output_dispatcher_register_sync(output_callback_t cb);

/**
 * @brief 注册异步输出通道（适用于网络发送 HTTP/MQTT、I2C 慢速 OLED 屏幕等高延迟外设）
 * 
 * @param cb 回调函数指针
 * @return esp_err_t ESP_OK 表示成功
 */
esp_err_t output_dispatcher_register_async(output_callback_t cb);

/**
 * @brief 分发推理结果到所有已注册的同步和异步输出通道
 * 
 * @param result 推理结果结构体指针
 */
void output_dispatcher_dispatch(const inference_result_t *result);

#ifdef __cplusplus
}
#endif

#endif // OUTPUT_DISPATCHER_H
