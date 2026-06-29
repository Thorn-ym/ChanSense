#ifndef MOTION_DETECTOR_H
#define MOTION_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SLIDING_WINDOW_SIZE 50
#define NUM_SUBCARRIERS     114
#define FUSION_CHANNELS     228
#define TOTAL_FEATURES      (SLIDING_WINDOW_SIZE * FUSION_CHANNELS)

// 运动检测状态结构体
typedef struct {
    float *window_buf;               // 滑动窗口缓冲区指针（在 PSRAM 中分配）
    int window_count;                // 当前填充的帧数
    bool is_moving;                  // 运动是否开始
    int motion_trigger_counter;      // 连续超出阈值帧计数器
    int idle_counter;                // 连续低于阈值帧计数器（消抖）
} motion_detector_t;

/**
 * @brief 创建运动检测器实例（动态在 PSRAM 中分配滑动窗口）
 * 
 * @return motion_detector_t* 返回创建的实例指针，失败返回 NULL
 */
motion_detector_t *motion_detector_create(void);

/**
 * @brief 销毁运动检测器实例并释放内存
 * 
 * @param detector 实例指针
 */
void motion_detector_destroy(motion_detector_t *detector);

/**
 * @brief 重置运动检测器的内部状态
 */
void motion_detector_reset(motion_detector_t *detector);

/**
 * @brief 输入一帧新的原始 CSI 载荷进行解析、滤波并更新运动检测状态机
 * 
 * @param detector 运动检测器实例
 * @param raw_payload 原始数据帧指针
 * @param raw_len 原始数据帧长度
 * @param out_motion_val 输出参数，当前帧计算得到的运动标准差
 * @param out_window_buf 输出缓冲区，当检测到动作并且有推理请求时，将当前 50*228 窗口数据拷贝至此
 * @param out_is_started 输出参数，如果这一帧动作刚开始触发（状态机从 standby 到 moving），设为 true
 * @param out_is_ended 输出参数，如果这一帧动作刚好结束（状态机从 moving 到 standby），设为 true
 * @return true 当前帧需要请求 AI 模型推理
 * @return false 当前帧无需进行推理
 */
bool motion_detector_process_frame(motion_detector_t *detector,
                                   const int8_t *raw_payload,
                                   size_t raw_len,
                                   float *out_motion_val,
                                   float *out_window_buf,
                                   bool *out_is_started,
                                   bool *out_is_ended);

#ifdef __cplusplus
}
#endif

#endif // MOTION_DETECTOR_H
