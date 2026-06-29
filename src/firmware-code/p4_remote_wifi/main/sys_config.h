#ifndef SYS_CONFIG_H
#define SYS_CONFIG_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// 系统配置参数结构体
typedef struct {
    float motion_threshold;        // 运动标准差触发阈值
    int debounce_frames;           // 运动结束消抖帧数
    int required_trigger_frames;   // 运动开始确认帧数
    float min_confidence;          // 实时预测最小置信度过滤阈值
} sys_config_t;

/**
 * @brief 初始化配置模块（设置默认值）
 */
void sys_config_init(void);

/**
 * @brief 获取完整的系统配置结构体（线程安全）
 * 
 * @param out_config 输出配置结构体指针
 */
void sys_config_get(sys_config_t *out_config);

/**
 * @brief 设置完整的系统配置结构体（线程安全）
 * 
 * @param in_config 输入配置结构体指针
 */
void sys_config_set(const sys_config_t *in_config);

/**
 * @brief 获取当前运动阈值
 */
float sys_config_get_motion_threshold(void);

/**
 * @brief 设置新运动阈值
 */
void sys_config_set_motion_threshold(float val);

/**
 * @brief 获取运动结束消抖帧数
 */
int sys_config_get_debounce_frames(void);

/**
 * @brief 设置运动结束消抖帧数
 */
void sys_config_set_debounce_frames(int val);

/**
 * @brief 获取运动开始确认帧数
 */
int sys_config_get_required_trigger_frames(void);

/**
 * @brief 设置运动开始确认帧数
 */
void sys_config_set_required_trigger_frames(int val);

/**
 * @brief 获取实时预测过滤的最小置信度阈值
 */
float sys_config_get_min_confidence(void);

/**
 * @brief 设置实时预测过滤的最小置信度阈值
 */
void sys_config_set_min_confidence(float val);

#ifdef __cplusplus
}
#endif

#endif // SYS_CONFIG_H
