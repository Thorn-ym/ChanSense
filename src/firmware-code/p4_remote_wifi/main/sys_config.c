#include "sys_config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// 全局配置实例，默认值与重构前一致
static sys_config_t g_config = {
    .motion_threshold = 3.0f,
    .debounce_frames = 15,
    .required_trigger_frames = 3,
    .min_confidence = 0.80f
};

// 用于多核保护的自旋锁
static portMUX_TYPE g_config_mux = portMUX_INITIALIZER_UNLOCKED;

void sys_config_init(void)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.motion_threshold = 3.0f;
    g_config.debounce_frames = 15;
    g_config.required_trigger_frames = 3;
    g_config.min_confidence = 0.80f;
    portEXIT_CRITICAL(&g_config_mux);
}

void sys_config_get(sys_config_t *out_config)
{
    if (out_config) {
        portENTER_CRITICAL(&g_config_mux);
        *out_config = g_config;
        portEXIT_CRITICAL(&g_config_mux);
    }
}

void sys_config_set(const sys_config_t *in_config)
{
    if (in_config) {
        portENTER_CRITICAL(&g_config_mux);
        g_config = *in_config;
        portEXIT_CRITICAL(&g_config_mux);
    }
}

float sys_config_get_motion_threshold(void)
{
    float val;
    portENTER_CRITICAL(&g_config_mux);
    val = g_config.motion_threshold;
    portEXIT_CRITICAL(&g_config_mux);
    return val;
}

void sys_config_set_motion_threshold(float val)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.motion_threshold = val;
    portEXIT_CRITICAL(&g_config_mux);
}

int sys_config_get_debounce_frames(void)
{
    int val;
    portENTER_CRITICAL(&g_config_mux);
    val = g_config.debounce_frames;
    portEXIT_CRITICAL(&g_config_mux);
    return val;
}

void sys_config_set_debounce_frames(int val)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.debounce_frames = val;
    portEXIT_CRITICAL(&g_config_mux);
}

int sys_config_get_required_trigger_frames(void)
{
    int val;
    portENTER_CRITICAL(&g_config_mux);
    val = g_config.required_trigger_frames;
    portEXIT_CRITICAL(&g_config_mux);
    return val;
}

void sys_config_set_required_trigger_frames(int val)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.required_trigger_frames = val;
    portEXIT_CRITICAL(&g_config_mux);
}

float sys_config_get_min_confidence(void)
{
    float val;
    portENTER_CRITICAL(&g_config_mux);
    val = g_config.min_confidence;
    portEXIT_CRITICAL(&g_config_mux);
    return val;
}

void sys_config_set_min_confidence(float val)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.min_confidence = val;
    portEXIT_CRITICAL(&g_config_mux);
}
