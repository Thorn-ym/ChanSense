#include "output_dispatcher.h"
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_log.h"

#define TAG "output_dispatcher"
#define MAX_CALLBACKS 4
#define DISPATCH_QUEUE_LEN 10

static output_callback_t g_sync_cbs[MAX_CALLBACKS] = {0};
static int g_sync_cb_count = 0;

static output_callback_t g_async_cbs[MAX_CALLBACKS] = {0};
static int g_async_cb_count = 0;

static QueueHandle_t g_output_queue = NULL;
static SemaphoreHandle_t g_dispatcher_mutex = NULL;

static void output_async_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting Output Async Task on Core %d...", xPortGetCoreID());
    inference_result_t result;
    while (1) {
        if (xQueueReceive(g_output_queue, &result, portMAX_DELAY) == pdPASS) {
            // 获取已注册的异步回调的快照
            xSemaphoreTake(g_dispatcher_mutex, portMAX_DELAY);
            int count = g_async_cb_count;
            output_callback_t callbacks[MAX_CALLBACKS];
            memcpy(callbacks, g_async_cbs, sizeof(g_async_cbs));
            xSemaphoreGive(g_dispatcher_mutex);

            // 逐个调用异步回调，即使其中一个慢，也仅影响该线程，不阻塞 AI 推理线程
            for (int i = 0; i < count; i++) {
                if (callbacks[i]) {
                    callbacks[i](&result);
                }
            }
        }
    }
}

esp_err_t output_dispatcher_init(void)
{
    g_dispatcher_mutex = xSemaphoreCreateMutex();
    if (!g_dispatcher_mutex) {
        ESP_LOGE(TAG, "Failed to create dispatcher mutex");
        return ESP_ERR_NO_MEM;
    }

    g_output_queue = xQueueCreate(DISPATCH_QUEUE_LEN, sizeof(inference_result_t));
    if (!g_output_queue) {
        ESP_LOGE(TAG, "Failed to create dispatcher queue");
        vSemaphoreDelete(g_dispatcher_mutex);
        return ESP_ERR_NO_MEM;
    }

    // 创建后台异步输出分发任务，由于存在高延迟网络 IO，这里将任务栈设定为 4096 字节
    // 优先级较低 (3)，绑定在 Core 1，保证当 Core 1 推理闲置时异步刷屏/发网络包
    BaseType_t ret = xTaskCreatePinnedToCore(
        output_async_task,
        "output_async_task",
        4096,
        NULL,
        3,
        NULL,
        1
    );

    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create output async task");
        vQueueDelete(g_output_queue);
        vSemaphoreDelete(g_dispatcher_mutex);
        return ESP_FAIL;
    }

    return ESP_OK;
}

esp_err_t output_dispatcher_register_sync(output_callback_t cb)
{
    if (!cb) return ESP_ERR_INVALID_ARG;
    if (!g_dispatcher_mutex) return ESP_ERR_INVALID_STATE;

    xSemaphoreTake(g_dispatcher_mutex, portMAX_DELAY);
    if (g_sync_cb_count >= MAX_CALLBACKS) {
        xSemaphoreGive(g_dispatcher_mutex);
        ESP_LOGE(TAG, "Max sync callbacks reached");
        return ESP_ERR_NO_MEM;
    }
    g_sync_cbs[g_sync_cb_count++] = cb;
    xSemaphoreGive(g_dispatcher_mutex);
    return ESP_OK;
}

esp_err_t output_dispatcher_register_async(output_callback_t cb)
{
    if (!cb) return ESP_ERR_INVALID_ARG;
    if (!g_dispatcher_mutex) return ESP_ERR_INVALID_STATE;

    xSemaphoreTake(g_dispatcher_mutex, portMAX_DELAY);
    if (g_async_cb_count >= MAX_CALLBACKS) {
        xSemaphoreGive(g_dispatcher_mutex);
        ESP_LOGE(TAG, "Max async callbacks reached");
        return ESP_ERR_NO_MEM;
    }
    g_async_cbs[g_async_cb_count++] = cb;
    xSemaphoreGive(g_dispatcher_mutex);
    return ESP_OK;
}

void output_dispatcher_dispatch(const inference_result_t *result)
{
    if (!result || !g_dispatcher_mutex) return;

    // 1. 在调用者（AI推理任务）线程上下文中，立即同步触发快速回调
    xSemaphoreTake(g_dispatcher_mutex, portMAX_DELAY);
    int sync_count = g_sync_cb_count;
    output_callback_t sync_callbacks[MAX_CALLBACKS];
    memcpy(sync_callbacks, g_sync_cbs, sizeof(g_sync_cbs));
    xSemaphoreGive(g_dispatcher_mutex);

    for (int i = 0; i < sync_count; i++) {
        if (sync_callbacks[i]) {
            sync_callbacks[i](result);
        }
    }

    // 2. 发送结果拷贝至异步处理队列（如果队列已满则非阻塞丢弃，绝不拖慢 AI 推理）
    if (xQueueSend(g_output_queue, result, 0) != pdPASS) {
        ESP_LOGW(TAG, "Output queue full! Async dispatch dropped.");
    }
}
