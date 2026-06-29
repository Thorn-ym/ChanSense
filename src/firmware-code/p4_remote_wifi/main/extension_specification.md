# ESP32-P4 WiFi CSI 固件后续扩展规范文档

本规范文档旨在为开发团队提供后续功能扩展的标准指南。重构后的系统遵循**高内聚、低耦合、多核异步**的设计原则。在后续开发（如增加旋转编码器、扩展显示设备、更换 AI 模型）时，请严格遵守本规范，以维持系统原有的高吞吐量与低延迟性能。

---

## 规范一：动态阈值与外设接入规范 (例如旋转编码器)

重构后的系统在 [sys_config](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/sys_config.h) 模块中对配置数据进行了统一封装，并通过自旋锁实现了多核间的线程安全访问。

### 1. 外设驱动任务的标准设计
*   **非阻塞原则**：旋转编码器或网络配置任务应运行在独立的低优先级线程（如优先级 2 或 3）中，或者直接在硬件中断 (ISR) 的下半部中通过队列触发任务。**禁止**在 UART 接收任务 (Core 0) 或 AI 推理任务 (Core 1) 中直接读取外设引脚或执行长延时消抖。
*   **自旋锁临界区**：修改阈值时，只需直接调用 `sys_config_set_*` 系列 API。这些 API 内部使用自旋锁完成保护，耗时在纳秒级，无需担心影响性能。

### 2. 代码实现示例
当您添加旋转编码器驱动时，参考如下实现：

```c
#include "driver/gpio.h"
#include "sys_config.h"
#include "esp_log.h"

#define ENCODER_A_GPIO GPIO_NUM_4
#define ENCODER_B_GPIO GPIO_NUM_5

static void rotary_encoder_task(void *pvParameters)
{
    ESP_LOGI("encoder", "旋转编码器驱动任务启动...");
    
    // 初始化 GPIO 并设置中断/轮询
    // ... 编码器硬件初始化代码 ...

    while (1) {
        // 阻塞等待编码器旋转事件
        if (wait_for_encoder_rotate()) {
            float current_thresh = sys_config_get_motion_threshold();
            float new_thresh = current_thresh;

            if (is_rotate_clockwise()) {
                new_thresh += 0.2f; // 顺时针旋转，提高动作触发阈值
            } else {
                new_thresh -= 0.2f; // 逆时针旋转，降低阈值
            }

            // 限制阈值合理范围，防止溢出
            if (new_thresh < 1.0f) new_thresh = 1.0f;
            if (new_thresh > 10.0f) new_thresh = 10.0f;

            // 线程安全写入配置中心
            sys_config_set_motion_threshold(new_thresh);
            ESP_LOGI("encoder", "动态阈值已更新: %.2f", new_thresh);
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
```

---

## 规范二：多 IO 输出设备扩展规范 (OLED屏幕/HTTP上报)

重构后的 [output_dispatcher](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/output_dispatcher.h) 支持注册多个独立的输出设备。**同步与异步的合理选择**是保障 AI 推理任务实时性的核心。

### 1. 同步 (Sync) 回调规范
*   **适用场景**：执行耗时极短（< 500 微秒）的操作，例如 GPIO 状态灯翻转、内存变量更新、快速 Console 打印。
*   **限制条件**：同步回调在 AI 推理线程的上下文中被直接执行。如果在此处执行耗时操作，会直接导致推理帧率下降或数据积压。**严禁在同步回调中调用网络发送、I2C 屏刷新、`vTaskDelay` 或带有大阻塞的锁。**

### 2. 异步 (Async) 回调规范
*   **适用场景**：慢速外设。例如 OLED 屏刷新 (一般耗时 15ms~50ms)、发送 HTTP/MQTT 请求 (一般耗时 100ms~1000ms)。
*   **工作机制**：当 AI 任务分发结果时，数据会被复制并推入 FreeRTOS 队列。后台异步任务 `output_async_task` 会在 CPU 空闲时唤醒，并逐个调用已注册的异步回调。
*   **防爆仓保护**：如果网络断开导致 HTTP 持续阻塞，异步队列填满后，分发器会自动**非阻塞丢弃**新结果，以死守 AI 任务的运行流畅度。

### 3. 代码实现规范示例
```c
#include "output_dispatcher.h"
#include "esp_http_client.h"

// 1. 实现您的 OLED 异步刷新回调
static void oled_draw_result_cb(const inference_result_t *res)
{
    // 本函数在 output_async_task (Core 1) 中执行
    if (res->is_final) {
        // 调用您的 oled 屏驱动库
        // oled_clear();
        // oled_draw_string(0, 0, res->class_name);
        // oled_show(); 
    }
}

// 2. 实现您的 HTTP 异步上报回调
static void http_post_result_cb(const inference_result_t *res)
{
    // 本函数在 output_async_task 中执行，网络阻塞时不会卡死 AI 推理
    if (res->is_final) {
        char post_data[128];
        snprintf(post_data, sizeof(post_data), "{\"gesture\":\"%s\",\"score\":%.2f}", 
                 res->class_name, res->confidence);

        esp_http_client_config_t config = {
            .url = "http://192.168.1.100:8080/api/result",
            .method = HTTP_METHOD_POST,
        };
        esp_http_client_handle_t client = esp_http_client_init(&config);
        esp_http_client_set_post_field(client, post_data, strlen(post_data));
        esp_http_client_set_header(client, "Content-Type", "application/json");
        
        esp_err_t err = esp_http_client_perform(client);
        if (err == ESP_OK) {
            ESP_LOGI("HTTP", "HTTP Post 成功");
        } else {
            ESP_LOGE("HTTP", "HTTP Post 失败: %s", esp_err_to_name(err));
        }
        esp_http_client_cleanup(client);
    }
}

// 3. 在 app_main 中注册回调
void app_main(void)
{
    // ... 初始化分发器后注册 ...
    output_dispatcher_register_async(oled_draw_result_cb);
    output_dispatcher_register_async(http_post_result_cb);
}
```

---

## 规范三：模型更换与部署规范 (神经网络升级)

当训练出新版 CNN 模型，或者更改了手势分类定义时，请按照以下步骤进行标准化更换：

### 1. 导出与模型更新
1.  **转换权重**：使用乐鑫 ESP-DL 转换工具将训练好的模型（如 `.onnx` 或 `.tflite` 格式）量化并生成 C 语言数组代码。
2.  **更新数据文件**：使用生成的数组更新 [model_data.c](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/model_data.c) 中的 `model_espdl` 数组。如果模型二进制字节大小改变，更新 `model_data.h` 中的 `model_espdl_len`。

### 2. 更新模型包装层 (`nn_model.h/cpp`)
1.  **类别数量**：如果手势数量发生了变化，必须修改 [nn_model.h](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/nn_model.h) 中的 `NUM_CLASSES` 宏。
2.  **输入与输出指数**：确保在 `nn_model_predict_cnn_with_probs` 中，输入与输出 Tensor 的 Exponent（量化指数）能够与新模型匹配。
3.  **前向传播适配**：如果有自定义的预处理，可以在 `preprocess_csi_fusion` 内部完成。

### 3. 主协调层与输出更新
1.  **手势名称表**：在 [main.c](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/main.c) 的 `gesture_names` 静态数组中，依次修改或增加对应的手势名称字符串（与新模型的分类输出顺序一一对应）。
2.  **模型参数自适应（推荐演进方向）**：如果频繁更换模型，可以在 `nn_model` 中提供 `nn_model_get_class_name(id)` 等接口，直接由模型提供分类名，彻底消除在 `main.c` 中修改手势名称的步骤。

---

## 规范四：多核内存分配与任务调度规范 (CPU/PSRAM)

ESP32-P4 是具有超大高性能 PSRAM 的双核处理器。为了极致的实时性，需遵守以下多核和内存法则：

### 1. 内存分配法则
*   **大数组分配**：由于输入特征为 50 帧 * 228 通道 = 11400 浮点数（约 45KB），**绝对不能**在任务栈中作为局部变量分配（容易导致 Stack Overflow）。
*   **PSRAM 优先**：大缓冲区（如 `motion_detector` 内的滑动窗口及多核共享的双缓冲 `inference_input_buf`）应使用 `heap_caps_malloc(..., MALLOC_CAP_SPIRAM)` 分配在外部 PSRAM 中，保证宝贵的内部 SRAM 留给任务栈及以太网/Wi-Fi 缓冲。

### 2. 双核调度分配
*   **Core 0 (PRO_CPU)**：专门用于时序非常严苛的数据 IO 接收和状态机。`csi_uart_rx_task` 绑定于 Core 0，优先级设为 **6**（稍高）。此线程中必须使用非阻塞的 `vTaskDelay` 让出 CPU，确保 Wi-Fi/串口中断能及时响应。
*   **Core 1 (APP_CPU)**：专门用于运算密集型的神经网络推理和输出分发。`ai_inference_task` 绑定于 Core 1，优先级设为 **5**。异步输出分发任务 `output_async_task` 亦运行在 Core 1，但优先级设为较矮的 **3**，以防止抢占 AI 推理的硬件算力。
