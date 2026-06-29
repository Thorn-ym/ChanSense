# ESP32-P4 WiFi CSI 固件代码重构总结与扩展指南

我们已经成功将原本高度耦合在 `main.c` 中的功能，重构为五个低耦合、高内聚、易于维护和扩展的独立子模块。以下是本次重构的成果总结和未来的扩展指南。

---

## 1. 代码结构与文件划分

重构后，[p4_remote_wifi/main](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main) 目录下的结构非常清晰：

| 模块名称 | 头文件 (Header) | 源文件 (Source) | 职责描述 |
| :--- | :--- | :--- | :--- |
| **配置中心** | [sys_config.h](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/sys_config.h) | [sys_config.c](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/sys_config.c) | 线程安全地存储和查询运行时的各项阈值（运动阈值、滤波置信度等）。 |
| **串口接收器** | [csi_receiver.h](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/csi_receiver.h) | [csi_receiver.c](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/csi_receiver.c) | 负责 UART1 初始化、同步字节头校验和数据完整性校验。 |
| **运动检测器** | [motion_detector.h](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/motion_detector.h) | [motion_detector.c](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/motion_detector.c) | 封装子载波对齐滤波、滑动窗口管理、标准差计算及动作状态机。 |
| **输出分发器** | [output_dispatcher.h](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/output_dispatcher.h) | [output_dispatcher.c](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/output_dispatcher.c) | 结果的发布订阅中心，维护同步/异步回调列表和后台异步分发线程。 |
| **系统协调器** | - | [main.c](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/main.c) | 启动入口、管理双核通信桥及 FreeRTOS 高优先级物理驱动任务。 |

---

## 2. 核心功能及性能保障

1. **零性能损耗**：
   * 所有内存（滑动窗口、推理缓冲区等）仍然采用静态/初始化阶段在 **PSRAM (SPIRAM)** 中预分配，消除了运行时的内存分配碎片与时延。
   * 配置中心的数据访问基于 **自旋锁 (portMUX_TYPE)**，多核并发读取时的开销在纳秒级。
2. **零时延异步 IO 分发**：
   * 输出分发器提供了**同步**和**异步**两种回调注册方式。
   * 慢速设备（如 HTTP 网络上传和慢速 I2C OLED 刷新）的回调运行在独立的低优先级后台线程 `output_async_task`，通过 FreeRTOS 队列串行化消费数据，**绝对不会拖慢 Core 1 上的神经网络实时推理主频**。
3. **低耦合状态机**：
   * 将状态机逻辑和子载波解析完全隐藏于 `motion_detector` 模块内部，简化了 `csi_uart_rx_task` 的代码，从而使得数据包接收循环逻辑极为轻量。

---

## 3. 后续扩展指南

### 扩展指南一：如何接入旋转编码器动态修改阈值？
有了配置中心，您只需在新开辟的硬件驱动任务（例如旋转编码器中断）中，调用以下 API 即可：

```c
// 假设旋转编码器被旋转，计算出新的运动阈值 new_val
float new_val = 3.5f;

// 直接在您的编码器任务中安全地设置新值
sys_config_set_motion_threshold(new_val);

// 运动检测状态机在下一帧运行时，便会自动从配置中心中线程安全地读取并采用该新阈值。
```

### 扩展指南二：如何添加 OLED 屏幕显示和 HTTP 上报？
在 [main.c](file:///d:/project/ddl-tcore/DaedalusLoom/src/firmware-code/p4_remote_wifi/main/main.c) 中，我们为您准备了占位符示例。您只需：

1. 实现您的具体驱动逻辑回调函数：
   ```c
   static void my_oled_output_cb(const inference_result_t *res) {
       if (res->is_final) {
           // OLED 绘图逻辑
           // oled_draw_string(0, 0, res->class_name);
       }
   }
   ```
2. 在 `app_main` 中注册为**异步回调**（让其自动在后台任务中排队调用，防止阻塞 AI 线程）：
   ```c
   void app_main(void) {
       // ...
       ESP_ERROR_CHECK(output_dispatcher_init());
       ESP_ERROR_CHECK(output_dispatcher_register_sync(console_logger_cb)); // 快速日志
       
       // 注册慢速 IO (OLED / HTTP) 到异步通道
       ESP_ERROR_CHECK(output_dispatcher_register_async(my_oled_output_cb));
       // ...
   }
   ```
