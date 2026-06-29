#ifndef CSI_RECEIVER_H
#define CSI_RECEIVER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CSI_FRAME_MAX_RAW_LEN   1024

/**
 * @brief 初始化 CSI 接收所需要的 UART 驱动与引脚配置
 * 
 * @return esp_err_t 初始化结果，ESP_OK 表示成功
 */
esp_err_t csi_receiver_init(void);

/**
 * @brief 从 UART 阻塞读取精确长度的 CSI 数据协议帧并校验
 * 
 * @param seq 输出参数，解析出的帧序号
 * @param raw_payload 输出缓冲区，用于存放解析出的载荷数据（最大长度必须为 CSI_FRAME_MAX_RAW_LEN）
 * @param raw_len 输出参数，解析出的载荷数据长度
 * @return true 成功接收且校验通过
 * @return false 接收失败或校验失败
 */
bool csi_receiver_read_frame(uint32_t *seq, int8_t *raw_payload, size_t *raw_len);

#ifdef __cplusplus
}
#endif

#endif // CSI_RECEIVER_H
