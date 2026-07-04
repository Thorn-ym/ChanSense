#ifndef NN_MODEL_H
#define NN_MODEL_H

#include <stdbool.h>
#include "esp_err.h"

#define NUM_CLASSES 4
#define NN_MODEL_MAX_SLOT_COUNT 8

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the quadrant classifier model (loads model into memory)
 */
void nn_model_init(void);

/**
 * @brief Rescan SD card model files without changing the active model.
 */
void nn_model_rescan(void);

/**
 * @brief Return the number of supported model slots.
 */
int nn_model_get_slot_count(void);

/**
 * @brief Return whether a model slot contains valid model bytes.
 */
bool nn_model_is_installed(int model_id);

/**
 * @brief Select and initialize a model slot. Returns ESP_OK only for installed slots.
 */
esp_err_t nn_model_select(int model_id);

/**
 * @brief Return the current model slot ID.
 */
int nn_model_get_active_id(void);

/**
 * @brief Return the current model display name.
 */
const char *nn_model_get_active_name(void);

/**
 * @brief Return a model slot display name.
 */
const char *nn_model_get_slot_name(int model_id);

/**
 * @brief Return the configured class name for the active model.
 */
const char *nn_model_get_class_name(int class_id);

/**
 * @brief Run inference on a coordinate pair (x, y)
 * 
 * @param x X-coordinate value
 * @param y Y-coordinate value
 * @return int Predicted quadrant class (0-3), or -1 on error
 */
int nn_model_predict(float x, float y);

/**
 * @brief Run inference on preprocessed 1D-CNN features
 * 
 * @param raw_csi Pointer to 11400 floats (50 time steps * 228 features)
 * @return int Predicted class (0: wave/cut, 1: grip, 2: circle/draw_o, 3: unknown), or -1 on error
 */
int nn_model_predict_cnn(float *raw_csi);

/**
 * @brief Run inference on preprocessed 1D-CNN features and output class probabilities
 * 
 * @param raw_csi Pointer to 11400 floats (50 time steps * 228 features)
 * @param out_probs Pointer to float array of size 4 to output probabilities
 * @return int Predicted class (0: wave/cut, 1: grip, 2: circle/draw_o, 3: unknown), or -1 on error
 */
int nn_model_predict_cnn_with_probs(float *raw_csi, float *out_probs);

#ifdef __cplusplus
}
#endif

#endif // NN_MODEL_H
