#ifndef NN_MODEL_H
#define NN_MODEL_H

#define NUM_CLASSES 4

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the quadrant classifier model (loads model into memory)
 */
void nn_model_init(void);

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
 * @param input_features Pointer to 11400 floats (50 time steps * 228 features)
 * @return int Predicted class (0: wave/cut, 1: grip, 2: circle/draw_o, 3: unknown), or -1 on error
 */
int nn_model_predict_cnn(float *input_features);

/**
 * @brief Run inference on preprocessed 1D-CNN features and output class probabilities
 * 
 * @param input_features Pointer to 11400 floats (50 time steps * 228 features)
 * @param out_probs Pointer to float array of size 4 to output probabilities
 * @return int Predicted class (0: wave/cut, 1: grip, 2: circle/draw_o, 3: unknown), or -1 on error
 */
int nn_model_predict_cnn_with_probs(float *input_features, float *out_probs);

#ifdef __cplusplus
}
#endif

#endif // NN_MODEL_H
