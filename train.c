

void train(void)
{

    /* Load dataset and preprocess the data */

    /* Initialize model's parameters */

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        /* Loop through the dataset in mini-batch */
        for (int i = 0; i < num_batch; i++) {
            /* Get the current mini-batch */
            /* Compute the forward pass of the model to get the predicted output */
            /* Compute the loss function between the predicted output and the true label */
            /* Compute the gradients of the model's parameters with respect to the loss */
            /* Update the model's parameters using optimizer */
        }
    }

}




//In this example, n is the number of samples, y_true is the array of true labels, and y_pred is the array of predicted labels. 
// The function computes the cross-entropy loss between the true labels and predicted labels and returns the average loss over 
// all samples
float calculateLoss(int n, int *y_true, int *y_pred)
{
    float loss = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float y_true_i = y_true[i];
        float y_pred_i = y_pred[i];
        loss += -y_true_i * log(y_pred_i) - (1 - y_true_i) * log(1 - y_pred_i);
    }
    return loss / n;
}



float calculate_dloss_dy(float y_pred, float y_true)
{
    float ret = (y_pred - y_true) / (y_pred * (1 - y_pred));
    return ret;
}



void dloss_dy(int n, float* y_true, float* y_pred, float* dloss_dy)
{
    for (int i = 0; i < n; ++i)
    {
        dloss_dy[i] = y_pred[i] - y_true[i];
    }
}

float* calculate_doutput_dinput(float* output, float* input, model_t* model)
{
    int n = model->input_size;
    int m = model->output_size;
    float* doutput_dinput = (float*) malloc(n * sizeof(float));
    
    // Calculate the gradient of the output with respect to the input
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < m; j++) {
            float dloss_dy = calculate_dloss_dy(output[j], model->y_true[j]);
            sum += dloss_dy * calculate_doutput_dy(output[j], input[i], model);
        }
        doutput_dinput[i] = sum;
    }
    return doutput_dinput;
}

void backpropagate(model_t model, float loss)
{
  float dloss_doutput = dloss_dy(expected_output, output);

  // Propagate gradients through the model
  float* doutput_dinput = calculate_doutput_dinput(output, input, model);
  float* dinput_dwte = calculate_dinput_dwte(input, model);
  float* dinput_dwelw = calculate_dinput_dwelw(input, model);
  float* dinput_dwelb = calculate_dinput_dwelb(input, model);
  float* dinput_dlnf_g = calculate_dinput_dlnf_g(input, model);
  float* dinput_dlnf_b = calculate_dinput_dlnf_b(input, model);
  float* dinput_dln1_b = calculate_dinput_dln1_b(input, model);
  float* dinput_dln1_g = calculate_dinput_dln1_g(input, model);
  float* dinput_dln2_b = calculate_dinput_dln2_b(input, model);
  float* dinput_dln2_g = calculate_dinput_dln2_g(input, model);
  float* dinput_dmlp_cfc_b = calculate_dinput_dmlp_cfc_b(input, model);
  float* dinput_dmlp_cfc_w = calculate_dinput_dmlp_cfc_w(input, model);
  float* dinput_dmlp_cproj_b = calculate_dinput_dmlp_cproj_b(input, model);
  float* dinput_dmlp_cproj_w = calculate_dinput_dmlp_cproj_w(input, model);
  float* dinput_dattn_cattn_b = calculate_dinput_dattn_cattn_b(input, model);
  float* dinput_dattn_cattn_w = calculate_dinput_dattn_cattn_w(input, model);
  float* dinput_dattn_cproj_b = calculate_dinput_dattn_cproj_b(input, model);
  float* dinput_dattn_cproj_w = calculate_dinput_dattn_cproj_w(input, model);

  // Compute gradients
  gradients->s_ln1_b_gradients = elementwise_mul(dloss_doutput, doutput_dinput, dinput_dln1_b, numLayers);
  // ...
  //gradients->s_ln1_g_gradients = elementwise_m
}



void train(int expectedToken, int actualToken, model_t model, int numLayers) {
    // Step 1: Calculate the loss between the expected token and actual token using a chosen loss function such as cross-entropy.
    float loss = calculateLoss(numTokens, expectedTokens, actualTokens);

    // Step 2: Backpropagate the loss through the model to calculate the gradients.
    backpropagate(model, loss);

    // Step 3: Update the model parameters using an optimization algorithm such as Adam or SGD.
    updateParameters(model, numLayers);

    // Step 4: Repeat steps 1 to 3 for multiple epochs until the model converges or reaches a stopping criterion.
    while (notConverged(model) && notReachedStoppingCriterion(model)) {
        train(nextExpectedToken, nextActualToken, model, numLayers);
    }
}
