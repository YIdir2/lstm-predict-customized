def model_predict_LSTM(model, input_pred):
        
    #Layer_0
    units0 = int(int(model.layers[0].trainable_weights[0].shape[1])/4) #units = 5
    #units0 = model.layers[0].get_config()['units']
    
    c_tm1 = np.zeros((1, units0))
    h_tm1 = np.zeros((1, units0))
    
    W0 = model.layers[0].get_weights()[0] #(3, 20)
    U0 = model.layers[0].get_weights()[1] #(5, 20)
    b0 = model.layers[0].get_weights()[2] #(20,)
    
    # Every LSTM cell contain 4 trainable weights (input gate, forget gate, new_candidate, output gate)
    # Keras API divide them to W, U and b. W is the trainable weights for the input sequences
    # U is the trainable weights for the hidden states and b is the bias vector. 
    
    W0_i = W0[:, :units0]
    W0_f = W0[:, units0: units0 * 2]
    W0_c = W0[:, units0 * 2: units0 * 3]
    W0_o = W0[:, units0 * 3:]
    
    U0_i = U0[:, :units0]
    U0_f = U0[:, units0: units0 * 2]
    U0_c = U0[:, units0 * 2: units0 * 3]
    U0_o = U0[:, units0 * 3:]
    
    b0_i = b0[:units0]#.reshape(units0, 1)
    b0_f = b0[units0: units0 * 2]#.reshape(units0, 1)
    b0_c = b0[units0 * 2: units0 * 3]#.reshape(units0, 1)
    b0_o = b0[units0 * 3:]#.reshape(units0, 1)
    
    
    i0_t = sigmoid(np.dot(input_pred, W0_i)+np.dot(h_tm1, U0_i)+b0_i) # (1,5) 
    f0_t = sigmoid(np.dot(input_pred, W0_f)+np.dot(h_tm1, U0_f)+b0_f)
    o0_t = sigmoid(np.dot(input_pred, W0_o)+np.dot(h_tm1, U0_o)+b0_o)
    new_candidate0_t = tanh(np.dot(input_pred ,W0_c)+np.dot(h_tm1, U0_c)+b0_c)
    c0_t = np.multiply(f0_t, c_tm1)+np.multiply(i0_t,  new_candidate0_t) # use * instead of np.dot
    h0_t = np.multiply(o0_t, tanh(c0_t))
    
        
    #Layer_1
    units1 = int(int(model.layers[1].trainable_weights[0].shape[1])/4)
    print("Num units1: ", units1)
    
    c1_tm1 = np.zeros((1, units1))
    h1_tm1 = np.zeros((1, units1))
    
    W1 = model.layers[1].get_weights()[0]
    U1 = model.layers[1].get_weights()[1]
    b1 = model.layers[1].get_weights()[2]
    
    W1_i = W1[:, :units1]
    W1_f = W1[:, units1: units1 * 2]
    W1_c = W1[:, units1 * 2: units1 * 3]
    W1_o = W1[:, units1 * 3:]
    
    U1_i = U1[:, :units1]
    U1_f = U1[:, units1: units1 * 2]
    U1_c = U1[:, units1 * 2: units1 * 3]
    U1_o = U1[:, units1 * 3:]
    
    b1_i = b1[:units1]#.reshape(units1, 1)
    b1_f = b1[units1: units1 * 2]#.reshape(units1, 1)
    b1_c = b1[units1 * 2: units1 * 3]#.reshape(units1, 1)
    b1_o = b1[units1 * 3:]#.reshape(units1, 1)
    
   
    
    h0_t = np.maximum(0, h0_t) # RELU activation function
    i1_t = sigmoid(np.dot(h0_t, W1_i)+np.dot(h1_tm1, U1_i)+b1_i)  #########  
    f1_t = sigmoid(np.dot(h0_t, W1_f)+np.dot(h1_tm1, U1_f)+b1_f)
    o1_t = sigmoid(np.dot(h0_t, W1_o)+np.dot(h1_tm1, U1_o)+b1_o)
    new_candidate1_t = tanh(np.dot(h0_t, W1_c)+np.dot(h1_tm1, U1_c)+b1_c)
    c1_t = np.multiply(f1_t, c1_tm1)+np.multiply(i1_t, new_candidate1_t)
    h1_t = np.multiply(o1_t, tanh(c1_t))
    
       
    #Layer_2
    h1_t = np.maximum(0, h1_t) # RELU activation function
    W2 = model.layers[2].get_weights()[0]
    b2 = model.layers[2].get_weights()[1]
    dense_output=np.dot(h1_t, W2)+b2 #.reshape(3,1)
    output_pred = dense_output.reshape(1, 3)
    
    return output_pred
