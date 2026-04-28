import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tcn import TCN

class AttentionLayer(Layer):
    """Custom Attention layer for the TCN output."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x is of shape (batch, time_steps, features)
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)
        
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

def build_workload_model(seq_length, num_features):
    """
    Builds the TCN + Attention model for workload prediction.
    Outputs predictive values for future cpu_usage and future request_rate.
    """
    inputs = Input(shape=(seq_length, num_features))
    
    # TCN Layer
    tcn_out = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8, 16], 
                  return_sequences=True, activation='relu')(inputs)
    
    # Attention Layer
    attention_out = AttentionLayer()(tcn_out)
    
    # Output layer: expecting 2 targets (cpu_usage, request_rate)
    outputs = Dense(2, activation='linear')(attention_out)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
