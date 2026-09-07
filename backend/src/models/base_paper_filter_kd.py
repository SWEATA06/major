import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Model

def build_bilstm_model(seq_length: int, num_features: int, hidden_units: int = 128, num_layers: int = 2, output_dim: int = 2) -> Model:
    """
    Builds a Bi-LSTM model as described in the base paper (Filter-KD).
    
    Args:
        seq_length: Number of time steps in sequence (e.g. 12)
        num_features: Number of features per timestep (e.g. 11)
        hidden_units: LSTM hidden size (e.g. 512 for Teacher, 128 for Student)
        num_layers: Number of Bi-LSTM layers
        output_dim: Output dimension (e.g. 2 for future_cpu_usage & future_request_rate)
    """
    inputs = Input(shape=(seq_length, num_features))
    x = inputs
    
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        x = Bidirectional(LSTM(hidden_units, return_sequences=return_seq))(x)
        
    outputs = Dense(output_dim, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs, name=f"BiLSTM_{hidden_units}x{num_layers}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss='mse', metrics=['mae'])
    return model


class FilterKDStudentModel(Model):
    """
    Custom Keras Model implementing Algorithm 1 & Eq (3)-(4) from the base paper:
    Filter-KD (Preventing Erroneous Prediction in Regression for Knowledge Distillation).
    
    Loss equation:
        L_K = (1/m) * sum( gamma * ||Psi^S(x_i) - y_i||^2 + (1-gamma) * L_filter )
        where L_filter = ||Psi^S(x_i) - Psi^T(x_i)||^2 if ||Psi^T(x_i) - y_i||^2 <= epsilon else 0
    """
    def __init__(self, student_network: Model, teacher_network: Model, gamma: float = 0.5, epsilon: float = 0.02):
        super(FilterKDStudentModel, self).__init__()
        self.student = student_network
        self.teacher = teacher_network
        self.teacher.trainable = False  # Freeze pretrained teacher
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.gt_loss_tracker = tf.keras.metrics.Mean(name="gt_loss")
        self.kd_loss_tracker = tf.keras.metrics.Mean(name="kd_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.gt_loss_tracker, self.kd_loss_tracker, self.val_loss_tracker]

    def train_step(self, data):
        x, y = data

        # Teacher predictions (no gradient required for teacher)
        teacher_preds = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)

            # Ground truth loss (MSE per sample)
            gt_error = tf.reduce_mean(tf.square(student_preds - y), axis=-1)

            # Teacher error per sample to filter erroneous teacher predictions
            teacher_error = tf.reduce_mean(tf.square(teacher_preds - y), axis=-1)

            # Filter-KD mask: 1.0 if teacher error <= epsilon, else 0.0
            filter_mask = tf.cast(teacher_error <= self.epsilon, dtype=tf.float32)

            # KD distillation loss per sample
            student_teacher_error = tf.reduce_mean(tf.square(student_preds - teacher_preds), axis=-1)
            filter_kd_loss = filter_mask * student_teacher_error

            # Combined Filter-KD Loss according to Equation 3
            batch_loss = self.gamma * gt_error + (1.0 - self.gamma) * filter_kd_loss
            total_loss = tf.reduce_mean(batch_loss)

        # Compute gradients & update student weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.total_loss_tracker.update_state(total_loss)
        self.gt_loss_tracker.update_state(tf.reduce_mean(gt_error))
        self.kd_loss_tracker.update_state(tf.reduce_mean(filter_kd_loss))

        return {
            "loss": self.total_loss_tracker.result(),
            "gt_loss": self.gt_loss_tracker.result(),
            "kd_loss": self.kd_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        student_preds = self.student(x, training=False)
        val_loss = tf.reduce_mean(tf.square(student_preds - y))
        self.val_loss_tracker.update_state(val_loss)
        return {"loss": self.val_loss_tracker.result()}
