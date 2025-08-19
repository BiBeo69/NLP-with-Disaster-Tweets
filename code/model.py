import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Bidirectional, GRU, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1],1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1],1),
                                 initializer='zeros',
                                 trainable=True)
        super().build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

def build_advanced_model(transformer, max_len=160):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    sequence_output = transformer(input_ids)[0]
    gru_output = Bidirectional(GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(sequence_output)
    attention_output = AttentionLayer()(gru_output)
    normalized = LayerNormalization()(attention_output)
    x = Dense(128, activation='relu')(normalized)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_ids, outputs=out)
    optimizer = Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model
