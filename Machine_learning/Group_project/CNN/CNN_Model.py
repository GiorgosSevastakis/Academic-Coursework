from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K
from kerastuner import HyperModel

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, scales_shape, fixed_params=None):
        self.input_shape = input_shape
        self.scales_shape = scales_shape
        self.fixed_params = fixed_params or {}

    def build(self, hp=None):
        image_input = Input(shape=self.input_shape, name='image_input')
        scales_input = Input(shape=self.scales_shape, name='scales_input')

        # Use fixed params if provided, otherwise search space
        num_conv_layers = self.fixed_params.get('num_conv_layers') or hp.Int('num_conv_layers', 1, 5)
        x = image_input
        for i in range(num_conv_layers):
            filters = self.fixed_params.get(f'conv_{i+1}_filters') or hp.Int(f'conv_{i+1}_filters', 48, 512, step=16)
            x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = MaxPooling2D(pool_size=2)(x)

        x = Flatten()(x)
        x = Lambda(lambda inputs: K.concatenate([inputs[0], inputs[1]], axis=-1))([x, scales_input])

        num_dense_layers = self.fixed_params.get('num_dense_layers') or hp.Int('num_dense_layers', 3, 10)
        for i in range(num_dense_layers):
            units = self.fixed_params.get(f'dense_{i+1}_units') or hp.Int(f'dense_{i+1}_units', 100, 2000)
            dropout_rate = self.fixed_params.get(f'dropout_{i+1}_rate') or hp.Float(f'dropout_{i+1}_rate', 0.2, 0.5, step=0.1)
            x = Dense(units, activation='relu')(x)
            x = Dropout(dropout_rate)(x)

        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[image_input, scales_input], outputs=output)
        learning_rate = self.fixed_params.get('learning_rate') or hp.Float('learning_rate', 1e-5, 1e-3, sampling='LOG')

        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        return model