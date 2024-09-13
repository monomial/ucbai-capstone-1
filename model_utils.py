# model_utils.py
from tensorflow import keras

def build_mlp_model(hp, input_shape, num_classes):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(shape=(input_shape,)))
    
    # Tune the number of layers
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=1024, step=32),
            activation='relu'
        ))
        model.add(keras.layers.Dropout(hp.Float(f'dropout_{i}', 0, 0.5, step=0.1)))
    
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model