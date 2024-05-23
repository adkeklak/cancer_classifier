import joblib
import tensorflow as tf
from cancer_classificator import ml
from cancer_classificator.ml import model,utils
from cancer_classificator.ml.utils import decode_image
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

AUTO = tf.data.experimental.AUTOTUNE

class MlModel:
    def __init__(self):
        self.model = None
        pass

    def train(self, X, y, epochs, verbose):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        #joblib.dump(self.model, path)
        self.model.save(path)

    def load_model(self, path):
        #self.model = joblib.load(path)
        self.model = load_model(path)

class LungsModel(MlModel):
    def __init__(self):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
        
    def train(self, X, y, epochs=10, verbose=1):
        y = to_categorical(y, num_classes=3)
        X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.15, random_state=42)

        train_ds = (
            tf.data.Dataset
            .from_tensor_slices((X_train, Y_train))
            .map(lambda x, y: decode_image(x, y, augment=True), num_parallel_calls=AUTO)
            .cache()
            .batch(16)
            .prefetch(AUTO)
        )

        val_ds = (
            tf.data.Dataset
            .from_tensor_slices((X_val, Y_val))
            .map(lambda x, y: decode_image(x, y, augment=False), num_parallel_calls=AUTO)
            .cache()
            .batch(16)
            .prefetch(AUTO)
        )

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=verbose)