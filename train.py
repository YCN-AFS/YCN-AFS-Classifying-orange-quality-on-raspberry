import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def load_and_preprocess_data(data_dir, img_height=224, img_width=224):
    data = []
    labels = []
    class_names = ['bad', 'good', 'mediocre']

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        class_label = class_names.index(class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).resize((img_width, img_height))
            img_array = np.array(img) / 255.0  # Normalize pixel values
            data.append(img_array)
            labels.append(class_label)

    return np.array(data), np.array(labels)


# Load data
data_dir = 'data'
X, y = load_and_preprocess_data(data_dir)


#-----------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
#danh gia
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
#save
model.save('orange_quality_model.h5')