from google.colab import files
uploaded = files.upload()
import zipfile
import os

zip_path = '/content/archive (3).zip'
extract_path = '/content/ecg_dataset'

# Unzip the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Check what's inside
print("Files/Folders in dataset directory:", os.listdir(extract_path))
for root, dirs, files in os.walk(extract_path):
    print("Root:", root)
    print("Dirs:", dirs)
    print("Files:", files[:5])  # Show first 5 files if there are many
    print("------")
    import pandas as pd

# Load the CSVs
mitbih_train = pd.read_csv('/content/ecg_dataset/mitbih_train.csv', header=None)
mitbih_test = pd.read_csv('/content/ecg_dataset/mitbih_test.csv', header=None)

print("Train shape:", mitbih_train.shape)
print("Test shape:", mitbih_test.shape)

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Split features and labels
X_train = mitbih_train.iloc[:, :-1].values
y_train = to_categorical(mitbih_train.iloc[:, -1].values)

X_test = mitbih_test.iloc[:, :-1].values
y_test = to_categorical(mitbih_test.iloc[:, -1].values)

# Normalize inputs
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

# Reshape for Conv1D input: (samples, time steps, 1)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

import os

dataset_path = '/content/ecg_dataset'

for root, dirs, files in os.walk(dataset_path):
    print(f"📁 Folder: {root}")
    print(f"📂 Subfolders: {dirs}")
    print(f"📄 Files: {files}")
    print("-----")

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
df = pd.read_csv('/content/ecg_dataset/mitbih_train.csv', header=None)

# Create folders to save images
os.makedirs("/content/ecg_images", exist_ok=True)
for label in df[187].unique():
    os.makedirs(f"/content/ecg_images/{int(label)}", exist_ok=True)

# Save first 100 samples as images
for i in range(100):  # you can increase this
    signal = df.iloc[i, :-1].values
    label = int(df.iloc[i, -1])

    plt.figure(figsize=(4, 2))
    plt.plot(signal)
    plt.axis('off')
    plt.tight_layout()

    # Save image
    plt.savefig(f"/content/ecg_images/{label}/img_{i}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

print("✅ 100 ECG signals converted to images.")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_path = "/content/ecg_images"
img_size = (224, 224)

datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(
    img_path,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)
import matplotlib.pyplot as plt

# Get a batch of images and labels
batch_images, batch_labels = next(train_gen)

# Show two images side by side
plt.figure(figsize=(10, 4))

for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(batch_images[i])
    plt.title(f"Label: {batch_labels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
