datapath = '/content/drive/MyDrive/Thesis3'
train_dir = '/content/drive/MyDrive/Thesis3/train'
val_dir = '/content/drive/MyDrive/Thesis3/validation'
test_dir = '/content/drive/MyDrive/Thesis3/test'
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
val_data = val_datagen.flow_from_directory(val_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=1, class_mode='categorical', shuffle=False)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, epochs=5, validation_data=val_data)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training and Validation Accuracy/Loss")
plt.show()
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

pred = model.predict(test_data)
y_pred = np.argmax(pred, axis=1)
y_true = test_data.classes

class_labels = list(train_data.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
from tensorflow.keras.preprocessing import image

img_path = '/content/drive/MyDrive/Thesis3/test/Myocardial Infarction/mi79 (1).jpg'

img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]
class_labels = list(train_data.class_indices.keys())

print("Predicted Class:", class_labels[predicted_class])
from tensorflow.keras.layers import Add, Input
from tensorflow.keras.models import Model

def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x
from tensorflow.keras.layers import Activation, GlobalAveragePooling2D,Dense,Dropout,Conv2D,MaxPooling2D,Flatten

input_layer = Input(shape=(64, 64, 3))
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2,2))(x)

# Residual Block 1
x = residual_block(x, 32)
x = MaxPooling2D((2,2))(x)

# Residual Block 2
x = residual_block(x, 32)
x = MaxPooling2D((2,2))(x)

x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(2, activation='softmax')(x)

resnet_model = Model(inputs=input_layer, outputs=output_layer)
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

resnet_history = resnet_model.fit(train_data, epochs=5, validation_data=val_data)
import matplotlib.pyplot as plt
plt.plot(history.history['val_accuracy'], label='Basic CNN')
plt.plot(resnet_history.history['val_accuracy'], label='ResNet-like CNN')
plt.title('Model Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
resnet_pred = resnet_model.predict(test_data)
y_pred_resnet = np.argmax(resnet_pred, axis=1)

print("ResNet-like Model Evaluation:")
print(classification_report(y_true, y_pred_resnet, target_names=class_labels))

cm_resnet = confusion_matrix(y_true, y_pred_resnet)
disp_resnet = ConfusionMatrixDisplay(confusion_matrix=cm_resnet, display_labels=class_labels)
disp_resnet.plot(cmap=plt.cm.Oranges)
plt.title("ResNet-like Confusion Matrix")
plt.show()
# For basic CNN
model.save('basic_cnn_model.h5')

# For ResNet-like model
resnet_model.save('resnet_cnn_model.h5')
from tensorflow.keras.models import load_model

# Load Basic CNN
basic_model = load_model('basic_cnn_model.h5')

# Load ResNet-like CNN
resnet_model = load_model('resnet_cnn_model.h5')
img_path = '/content/drive/MyDrive/Thesis3/test/Myocardial Infarction/mi799.jpg'

# Load image with target size same as training images
img = image.load_img(img_path, target_size=(64, 64))

# Convert image to array
img_array = image.img_to_array(img)

# Normalize pixel values (if you normalized training data)
img_array = img_array / 255.0

# Expand dimensions to match model input shape
img_array = np.expand_dims(img_array, axis=0)


# Predict
prediction = resnet_model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

# Print result
print("Predicted Class:", class_labels[predicted_class])
resnet_model.save('resnet_model.keras')
# And load with:
# resnet_model = load_model('resnet_model.keras')
# Save the Basic CNN model
basic_model.save('basic_model.h5')

# Save the ResNet-like model
resnet_model.save('resnet_model.h5')
import zipfile

# Create a zip file and add both model files
with zipfile.ZipFile('ecg_models.zip', 'w') as zipf:
    zipf.write('basic_model.h5')
    zipf.write('resnet_model.h5')

from google.colab import files

# Download the zip file
files.download('ecg_models.zip')
val_data = val_datagen.flow_from_directory(
    '/content/drive/MyDrive/Thesis3/validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
loss1, acc1 = basic_model.evaluate(val_data)
print(f"Basic CNN Accuracy: {acc1 * 100:.2f}%")
loss2, acc2 = resnet_model.evaluate(val_data)
print(f"ResNet-like Model Accuracy: {acc2 * 100:.2f}%")
basic_model = load_model('basic_cnn_model.h5')
# Compile the basic model
basic_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Load ResNet-like CNN
resnet_model = load_model('resnet_cnn_model.h5')
# Compile the ResNet-like model
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Add, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Define improved ResNet block
def resnet_block(x, filters):
    shortcut = x

    # Match the shape of shortcut to the main path if needed
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

# Full ResNet-like model
def build_improved_resnet(input_shape=(64, 64, 3), num_classes=2):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = resnet_block(x, 128)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Load data
train_dir = '/content/drive/MyDrive/Thesis3/train'
val_dir = '/content/drive/MyDrive/Thesis3/validation'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
resnet_model = build_improved_resnet()
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, min_lr=1e-6, verbose=1)

history = resnet_model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[lr_reduction]
)

# Evaluate model
val_loss, val_accuracy = resnet_model.evaluate(val_data)
print(f"✅ Improved ResNet Accuracy: {val_accuracy * 100:.2f}%")
   # Save the Basic CNN model
basic_model.save('basic_model.h5')

# Save the ResNet-like model
resnet_model.save('resnet_model.h5')
import zipfile

# Create a zip file and add both model files
with zipfile.ZipFile('ecg_models.zip', 'w') as zipf:
    zipf.write('basic_model.h5')
    zipf.write('resnet_model.h5')
from google.colab import files

# Download the zip file
files.download('ecg_models.zip')
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('/content/basic_model.h5') #('/content/resnet_model.h5')
img_path = '/content/drive/MyDrive/Thesis3/test/Myocardial Infarction/mi800.jpg'

img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])

class_labels = list(train_data.class_indices.keys())
predicted_class = class_labels[predicted_class_index]

print(f"Predicted Class: {predicted_class}")
print(f"Raw Prediction: {predictions}")

plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('/content/basic_model.h5') #('/content/resnet_model.h5')
img_path = '/content/drive/MyDrive/Thesis3/test/Normal/mi1.jpg'

img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])

class_labels = list(train_data.class_indices.keys())
predicted_class = class_labels[predicted_class_index]

print(f"Predicted Class: {predicted_class}")
print(f"Raw Prediction: {predictions}")

plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()
