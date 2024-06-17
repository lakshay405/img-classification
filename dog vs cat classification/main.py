# Ensure necessary libraries are installed
!pip install tensorflow opencv-python matplotlib

# Import necessary libraries
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow 
import cv2
import glob

# Setting up GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

# Define the directory for your dataset
data_dir = 'data' 
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# Check and remove unsupported images
for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print(f'Image not in ext list {image_path}')
                os.remove(image_path)
        except Exception as e: 
            print(f'Issue with image {image_path}')
            # os.remove(image_path)  # Uncomment if you want to remove the problematic images

# Count the number of images
path, dirs, files = next(os.walk(data_dir))
file_count = len(files)
print(f'Number of images: {file_count}')

# Display some sample images
file_names = os.listdir(data_dir)
img = mpimg.imread(os.path.join(data_dir, file_names[0]))
imgplt = plt.imshow(img)
plt.show()

img = mpimg.imread(os.path.join(data_dir, file_names[1]))
imgplt = plt.imshow(img)
plt.show()

# Count the number of dog and cat images
dog_count = 0
cat_count = 0
for img_file in file_names:
    if 'dog' in img_file:
        dog_count += 1
    else:
        cat_count += 1

print(f'Number of dog images: {dog_count}')
print(f'Number of cat images: {cat_count}')

# Create directory for resized images
resized_folder = os.path.join(data_dir, 'resized_images')
os.makedirs(resized_folder, exist_ok=True)

# Resize images to a uniform size
for img_file in file_names:
    img_path = os.path.join(data_dir, img_file)
    img = Image.open(img_path)
    img = img.resize((224, 224)).convert('RGB')
    new_img_path = os.path.join(resized_folder, img_file)
    img.save(new_img_path)

# Display resized images
img = mpimg.imread(os.path.join(resized_folder, file_names[0]))
imgplt = plt.imshow(img)
plt.show()

img = mpimg.imread(os.path.join(resized_folder, file_names[1]))
imgplt = plt.imshow(img)
plt.show()

# Assign labels to images
labels = [1 if 'dog' in img_file else 0 for img_file in file_names]
print(labels[:5])

# Count images of dogs and cats
values, counts = np.unique(labels, return_counts=True)
print(values)
print(counts)

# Load and preprocess images
image_extension = ['png', 'jpg']
files = []
[files.extend(glob.glob(os.path.join(resized_folder, '*.' + e))) for e in image_extension]
images = np.asarray([cv2.imread(file) for file in files])
print(images.shape)

# Split the dataset
X = images
Y = np.asarray(labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize the image data
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# Load pre-trained MobileNetV2 model and build the final model
import tensorflow_hub as hub
mobilenet_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
pretrained_model = hub.KerasLayer(mobilenet_url, input_shape=(224, 224, 3), trainable=False)
num_classes = 2

model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.Dense(num_classes)
])

model.summary()

# Compile and train the model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

model.fit(X_train_scaled, Y_train, epochs=5)

# Evaluate the model
score, acc = model.evaluate(X_test_scaled, Y_test)
print(f'Test Loss: {score}')
print(f'Test Accuracy: {acc}')

# Predict on a new image
input_image_path = input('Enter the path of the image to be predicted: ')
input_image = cv2.imread(input_image_path)
cv2_imshow(input_image)

input_image_resized = cv2.resize(input_image, (224, 224)) / 255.0
input_image_reshaped = np.reshape(input_image_resized, [1, 224, 224, 3])

prediction = model.predict(input_image_reshaped)
predicted_label = np.argmax(prediction)

if predicted_label == 0:
    print('The image represents a Cat')
else:
    print('The image represents a Dog')

# Save and load the model
model.save(os.path.join('models', 'image_classifier.h5'))
new_model = tf.keras.models.load_model('models/image_classifier.h5', custom_objects={'KerasLayer': hub.KerasLayer})
new_model.predict(input_image_reshaped)
