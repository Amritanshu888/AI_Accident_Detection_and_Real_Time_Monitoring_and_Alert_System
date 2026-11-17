#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape
from keras.models import Model,load_model
from keras.losses import binary_crossentropy
from imutils import paths
from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling1D
from keras.models import Sequential
from google.colab.patches import cv2_imshow
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import random
from PIL import Image


# In[3]:


# Read the kaggle API token to interact with your kaggle account
from google.colab import files
files.upload()
# Series of commands to set-up for download
get_ipython().system('ls -lha kaggle.json')
get_ipython().system('pip install -q kaggle # installing the kaggle package')
get_ipython().system('mkdir -p ~/.kaggle # creating .kaggle folder where the key should be placed')
get_ipython().system('cp kaggle.json ~/.kaggle/ # move the key to the folder')
get_ipython().system('pwd # checking the present working directory')
# giving rw access (if 401-nathorized)
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[4]:


# Download data command
get_ipython().system('kaggle datasets download -d ckay16/accident-detection-from-cctv-footage')

# # Unzip
get_ipython().system('unzip /content/accident-detection-from-cctv-footage.zip -d .')


# In[5]:


# Get all image paths from the folder
imagePaths = list(paths.list_images("/content/data/train"))
# Function to display a grid of images
def display_images(image_paths, title, cols=5, num_images=25):  # Set cols and add num_images
    random.shuffle(image_paths)  # Shuffle the image paths randomly
    num_images = min(len(image_paths), num_images)  # Limit the number of images to display
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(12, 12))  # Adjust figure size to accommodate more rows
    plt.suptitle(title)

    for i, image_path in enumerate(image_paths[:num_images]):
        plt.subplot(rows, cols, i + 1)
        img = Image.open(image_path)
        # Convert the image to RGB format
        img = img.convert('RGB')
        plt.imshow(img)
        plt.title(image_path.split(os.path.sep)[4])
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Display a random set of images in a 5x5 grid
display_images(imagePaths, title="Random Sample of Images from Different Classes", num_images=25)


# In[6]:


class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []

		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			if self.preprocessors is not None:
				# loop over the preprocessors and apply each to
				# the image
				for p in self.preprocessors:
					image = p.preprocess(image)

			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)

			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1,
					len(imagePaths)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))


# In[7]:


class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)


# In[8]:


class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		# store the image data format
		self.dataFormat = dataFormat

	def preprocess(self, image):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the image
		return img_to_array(image, data_format=self.dataFormat)


# In[9]:


# initialize the image preprocessors
sp = SimplePreprocessor(128,128)
iap = ImageToArrayPreprocessor()


# In[10]:


# load train image paths
train_paths = list(paths.list_images("/content/data/train"))
val_paths = list(paths.list_images("/content/data/val"))
train_imagepaths = train_paths+val_paths
classNames = [pt.split(os.path.sep)[-2] for pt in train_imagepaths]
classNames = [str(x) for x in np.unique(classNames)]
classNames


# # Visualize train image

# In[11]:


img_read = cv2.imread(train_imagepaths[0])
img_resize = cv2.resize(img_read,(200,200))
cv2_imshow(img_resize)


# In[12]:


img_read = cv2.imread(train_imagepaths[25])
img_resize = cv2.resize(img_read,(200,200))
cv2_imshow(img_resize)


# In[13]:


# load the trian dataset from disk then scale the raw pixel intensities to
# the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(train_data, train_label) = sdl.load(train_imagepaths, verbose=500)
train_data = train_data.astype('float32') / 255.


# In[14]:


# load validaiton image paths
validation_imagePaths = list(paths.list_images("/content/data/test"))
classNames = [pt.split(os.path.sep)[-2] for pt in validation_imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
classNames


# # visualize validation image

# In[15]:


img_read = cv2.imread(validation_imagePaths[50])
img_resize = cv2.resize(img_read,(200,200))
cv2_imshow(img_resize)


# In[16]:


# load the validation dataset from disk then scale the raw pixel intensities to
# the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(validation_data,validation_label) = sdl.load(validation_imagePaths, verbose=10)
validation_data = validation_data.astype('float32') / 255.


# In[17]:


# Define the image path
image_path = '/content/data/train/Accident/acc1 (2).jpg'

# Load the image using OpenCV
image = cv2.imread(image_path)
print("original image shape",image.shape)
# Initialize the SimplePreprocessor with desired width and height
preprocessor = SimplePreprocessor(128, 128)

# Preprocess the image (resize it)
preprocessed_image = preprocessor.preprocess(image)
print("preprocessed image shape",preprocessed_image.shape)
# Create a subplot to display images side by side
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Preprocessed Image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
plt.title('Preprocessed Image')

plt.show()


# In[18]:


# Apply Label encoder for train labels
le = LabelEncoder()
le.fit(train_label)
train_labels = le.transform(train_label)
train_labels = to_categorical(train_labels)


# In[19]:


# Apply Label encoder for validation  labels
le.fit(validation_label)
validation_labels = le.transform(validation_label)
validation_labels = to_categorical(validation_labels)


# In[20]:


train_count = train_labels.shape[0]
validation_count = validation_labels.shape[0]
counts = [train_count, validation_count]
fig = plt.figure(figsize=(10, 5))
label_name = ['train', 'validation']
plt.bar(label_name, counts, color=["blue", "red"], width=0.4)
plt.xlabel("Label Name")
plt.ylabel("Count")
plt.title("Number of Training and Validation Samples")
plt.show()


# In[21]:


import pandas as pd
train_count = train_labels.shape[0]
validation_count = validation_labels.shape[0]
label_counts = {'train': train_count, 'validation': validation_count}
df_counts = pd.DataFrame(label_counts.items(), columns=['Label', 'Count'])
plt.figure(figsize=(8, 8))
plt.pie(df_counts['Count'], labels=df_counts['Label'], autopct='%1.1f%%')
plt.title('Class Distribution')
plt.show()


# # Training using Convolution neural network(CNN Model)

# In[22]:


from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
input_shape = (128,128,1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128,kernel_size=(3, 3), activation='relu',padding = 'Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()


# In[23]:


if not os.path.exists("/content/drive/MyDrive/accident_detection"):
    os.mkdir("/content/drive/MyDrive/accident_detection")


# In[24]:


# Define the checkpoint callback
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
checkpoint = ModelCheckpoint(
    filepath='/content/drive/MyDrive/accident_detection/car_accident_model.weights.h5',  # Filepath to save the best weights
    monitor='val_accuracy',                     # Metric to monitor
    save_best_only=True,                    # Save only the best weights
    save_weights_only=True,                 # Save only weights, not the full model
    mode='max',                             # 'min' for validation loss
    verbose=1                               # Print messages when saving
)

# Define your optimizer using legacy Adam
optimizer = Adam(learning_rate=0.0001)

# Compile your model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming you have your train_data, train_labels, validation_data, and validation_labels defined
print("[INFO] training model...")
H = model.fit(train_data, train_labels, batch_size=16,
              validation_data=(validation_data, validation_labels), epochs=50, callbacks=[checkpoint])


# # Validation

# In[25]:


# model load
model.load_weights("/content/drive/MyDrive/accident_detection/car_accident_model.weights.h5")
prediction = model.predict(validation_data)


# # classification report

# In[26]:


classNames = ['Accident', 'Not Accident']
print(classification_report(validation_labels.argmax(axis=1),
	prediction.argmax(axis=1), target_names=classNames))


# Confusion matrix

# In[27]:


import seaborn as sns
CM = confusion_matrix(validation_labels.argmax(axis=1),
	prediction.argmax(axis=1))
# drawing confusion matrix
sns.heatmap(CM, center = True , annot=True, fmt="d" ,cmap="Blues", xticklabels=classNames, yticklabels=classNames)
plt.show()


# Prediction

# In[28]:


image_path = "/content/data/train/Non Accident/5_17.jpg"
image = cv2.imread(image_path)
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
sp = SimplePreprocessor(128,128)
data =  sp.preprocess(gray_img)
data = data.astype("float") / 255.0
data = np.expand_dims(data, axis=0)
data = np.expand_dims(data, axis=-1)
model.load_weights('/content/drive/MyDrive/accident_detection/car_accident_model.weights.h5')
preds = model.predict(data)
classNames = ['Accident', 'Not Accident']
print("preicted class is:", classNames[preds.argmax()])
image= cv2.resize(image, (200, 200))
cv2_imshow(image)

