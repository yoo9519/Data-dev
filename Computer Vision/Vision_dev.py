# -*- coding: utf-8 -*-

"""
Author: Cline

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from glob import glob
import seaborn as sns

AUTO = tf.data.experimental.AUTOTUNE
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

# GCS_DS_PATH = KaggleDatasets().get_gcs_path()

IMG_SIZE = 784
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
nb_classes = 2

df = pd.read_csv('/content/drive/MyDrive/../df2_merge_toy.csv')
glob_image = glob('/content/drive/MyDrive/../raw_images_merge/*.jpg')
print(len(glob_image))
df

data_image_paths = {os.path.basename(x): x for x in glob_image}
df['path'] = df['filename'].map(data_image_paths.get)
df['ahi_osa'] = df['ahi_osa'].map(lambda x: x.replace('no', 'normal'))

normal_images = glob_image
normal_data = {'path': normal_images, 'ahi_osa': 'Normal'}
df1 = pd.DataFrame(normal_data)
df = pd.concat([df,df1], ignore_index=True, axis=1)
df = df[[6, 8, 10, 11, 12]]
df = df.dropna(axis=0)
df.to_csv('last_check.csv')
df[8] = df[8].replace('normal', 'normal|mild')
df[8] = df[8].replace('mild', 'normal|mild')
df[8] = df[8].replace('moderate', 'moderate|severe')
df[8] = df[8].replace('severe', 'moderate|severe')
df

train_df = df[df[11] == 'train']
valid_df = df[df[11] == 'valid']
test_df = df[df[11] == 'test']
train_df.sample(5)

label_counts = df[8].value_counts()
print(label_counts)
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 0)

all_labels = ['normal|mild', 'moderate|severe']
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label) > 1: # leave out empty labels
        df[c_label] = df[8].map(lambda finding: 1 if c_label in finding else 0)

df['osa_vec'] = df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
df['osa_vec']

df.shape

path='/content/drive/MyDrive/../raw_images_merge'

train = df[6]
# train_id = df[8]

y_train = df[8]
category_names = ['normal|mild', 'moderate|severe']

# root = 'images'
images_paths = df[12]

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(df, y_train, test_size=0.2)

labels = ['normal|mild', 'moderate|severe']

def compute_class_freqs(labels):
    
    N = len(labels)
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies

train_labels = []
ds_len = df.shape[0]

for inx in range(ds_len):
    row = df.iloc[inx]
    vec = np.array(row['osa_vec'], dtype=np.int)
    train_labels.append(vec)

freq_pos, freq_neg = compute_class_freqs(train_labels)
print(freq_pos), print(freq_neg)

data = pd.DataFrame({"Class": labels, "Label": "Negative", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Positive", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
print(data)
plt.xticks(rotation=0)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)

def weighted_loss(y_true, y_pred, pos_weights, neg_weights, epsilon=1e-7):

        # initialize loss to zero
        loss = 0.0

        for i in range(len(pos_weights)):
            # 각각 class 평균 loss weights를 준다.
            loss += -(torch.mean( pos_weights[i] * y_true[:,i] * torch.log(y_pred[:,i] + epsilon) + \
                                neg_weights[i] * (1 - y_true[:,i]) * torch.log(1 - y_pred[:,i] + epsilon), axis = 0))
            
        return loss

pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
print(data)
plt.xticks(rotation=0)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data)

# from sklearn.utils.class_weight import compute_class_weight
# # class_weights = compute_class_weight('balanced', np.unique(y_train.argmax(axis=1)), y_train.argmax(axis=0))
# class_weights = compute_class_weight('balanced', np.unique(y_train.argmax), y_train.argmax)
# print('class weights: ',class_weights)

# plt.bar(range(2),1/class_weights,color=['normal|mild', 'moderate|severe'],width=0.9)
# plt.xticks(range(2), category_names) 

# plt.title("Loss weights");
# plt.ylabel('Probability')
# plt.xlabel('Data')
# plt.show()

# #class weights to dict
# c_w = dict(zip(range(2),class_weights))

def decode_image(filename, label=None, image_size=(IMG_SIZE, IMG_SIZE)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    #convert to numpy and do some cv2 staff mb?
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None, seed=5050):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
           
    if label is None:
        return image
    else:
        return image, label

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )

val_dataset = (tf.data.Dataset
               .from_tensor_slices((x_val,y_val))
               .map(decode_image,num_parallel_calls=AUTO)
               .batch(BATCH_SIZE)
               .cache()
               .prefetch(AUTO)
              )

!pip install efficientnet
import efficientnet.tfkeras as efn
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

def get_model():
    base_model = efn.EfficientNetB7(weights='imagenet',
                          include_top=False,
                          input_shape=(IMG_SIZE,IMG_SIZE, 3),
                          pooling='avg')
    x = base_model.output
    predictions = Dense(nb_classes, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=predictions)

from tensorflow.keras.optimizers import Adam

with strategy.scope():
    model = get_model()

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

model_name = 'xrayosa.h5'

#good callbacks
best_model = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True,mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,min_lr=0.000001,patience=6)

train_df1 = []
for tr in train_df[6]:
    train_df1.append(tr)

valid_df1 = []
for vl in valid_df[6]:
    valid_df1.append(vl)

valid_df1[:5]

print(len(train_df1))

print(y_train.shape)
print(BATCH_SIZE)
len(train_df1)

3664 / 64

# history = model.fit(train_df,
#                     steps_per_epoch=y_train.shape[0]//BATCH_SIZE,
#                     epochs=5,
#                     verbose=1,
#                     validation_data=valid_df,
#                     callbacks=[reduce_lr,best_model]
#                     )

history = model.fit(train_df1,
                    steps_per_epoch=y_train.shape[0]//BATCH_SIZE,
                    epochs=5,
                    verbose=1,
                    validation_data=valid_df1
                    )

plt.title('model accuracy')
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

path='../input/plant-pathology-2020-fgvc7/'

test = pd.read_csv(path+'test.csv')
test_id = test['image_id']

root = 'images'
x_test = [(os.path.join(GCS_DS_PATH,root,idee+'.jpg')) for idee in test_id]

model.load_weights(model_name)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)

y_pred = model.predict(test_dataset,verbose=1)

def save_results(y_pred):
    
    path='../input/plant-pathology-2020-fgvc7/'
    test = pd.read_csv(path + 'test.csv')
    test_id = test['image_id']

    res = pd.read_csv(path+'train.csv')
    res['image_id'] = test_id
  
    labels = res.keys()

    for i in range(1,5):
        res[labels[i]] = y_pred[:,i-1]

    res.to_csv('submission.csv',index=False)
  
    print(res.head)

save_results(y_pred)
