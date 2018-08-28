import os

import tensorflow as tf
import numpy as np

from data.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from data.reader import DatasetReader
from data.image import (
        preprocess_image)
from data.utils import (
        calc_n_batches_per_epoch, read_json,
        n_records_in_tfr, find_tfr_files_pattern)
from config.config import ConfigLoader
import matplotlib.pyplot as plt
from training.prepare_model import create_model

tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()

data_reader = DatasetReader(tfr_encoder_decoder.decode_record)


tfr_train = find_tfr_files_pattern('./test_big/cats_vs_dogs/tfr_files/',
                            'train')

output_labels = ['class']
output_labels_clean = ['label/class']

class_mapp = './test_big/cats_vs_dogs/tfr_files/label_mapping.json'

class_mapping = read_json(class_mapp)
n_classes_per_label_dict = {c: len(class_mapping[o]) for o, c in
                            zip(output_labels, output_labels_clean)}
n_classes_per_label = [n_classes_per_label_dict[x]
                       for x in output_labels_clean]
index_to_class = {k: {vv: kk for kk, vv in v.items()} for k, v in class_mapping.items()}


# Load model config
model_cfg = ConfigLoader('./config/models.yaml')

image_processing = model_cfg.cfg['models']['small_cnn']['image_processing']

# Calculate Dataset Image Means and Stdevs for a dummy batch
dataset = data_reader.get_iterator(
        tfr_files=tfr_train,
        batch_size=10,
        is_train=False,
        n_repeats=1,
        #output_labels=['label_num/class'],
        output_labels=output_labels,
        label_to_numeric_mapping=class_mapping,
        #label_to_numeric_mapping={'class': {'cat': 0, 'dog': 2}},
        image_pre_processing_fun=preprocess_image,
        image_pre_processing_args={**image_processing,
                                   'is_training': False},
        buffer_size=32,
        num_parallel_calls=2,
        only_return_one_label=False,
        return_only_ml_data=False)
iterator = dataset.make_one_shot_iterator()
batch_data = iterator.get_next()


iterator = dataset.make_initializable_iterator()
#iterator = dataset.make_one_shot_iterator()
batch_data = iterator.get_next()


with tf.Session() as sess:
    tf.tables_initializer().run()
    sess.run(iterator.initializer)
    for i in range(0,1):
        features, labels = sess.run(batch_data)
        for j in range(0, labels['label/class'].shape[0]):
            print("Class: %s" % index_to_class['class'][labels['label/class'][j][0]])
            plt.imshow(features['images'][j,:,:,:])
            plt.show()



with tf.Session() as sess:
    for i in range(0,1):
        data = sess.run(batch_data)
        for j in range(2000, data['label/0/class'].shape[0]):
            print("Class: %s" % data['label/0/class'][j])
            plt.imshow(data['images'][j,:,:,:])
            plt.show()



# Calculate Dataset Image Means and Stdevs for a dummy batch
image_processing['image_means'] = [0, 0, 0]
image_processing['image_stdevs'] = [1, 1, 1]
batch_data = data_reader.get_iterator(
        tfr_files=tfr_file,
        batch_size=4096,
        is_train=False,
        n_repeats=1,
        output_labels=labels,
        image_pre_processing_fun=preprocess_image,
        image_pre_processing_args={**image_processing,
                                   'is_training': False},
        max_multi_label_number=None,
        buffer_size=32,
        num_parallel_calls=2)


with tf.Session() as sess:
    data = sess.run(batch_data)


# calculate and save image means and stdvs of each color channel
# for pre processing purposes
image_means = [round(float(x), 4) for x in
               list(np.mean(data['images'], axis=(0, 1, 2), dtype=np.float64))]
image_stdevs = [round(float(x), 4) for x in
                list(np.std(data['images'], axis=(0, 1, 2), dtype=np.float64))]

image_processing['image_means'] = image_means
image_processing['image_stdevs'] = image_stdevs



def input_feeder_train():
        return data_reader.get_iterator(
                    tfr_files=tfr_file,
                    batch_size=64,
                    is_train=True,
                    n_repeats=None,
                    output_labels=labels,
                    image_pre_processing_fun=preprocess_image,
                    image_pre_processing_args={
                        **image_processing,
                        'is_training': True},
                    max_multi_label_number=None,
                    numeric_labels=True
                    buffer_size=2,
                    num_parallel_calls=2)



batch_data = input_feeder_train()
with tf.Session() as sess:
    for i in range(0,3):
        data = sess.run(batch_data)
        image_means = [round(float(x), 4) for x in
               list(np.mean(data['images'], axis=(0, 1, 2), dtype=np.float64))]
        image_stdevs = [round(float(x), 4) for x in
                list(np.std(data['images'], axis=(0, 1, 2), dtype=np.float64))]
        print("means: %s" % image_means)
        print("stdevs: %s" % image_stdevs)
        for j in range(0, data['label/0/class'].shape[0]):
            print("Class: %s" % data['label/0/class'][j])
            plt.imshow(data['images'][j,:,:,:])
            plt.show()


image_means = [round(float(x), 4) for x in
               list(np.mean(data['images'], axis=(0, 1, 2), dtype=np.float64))]
image_stdevs = [round(float(x), 4) for x in
                list(np.std(data['images'], axis=(0, 1, 2), dtype=np.float64))]



n_records_train = n_records_in_tfr(tfr_file)
n_batches_per_epoch_train = calc_n_batches_per_epoch(
    n_records_train, 64)



train_model, train_model_base = create_model(
    model_name='cats_vs_dogs',
    input_feeder=input_feeder_train,
    target_labels=labels,
    n_classes_per_label_type=n_classes_per_label,
    n_gpus=1,
    continue_training=False,
    transfer_learning=False,
    path_of_model_to_load=None)

val_model, val_model_base = create_model(
    model_name='cats_vs_dogs',
    input_feeder=input_feeder_val,
    target_labels=labels,
    n_classes_per_label_type=n_classes_per_label,
    n_gpus=1)


train_model.summary()
train_model.loss
train_model.loss_functions

train_model.fit(epochs=1,
                steps_per_epoch=n_batches_per_epoch_train,
                initial_epoch=0)


test_data = input_feeder_train()
tt = train_model.test_on_batch(x=test_data['images'], y=test_data['label/0/class'])


x = test_data['images']
y = test_data['label/0/class']

tt = train_model.predict_on_batch(x)

with tf.Session() as sess:
    ys = sess.run(y)

np.sum(ys == np.argmax(tt, axis=1)) / ys.shape



test_data

ys = test_data['label/0/class']


# Example for inspecting TFR files
import matplotlib.pyplot as plt

dataset = get_my_dataset(tfr_files, batch_size)
iterator = dataset.make_one_shot_iterator()
batch_data = iterator.get_next()

index_to_class = {0: 'wildebeest', 1: 'zebra'}

with tf.Session() as sess:
    for i in range(0, 1):
        features, labels = sess.run(batch_data)
        for j in range(0, labels['label/class'].shape[0]):
            print("Class: %s" % index_to_class['class'][labels['label/class'][j][0]])
            plt.imshow(features['image'][j, :, :, :])
            plt.show()
