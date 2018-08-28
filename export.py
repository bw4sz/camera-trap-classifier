""" Export a Trained model as Tensorflow Estimator for Deployment
    with Tensorflow-Serving

    WARNING: Requires Tensorflow 1.9 or higher

    Example Usage:
    --------------

    python3 export.py -model /my_experiment/model_save_dir/prediction_model.hdf5 \
    -class_mapping_json /my_experiment/model_save_dir/label_mappings.json \
    -pre_processing_json /my_experiment/model_save_dir/pre_processing.json \
    -output_dir /my_experiment/my_model_exports/ \
    -estimator_save_dir /my_experiment/my_estimators/

"""
import argparse
import logging

import tensorflow as tf
from tensorflow.keras.estimator import model_to_estimator

from training.prepare_model import load_model_from_disk
from data.image import preprocess_image
from data.utils import read_json
from config.config_logging import setup_logging


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True,
                        help="path to a prediction model (.hdf5 file)")
    parser.add_argument(
        "-class_mapping_json", type=str, required=True,
        help="path to label_mappings.json")
    parser.add_argument(
        "-pre_processing_json", type=str, required=True,
        help="path to the image_processing.json")
    parser.add_argument(
        "-output_dir", type=str, required=True,
        help="Root directory to which model is exported")
    parser.add_argument(
        "-log_outdir", type=str, required=False, default=None,
        help="The directory to write logfiles to (defaults to output_dir)")
    parser.add_argument(
        "-estimator_save_dir", type=str, required=False,
        help="Directory to which estimator is saved (if not specified)\
              a temporary location is chosen")

    # Parse command line arguments
    args = vars(parser.parse_args())

    # Configure Logging
    if args['log_outdir'] is None:
        args['log_outdir'] = args['output_dir']

    setup_logging(log_output_path=args['log_outdir'])

    logger = logging.getLogger(__name__)

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s: %s" % (k, v))

    args = vars(parser.parse_args())

    # Load Model and extract input/output layers
    keras_model = load_model_from_disk(args['model'])

    input_names = keras_model.input_names
    output_names = keras_model.output_names

    label_mapping = read_json(args['class_mapping_json'])
    pre_processing = read_json(args['pre_processing_json'])
    estimator = model_to_estimator(
        keras_model,
        model_dir=args['estimator_save_dir'])

    def decode_and_process_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = preprocess_image(image, **pre_processing)
        return image

    def generate_dataset_iterator(image_list):
        """ Dataset Iterator from a list of Image Bytes """
        dataset = tf.data.Dataset.from_tensor_slices(image_list)
        dataset = dataset.map(decode_and_process_image)
        dataset = dataset.batch(128)
        next_example = tf.contrib.data.get_single_element(dataset)
        return next_example

    def serving_input_receiver_fn():
        """
        This is used to define inputs to serve the model.

        :return: ServingInputReciever
        """
        # Input Tensor (list of image bytes)
        list_of_image_bytes = tf.placeholder(shape=[1], dtype=tf.string)
        receiver_tensors = {
            'image': list_of_image_bytes
        }

        # Generate an iterator for the images
        image_batch = generate_dataset_iterator(list_of_image_bytes)
        features = {
            input_names[0]: image_batch
        }
        return tf.estimator.export.ServingInputReceiver(
            receiver_tensors=receiver_tensors,
            features=features)

    # Save the model
    estimator.export_savedmodel(
        args['output_dir'],
        serving_input_receiver_fn=serving_input_receiver_fn,
        assets_extra={'label_mappings.json': args['class_mapping_json']})
