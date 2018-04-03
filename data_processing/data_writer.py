""" Write Data Inventory to Disk """
import random
import os
import math

import tensorflow as tf

from config.config import logging
from pre_processing.image_transformations import read_jpeg
from data_processing.data_inventory import DatasetInventory
from data_processing.tfr_file import TFRFile


class DatasetWriter(object):
    def __init__(self, tfr_encoder):
        self.tfr_encoder = tfr_encoder

    def _save_tfr_data_to_disk(self):
        """ Save additional dataset inventory to disk """

    def encode_inventory_to_tfr(
         self, data_inventory, output_file,
         image_pre_processing_fun=None,
         image_pre_processing_args=None,
         random_shuffle_before_save=True,
         overwrite_existing_file=True,
         prefix_to_labels='',
         generate_new_file_every_n=None):
        """ Export Data Inventory to a TFRecord file """

        if os.path.exists(output_file) and not overwrite_existing_file:
            logging.info("File: %s exists - not gonna overwrite" % output_file)
            return None

        logging.info("Starting to Encode Inventory to Dictionary")

        if not isinstance(data_inventory, DatasetInventory):
            raise ValueError("data_inventory must be a DatasetInventory")

        all_label_types = data_inventory.label_handler.get_all_label_types()
        logging.info("Found following label types: %s" % all_label_types)

        n_records = data_inventory.get_number_of_records()
        logging.info("Found %s records in inventory" % n_records)

        # generate file names
        if generate_new_file_every_n is not None:
            n_files_to_generate = math.ceil(n_records / generate_new_file_every_n)
            prefix_filename = output_file.split('.')[0]
            postfix_filename = output_file.split('.')[1]
            output_files = ['%s_%03d.%s' %
                            (prefix_filename, i, postfix_filename)
                            for i in range(0, n_files_to_generate)]
            file_sizes = [generate_new_file_every_n for i in range(0, n_files_to_generate)]
        else:
            output_files = [output_file]
            file_sizes = [n_records]

        record_ids = data_inventory.get_all_record_ids()

        # Randomly shuffle records before saving, this is better for
        # model training
        if random_shuffle_before_save:
            random.seed(123)
            random.shuffle(record_ids)

        start_id_position = 0
        for file_number, (output_file, file_size) in enumerate(zip(output_files, file_sizes)):

            batch_record_ids = record_ids[0:start_id_position+file_size]
            start_id_position += file_size + 1

            data_inventory_batch = \
                data_inventory.copy_data_inv_with_only_ids(set(batch_record_ids))

            # Create and Write Records to TFRecord file
            with tf.python_io.TFRecordWriter(output_file) as writer:

                logging.info("Start Writing Record to TFRecord %s- Total %s" %
                             (output_file, n_records))

                # Loop over all records and write to TFRecord
                successfull_writes = 0
                for i, record_id in enumerate(batch_record_ids):

                    if i % 1000 == 0:
                        logging.info("Wrote %s / %s files" % (i, n_records))

                    record_data = data_inventory_batch.get_record_id_data(record_id)

                    # Process all images in a record
                    raw_images = list()
                    for image_path in record_data['images']:
                        try:
                            if image_pre_processing_fun is not None:
                                image_pre_processing_args['image'] = image_path
                                image_raw = image_pre_processing_fun(
                                     **image_pre_processing_args)
                            else:
                                image_raw = read_jpeg(image_path)

                        except Exception as e:
                            logging.debug("Failed to read file: %s , error %s" %
                                          (image_path, str(e)))
                            continue

                        raw_images.append(image_raw)

                    # check if at least one image is available
                    if len(raw_images) == 0:
                        logging.info("Discarding record %s - no image avail" %
                                     record_id)
                        data_inventory_batch.remove_record(record_id)
                        continue

                    # add prefix to labels
                    label_data = {prefix_to_labels + k: v for k, v in
                                  record_data['labels'].items()}

                    # Create Record to Serialize
                    record_to_serialize = dict()
                    record_to_serialize['id'] = record_id
                    record_to_serialize['labels'] = label_data
                    record_to_serialize['images'] = raw_images

                    serialized_record = self.tfr_encoder(record_to_serialize,
                                                         labels_are_numeric=False)

                    # Write the serialized data to the TFRecords file.
                    writer.write(serialized_record)
                    successfull_writes += 1

                logging.info(
                    "Finished Writing Records to TFRecord - Wrote %s of %s" %
                    (successfull_writes, n_records))

                # Create a TFRecord class object and export information about
                # the file
                tfr_file = TFRFile(file_path=output_file,
                                   data_inventory=data_inventory_batch)
