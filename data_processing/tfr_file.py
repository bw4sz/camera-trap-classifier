""" Class to Manage a TFR File """
from collections import OrderedDict
import os

import tensorflow as tf

from data_processing.data_reader import DatasetReader
from data_processing.data_inventory import DatasetInventory
from data_processing.label_handler import LabelHandler
from config.config import logging
from data_processing.utils import n_records_in_tfr


class TFRFile(object):
    def __init__(self, file_path, data_inventory=None, tfr_decoder=None,
                 output_labels=None, labels_are_numeric=False,
                 overwrite_data_inv=False):
        self.file_path = file_path
        self.data_inv = data_inventory
        self.tfr_decoder = tfr_decoder
        self.output_labels = output_labels
        self.overwrite_data_inv = overwrite_data_inv
        self.labels_are_numeric = labels_are_numeric

        assert self.file_path.endswith('.tfrecord'),\
            "File path must end with .tfrecord, currently is: %s" % \
            self.file_path

        self._create_data_inv_path()

        if self.data_inv is not None:
            self.data_inv.export_to_json(self.data_inv_path)

        elif os.path.exists(self.data_inv_path):
            logging.info("Creating Data Inventory from %s" % self.data_inv_path)
            self.data_inv = DatasetInventory()
            self.data_inv.create_from_data_inv_export(self.data_inv_path)

        else:
            assert tfr_decoder is not None, "tfr_decoder cannot be None"
            assert output_labels is not None, "output_labels cannot be None"
            self._create_data_inventory(self.file_path)

    def get_record_number(self):
        """ get number of records """
        if self.data_inv is not None:
            return len(self.data_inv.get_all_record_ids())
        else:
            return n_records_in_tfr()

    def _create_data_inv_path(self):
        """ Create a pre-defined path to the dataset inventory """

        prefix = self.file_path.split('.tfrecord')[0]
        self.data_inv_path = prefix + '.json'

    def _export_data_inventory(self):
        """ Write TFRecord meta data as data inventory """
        self.data_inv.export_to_json(self.data_inv_path)

    def _create_data_inventory(self):
        """ Create a dataset inventory from the TFRecord file """

        data_inventory_exists_on_disk = os.path.exists(self.data_inv_path)
        if not self.overwrite_data_inv and data_inventory_exists_on_disk:
            return None

        output_labels_clean = ['labels/' + x for x in self.output_labels]

        # get all ids and their labels from the input file
        dataset_reader = DatasetReader(self.tfr_decoder)

        iterator = dataset_reader.get_iterator(
             self.file_path, batch_size=2048,
             is_train=False, n_repeats=1,
             output_labels=self.output_labels,
             buffer_size=10192,
             decode_images=False,
             labels_are_numeric=self.labels_are_numeric,
             max_multi_label_number=None,
             drop_batch_remainder=False)

        logging.info("Iterator: %s" % iterator)
        logging.info("Labels numeric:%s" % self.labels_are_numeric)
        logging.info("File PAth: %s" % self.file_path)
        logging.info("Looking for lables: %s" % self.output_labels)

        id_label_dict = OrderedDict()
        with tf.Session() as sess:
            while True:
                try:
                    batch_data = sess.run(iterator)
                    self._extract_id_labels(id_label_dict,
                                            batch_data,
                                            output_labels_clean,
                                            labels_num=self.labels_are_numeric)
                except tf.errors.OutOfRangeError:
                    break

        # convert label dict to inventory
        logging.debug("Converting label dictionary to data inventory")
        data_inv = self._convert_id_label_dict_to_inventory(id_label_dict)

        logging.info("Label Inv sample: %s" % data_inv.data_inventory[list(data_inv.data_inventory.keys())[0]])
        self.data_inv = data_inv

        self._export_data_inventory()

    def _convert_id_label_dict_to_inventory(self, id_label_dict):
        """ convert id label dict to inventory """
        data_inv = DatasetInventory()
        data_inv.data_inventory = OrderedDict()
        inv = data_inv.data_inventory
        for record_id, label_types in id_label_dict.items():
            inv[record_id] = {'labels': dict()}
            for label_type, label_list in label_types.items():
                inv[record_id]['labels'][label_type] = label_list
        data_inv.label_handler = LabelHandler(data_inv.data_inventory)
        return data_inv

    def _extract_id_labels(self, dict_all, data_batch, output_labels,
                           labels_num=True):
        """ Extract ids and labels from dataset and add to dict
            {'1234': {'labels/primary': ['cat', 'dog']}}
        """
        for i, idd in enumerate(list(data_batch['id'])):
            id_clean = str(idd, 'utf-8')
            dict_all[id_clean] = dict()
            for lab in output_labels:
                lab_i = data_batch[lab][i]
                if labels_num:
                    lab_i = [int(str(x)) for x in lab_i]
                else:
                    lab_i = [str(x, 'utf-8') for x in lab_i]
                dict_all[id_clean][lab] = lab_i
