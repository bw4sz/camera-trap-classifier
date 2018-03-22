""" Class To Create Dataset Inventory """
from config.config import logging
from data_processing.data_importer import ImportFromJson, ImportFromImageDirs
from data_processing.label_handler import LabelHandler


class DatasetInventory(object):
    """ Creates Datset Dictionary - Contains labels, links and data about each
        Record
    """
    def __init__(self):
        self.data_inventory = None
        self.label_handler = None

    def get_all_record_ids(self):
        """ Get all ids of the inventory """
        return list(self.data_inventory.keys())

    def get_record_id_data(self, record_id):
        """ Get content of record id """
        return self.data_inventory[record_id]

    def get_number_of_records(self):
        """ Count and Return number of records """
        return len(self.data_inventory.keys())

    def remove_record(self, id_to_remove):
        """ Remove specific record """
        self.data_inventory.pop(id_to_remove, None)

    def create_from_class_directories(self, root_path):
        """ Create inventory from path which contains class-specific
            directories
        """
        class_dir_reader = ImportFromImageDirs()
        self.data_inventory = \
            class_dir_reader.read_from_image_root_dir(root_path)

        self.label_handler = LabelHandler(self.data_inventory)
        self.label_handler.remove_not_all_label_types_present()

    def create_from_json(self, json_path):
        """ Create inventory from json file """
        json_reader = ImportFromJson()
        self.data_inventory = \
            json_reader.read_from_json(json_path)

        self.label_handler = LabelHandler(self.data_inventory)
        self.label_handler.remove_not_all_label_types_present()

    def log_stats(self):
        """ Logs Statistics about Data Inventory """

        # Calculate and log statistics about labels
        label_stats = dict()
        label_type_stats = dict()
        for k, v in self.data_inventory.items():
            # For each record get and count label types and labels
            for label_type, label_list in v['labels'].items():
                if label_type not in label_stats:
                    label_stats[label_type] = dict()
                    label_type_stats[label_type] = 0

                # Count if multiple labels
                if len(label_list) > 1:
                    label_type_stats[label_type] += 1

                for label in label_list:
                    if label not in label_stats[label_type]:
                        label_stats[label_type][label] = 0
                    label_stats[label_type][label] += 1

        # Log stats
        for k, v in label_stats.items():
            for label, label_count in v.items():
                logging.info("Label Type: %s - %s records for %s" %
                             (k, label_count, label))

        # Multiple Labels per Label Type
        for k, v in label_type_stats.items():
            logging.info("Label Type %s has %s records with multiple labels" %
                         (k, v))