""" Class To Create Dataset Inventory """
import random
import json
import logging
import copy


from data.utils import (
    randomly_split_dataset, map_label_list_to_numeric_dict,
    export_dict_to_json, _balanced_sampling)
from data.importer import DatasetImporter


logger = logging.getLogger(__name__)


class DatasetInventory(object):
    """ Defines a Datset Inventory - Contains labels, links and data about each
        Record
    """

    missing_label_value = '-1'
    missing_label_value_num = -1

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

    def _get_all_labels(self):
        """ Extract all labels
            Returns: {'species': ('elephant', 'zebra'),
                      'count': ('1', '2')}
        """
        all_labels = dict()
        for k, v in self.data_inventory.items():
            for label_entry in v['labels']:
                # For each record get and count label types and labels
                for label_name, label_value in label_entry.items():
                    if label_name not in all_labels:
                        all_labels[label_name] = set()
                    if not label_value == type(self).missing_label_value:
                        all_labels[label_name].add(label_value)
        return all_labels

    def _calc_label_stats(self):
        """ Calculate Label Stats
            Returns: {'species': {'Zebra': 3, 'Elephant': 6},
                      'counts': {'1': 5, '2': 10}}
        """
        # Calculate and log statistics about labels
        label_stats = dict()
        for _id, data in self.data_inventory.items():
            # For each record get and count label types and labels
            for label_entry in data['labels']:
                for label_name, label_val in label_entry.items():
                    if label_name not in label_stats:
                        label_stats[label_name] = dict()
                    if label_val not in label_stats[label_name]:
                        label_stats[label_name][label_val] = 0
                    label_stats[label_name][label_val] += 1
        return label_stats

    def log_stats(self, debug_only=False):
        """ Logs Statistics about Data Inventory """
        label_stats = self._calc_label_stats()
        # Log Stats
        for label_type, labels in label_stats.items():
            label_list = list()
            count_list = list()
            for label, count in labels.items():
                label_list.append(label)
                count_list.append(count)
            total_counts = sum(count_list)
            sort_index = sorted(range(len(count_list)), reverse=True,
                                key=lambda k: count_list[k])
            for idx in sort_index:
                if debug_only:
                    logger.debug(
                        "Label Type: %s Label: %s Records: %s / %s (%s %%)" %
                        (label_type, label_list[idx], count_list[idx],
                         total_counts,
                         round(100 * (count_list[idx]/total_counts), 4)))
                else:
                    logger.info(
                        "Label Type: %s Label: %s Records: %s / %s (%s %%)" %
                        (label_type, label_list[idx], count_list[idx],
                         total_counts,
                         round(100 * (count_list[idx]/total_counts), 4)))

    def export_to_json(self, json_path):
        """ Export Inventory to Json File """

        if self.data_inventory is not None:
            with open(json_path, 'w') as fp:
                json.dump(self.data_inventory, fp)

            logger.info("Data Inventory saved to %s" % json_path)
        else:
            logger.warning("Cant export data inventory to json - no\
                            inventory created yet")

    def export_to_tfrecord(self, tfr_writer, tfr_path,
                           **kwargs):
        """ Export Dataset to TFRecod """

        # create tfrecord dictionary
        tfrecord_dict = dict()
        for _id, record_values in self.data_inventory.items():
            tfr_record = self._convert_record_to_tfr_format(
                _id, record_values)
            tfrecord_dict[_id] = tfr_record

        # Write to disk
        tfr_writer.encode_to_tfr(tfrecord_dict, tfr_path, **kwargs)

    def _convert_record_to_tfr_format(self, id, record):
        """ Convert a record to a tfr format """

        # Extract and convert meta data information
        if 'meta_data' in record.keys():
            if isinstance(record['meta_data'], str):
                meta_data = record['meta_data']
            elif isinstance(record['meta_data'], dict):
                meta_data = json.dumps(record['meta_data'])
            else:
                meta_data = ''
        else:
            meta_data = ''

        # Generate concatenated labels text
        label_text = list()
        for label in record['labels']:
            for label_name, label_value in label.items():
                label_text += ['#' + label_name + ':' + label_value]
        label_text = ''.join(label_text)

        # generate labels dict, save string and numeric labels
        labels_dict = dict()
        labels_num_dict = dict()
        for label in record['labels']:
            for label_name, label_value in label.items():
                label_id = 'label/' + label_name
                label_id_num = 'label_num/' + label_name
                if label_name not in labels_dict:
                    labels_dict[label_id] = []
                    labels_num_dict[label_id_num] = []
                if label_value == type(self).missing_label_value:
                    val_num = type(self).missing_label_value_num
                else:
                    val_num = self.labels_numeric_map[label_name][label_value]
                val = label_value
                labels_num_dict[label_id_num].append(val_num)
                labels_dict[label_id].append(val)

        tfr_data = {
            "id": str(id),
            "n_images": len(record['images']),
            "n_labels": len(record['labels']),
            "image_paths": record['images'],
            "meta_data": meta_data,
            "labelstext": label_text,
            **labels_dict,
            **labels_num_dict
        }

        return tfr_data

    def export_label_mapping(self, path):
        """ Export Label Mapping to Json file """
        assert self.labels_numeric_map is not None, \
            "Numeric Label Mapping has not been generated"

        export_dict_to_json(self.labels_numeric_map, path)


class DatasetInventorySplit(DatasetInventory):
    """ Datset Dictionary Split - Does not allow further
        manipulations
    """
    def __init__(self, data_inventory, labels, labels_numeric_map):
        self.data_inventory = data_inventory
        self.labels = labels
        self.labels_numeric_map = labels_numeric_map


class DatasetInventoryMaster(DatasetInventory):
    """ Creates Datset Dictionary from a source and allows to
        manipulate labels and create splits
    """
    def __init__(self, labels_numeric_map=None):
        self.data_inventory = None
        self.labels = None
        self.labels_numeric_map = labels_numeric_map

    def _map_labels_to_numeric(self):
        """ Map all labels to numerics """

        if self.labels_numeric_map is None:
            self.labels = self._get_all_labels()
            labels_numeric_map = dict()

            for label_name, label_set in self.labels.items():
                mapped = map_label_list_to_numeric_dict(list(label_set))
                labels_numeric_map[label_name] = mapped

            self.labels_numeric_map = labels_numeric_map

        # create numeric to text labels as well
        self.label_mapping_from_num = \
            {k: {kk: vv for vv, kk in v.items()}
             for k, v in self.labels_numeric_map.items()}

    def create_from_source(self, type, params):
        """ Create Dataset Inventory from a specific Source """
        importer = DatasetImporter().create(type, params)
        self.data_inventory = importer.import_from_source()
        # self.label_handler = LabelHandler(self.data_inventory)
        # self.label_handler.remove_not_all_label_attributes()

    def remove_multi_label_records(self):
        """ Remove records with multiple labels / observations """
        to_remove = list()
        for record_id, data in self.data_inventory.items():
            if len(data['labels']) > 1:
                to_remove.append(record_id)
        logger.info("Removing %s records with multiple labels" %
                    len(to_remove))
        for record_id in to_remove:
            self.remove_record(record_id)

    def randomly_remove_samples_to_percent(self, p_keep):
        """ Randomly sample a percentage of all records """
        if not p_keep <= 1:
            raise ValueError("p has to be between 0 and 1")

        new_data_inv = dict()
        all_ids = list(self.data_inventory.keys())
        n_total = len(all_ids)
        n_choices = int(n_total * p_keep)
        choices = random.sample(all_ids, k=n_choices)

        for id in choices:
            new_data_inv[id] = self.data_inventory[id]

        self.data_inventory = new_data_inv

    def remove_records_with_label(self, label_name_list, label_value_list):
        """ Remove all records with labels in label_name and corresponding
            label values
            Example: label_name : [species, species]
                     label_value: ['zebra', 'elephant']
        """
        assert all([isinstance(label_name_list, list),
                    isinstance(label_value_list, list)]), \
            "label_name_list and label_value_list must be lists"

        for label_name, label_value in zip(label_name_list, label_value_list):
            self._remove_records_with_label(label_name, label_value)

    def _remove_records_with_label(self, label_name, label_value):
        """ Remove all records with 'label_value' for 'label_name'
            Example: label_name: 'species' label_value: 'Zebra'
        """
        ids_to_remove = list()

        for record_id, record_value in self.data_inventory.items():
            labels_list = record_value['labels']
            for label in labels_list:
                for l_name, l_val_list in label.items():
                    if (label_name == l_name):
                        if label_value in l_val_list:
                            ids_to_remove.append(record_id)

        logger.info("Removing %s records from label %s with value %s" %
                    (len(ids_to_remove), label_name, label_value))

        for id_to_remove in ids_to_remove:
            self.remove_record(id_to_remove)

    def keep_only_records_with_label(self, label_name_list, label_value_list):
        """ Keep only records with (at least one) of the specified
            label_name and corresponding label values
        """
        assert all([isinstance(label_name_list, list),
                    isinstance(label_value_list, list)]), \
            "label_name_list and label_value_list must be lists"

        to_keep = set()
        for label_name, label_value in zip(label_name_list, label_value_list):
            to_keep = to_keep.union(
                self._keep_only_record_with_label(label_name, label_value))

        logger.info("Keeping %s records" % len(to_keep))

        to_remove = self.data_inventory.keys() - to_keep
        for id_to_remove in to_remove:
            self.remove_record(id_to_remove)

    def _keep_only_record_with_label(self, label_name, label_value):
        """ Keep only records with the label_value of the label_name
        """
        ids_to_keep = set()
        for record_id, record_value in self.data_inventory.items():
            labels_list = record_value['labels']
            for label in labels_list:
                for l_name, l_val in label.items():
                    if (label_name == l_name):
                        if label_value == l_val:
                            ids_to_keep.add(record_id)
        return ids_to_keep

    def _remove_records_with_any_missing_label(self):
        """ Remove any records with the default missing value of -1 """
        ids_to_remove = set()
        for record_id, record_value in self.data_inventory.items():
            labels_list = record_value['labels']
            for label in labels_list:
                for l_vals in label.values():
                    if l_vals == type(self).missing_label_value:
                        ids_to_remove.add(record_id)

        logger.info("Removing %s records with missing labels" %
                    len(ids_to_remove))

        for id_to_remove in ids_to_remove:
            self.remove_record(id_to_remove)

    def split_inventory_by_random_splits_with_balanced_sample(
            self,
            split_label_min,
            split_names,
            split_percent):
        """ Split inventory randomly into different sets
            according to
                split_label_min: e.g 'species'
            Returns dict: {'id1': 'test', 'id2': 'train'}
        """

        # Create a dictionary mapping each record to label for sampling
        ids_to_split_label = dict()

        for record_id, record_value in self.data_inventory.items():
            first_labels_entry = record_value['labels'][0]
            # only consider first entry in labels list
            if split_label_min in first_labels_entry:
                split_label = first_labels_entry[split_label_min]
                ids_to_split_label[record_id] = split_label

        split_ids = list(ids_to_split_label.keys())
        logging.debug("Found %s record to split randomly" % len(split_ids))

        split_assignments = randomly_split_dataset(
            split_ids,
            split_names,
            split_percent,
            balanced_sampling_min=True,
            balanced_sampling_id_to_label=ids_to_split_label)

        logging.debug("Found %s records with split assignments" %
                      len(split_assignments.keys()))

        return self._convert_splits_to_dataset_inventorys(split_assignments)

    def split_inventory_by_random_splits(
            self,
            split_names,
            split_percent):
        """ Split inventory randomly into different sets
            Returns dict: {'id1': 'test', 'id2': 'train'}
        """

        split_ids = list(self.data_inventory.keys())

        split_assignments = randomly_split_dataset(
            split_ids,
            split_names,
            split_percent,
            balanced_sampling_min=False,
            balanced_sampling_id_to_label=None)

        return self._convert_splits_to_dataset_inventorys(split_assignments)

    def split_inventory_by_meta_data_column(
            self,
            meta_colum
            ):
        """ Split inventory into different sets based on
            meta_data_column
        """

        split_assignments = dict()

        for record_id, record_value in self.data_inventory.items():
            meta_val = record_value['meta_data'][meta_colum]
            split_assignments[record_id] = meta_val

        return self._convert_splits_to_dataset_inventorys(split_assignments)

    def split_inventory_by_meta_data_column_and_balanced_sampling(
            self,
            meta_colum,
            split_label_min
            ):
        """ Split inventory into different sets based on
            meta_data_column after balanced sampling
        """

        id_to_label = dict()

        for record_id, record_data in self.data_inventory.items():
            # take only the first entry of the labels / observations to assign
            # a label for that record
            if split_label_min in record_data['labels'][0]:
                label = record_data['labels'][0][split_label_min]
                id_to_label[record_id] = label

        remaining_ids = set(_balanced_sampling(id_to_label))

        split_assignments = dict()

        for record_id, record_value in self.data_inventory.items():
            meta_val = record_value['meta_data'][meta_colum]
            if record_id in remaining_ids:
                split_assignments[record_id] = meta_val

        return self._convert_splits_to_dataset_inventorys(split_assignments)

    def _convert_splits_to_dataset_inventorys(self, split_assignments):
        """ Convert split assignments to new splitted dataset inventories """

        # label overview
        all_labels = self._get_all_labels()
        self._map_labels_to_numeric()

        # Create dictionary with split_name to id mapping
        split_to_record = {}
        for k, v in split_assignments.items():
            split_to_record[v] = split_to_record.get(v, [])
            split_to_record[v].append(k)

        # Create new splitted data inventories
        splitted_inventories = dict()

        for split, record_list in split_to_record.items():
            split_dict = dict()
            for record_id in record_list:
                split_dict[record_id] = self.data_inventory[record_id]
            logging.debug("Creating dataset split %s with %s records" %
                          (split, len(split_dict.keys())))
            splitted_inventories[split] = DatasetInventorySplit(
                                            split_dict,
                                            all_labels,
                                            self.labels_numeric_map)

        return splitted_inventories

    def remap_labels(self, label_map_dict):
        """ Remap labels according to mapping dictionary

            label_map_dict (dict):
                {'species': {'Zebra': 'species', 'Elephant': 'species',
                             'blank': 'blank'},
                 'counts': {'1': '1-5'}}
        """
        new_inventory = copy.deepcopy(self.data_inventory)
        # Loop over records
        for record_id, record_value in self.data_inventory.items():
            # loop over list of label entries [{species:}, {species:}]
            for i, labels in enumerate(record_value['labels']):
                # Loop over label names
                for label_name, label_value_list in labels.items():
                    if label_name in label_map_dict:
                        # loop over label name entries
                        for j, label_value in enumerate(label_value_list):
                            if label_value in label_map_dict[label_name]:
                                new_label = label_map_dict[label_name][label_value]
                                new_inventory[record_id]['labels'][i][label_name][j][new_label]

        self.data_inventory = new_inventory
