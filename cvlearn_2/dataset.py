import os
import glob
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import sys
import torch
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, directory, split, transforms=None):
        self.split = split
        self.directory = directory
        self.transforms = transforms
        self.labels_dict = self.get_labels_dict()
        self.label_count, self.data = self._load_all_image_paths_labels(split)
        self.classes_count = self._count_classes()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self._load_image(self.data[idx]['image_path'])
        if self.transforms is not None:
            image = self.transforms(image)
        labels = self.data[idx]['labels']
        return (image, labels)

    def get_labels_dict(self):
        return {
            'bird': 0,
            'bus' : 1,
            'car' : 2,
            'cat' : 3,
            'dog' : 4,
        }

    def _count_classes(self):
        count_dict = {x: 0 for x in self.labels_dict}
        for pairs in self.data:
            for label_list in pairs['labels']:
                for label in np.unique(label_list):
                    count_dict[label] += 1
        return count_dict

    def _load_image(self, image_path):
        img = Image.open(image_path)
        assert (img.mode == 'RGB')
        return img

    def _get_images_list(self, split):
        image_paths = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.lower().endswith('.jpg') and split in root:
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _get_xml_file_path(self, image_name):
        return image_name.replace('JPEGImages','Annotations').replace('.jpg','.xml')

    def _load_all_image_paths_labels(self, split):
        label_count = 0
        all_image_paths_labels = []
        images_list = self._get_images_list(split)
        xml_path_list = [self._get_xml_file_path(image_path)
                         for image_path in images_list]
        for image_path, xml_path in zip(images_list, xml_path_list):
            assert (image_path not in all_image_paths_labels)
            labels = list(np.unique(self._get_labels_from_xml(xml_path)))
            label_count += len(labels)
            image_path_labels = {'image_path': image_path,
                                 'labels': labels}
            all_image_paths_labels.append(image_path_labels)

        print("SET: {} | TOTAL IMAGES: {}".format(self.split, len(all_image_paths_labels)))
        print("SET: {} | TOTAL LABELS: {}".format(self.split, label_count))
        return label_count, all_image_paths_labels

    def _get_labels_from_xml(self, xml_path):
        labels = []
        candidate = ['bird','bus','car','cat','dog']
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for child in root.iter('object'):
            name=child.find('name').text
            if name in candidate:
                labels.append(name)
        #print(labels)
        return labels

class VOCBatch:
    def __init__(self, data):
        self.transposed_data = list(zip(*data))
        self.image = torch.stack(self.transposed_data[0], 0)
        self.labels = self.construct_int_labels()

    def construct_int_labels(self):
        remap_dict = VOCDataset.get_labels_dict(None)
        labels = self.transposed_data[1]
        batch_size = self.image.shape[0]
        num_classes = len(remap_dict)
        one_hot_int_labels = torch.zeros((batch_size, num_classes))
        for i in range(len(labels)):
            sample_labels = labels[i]
            one_hot = torch.zeros(num_classes)
            for string_label in sample_labels:
                int_label = remap_dict[string_label]
                one_hot[int_label] = 1.
            one_hot_int_labels[i] = one_hot
        return one_hot_int_labels

    def pin_memory(self):
        self.image = self.image.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

def collate_wrapper(batch):
    return VOCBatch(batch)