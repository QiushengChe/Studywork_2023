import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


class customPASCALdataset(Dataset):
    # consider the root_dir = train_val; since the structure of the test folder and train_val folder nearly the same and are gathered to become pascal-voc
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        self.image_list = os.listdir(self.image_dir)
        self.annotation_list = os.listdir(self.annotation_dir)

    def __len__(self):
        return len(self.annotation_list)

    # test folder
    def __getitem__(self, index):
        img_name_anno = self.annotation_list[index]
        annotation_path = os.path.join(self.annotation_dir, img_name_anno)
        img_path = os.path.join(self.image_dir, img_name_anno.replace('.xml', '.jpg'))
        img_name_time = os.listdir(img_path)
        annotation_name_time = os.listdir(annotation_path)

        image = Image.open(img_path + '/' + img_name_time[0]).convert('RGB')
        label = self.parse_annotation(annotation_path + '/' + annotation_name_time[0])

        return image, label

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax]
            })

        return objects


def visualize_data(dataset):
    num_images = 10  #
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(6, 6 * num_images))

    for i in range(num_images):
        image, label = dataset[i]
        print(image, label)

        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f"Image {i + 1}")

        # You can customize the visualization of the target here
        # For example, plotting bounding boxes or labels

    plt.tight_layout()
    plt.show()


def visualize_data_bbox(dataset):
    num_images = 10
    fig, axes = plt.subplots(nrows=num_images, ncols=1, figsize=(6, 6 * num_images))

    for i in range(num_images):
        image, target = dataset[i]

        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f"Image {i + 1}")

        # Plot bounding boxes
        for obj in target:
            bbox = obj['bbox']
            xmin, ymin, xmax, ymax = bbox

            # Plot bounding box rectangle
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, edgecolor='r', linewidth=2)
            axes[i].add_patch(rect)

            # Add label text
            label = obj['name']
            axes[i].text(xmin, ymin - 2, label, fontsize=8, color='r',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root_dir = 'D:\\Code\\Train_data\\studywork_2023\\Homework2\\Data'
    dataset = customPASCALdataset(root_dir, transforms=None)
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    visualize_data_bbox(dataset)
