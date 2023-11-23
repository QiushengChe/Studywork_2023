import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import xml.dom.minidom as minidom

data_path = 'D:/Code/Train_data/studywork_2023/Homework2/Data'
anno_path = f'{data_path}/Annotations'
image_path = f'{data_path}/JPEGImages'

# collect 100 samples from files in anno_path
sample_list = []

for xml in os.listdir(annotation_path):
    file_id = os.path.splitext(xml)[0]

    xml_file = f'{anno_path}/{file_id}.xml'
    jpg_file = f'{image_path}/{file_id}.jpg'

    sample = {'id': file_id, 'xml_file': xml_file, 'jpg_file': jpg_file}
    sample_list.append(sample)

    if len(sample_list) >= 100:
        break
