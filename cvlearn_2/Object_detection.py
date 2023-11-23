import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from torchvision.datasets import CocoDetection, VOCDetection
from torch.utils.data import DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import matplotlib.pyplot as plt
import numpy as np

# Define the transforms for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define the dataset
dataset = CocoDetection(root='D:/Code/Train_data/studywork_2023/Homework2/Data/JPEGImages',
                        annFile='D:/Code/Train_data/studywork_2023/Homework2/Data/Annotations', transform=transform)
# or
# dataset = VOCDetection(root='<path_to_dataset>', year='2007', image_set='train', transform=transform)

# Define the data loaders
dataloaders = {
    'train': DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4),
    'test': DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
}
# Load the ResNet backbone
backbone = torchvision.models.resnet50(pretrained=True)

# Modify the backbone architecture for object detection
backbone.out_channels = 256

# Define anchor generator
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# Define the model using the ResNet backbone and anchor generator
model = fasterrcnn_resnet50_fpn(pretrained_backbone=False,
                                backbone=backbone,
                                num_classes=91,  # Update with the appropriate number of classes
                                rpn_anchor_generator=anchor_generator)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 10  # Define the number of epochs
for epoch in range(num_epochs):
    for images, targets in dataloaders['train']:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.train()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()
model.eval()
with torch.no_grad():
    for images, targets in dataloaders['test']:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        # Process the outputs for visualization or further analysis
        # ...


def plot_detections(image, boxes, labels):
    plt.imshow(image.permute(1, 2, 0))
    ax = plt.gca()

    for box, label in zip(boxes, labels):
        box = box.cpu().numpy()
        xmin, ymin, xmax, ymax = box
        label = label.cpu().numpy()

        color = np.random.rand(3, )
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor=color, linewidth=2))
        ax.text(xmin, ymin - 6, dataset.classes[label],
                bbox=dict(facecolor=color, alpha=0.5), fontsize=6, color='white')


# Example usage for visualization
plot_detections(images[0].cpu(), outputs[0]['boxes'], outputs[0]['labels'])
plt.show()
