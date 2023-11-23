import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from dataset import VOCDataset, collate_wrapper


class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes, backbone='resnet18'):
        super(ObjectDetectionModel, self).__init__()
        if backbone == 'resnet18':
            self.backbone = resnet18(weights="IMAGENET1K_V1")
        else:
            raise Exception("Resnet model configure failed!")
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def calculate_mAP(ground_truth, predictions, num_classes, iou_threshold=0.5):
    average_precisions = []
    for class_id in range(num_classes):
        true_positives = []
        false_positives = []
        scores = []

        # Collect ground truth and predictions for a specific class
        gt_boxes = ground_truth[class_id]  # Ground truth bounding boxes for the class
        pred_boxes = predictions[class_id]  # Predicted bounding boxes for the class

        # Sort the predicted bounding boxes by confidence score in descending order
        sorted_indices = np.argsort(pred_boxes[:, 4])[::-1]
        pred_boxes = pred_boxes[sorted_indices]

        # Initialize empty arrays to store cumulative true positives and false positives
        cum_tp = np.zeros(len(pred_boxes))
        cum_fp = np.zeros(len(pred_boxes))

        # Iterate over each predicted bounding box
        for i, pred_box in enumerate(pred_boxes):
            pred_box_class = pred_box[5]  # Predicted class
            pred_box_coords = pred_box[:4]  # Predicted bounding box coordinates
            pred_box_score = pred_box[4]  # Predicted confidence score

            # Extract ground truth bounding boxes of the same class
            gt_class_boxes = gt_boxes[gt_boxes[:, 4] == pred_box_class][:, :4]

            if gt_class_boxes.shape[0] > 0:
                iou_scores = calculate_iou(pred_box_coords, gt_class_boxes)

                max_iou_idx = np.argmax(iou_scores)
                max_iou = iou_scores[max_iou_idx]

                if max_iou >= iou_threshold:
                    if not true_positives[max_iou_idx]:
                        true_positives[max_iou_idx] = 1
                        scores.append(pred_box_score)
                    else:
                        false_positives[i] = 1
                else:
                    false_positives[i] = 1
            else:
                false_positives[i] = 1

        # Compute cumulative sums of true positives and false positives
        cum_tp = np.cumsum(true_positives)
        cum_fp = np.cumsum(false_positives)

        # Compute precision and recall values for each threshold
        precision, recall, _ = precision_recall_curve(np.concatenate([true_positives, false_positives]), scores)

        # Compute Average Precision (AP) for the class
        ap = compute_ap(precision, recall)

        average_precisions.append(ap)

    # Compute mean Average Precision (mAP) across all classes
    mAP = np.mean(average_precisions)

    return mAP


# Helper function to calculate IoU (Intersection over Union) between bounding boxes
def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = np.maximum(box1[:, 0], box2[:, 0])
    y1 = np.maximum(box1[:, 1], box2[:, 1])
    x2 = np.minimum(box1[:, 2], box2[:, 2])
    y2 = np.minimum(box1[:, 3], box2[:, 3])

    # Calculate intersection area
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate box areas
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Calculate Union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


# Helper function to compute Average Precision (AP) given precision and recall values
def compute_ap(precision, recall):
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    # Find indices where recall values change
    change_indices = np.where(recall[1:] != recall[:-1])[0] + 1

    # Compute Average Precision (AP)
    ap = np.sum((recall[change_indices] - recall[change_indices - 1]) * precision[change_indices])

    return ap


def visualize_curves(train_loss, val_loss, mAP):
    # Visualize loss curves and mAP curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mAP, label='mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()

    plt.show()


def main():
    # Set up hyperparameters and configuration
    num_classes = 5
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the PASCAL VOC dataset
    directory = 'D:/Code/Train_data/studywork_2023/Homework2/Data'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_dataset = VOCDataset(directory, 'train', transforms=data_transforms['train'])
    val_dataset = VOCDataset(directory, 'val', transforms=data_transforms['val'])
    # train_dataset = VOCDetection(root='D:/Code/Train_data/studywork_2023/Homework2/Data', year='2012',
    #                              image_set='train', download=False, transform=ToTensor())
    # val_dataset = VOCDetection(root='D:/Code/Train_data/studywork_2023/Homework2/Data', year='2012',
    #                            image_set='val', download=False, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, criterion, and optimizer
    model = ObjectDetectionModel(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move the model to the device
    model = model.to(device)

    # Training loop
    train_losses = []
    val_losses = []
    mAP_scores = []

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = test_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        mAP = calculate_mAP(model, val_loader, device)
        mAP_scores.append(mAP)

        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mAP: {mAP:.4f}')

    visualize_curves(train_losses, val_losses, mAP_scores)


if __name__ == '__main__':
    main()
    debug = 1
