import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataset import VOCDataset, collate_wrapper
import torchvision.models as models
import numpy as np
from sklearn.metrics import average_precision_score
import torch.optim as optim

batch_size = 48
num_workers = 16
lr=1e-3
directory='/GPFS/data/heyangliu/Data'
num_classes=5
num_epochs=100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

train_set = VOCDataset(directory, 'train', transforms=data_transforms['train'])
train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True,
                          num_workers=num_workers)

val_set = VOCDataset(directory, 'val', transforms=data_transforms['val'])
val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True,
                        num_workers=num_workers)

model = models.resnet34(pretrained=True)
model.fc = torch.nn.Linear(512, num_classes)

train_losses = []
val_losses = []

model.to(device)
print('Starting optimizer with LR={}'.format(lr))
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

classwise_frequencies = np.array(list(train_set.classes_count.values()))
minimum_frequency = np.min(classwise_frequencies)
loss_weights = minimum_frequency / classwise_frequencies
loss_weights = torch.Tensor(loss_weights).to(device)
loss_function = torch.nn.BCELoss(weight=loss_weights)

for epoch in range(1, num_epochs + 1):
    model.train()
    losses = []
    for idx, batch in enumerate(train_loader):
        data = batch.image.to(device)
        target = batch.labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(torch.sigmoid(output), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        #print('Epoch: {}, Samples: {}/{}, Loss: {}'.format(epoch, idx * batch_size,
        #                                                   len(train_loader) * batch_size,
        #                                                   loss.item()))
        train_loss = torch.mean(torch.tensor(losses))
    print('Epoch: {}'.format(epoch))
    print('Training set: Average loss: {:.4f}'.format(train_loss))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            data = batch.image.to(device)
            target = batch.labels.to(device)
            output = model(data)
            batch_loss = loss_function(torch.sigmoid(output), target)
            val_loss += batch_loss.item()
            pred = torch.sigmoid(output)
            if idx == 0:
                predictions = pred
                targets = target
            else:
                predictions = torch.cat((predictions, pred))
                targets = torch.cat((targets, target))


    val_loss /= len(val_loader)
    val_mAP= average_precision_score(target.reshape(-1, num_classes).cpu(), pred.reshape(-1, num_classes).cpu())
    print('Validation set: Average loss: {:.4f}, mAP: {:.4f}'.format(val_loss,val_mAP))

    if (len(val_losses) > 0) and (val_loss < min(val_losses)):
        torch.save(model.state_dict(), "checkpoints/lr{}_model_{}_{:.4f}.pt".format(lr, epoch, val_mAP))
        print("Saving model (epoch {}) with lowest validation mAP: {}"
              .format(epoch, val_mAP))

    train_losses.append(train_loss)
    val_losses.append(val_loss)


