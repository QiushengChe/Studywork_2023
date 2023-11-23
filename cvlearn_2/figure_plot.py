import matplotlib.pyplot as plt
import numpy as np


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
