import fire
import numpy as np
import torch
from models.pointnet2_seg import pointnet2_seg_ssg
import os
import json
import matplotlib.pyplot as plt


def pc_normalize(pc):
    mean = np.mean(pc, axis=0)
    pc -= mean
    m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)))
    pc /= m
    return pc

def plot_results(xyz_points, labels, preds):
    fig = plt.figure(figsize=(18, 6))

    # Original labels
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2], c=labels, cmap='viridis')
    ax1.set_title('Original Labels')

    # Predicted labels
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2], c=preds, cmap='viridis')
    ax2.set_title('Predicted Labels')

    # Correct predictions
    correct = (preds == labels)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2], c=correct, cmap='viridis', marker='o')
    ax3.set_title('Correct Predictions')

    plt.show()

def evaluate_seg_one(data_root, checkpoint, dims=6, nclasses=50):
    cat = {}
    with open(os.path.join(data_root, 'synsetoffset2category.txt'), 'r') as f:
        for line in f.readlines():
            cat[line.strip().split()[0]] = line.strip().split()[1]
    show_json_path = os.path.join(data_root, 'train_test_split', 'shuffled_show_file_list.json')
    with open(show_json_path, 'r') as file:
        data = json.load(file)
    l = os.path.join(data_root, data[0].split('/')[1], data[0].split('/')[2] + '.txt')
    pc = np.loadtxt(str(l)).astype(np.float32)
    print(pc.shape)
    unique_labels = np.unique(pc[:, -1])
    uninclasses = len(unique_labels)
    print(f"Number of unique classes: {uninclasses}")
    device = torch.device('cuda')
    model = pointnet2_seg_ssg(dims, nclasses)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print('Loading {} completed'.format(checkpoint))

    xyz_points = pc[:, :6]
    labels = pc[:, -1].astype(np.int32)
    # labels = np.concatenate(labels, axis=0)
    xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
    xyz = torch.tensor(xyz_points[:, :3]).unsqueeze(0).to(device)
    points = torch.tensor(xyz_points[:, 3:]).unsqueeze(0).to(device)
    preds = []
    with torch.no_grad():
        pred = model(xyz, points)
        pred = torch.max(pred, dim=1)[1].cpu().detach().numpy()
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    accuracy = np.sum(pred == labels) / float(len(labels))
    print(f"Accuracy: {accuracy:.4f}")
    plot_results(xyz_points, labels, preds)


if __name__ == '__main__':
    fire.Fire()