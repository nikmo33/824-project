from torch.utils.data import DataLoader
import torch
import numpy as np
from tensorboardX import SummaryWriter
from Loss import *


model = Net()
num_epochs = num_epochs
log_freq = 10  # Steps
test_freq = 1000  # Steps

loss = MultiLossLayer()

train_dataset_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

# If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
val_dataset_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

# Use the GPU if it's available.
cuda = torch.cuda.is_available()

if cuda:
model = model.cuda()

optimizer = torch.optim.SGD(
    'params': model.parameters(), lr=2.5e-4, momentum=0.9, weight_decay=1e-4)


def validate():
    for batch_id, batch_data in enumerate(val_dataset_loader):
        model.eval()
        image_data = batch_data['image'].cuda()

        segmented_out, depth_out = model(
            image_data)

        segmented_gt = batch_data['segmented_image'].cuda()
        depth_gt = batch_data['depth_image'].cuda()

        # Optimize the model according to the predictions
        # loss = self._optimize(
        #     predicted_answer, ground_truth_answer, train_flag=False)
        seg_acc = segmentation_acc(segmented_out, segmented_gt)
        depth_error = depth_error(depth_out, depth_gt)
        acc_val = accuracy(
            predicted_answer, ground_truth_answer, topk=(1,))
    return seg_acc, depth_error


def segmentation_acc(segmented_out, sgemented_gt):
    correct_preds = (torch.eq(segmented_gt, segmented_out)).sum(dim=(1, 2))
    correct_preds = correct_preds.numpy()/(gt.shape[1]*gt.shape[2])
    acc = correct_preds.sum()/gt.shape[0]
    acc = torch.Tensor(acc)
    return acc


def depth_error(depth_out, depth_gt):
    rms_val = torch.sum(torch.Tensor((depth_out-depth_gt)**2), dim=(1, 2))
    rms_val = out**0.5
    error = out.numpy()/(gt.shape[1]*gt.shape[2])
    error = torch.Tensor(error)
    return error


def train():
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        num_batches = len(train_dataset_loader)

        for batch_id, batch_data in enumerate(train_dataset_loader):
            model.train()  # Set the model to train mode
            current_step = epoch * num_batches + batch_id

            # ============
            image_data = batch_data['image'].cuda()

            segmented_out, depth_out = model(
                image_data)

            segmented_gt = batch_data['segmented_image'].cuda()
            depth_gt = batch_data['depth_image'].cuda()
            # ============

            loss = optimize(
                segmented_out, depth_out, segmenetd_gt, depth_gt, writer, current_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            seg_acc = segmentation_acc(segmented_out, segmented_gt)
            depth_error = depth_error(depth_out, depth_gt)

            if current_step % log_freq == 0:
                print("Epoch: {}, Batch {}/{} has loss {}".format(epoch,
                                                                  batch_id, num_batches, loss))

            if current_step % test_freq == 0:
                val_seg_accuracy, val_depth_loss = validate()
                print("Epoch: {} has val accuracy {}".format(
                    epoch, val_accuracy[0][0]))
