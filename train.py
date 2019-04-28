from torch.utils.data import DataLoader
import torch
import numpy as np
from tensorboardX import SummaryWriter
from Loss import *
import dataloader
import deeplab
import pdb
from Loss import Decoder


num_epochs = 10
log_freq = 100  # Steps
test_freq = 1000  # Steps

loss_func = MultiLossLayer()

enc_model = deeplab.DeepLab(backbone='resnet', output_stride=8)
dec_model = Decoder(13)
dataset = dataloader.ProjectDataset(base_dir='episodes/')

train_dataloader = DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=10)

# train_dataset_loader = DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

# If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
# val_dataset_loader = DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

# Use the GPU if it's available.
cuda = torch.cuda.is_available()

if cuda:
    enc_model = enc_model.cuda()
    dec_model = dec_model.cuda()
    loss_func = loss_func.cuda()
optimizer = torch.optim.SGD([
    {'params': enc_model.parameters(), 'lr': 25e-5},
    {'params': dec_model.parameters(), 'lr': 25e-5},
    {'params': loss_func.seg_sigma,
        'lr': 1e-2}, {'params': loss_func.depth_sigma, 'lr': 1e-2}
], lr=2.5e-4, momentum=0.9, weight_decay=1e-4)


def validate():
    for batch_id, batch_data in enumerate(val_dataset_loader):
        enc_model.eval()
        dec_model.eval()
        loss_func.eva()
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


def segmentation_acc(segmented_out, segmented_gt):
    # print(torch.max(segmented_gt), torch.min(segmented_gt))
    # print(segmented_out, segmented_out.shape)
    # print(segmented_gt.shape)
    _, indices = torch.max(torch.Tensor(segmented_out), 1)
    # print(indices.shape)
    indices = indices.type('torch.FloatTensor')
    correct_preds = (torch.eq(segmented_gt, indices)).sum(dim=(1, 2))
    correct_preds = correct_preds.float().numpy(
    )/(segmented_gt.shape[1]*segmented_gt.shape[2])
    # print(correct_preds)
    acc = correct_preds.sum()/segmented_gt.shape[0]
    # print(acc)
    # acc = torch.Tensor(acc)
    # print(acc)
    return acc


def depth_err(depth_out, depth_gt):
    print(depth_out.shape)
    print(depth_gt.shape)
    # print(depth_gt)
    # print()
    # x = torch.randn(3, 2)

    # y = torch.zeros(512, 512)
    # # # x
    # # # tensor([[-0.4620,  0.3139],
    # # #     [0.3898, -0.7197],
    # # #     [0.0478, -0.1657]])
    # depth_gt = torch.where(depth_gt < 2., depth_gt, y)
    # # print(depth_gt)
    # # return 0
    diff = torch.Tensor(
        (depth_out.squeeze(1)-depth_gt)**2)

    diff = torch.where(depth_gt > 100, diff, torch.zeros(diff.shape))
    rms_val = torch.sum(diff), dim=(1, 2))
    rms_val = rms_val**0.5
    error = rms_val.detach().numpy()/(depth_gt.shape[1]*depth_gt.shape[2])
    error = error.sum()/depth_gt.shape[0]
    # error = torch.Tensor(error)
    #criterion = torch.nn.MSELoss()
    #error = torch.sqrt(criterion(depth_out,depth_gt))
    return error


def train():
    enc_model.train()
    dec_model.train()
    loss_func.train()
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        num_batches = len(train_dataloader)

        for batch_id, batch_data in enumerate(train_dataloader):
            enc_model.train()
            dec_model.train()
            loss_func.train()  # Set the model to train mode
            current_step = epoch * num_batches + batch_id

            # ============
            image_data = batch_data[0].cuda()
            output = enc_model(image_data)
            output = output.cuda()
            segmented_out, depth_out = dec_model.forward(output)
            segmented_out = segmented_out.cuda()
            depth_out = depth_out.cuda()
            segmented_gt = batch_data[1].cuda()
            depth_gt = batch_data[2].cuda()
            # ============
            # pdb.set_trace()
            loss = loss_func(segmented_out, depth_out,
                             segmented_gt, depth_gt, writer, current_step)
            print(loss)
            loss = loss.cuda()
            # break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            seg_acc = segmentation_acc(segmented_out, segmented_gt)
            depth_error = depth_err(depth_out, depth_gt)
            # break
            print(seg_acc)
            print(depth_error)
            if current_step % log_freq == 0:
                print("Epoch: {}, Batch {}/{} has loss {}, seg_accuracy {}, adn depth error {}".format(epoch,
                                                                                                       batch_id, num_batches, loss, seg_acc, depth_error))
                writer.add_scalar('train/seg_accuracy',
                                  seg_acc, current_step)
                writer.add_scalar('train/depth_error',
                                  depth_error, current_step)

            # if current_step % test_freq == 0:
            #     val_seg_accuracy, val_depth_loss = validate()
            #     print("Epoch: {} has val_seg accuracy {} and val_depth error {}".format(
            #         epoch, val_seg_accuracy, val_depth_loss))
            #     writer.add_scalar('validation/seg_accuracy',
            #                       val_seg_accuracy, current_step)
            #     writer.add_scalar('validation/depth_accuracy',
            #                       val_depth_loss, current_step)


if __name__ == '__main__':
    train()
