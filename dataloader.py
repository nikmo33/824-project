from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms


class ProjectDataset(Dataset):

    def __init__(self, base_dir):
        self.root_dir = base_dir
        self.episodes = [p for p in os.listdir(
            base_dir) if p.startswith('episode')]
        self.num_episodes = len(self.episodes)
        episode_len = []
        for episode in self.episodes:
            episode_dir = os.path.join(base_dir, episode)
            episode_len.append(len(os.listdir(episode_dir)) // 4)

        self.episode_len = episode_len

        self.transform = transforms.Compose([transforms.CenterCrop(512),
                                             transforms.ToTensor()])

    def __len__(self):
        return sum(self.episode_len)

    def __getitem__(self, idx):
        episode_num = idx % self.num_episodes
        episode = self.episodes[episode_num]
        point_ID = idx % self.episode_len[episode_num]
        img = self.transform(self.get_raw_img(episode, point_ID))
        seg_GT = self.transform(
            self.get_segmentation_img(episode, point_ID))[0, :, :]*255
        depth_img = self.transform(self.get_depth_img(episode, point_ID))
        depth_GT = torch.Tensor(self.depth_to_array(depth_img))

        return img, seg_GT, depth_GT

    def depth_to_array(self, image):
        img = np.array(image.permute((1, 2, 0)))
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        normalized_depth = np.dot(img, [1.0, 256.0, 65536.0])
        normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        normalized_depth *= 1000 * 255
        return normalized_depth

    def get_raw_img(self, episode, point_ID):
        file_name = 'CentralRGB_' + str(point_ID).zfill(5) + '.png'
        file_path = os.path.join(
            self.root_dir, os.path.join(episode, file_name))
        return Image.open(file_path)

    def get_segmentation_img(self, episode, point_ID):
        file_name = 'CentralSemanticSeg_' + str(point_ID).zfill(5) + '.png'
        file_path = os.path.join(
            self.root_dir, os.path.join(episode, file_name))
        return Image.open(file_path)

    def get_depth_img(self, episode, point_ID):
        file_name = 'CentralDepth_' + str(point_ID).zfill(5) + '.png'
        file_path = os.path.join(
            self.root_dir, os.path.join(episode, file_name))
        return Image.open(file_path)


if __name__ == "__main__":
    dataset = ProjectDataset(base_dir='episodes/')

    dataloader = DataLoader(
<<<<<<< HEAD
        dataset, batch_size=1, shuffle=True, num_workers=10)
=======
        dataset, batch_size=1, shuffle=True, num_workers=1)
>>>>>>> 7aff9dbb1b60b0f8cf9743f8c74395e6e948b4fe
    # dataloader = DataLoader(vqa_dataset, batch_size=100)
    print(dataloader)
    # sample = vqa_dataset[1]
    # print(sample)

    for i_batch, sample_batched in enumerate(dataloader):
<<<<<<< HEAD
        print(i_batch, torch.max(
            sample_batched[1]), torch.min(sample_batched[1]))

=======
        img, seg_GT, depth_GT = sample_batched
        print(img.shape)
        print(depth_GT.min())
        print(depth_GT.max())
        print(seg_GT.shape)
        print(seg_GT.max())
        print(seg_GT.min())
        print(depth_GT.shape)
        if(i_batch == 10):
            break
>>>>>>> 7aff9dbb1b60b0f8cf9743f8c74395e6e948b4fe
    # print(i, sample['image'].shape, sample['landmarks'].shape)
