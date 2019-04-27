from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch

class ProjectDataset(Dataset):

	def __init__(self, base_dir):
		self.base_dir = base_dir
		self.episodes = os.listdirs(base_dir)
		num_episodes = len(self.episodes)
		episode_len = []
		for episode in self.episodes:



	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		
		return img, seg_GT, depth_GT


	def get_raw_img(self, episode, point_ID):
		file_name = 'CentralRGB_' + str(point_ID) +'.png'
		file_path = os.path.join(self.root_dir,os.path.join(episode,file_name))
		return np.array(Image.open(file_path))

	def get_segmentation_img(self, episode, point_ID):
		file_name = 'CentralSemanticSeg_' + str(point_ID) +'.png'
		file_path = os.path.join(self.root_dir,os.path.join(episode,file_name))
		img = np.array(Image.open(file_path))
		return img[:, :, 0]

	def get_depth_img(self, episode, point_ID):
		file_name = 'CentralDepth_' + str(point_ID) +'.png'
		file_path = os.path.join(self.root_dir,os.path.join(episode,file_name))
		return np.array(Image.open(file_path))