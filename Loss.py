import torch
import torch.nn as nn

class MultiLossLayer(nn.Module):
	def __init__(self):
		super(MultiLossLayer, self).__init__()
		self.num_outputs = 2
		self.seg_sigma = nn.Parameter(torch.rand(1))
		self.depth_sigma = nn.Parameter(torch.rand(1))
		self.seg_loss = nn.CrossEntropyLoss()

	def forward(self, seg_out, depth_out, seg_GT, depth_GT, writer, curr_step):
		seg_loss = (1 / torch.exp(self.seg_sigma)) * self.seg_loss(seg_out, seg_GT)
		depth_loss = (1 / (2 * torch.exp(self.depth_sigma))) * self.depth_loss(depth_out, depth_GT)
		sigma_reg = self.seg_sigma + self.depth_sigma
		final_loss = seg_loss + depth_loss + sigma_reg

		writer.add_scalar("Loss/Segmentation", seg_loss, curr_step)
		writer.add_scalar("Loss/Depth", depth_loss, curr_step)
		writer.add_scalar("Loss/Reg", sigma_reg, curr_step)
		writer.add_scalar("Loss/Final", final_loss, curr_step)
		writer.add_scalar("Uncertainty/Depth", self.depth_sigma, curr_step)
		writer.add_scalar("Uncertainty/Segmentation", self.seg_sigma, curr_step)

		return final_loss

	def depth_loss(depth_out, depth_GT):
		pred = 1 / depth_out
		GT = 1 / depth_GT
		cum_loss = torch.sum(torch.abs(pred - GT))
		loss = cum_loss.mean()

		return loss


class Decoder(nn.Module):

	def __init__(self,num_classes):
		super(Decoder, self).__init__()
		self.num_classes = num_classes
		self.seg_decoder = nn.Sequential(nn.Conv2d(256, 256, 1),
										nn.BatchNorm2d(256),
										nn.ReLU(),
										nn.Conv2d(256, num_classes, 1))
		self.depth_decoder = nn.Sequential(nn.Conv2d(256, 256, 1),
										nn.BatchNorm2d(256),
										nn.ReLU(),
										nn.Conv2d(256, 1, 1))
		self.context_decoder = nn.Sequential(nn.Conv2d(num_classes + 1, 256, 3, padding = 1),
										nn.BatchNorm2d(256),
										nn.ReLU(),
										nn.Conv2d(256, num_classes + 1, 1))

	def forward(self, encoder_output):
		seg_out = self.seg_decoder(encoder_output)
		depth_out = self.depth_decoder(encoder_output)
		
		return seg_out, depth_out

	def context_forward(self, encoder_output):
		seg_out = self.seg_decoder(encoder_output)
		depth_out = self.depth_decoder(encoder_output)
		
		context_out = torch.cat((depth_out, seg_out), dim = 1)
		context_out = self.context_decoder(context_out)
		
		seg_out = context_out[:,:self.num_classes,:,:]
		depth_out = context_out[:,-1:,:,:]

		return seg_out, depth_out
