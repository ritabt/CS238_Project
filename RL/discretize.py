import numpy as np


class DiscretePos():
	def __init__(self, world_width, world_height, width_bins, height_bins):
		self.world_width = world_width
		self.world_height = world_height
		self.width_bins = width_bins
		self.height_bins = height_bins
		self.x_vals = np.linspace(0, world_width, num=width_bins+1)
		self.y_vals = np.linspace(0, world_height, num=height_bins+1)

	def get_pos(self, out_idx):
		idx_x = out_idx%self.width_bins
		idx_y = out_idx//self.width_bins
		return self.x_vals[idx_x], self.y_vals[idx_y]

	def snap_to_grid(self, pos, dim, num_bins):
		step_size = float(dim/num_bins)
		bin_idx = int(pos//step_size)
		lower = bin_idx * step_size
		higher = (bin_idx+1) * step_size
		if abs(pos-lower) <= abs(pos-higher):
			return bin_idx
		else:
			return bin_idx + 1

	def map_to_idx(self, idx_x, idx_y):
		return idx_x + idx_y * (self.width_bins+1)

	def discretize(self, pos_x, pos_y):
		idx_x = self.snap_to_grid(pos_x, self.world_width, self.width_bins)
		idx_y = self.snap_to_grid(pos_y, self.world_height, self.height_bins)
		output = self.map_to_idx(idx_x, idx_y)
		return output

class DiscreteHeading():
	def __init__(self, num_bins, low=0, high=2*np.pi):
		self.num_bins = num_bins
		self.low = low
		self.high = high
		self.step_size = float((high - low)/num_bins)
		self.vals = np.linspace(low, high, num=num_bins+1)

	def get_val(self, idx):
		return self.vals[idx]

	def discretize(self, alpha):
		alpha = alpha%(2*np.pi)
		bin_idx = alpha//self.step_size
		lower = bin_idx * self.step_size
		higher = (bin_idx+1) * self.step_size

		if abs(alpha-lower) <= abs(alpha-higher):
			return bin_idx
		else:
			return bin_idx+1

class DiscreteAction():
	def __init__(self, low, high, num_bins):
		self.low = low
		self.high = high
		self.num_bins = num_bins
		self.step_size = float((high - low)/num_bins)
		self.vals = np.linspace(low, high, num=num_bins+1)

	def get_val(self, idx):
		return self.vals[idx]

	def discretize(self, val):
		bin_idx = val//self.step_size
		lower = bin_idx * self.step_size
		higher = (bin_idx+1) * self.step_size

		if abs(val-lower) <= abs(val-higher):
			return bin_idx
		else:
			return bin_idx+1
