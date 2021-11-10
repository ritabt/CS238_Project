import numpy as np


class DiscretePos():
	def __init__(self, world_width, world_height, width_bins, height_bins):
		self.world_width = world_width
		self.world_height = world_height
		self.width_bins = width_bins
		self.height_bins = height_bins

	def snap_to_grid(self, pos, dim, num_bins):
		step_size = float(dim/num_bins)
		bin_idx = pos//step_size
		lower = bin_idx * step_size
		higher = (bin_idx+1) * step_size
		if abs(pos-lower) <= abs(pos-higher):
			return lower
		else:
			return higher

	def discretize(self, pos_x, pos_y):
		# Turn space into grid and snap position to closest grid spot
		pos_x = self.snap_to_grid(pos_x, self.world_width, self.width_bins)
		pos_y = self.snap_to_grid(pos_y, self.world_height, self.height_bins)
		return pos_x, pos_y

class DiscreteHeading():
	def __init__(self, num_bins, low=0, high=2*np.pi):
		self.num_bins = num_bins
		self.low = low
		self.high = high
		self.step_size = float((high - low)/num_bins)

	def discretize(self, alpha):
		alpha = alpha%(2*np.pi)
		bin_idx = alpha//self.step_size
		lower = bin_idx * self.step_size
		higher = (bin_idx+1) * self.step_size

		if abs(alpha-lower) <= abs(alpha-higher):
			return lower
		else:
			return higher

class DiscreteAction():
	def __init__(self, low, high, num_bins):
		self.low = low
		self.high = high
		self.num_bins = num_bins
		self.step_size = float((high - low)/num_bins)

	def discretize(self, val):
		bin_idx = val//self.step_size
		lower = bin_idx * self.step_size
		higher = (bin_idx+1) * self.step_size

		if abs(val-lower) <= abs(val-higher):
			return lower
		else:
			return higher
