import numpy


class Discrete_Pos():
	def __init__(self, world_width, world_height, width_bins, height_bins):
		self.world_width = world_width
		self.world_height = world_height
		self.width_bins = width_bins
		self.height_bins = height_bins

	def discretize(self, pos_x, pos_y):
		# Turn space into grid and snap position to closest grid spot
		return pos_x, pos_y

class Discrete_Heading():
	def __init__(self, num_bins):
		self.num_bins = num_bins

	def discretize(self, alpha):
		# Use same method as below but with low 0 and high 2*pi
		return alpha

class Discrete_Action():
	def __init__(self, low, high, n_bins):
		self.low = low
		self.high = high
		self.n_bins = n_bins

	def discretize(self, val):
		# snap value to correct bin
		return val
