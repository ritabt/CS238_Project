import numpy as np
import torch 

class Action():
	def __init__(self, Car, Acceleration, Steering):
		self.Acceleration = Acceleration
		self.Steering = Steering
		self.acc_idx = Acceleration.discretize(Car.inputAcceleration)
		self.st_idx = Steering.discretize(Car.inputSteering)

	def get_num_actions(self):
		output = self.Acceleration.num_bins
		output *= self.Steering.num_bins
		return output

	def vectorize(self):
		return np.array([self.acc_idx, self.st_idx])

	def linearize(self):
		# start by adding acceleration
		output = self.acc_idx

		# shift by steering num possible vals and add steering
		st_num_vals = self.Steering.num_bins + 1
		output = output*st_num_vals + self.st_idx

		# add 1 to avoid 0 output
		#output += 1
		return int(output)