import numpy as np

class State():

	def __init__(self, Car, Pos, Heading, GoalPos):
		self.Pos = Pos
		self.car = Car
		self.goal_pos = GoalPos
		self.Heading = Heading
		self.pos_idx = Pos.discretize(Car.center.x, Car.center.y)
		self.h_idx = Heading.discretize(Car.heading)
		self.goal_pos_idx = Pos.discretize(GoalPos.center.x, GoalPos.center.y)

	def get_num_states(self):
		output = self.Heading.num_bins + 1
		output *= (self.Pos.width_bins + 1) 
		output *= (self.Pos.height_bins + 1)
		return output

	def vectorize(self):
		return np.array([self.pos_idx, self.h_idx, self.goal_pos_idx])

	def linearize(self):
		# start by adding position
		output = self.pos_idx

		# shift by heading num possible vals and add heading
		h_num_vals = self.Heading.num_bins + 1
		output = output*h_num_vals + self.h_idx

		goal_pose_num_vals = (self.Pos.width_bins + 1) * (self.Pos.height_bins + 1)
		output = output*goal_pose_num_vals + self.goal_pos_idx

		return int(output)

		