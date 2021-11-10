class State():
	def __init__(self, Car, Pos, Heading, Acceleration, Steering):
		self.Pos = Pos
		self.Heading = Heading
		self.Acceleration = Acceleration
		self.Steering = Steering
		self.pos_idx = Pos.discretize(Car.center.x, Car.center.y)
		self.h_idx = Heading.discretize(Car.heading)
		self.acc_idx = Acceleration.discretize(Car.inputAcceleration)
		self.st_idx = Steering.discretize(Car.inputSteering)

	def linearize(self):
		# start by adding position - add 1 to avoid 0 output
		output = 1 + self.pos_idx

		# shift by acceleration num possible vals and add acc
		acc_num_vals = self.Acceleration.num_bins + 1
		output = output*acc_num_vals + self.acc_idx

		# shift by steering num possible vals and add steering
		st_num_vals = self.Steering.num_bins + 1
		output = output*st_num_vals + self.st_idx

		# shift by heading num possible vals and add heading
		h_num_vals = self.Heading.num_bins + 1
		output = output*h_num_vals + self.h_idx

		return int(output)

		