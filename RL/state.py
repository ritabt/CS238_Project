class State():
	def __init__(self, Car, Pos, Heading, Acceleration, Steering):
		self.Pos = Pos
		self.Heading = Heading
		self.pos_idx = Pos.discretize(Car.center.x, Car.center.y)
		self.h_idx = Heading.discretize(Car.heading)

	def linearize(self):
		# start by adding position - add 1 to avoid 0 output
		output = 1 + self.pos_idx

		# shift by heading num possible vals and add heading
		h_num_vals = self.Heading.num_bins + 1
		output = output*h_num_vals + self.h_idx

		return int(output)

		