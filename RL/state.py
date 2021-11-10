class State():
	def __init__(self, Car, Pos, Heading, Acceleration, Steering):
		self.Pos = Pos
		self.pos_x, self.pos_y = Pos.discretize(Car.center.x, Car.center.y)
		self.heading = Heading.discretize(Car.heading)
		self.acceleration = Acceleration.discretize(Car.inputAcceleration)
		self.steering = Steering.discretize(Car.inputSteering)

	def linearize(self):
		'''
			- range of acceleration = [-1, 1]
			- range of steering = [0, 2*pi] = [0, 6.3]
			- range of heading = [0, 2*pi] = [0, 6.3]
			- range of pos_x = [0, 120] = [0, Pos.world_width]
			- range of pos_y = [0, 120] = [0, Pos.world_height]

			Method: shift each state value and add everything together 
				    to create a unique value for each state
		'''
		
		# counting digits from the right
		digit_1 = 1 + self.acceleration
		digit_2_3 = 100*self.steering
		digit_4_5 = 10000*self.heading
		pos = self.pos_x + 1000*self.pos_y
		digit_6_11 = pos*(10^6)
		return digit_1 + digit_2_3 + digit_4_5 + digit_6_11
		