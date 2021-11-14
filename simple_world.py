import numpy as np
import carlo 
import time 
import RL

DEBUG = True
# how fast does it refresh
dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = carlo.World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# Let's add some sidewalks and RectangleBuildings.
# A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
# A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
# For both of these objects, we give the center point and the size.

# figure out wrapper to discretize position + heading

# Right Side
w.add(carlo.Painting(carlo.Point(105, 60), carlo.Point(30, 120), 'gray80')) # We build a sidewalk.
w.add(carlo.RectangleBuilding(carlo.Point(110, 60), carlo.Point(25, 120), '#71C671')) # We add greenspace/building.

# Left Side
w.add(carlo.Painting(carlo.Point(15, 60), carlo.Point(30, 120), 'gray80')) # We build a sidewalk.
w.add(carlo.RectangleBuilding(carlo.Point(10, 60), carlo.Point(25, 120), '#71C671')) # We add greenspace/building.

# Road Lines
for i in range(20):
	w.add(carlo.Painting(carlo.Point(50, 10 + i*13), carlo.Point(1, 5), 'white'))
	w.add(carlo.Painting(carlo.Point(70, 10 + i*13), carlo.Point(1, 5), 'white'))

# Obstacles - No obstacles for now
# w.add(carlo.RectangleBuilding(carlo.Point(40, 70), carlo.Point(15, 2), '#FF6103'))
# w.add(carlo.RectangleBuilding(carlo.Point(40, 77), carlo.Point(15, 2), '#FF6103'))

goal = carlo.Painting(carlo.Point(60, 100), carlo.Point(40, 1), 'green')

w.add(goal) # We build a goal.

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
# Red Car
c1 = carlo.Car(carlo.Point(40,10), np.pi/2)
w.add(c1)
c1.set_control(0, 0.55)

# Blue Car - Only 1 car for now
# c2 = carlo.Car(carlo.Point(60,10), np.pi/2, 'blue')
# w.add(c2)
# c2.set_control(0, 0.35)


Pos = RL.DiscretePos(120, 120, 12, 12)
Heading = RL.DiscreteHeading(6)
Acceleration = RL.DiscreteAction(-1, 1, 10)
Steering = RL.DiscreteAction(0, 2*np.pi, 12)

while True:
	w.render()
	w.tick()
	time.sleep(dt/4)

	RedCarState = RL.State(c1, Pos, Heading, goal)
	RedCarAction = RL.Action(c1, Acceleration, Steering)
	curr_reward = RL.get_reward(RedCarState, RedCarAction, w)

	if DEBUG:
		print("Red car position: ", c1.center)
		print("Red car heading: ", c1.heading)
		print("Red car acceleration: ", c1.inputAcceleration)
		print("Discrete Red car position: ", Pos.discretize(c1.center.x, c1.center.y))
		print("Discrete Red car heading: ", Heading.discretize(c1.heading))
		print("Discrete Red car acceleration: ", Acceleration.discretize(c1.inputAcceleration))
		print("Linear Red Car State: ", RedCarState.linearize())

	if curr_reward > 500:
		print('Red car reached goal...')
		break
	# if w.collision_exists(c2):
	# 	print('Blue car collided with something...')


