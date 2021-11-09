import numpy as np
import carlo 
import time 

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = carlo.World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# Let's add some sidewalks and RectangleBuildings.
# A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
# A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
# For both of these objects, we give the center point and the size.

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

# Obstacles
w.add(carlo.RectangleBuilding(carlo.Point(40, 70), carlo.Point(15, 2), '#FF6103'))
w.add(carlo.RectangleBuilding(carlo.Point(40, 77), carlo.Point(15, 2), '#FF6103'))

goal = carlo.Painting(carlo.Point(60, 100), carlo.Point(40, 1), 'green')

w.add(goal) # We build a goal.

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = carlo.Car(carlo.Point(40,10), np.pi/2)
w.add(c1)
c1.set_control(0.0, 0.55)
# TODO: make this a constant size to run faster
r1 = []

c2 = carlo.Car(carlo.Point(60,10), np.pi/2, 'blue')
w.add(c2)
c2.set_control(0, 0.35)
while True:
	w.render()
	w.tick()
	time.sleep(dt/4)

	c1_collided = w.collision_exists(c1)
	c2_collided = w.collision_exists(c2)
	c1_reached_goal = c1.collidesWith(goal)
	r1.append(c1.get_reward(c1_collided, c1_reached_goal))

	print(c1.center)

	print("Reward: ", c1.get_reward(c1_collided, c1_reached_goal))

	# print("Red car position: ", c1.center)
	if c1_collided:
		print('Red car collided with something...')
	if c2_collided:
		print('Blue car collided with something...')
	if c1_reached_goal:
		print('Red car reached goal')



