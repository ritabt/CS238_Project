import random
import collections
import numpy as np
import carlo
import time
import RL
import pickle

GAMMA = 1.0
MAX_DEPTH = 3

def make_index_to_action(acceleration_discretizer, steering_discretizer):
    index_to_action = {}
    # loop over all accel commands
    for acc in acceleration_discretizer.vals:
        # loop over all steer commands
        for theta in steering_discretizer.vals:
            # build a car object, state doesn't matter can be anything
            car = carlo.Car(carlo.Point(40, 10), np.pi / 2)
            car.set_control(theta, acc)
            car_action = RL.Action(car, acceleration_discretizer,
                             steering_discretizer)
            index_to_action[car_action.linearize()] = car_action
    return index_to_action

def calculateAndExecuteBestActionForwardSearch(car: Car,  Pos: DiscretePos,
	Heading: DiscreteHeading, Acceleration: DiscreteAction,
	Steering: DiscreteAction, world: carlo.World, index_to_action: Dict,
	goal, dt)
	# Step 1: Get index of current state and action (discrete)
	# Step 2: Build transition model function T (s', s|a). Get closest discrete state
	# Step 3: 
	(best_u, best_a) = runForwardSearch(car, index_to_action, MAX_DEPTH)
    steering_idx = int(best_a.st_idx)
    accel_idx = int(best_a.acc_idx)

    car.set_control(best_a.Steering.get_val(steering_idx),
                         best_a.Acceleration.get_val(accel_idx))
	

def runForwardSearch(car, index_to_action, d):
	if d == 0:
		return (None, 0)

	best = (None, -Inf)
	car_state = RL.State(car, Pos,
                             Heading, goal)
	#Iterate over all actions
	for action_index in index_to_action.keys():
		action = index_to_action[action_index]
		new_car = carlo.Car(car.center, car.heading)
		steering_idx = int(action.st_idx)
        accel_idx = int(action.acc_idx)
        new_car.set_control(action.Steering.get_val(steering_idx),
                             action.Acceleration.get_val(accel_idx))
		# deterministically update the AVs position
		# TODO: Add step size
		new_car.tick(dt)
		new_car_state = RL.State(new_car, Pos,
                             Heading, goal)
		# get index for AVs new position
		u = RL.get_reward(car_state, car_action, world) + GAMMA * 
			runForwardSearch(new_car, index_to_action, d - 1):
		if u > best[1]:
			best = (action, u)
	return best


