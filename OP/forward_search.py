import random
import collections
import numpy as np
import carlo
import time
import RL
import pickle

GAMMA = 1.0
MAX_DEPTH = 2

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

def runForwardSearch(car: carlo.Car,  Pos: RL.DiscretePos,
    Heading: RL.DiscreteHeading, world: carlo.World, index_to_action: dict, goal, dt, d):
    if d == 0:
        return (None, 0)

    best = (None, -np.Inf)
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
        # if d > 1 and action.Steering.get_val(steering_idx) < -3:
        #     print("Pos before tick: ", new_car.center)
        #     print("Speed before tick: ", new_car.speed)
        new_car.tick(dt)
        # if d > 1 and action.Steering.get_val(steering_idx) < -3:
        #     print("Pos after tick: ", new_car.center)
        #     print("Speed after tick: ",new_car.speed)
        new_car_state = RL.State(new_car, Pos,
                             Heading, goal)
        # get index for AVs new position
        u = RL.get_reward(car_state, action, world) + GAMMA * runForwardSearch(new_car, Pos, Heading, world, index_to_action, goal, dt, d - 1)[1]
        
        # if d > 1 and action.Steering.get_val(steering_idx) < -3:
        #     print("Accel val: ", action.Acceleration.get_val(accel_idx))
        #     print("Steer val: ", action.Steering.get_val(steering_idx))
        #     print("Curr Reward: ", RL.get_reward(car_state, action, world))
        #     print("Future Reward: ", u - RL.get_reward(car_state, action, world))
        #     print("Future Reward v2: ", RL.get_reward(new_car_state, action, world))
        if u > best[1]:
            best = (action, u)
    return best

def calculateAndExecuteBestActionForwardSearch(car: carlo.Car,  Pos: RL.DiscretePos,
    Heading: RL.DiscreteHeading, world: carlo.World, index_to_action: dict, goal, dt):
    # Step 1: Get index of current state and action (discrete)
    # Step 2: Build transition model function T (s', s|a). Get closest discrete state
    # Step 3: 
    print("Starting search")
    (best_a, best_u) = runForwardSearch(car, Pos, Heading, world, index_to_action, goal, dt, MAX_DEPTH)
    steering_idx = int(best_a.st_idx)
    accel_idx = int(best_a.acc_idx)

    car.set_control(best_a.Steering.get_val(steering_idx),
                         best_a.Acceleration.get_val(accel_idx))
    print(best_a.Acceleration.get_val(accel_idx))
    




