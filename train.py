#!/usr/bin/env python3
import gym
import random
import collections
import numpy as np
import carlo
import time
import RL

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2
EPISODES_NUM = 20

def build_world():
    # how fast does it refresh
    dt = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.
    w = carlo.World(dt, width=120, height=120, ppm=6)  # The world is 120 meters by 120 meters. ppm is the pixels per meter.

    # Let's add some sidewalks and RectangleBuildings.
    # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
    # A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
    # For both of these objects, we give the center point and the size.

    # figure out wrapper to discretize position + heading

    # Right Side
    w.add(carlo.Painting(carlo.Point(105, 60), carlo.Point(30, 120), 'gray80'))  # We build a sidewalk.
    w.add(carlo.RectangleBuilding(carlo.Point(110, 60), carlo.Point(25, 120), '#71C671'))  # We add greenspace/building.

    # Left Side
    w.add(carlo.Painting(carlo.Point(15, 60), carlo.Point(30, 120), 'gray80'))  # We build a sidewalk.
    w.add(carlo.RectangleBuilding(carlo.Point(10, 60), carlo.Point(25, 120), '#71C671'))  # We add greenspace/building.

    # Road Lines
    for i in range(20):
        w.add(carlo.Painting(carlo.Point(50, 10 + i * 13), carlo.Point(1, 5), 'white'))
        w.add(carlo.Painting(carlo.Point(70, 10 + i * 13), carlo.Point(1, 5), 'white'))

    # Obstacles - No obstacles for now
    # w.add(carlo.RectangleBuilding(carlo.Point(40, 70), carlo.Point(15, 2), '#FF6103'))
    # w.add(carlo.RectangleBuilding(carlo.Point(40, 77), carlo.Point(15, 2), '#FF6103'))

    goal = carlo.Painting(carlo.Point(60, 100), carlo.Point(40, 1), 'green')

    w.add(goal)  # We build a goal.
    return w, goal

class Environment:
    def __init__(self):
        self.w, self.goal = build_world()
        # these moved to reset and should be ok to remove from here
        #self.car = carlo.Car(carlo.Point(40, 10), np.pi / 2)
        #self.w.add(self.car)
        #self.car.set_control(0, 0.55)

        # discretizer
        self.position_discretizer = RL.DiscretePos(120, 120, 12, 12)
        self.heading_discretizer = RL.DiscreteHeading(6)
        self.num_accel_actions = 11
        self.num_steer_actions = 13
        # num_bins = num_actions - 1
        self.acceleration_discretizer = RL.DiscreteAction(-1, 1, self.num_accel_actions - 1)
        self.steering_discretizer = RL.DiscreteAction(0, 2 * np.pi, self.num_steer_actions - 1)

        # dictionary which maps linearlized action state index to an RL.Action object
        self.index_to_action = self.make_index_to_action()

    def make_index_to_action(self):
        index_to_action = {}
        # loop over all accel commands
        for acc in self.acceleration_discretizer.vals:
            # loop over all steer commands
            for theta in self.steering_discretizer.vals:
                # build a car object, state doesn't matter can be anything
                car = carlo.Car(carlo.Point(40, 10), np.pi / 2)
                car.set_control(theta, acc)
                car_action = RL.Action(car, self.acceleration_discretizer,
                                 self.steering_discretizer)
                index_to_action[car_action.linearize()] = car_action
        return index_to_action

    # reset the world and restart
    def reset(self):
        self.w.reset()
        self.car = carlo.Car(carlo.Point(40, 10), np.pi / 2)
        self.w.add(self.car)
        self.car.set_control(0, 0.55)

        # return the car state
        car_state = RL.State(self.car, self.position_discretizer,
                             self.heading_discretizer, self.goal)
        return car_state.linearize()

    # sample an action randomly from the action space
    def action_space_sample(self):
        # 3. Need a function returning a random action
        # should this be an action class or the linearized index?
        return random.choice(self.index_to_action.keys())

    # one step forward with the given action
    # returns the new state, reward, is_done
    def step(self, action):

        car_state = RL.State(self.car, self.position_discretizer,
                             self.heading_discretizer, self.goal)
        car_action = RL.Action(self.car, self.acceleration_discretizer,
                                 self.steering_discretizer)

        # 4. ***Given a linear indexed action, how to command the car?
        # Ideally we store both the linear indexed action and the corresponding action class
        # (maybe in a dict)? Then we can just call car.set_control(steer, accel) to command

        self.w.render()
        self.w.tick()

        #1. Need a “is_done” - we are done if we collide with something (including the goal)
        is_done = self.w.collision_exists(car_state.car)
        #5. Confirm if this(same comment in the source) is right
        return car_state.linearize(), \
               RL.get_reward(car_state, car_action, self.w), \
               is_done

    # return the number of the actions available
    def action_space_number(self):
        # 2. Need a function returning the number of actions available
        return self.num_accel_actions * self.num_steer_actions

class Agent:
    def __init__(self):
        self.env = Environment()
        self.s = self.env.reset()
        self.Q = collections.defaultdict(float)

    def sample_env(self):
        a = self.env.action_space_sample()
        old_s = self.s
        new_s, r, is_done = self.env.step(a)
        self.s = self.env.reset() if is_done else new_s
        return (old_s, a, r, new_s)

    def best_value_and_action(self, s):
        best_action_value, best_action = None, None
        for a in range(self.env.action_space_number()):
            action_value = self.Q[(s, a)]
            if best_action_value is None or best_action_value < action_value:
                best_action_value = action_value
                best_action = a

        if best_action_value == None:
            best_action_value = 0.0

        if best_action == None:
            best_action = 0.0

        return best_action_value, best_action

    def value_update(self, s, a, r, s_next):
        best_Q, _ = self.best_value_and_action(s_next)
        new_Q = r + GAMMA * best_Q
        old_Q = self.Q[(s, a)]
        self.Q[(s, a)] = old_Q * (1 - ALPHA) + new_Q * ALPHA

    def play_episode(self, env):
        total_reward = 0.0
        s = env.reset()
        while True:
            _, a = self.best_value_and_action(s)
            new_s, reward, is_done, _ = env.step(a)
            total_reward += reward
            if is_done:
                break
            s = new_s

        return total_reward

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()

    iteration_count = 0
    best_reward = 0.0
    while True:
        iteration_count += 1
        s, a, r, s_next = agent.sample_env()
        agent.value_update(s, a, r, s_next)

        reward = 0.0
        for _ in range(EPISODES_NUM):
            reward += agent.play_episode(test_env)
        reward /= EPISODES_NUM
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iteration_count)
            break
