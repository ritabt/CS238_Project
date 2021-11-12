#!/usr/bin/env python3
import gym
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
    return w

class Environment:
    def __init__(self):
        self.w = build_world()
        self.car = carlo.Car(carlo.Point(40, 10), np.pi / 2)
        self.w.add(self.car)
        self.car.set_control(0, 0.55)

    # reset the world and restart
    def reset(self):
        # how do we reset the car?
        # return the rested car state
        return 0

    # sample an action randomly from the action space
    def action_space_sample(self):
        return 0

    # one step forward with the given action
    def step(self, action):
        return 0, 0, True

    # return the number of the actions available
    def action_space_number(self):
        return 0

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
