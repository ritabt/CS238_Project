#!/usr/bin/env python3
import random
import collections
import numpy as np
import carlo
import time
import RL
import pickle


DEBUG = False
Q_SAVE_JSON_NAME = "./saved_Q.pkl"

def save_Q(Q):
    with open(Q_SAVE_JSON_NAME, 'wb') as f:
        pickle.dump(Q, f)

def load_Q():
    with open(Q_SAVE_JSON_NAME, "rb") as f:
        Q = pickle.load(f)

    return Q

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

    goal = carlo.RectangleBuilding(carlo.Point(60, 100), carlo.Point(40, 1), 'green')

    w.add(goal)  # We build a goal.
    return w, goal

class Environment:
    def __init__(self):
        self.w, self.goal = build_world()

        # discretizer
        self.position_discretizer = RL.DiscretePos(120, 120, 12, 12)
        self.heading_discretizer = RL.DiscreteHeading(6)
        self.num_accel_actions = 11
        self.num_steer_actions = 13
        # num_bins = num_actions - 1
        self.acceleration_discretizer = RL.DiscreteAction(-1, 1, self.num_accel_actions - 1)
        self.steering_discretizer = RL.DiscreteAction(-np.pi, np.pi, self.num_steer_actions - 1)

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
        return random.choice(list(self.index_to_action.keys()))

    # one step forward with the given action
    # returns the new state, reward, is_done
    def step(self, action, render):

        car_state = RL.State(self.car, self.position_discretizer,
                             self.heading_discretizer, self.goal)

        # 4. ***Given a linear indexed action, how to command the car?
        # Ideally we store both the linear indexed action and the corresponding action class
        # (maybe in a dict)? Then we can just call car.set_control(steer, accel) to command
        car_action = self.index_to_action[action]
        steering_idx = int(car_action.st_idx)
        accel_idx = int(car_action.acc_idx)

        self.car.set_control(car_action.Steering.get_val(steering_idx),
                             car_action.Acceleration.get_val(accel_idx))

        if render:
            self.w.render()
        self.w.tick()

        #1. Need a “is_done” - we are done if we collide with something (including the goal)
        is_done = (self.w.collision_exists(car_state.car) or car_state.car.collidesWith(car_state.goal_pos)
                  )
        #5. Confirm if this(same comment in the source) is right
        linear_num = car_state.linearize()
        rew = RL.get_reward(car_state, car_action, self.w)
        return linear_num, \
               rew, \
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
        for s in range(13*13*13*13*7):
            for a in range(self.env.num_accel_actions * self.env.num_steer_actions):
                    self.Q[(s, a)] = -1e8

    def sample_env(self):
        a = self.env.action_space_sample()
        old_s = self.s
        new_s, r, is_done = self.env.step(a, False)
        self.s = self.env.reset() if is_done else new_s
        return (old_s, a, r, new_s)

    def best_value_and_action(self, s):
        best_action_value = None
        best_action = []
        for a in range(self.env.action_space_number()):
            action_value = self.Q[(s, a)]

            if best_action_value is None or best_action_value < action_value:
                best_action_value = action_value
                if len(best_action) != 0:
                    best_action = []

                best_action.append(a)
            elif best_action_value == action_value:
                best_action.append(a)

        return best_action_value, random.choice(best_action)

    def value_update(self, s, a, r, s_next):
        best_Q, _ = self.best_value_and_action(s_next)
        new_Q = r + GAMMA * best_Q
        old_Q = self.Q[(s, a)]
        self.Q[(s, a)] = old_Q * (1 - ALPHA) + new_Q * ALPHA

    def play_episode(self, render=False):
        total_reward = 0.0
        s = self.env.reset()
        num_steps = 0
        while True:
            _, a = self.best_value_and_action(s)
            new_s, reward, is_done = self.env.step(a, render)
            num_steps += 1
            #print("reward %f" %  reward)
            #print(is_done)
            total_reward += reward
            if is_done:
                if DEBUG:
                    print("!!!!!!!!!!!!!Done, ending an episode")
                    print(reward)
                break
            if num_steps >= MAX_NUM_STEPS_PER_EPISODE_STATIONARY and self.env.car.speed < 0.1:
                # car is stopped, break with a large neg reward
                if DEBUG:
                    print("Too manuy steps done and currently stopped, ending an episode")
                total_reward -= 10000
                break
            if (DEBUG):
                if (num_steps % 500 == 0):
                    print(num_steps)
            s = new_s

        return total_reward

GAMMA = 0.9
ALPHA = 0.2
EPISODES_NUM = 10
# after 50 steps, check whether that car is still stationary. If that happens, break
MAX_NUM_STEPS_PER_EPISODE_STATIONARY = 100
TOTAL_ITER_COUNT = 10000

def train():
    agent = Agent()

    iteration_count = 0
    best_reward = 0.0
    while iteration_count <= TOTAL_ITER_COUNT:
        iteration_count += 1
        s, a, r, s_next = agent.sample_env()
        agent.value_update(s, a, r, s_next)

        reward = 0.0
        for _ in range(EPISODES_NUM):
            reward += agent.play_episode()
        reward /= EPISODES_NUM
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
            save_Q(agent.Q)

        if iteration_count % 50 == 0:
            print("Current iteration: %d best reward: %f" % (iteration_count, best_reward))

PLAY_NUM = 100
def play():
    agent = Agent()
    print("Loading the saved Q into the agent...")
    agent.Q = load_Q()
    total_reward = 0.0
    for i in range(PLAY_NUM):
        reward = agent.play_episode(True)
        print("Play: %d reward %f" % (i, reward))
        total_reward += reward

    total_reward /= PLAY_NUM
    print("Done playing with avg. reward %f" % total_reward)



if __name__ == "__main__":
    #train()
    play()