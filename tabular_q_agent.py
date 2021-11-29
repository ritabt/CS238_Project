import collections
import pickle
import random
from environment import Environment

Q_SAVE_JSON_NAME = "./saved_Q.pkl"
# after 50 steps, check whether that car is still stationary. If that happens, break
MAX_NUM_STEPS_PER_EPISODE_STATIONARY = 100

DEBUG = False
GAMMA = 0.9
ALPHA = 0.2

TOTAL_ITER_COUNT = 10000
EPISODES_NUM = 10

PLAY_NUM = 10

class TabQAgent:
    def __init__(self):
        self.env = Environment()
        self.s = self.env.reset()
        self.Q = collections.defaultdict(float)
        for s in range(13*13*13*13*7):
            for a in range(self.env.num_accel_actions * self.env.num_steer_actions):
                    self.Q[(s, a)] = -1e8

    def save_Q(self, Q):
        with open(Q_SAVE_JSON_NAME, 'wb') as f:
            pickle.dump(Q, f)

    def load_Q(self):
        with open(Q_SAVE_JSON_NAME, "rb") as f:
            Q = pickle.load(f)

        return Q

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

    def train(self):
        agent = self

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
                agent.save_Q(agent.Q)

            if iteration_count % 50 == 0:
                print("Current iteration: %d best reward: %f" % (iteration_count, best_reward))

    def play(self):
        agent = self
        print("Loading the saved Q into the agent...")
        agent.Q = agent.load_Q()
        total_reward = 0.0
        for i in range(PLAY_NUM):
            reward = agent.play_episode(True)
            print("Play: %d reward %f" % (i, reward))
            total_reward += reward

        total_reward /= PLAY_NUM
        print("Done playing with avg. reward %f" % total_reward)