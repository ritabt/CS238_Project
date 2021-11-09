#!/usr/bin/env python3
import gym
import collections

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2
EPISODES_NUM = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.s = self.env.reset()
        self.Q = collections.defaultdict(float)

    def sample_env(self):
        a = self.env.action_space.sample()
        old_s = self.s
        new_s, r, is_done, _ = self.env.step(a)
        self.s = self.env.reset() if is_done else new_s
        return (old_s, a, r, new_s)

    def best_value_and_action(self, s):
        best_action_value, best_action = None, None
        for a in range(self.env.action_space.n):
            action_value = self.Q[(s, a)]
            if best_action_value is None or best_action_value < action_value:
                best_action_value = action_value
                best_action = a
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
