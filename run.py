#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from tabular_q_agent import TabQAgent
from neural_q_agent import NeuralQAgent

def train():
    agent = NeuralQAgent(seed=0)
    scores = agent.train()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    plt.show()

def play():
    agent = NeuralQAgent(seed=0)
    agent.play()

if __name__ == "__main__":
    #train()
    play()