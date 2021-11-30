#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from tabular_q_agent import TabQAgent
from neural_q_agent import NeuralQAgent

if __name__ == "__main__":
    #agent = TabQAgent()
    agent = NeuralQAgent(state_size=3, action_size=2, seed=0)
    scores = agent.train()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    plt.show()
    #agent.play()