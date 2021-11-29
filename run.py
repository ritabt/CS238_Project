#!/usr/bin/env python3
from tabular_q_agent import TabQAgent
from neural_q_agent import NeuralQAgent

if __name__ == "__main__":
    #agent = TabQAgent()
    agent = NeuralQAgent(0)
    #agent.train()
    #agent.play()