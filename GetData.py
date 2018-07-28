import gym
import random
import numpy as np
from collections import Counter
from statistics import mean, median

env = gym.make('CartPole-v0')
env.reset()

INITIAL_GAMES = 10000
GOAL_STEPS = 500
MIN_SCORE = 50

def Populate():
    training_data = []
    scores = []
    accepted_scores = []
    for episodes in range(INITIAL_GAMES):
        score = 0
        GameMemory = []
        PrevObservation = []
        
        for j in range(GOAL_STEPS):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)
           
            if len(PrevObservation) > 0:
                GameMemory.append([PrevObservation, action])
            
            PrevObservation = observation
            score += reward
            if done:
                break
        
        if score >= MIN_SCORE:
            accepted_scores.append(score)
            #HotEncode data
            for data in GameMemory:
                if data[1]:
                    output = [0, 1]
                else:
                    output = [1, 0]
            
                training_data.append([data[0], output])        
        
        env.reset()
        scores.append(score)
        
    training_data_save = np.array(training_data)
    
    X_train = np.array([data[0] for data in training_data])
    Y_train = np.array([data[1] for data in training_data])
    
    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))

    return (X_train, Y_train)