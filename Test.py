import gym
import random
import numpy as np
from statistics import mean, median

def PlayGame(classifier, nb_games = 10, nb_goal_steps = 500):
    env = gym.make('CartPole-v0')
    
    scores = []
    env.reset()
    
    for game in range(nb_games):
        score = 0
        PrevObservation = []
        env.reset()

        for j in range(nb_goal_steps):
            env.render()
            
            if len(PrevObservation) == 0:
                action = random.randrange(0, 2)
            else:
               action = np.argmax(classifier.predict(np.array([PrevObservation, ]))[0])             
            observation, reward, done, info = env.step(action)
            PrevObservation = observation
            score += reward
            if done:
                break
        
        scores.append(score)

        print('Score: \n', score)
    print('Average accepted score: ', mean(scores))