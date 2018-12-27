import tensorflow as tf
import numpy as np  
from vizdoom import *

import random                
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore') 

def create_environment():
    game = DoomGame()

    # Richtige Spielconfig laden
    game.load_config("basic.cfg")

    # Richtiges Spielszenario laden
    game.set_doom_scenario_path("basic.wad")

    game.init()

    # Mögliche Actionen
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions


def test_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    episodes = 10

    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(possible_actions)
            print(action)
            reward = game.make_action(action)
            print("\treward: " + reward)
            time.sleep(2)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()
