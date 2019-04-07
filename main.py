import tensorflow as tf 
import gym
from gym import wrappers
import numpy as np 
import json, sys, os
from os import path
import random 

from replay_buffer import ReplayBuffer as Memory 
from actor_network import ActorNetwork
from critic_network import CriticNetwork

import utils

GAMMA = 0.99
ENV_NAME = 'Pendulum-v0'
HIDDEN_1_ACTOR = 8
HIDDEN_2_ACTOR = 8
HIDDEN_3_ACTOR = 8
HIDDEN_1_CRITIC = 8
HIDDEN_2_CRITIC = 8
HIDDEN_3_CRITIC = 8
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3 #TODO
L2_DECAY = 1
L2_REG_ACTOR = 1e-6
L2_REG_CRITIC = 1e-6
DROPOUT_ACTOR = 0
DROPOUT_CRITIC = 0
NUM_EPISODES = 15000
MAX_STEPS_PER_EPISODE = 10000
TAU = 1e-2
TRAIN_EVERY = 1 #TODO add doc
REPLAY_MEM_CAPACITY = int(1e5)
MINI_BATCH_SIZE = 1024 #TODO
INITIAL_NOISE_SCALE = 0.1
NOISE_DECAY = 0.99
EXPLORATION_MU = 0.0
EXPLORATION_THETA = 0.15
EXPLORATION_SIGMA = 0.2
STATE_DIM = 3
ACTION_DIM = 1
OUTPUT_DIR = "output"


def write_to_file(file_name, s):
    with open(path.join(OUTPUT_DIR, file_name), 'w') as fh: 
        fh.write(s)

def main():
    env = gym.make(ENV_NAME)
    
    env.seed(0)
    np.random.seed(0)

    env = wrappers.Monitor(env, OUTPUT_DIR, force=True)

    info = {}
    info['env_id'] = env.spec.id #TODO
    info['parameters'] = dict( gamma=GAMMA,
                               h1_actor=HIDDEN_1_ACTOR,
                               h2_actor=HIDDEN_2_ACTOR,
                               h3_actor=HIDDEN_3_ACTOR,
                               h1_critic=HIDDEN_1_CRITIC,
                               h2_critic=HIDDEN_2_CRITIC,
                               h3_critic=HIDDEN_3_CRITIC,
                               lr_actor=LEARNING_RATE_ACTOR,
                               lr_critic=LEARNING_RATE_CRITIC,
                               lr_decay=L2_DECAY,
                               l2_reg_actor=L2_REG_ACTOR,
                               l2_reg_critic=L2_REG_CRITIC,
                               dropout_actor=DROPOUT_ACTOR,
                               dropout_critic=DROPOUT_CRITIC,
                               num_episodes=NUM_EPISODES,
                               max_steps_ep=MAX_STEPS_PER_EPISODE,
                               tau=TAU,
                               train_every=TRAIN_EVERY,
                               replay_memory_capacity=REPLAY_MEM_CAPACITY,
                               minibatch_size=MINI_BATCH_SIZE,
                               initial_noise_scale=INITIAL_NOISE_SCALE,
                               noise_decay=NOISE_DECAY,
                               exploration_mu=EXPLORATION_MU,
                               exploration_theta=EXPLORATION_THETA,
                               exploration_sigma=EXPLORATION_SIGMA)
    
    np.set_printoptions(threshold=np.nan)

    replay_memory = Memory(REPLAY_MEM_CAPACITY)

    # Tensorflow part starts here!
    tf.reset_default_graph()

    # Placeholders
    state_placeholder = tf.placeholder(dtype=tf.float32, \
                                       shape=(None, STATE_DIM))
    action_placeholder = tf.placeholder(dtype=tf.float32, \
                                        shape=(None, ACTION_DIM))
    reward = tf.placeholder(dtype=tf.float32)
    next_state_placeholder = tf.placeholder(dtype=tf.float32,
                                       shape=(None, STATE_DIM))
    is_not_terminal_placeholder = tf.placeholder(dtype=tf.float32)
    is_training_placeholder = tf.placeholder(dtype=tf.float32, shape=())

    # Episode counter
    episodes = tf.Variable(0.0, trainable=False, name='episodes')
    episode_incr_op = episodes.assign_add(1)

    actor = ActorNetwork(STATE_DIM, ACTION_DIM, 
                         HIDDEN_1_ACTOR, HIDDEN_2_ACTOR, HIDDEN_3_ACTOR,
                         DROPOUT_ACTOR)
    target_actor = ActorNetwork(STATE_DIM, ACTION_DIM,
                                HIDDEN_1_ACTOR, HIDDEN_2_ACTOR, HIDDEN_3_ACTOR,
                                DROPOUT_ACTOR)
    critic = CriticNetwork(STATE_DIM, ACTION_DIM,
                           HIDDEN_1_CRITIC, HIDDEN_2_CRITIC, HIDDEN_3_CRITIC,
                           DROPOUT_CRITIC)
    target_critic = CriticNetwork(STATE_DIM, ACTION_DIM,
                                  HIDDEN_1_CRITIC, HIDDEN_2_CRITIC, 
                                  HIDDEN_3_CRITIC, DROPOUT_CRITIC)

    with tf.variable_scope('actor'):
        unscaled_actions = actor.model.predict(state_placeholder)
        actions = utils.scale_actions(unscaled_actions, env.action_space.low, 
                                      env.action_space.low)
    
    with tf.variable_scope('target_actor'):
        unscaled_actions = target_actor.model.predict(state_placeholder)
        actions_target = utils.scale_actions(unscaled_actions, 
                                             env.action_space.low,
                                             env.action_space.low)
        target_next_actions = tf.stop_gradient(actions_target)

    with tf.variable_scope('critic'):
        q_values_of_given_actions = critic.model.predict( \
            tf.concat([state_placeholder, action_placeholder], axis=1))

        q_values_of_suggested_actions = critic.model.predict( \
            tf.concat([state_placeholder, actions], axis=1))

    with tf.variable_scope('target_critic'):
        next_target_q_values = tf.stop_gradient(tf.concat( \
            [next_state_placeholder, target_next_actions], axis=1))






                        



main()


