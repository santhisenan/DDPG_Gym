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

from utils import *

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
LR_DECAY = 1
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
                               lr_decay=LR_DECAY,
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
    
    # np.set_printoptions(threshold=np.nan)

    replay_memory = Memory(REPLAY_MEM_CAPACITY)

    # Tensorflow part starts here!
    tf.reset_default_graph()

    # Placeholders
    state_placeholder = tf.placeholder(dtype=tf.float32, \
                                       shape=(None, STATE_DIM))
    action_placeholder = tf.placeholder(dtype=tf.float32, \
                                        shape=(None, ACTION_DIM))
    reward_placeholder = tf.placeholder(dtype=tf.float32)
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
        # unscaled_actions = actor.model.predict(state_placeholder, steps=1)
        unscaled_actions = actor.predict_actions(state_placeholder, steps=1)
        actions = scale_actions(unscaled_actions, env.action_space.low, 
                                      env.action_space.low)
    
    with tf.variable_scope('target_actor'):
        unscaled_actions = target_actor.model.predict(state_placeholder)
        actions_target = scale_actions(unscaled_actions, 
                                             env.action_space.low,
                                             env.action_space.low)
        actions_target = tf.stop_gradient(actions_target)

    with tf.variable_scope('critic'):
        q_values_of_given_actions = critic.model.predict( \
            tf.concat([state_placeholder, action_placeholder], axis=1))

        q_values_of_suggested_actions = critic.model.predict( \
            tf.concat([state_placeholder, actions], axis=1))

    with tf.variable_scope('target_critic'):
        next_target_q_values = target_critic.model.predict(tf.concat( \
            [next_state_placeholder, actions_target], axis=1))
        next_target_q_values = tf.stop_gradient(next_target_q_values)

    # Isolate variables in each network
    actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                     scope='actor')
    target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                     scope='target_actor')
    critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='critic')
    target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='target_critic')

    targets = tf.expand_dims(reward_placeholder, 1) + \
        tf.expand_dims(is_not_terminal_placeholder, 1) * GAMMA * \
            next_target_q_values
    td_errors = targets - q_values_of_given_actions
    critic_loss = tf.reduce_mean(tf.square(td_errors))

    # Update critic networks after computing loss
    for var in critic_vars:
        if not 'bias' in var.name:
            critic_loss += L2_REG_CRITIC * 0.5 * tf.nn.l2_loss(var)

    # optimize critic
    critic_train_op = tf.train.AdamOptimizer(LEARNING_RATE_CRITIC * LR_DECAY **
                                             episodes).minimize(critic_loss)
    
    # Actor's loss
    actor_loss = -1 * tf.reduce_mean(q_values_of_suggested_actions)
    for var in actor_vars:
        if not 'bias' in var.name:
            actor_loss += L2_REG_ACTOR * 0.5 * tf.nn.l2_loss(var)

    # Optimize actor
    actor_train_op = tf.train.AdamOptimizer(LEARNING_RATE_ACTOR * LR_DECAY ** 
        episodes).minimize(actor_loss, var_list=actor_vars)

    # Init session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Training
    num_steps = 0
    for episode in range(NUM_EPISODES):
        total_reward = 0
        num_steps_in_episode = 0

        # Create noise
        noise = np.zeros(ACTION_DIM)
        noise_scale = (INITIAL_NOISE_SCALE * NOISE_DECAY ** episode) * \
            (env.action_space.high - env.action_space.low) #TODO: uses env
        
        # Initial state
        state = env.reset() #TODO: uses env

        for t in range(MAX_STEPS_PER_EPISODE):
            env.render()
            
            print("State" + str(state[None]))
            # Choose an action
            action = sess.run(actions, feed_dict={ \
                state_placeholder: state[None],
                is_training_placeholder: False})

            # Add Noise to actions
            noise = EXPLORATION_THETA * (EXPLORATION_MU - noise) + \
                EXPLORATION_SIGMA * np.random.randn(ACTION_DIM)

            action += noise_scale * noise

            # Take action on env
            next_state, reward, done, _info = env.step(action) #TODO: uses env
            total_reward += reward
            replay_memory.insert(state, action, reward, done, next_state)

            if num_steps % TRAIN_EVERY == 0 and \
                replay_memory.size() >= MINI_BATCH_SIZE :
                minibatch = replay_memory.sample_batch(MINI_BATCH_SIZE)
                _, _ = sess.run([critic_train_op, actor_train_op], \
                    feed_dict={
                        state_placeholder: np.asarray(
                            [e[0] for e in minibatch]),

                        action_placeholder: np.asarray(
                            [e[1] for e in minibatch]),

                        reward_placeholder: np.asarray(
                            [e[2] for e in minibatch]),

                        next_state_placeholder: np.asarray(
                            [e[3] for e in minibatch]),

                        is_not_terminal_placeholder: np.asarray(
                            [e[4] for e in minibatch]),

                        is_training_placeholder: True
                })
                update_target_networks(sess, TAU, 
                                       target_actor_vars, actor_vars, 
                                       target_critic_vars, critic_vars)

            state = next_state
            num_steps += 1
            num_steps_in_episode += 1

            if done:
                _ = sess.run(episode_incr_op)
                break     

        print(str((episode, total_reward, num_steps_in_episode, noise_scale)))

    write_to_file('info.json', json.dumps(info))
    env.close()
    # gym.upload(OUTPUT_DIR)
            
main()


