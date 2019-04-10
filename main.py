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
LEARNING_RATE_CRITIC = 1e-4 #TODO
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
    ''' Create the environment
    '''
    env = gym.make(ENV_NAME)
    
    # env = wrappers.Monitor(env, outdir, force=True)

    env.seed(0)
    np.random.seed(0)

    ''' Create the replay memory
    '''
    replay_memory = Memory(REPLAY_MEM_CAPACITY)

    # Tensorflow part starts here!
    tf.reset_default_graph()

    ''' Create placeholders 
    '''
    # Placeholders
    state_placeholder = tf.placeholder(dtype=tf.float32, \
                                       shape=[None, STATE_DIM])
    action_placeholder = tf.placeholder(dtype=tf.float32, \
                                        shape=(None, ACTION_DIM))
    reward_placeholder = tf.placeholder(dtype=tf.float32, shape=(None))
    next_state_placeholder = tf.placeholder(dtype=tf.float32,
                                       shape=(None, STATE_DIM))
    is_not_terminal_placeholder = tf.placeholder(dtype=tf.float32)
    is_training_placeholder = tf.placeholder(dtype=tf.float32, shape=())

    ''' A counter to count the number of episodes
    '''
    episodes = tf.Variable(0.0, trainable=False, name='episodes')
    episode_incr_op = episodes.assign_add(1)

    ''' Create the actor network inside the actor scope and calculate actions
    '''
    with tf.variable_scope('actor'):
        actor = ActorNetwork(STATE_DIM, ACTION_DIM,
                             HIDDEN_1_ACTOR, HIDDEN_2_ACTOR, HIDDEN_3_ACTOR,
                             trainable=True)
        unscaled_actions = actor.call(state_placeholder)

        ''' Scale the actions to fit within the bounds provided by the 
        environment
        '''
        actions = scale_actions(unscaled_actions, env.action_space.low,
                                env.action_space.high)

    ''' Create the target actor network inside target_actor scope and calculate 
    the target actions. Apply stop_gradient to the target actions so that 
    thier gradient is not taken at any point of time.
    '''
    with tf.variable_scope('target_actor', reuse=False):
        target_actor = ActorNetwork(STATE_DIM, ACTION_DIM,
                                    HIDDEN_1_ACTOR, HIDDEN_2_ACTOR, 
                                    HIDDEN_3_ACTOR,
                                    trainable=True)

        unscaled_target_actions = target_actor.call(next_state_placeholder)
        
        ''' Scale the actions to fit within the bounds provided by the 
        environment
        '''
        target_actions_temp = scale_actions(unscaled_target_actions,
                                       env.action_space.low,
                                       env.action_space.low)
        target_actions = tf.stop_gradient(target_actions_temp)

    ''' Create the critic network inside the critic variable scope. Get the 
    Q-values of given actions and Q-values of actions suggested by the actor 
    network.
    '''
    with tf.variable_scope('critic'):
        critic = CriticNetwork(STATE_DIM, ACTION_DIM,
                               HIDDEN_1_CRITIC, HIDDEN_2_CRITIC, 
                               HIDDEN_3_CRITIC,
                               trainable=True)

        q_values_of_given_actions = critic.call(state_placeholder,
                                                action_placeholder)
        q_values_of_suggested_actions = critic.call(state_placeholder, actions)

    ''' Create the target critic network inside the target_critic variable 
    scope. Calculate the target Q-values and apply stop_gradient to it.
    '''
    with tf.variable_scope('target_critic', reuse=False):
        target_critic = CriticNetwork(STATE_DIM, ACTION_DIM,
                                      HIDDEN_1_CRITIC, HIDDEN_2_CRITIC,
                                      HIDDEN_3_CRITIC, trainable=True)
        
        target_q_values_temp = target_critic.call(next_state_placeholder, 
                                                  target_actions)
        target_q_values = tf.stop_gradient(target_q_values_temp)

    ''' Calculate 
    - trainable variables in actor (Weights of actor network), 
    - Weights of target actor network
    - trainable variables in critic (Weights of critic network),
    - Weights of target critic network
    '''
    actor_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

    target_actor_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')

    critic_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

    target_critic_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')

    ''' Get the operators for updating the target networks. The 
    update_target_networks function defined in utils returns a list of operators 
    to be run from tf session inorder to update the target networks using 
    soft update.
    '''
    update_targets_op = update_target_networks(TAU, \
        target_actor_vars, actor_vars, target_critic_vars, \
            critic_vars)

    ''' Create the tf operation to train the critic network:
    - calculate TD-target 
    - calculate TD-Error = TD-target - q_values_of_given_actions
    - calculate Critic network's loss (Mean Squared Error of TD-Errors)
    - ?
    - create a tf operation to train the critic network
    '''
    targets = tf.expand_dims(reward_placeholder, 1) + \
        tf.expand_dims(is_not_terminal_placeholder, 1) * GAMMA * \
            target_q_values
    td_errors = targets - q_values_of_given_actions
    critic_loss = tf.reduce_mean(tf.square(td_errors))

    # Update critic networks after computing loss
    for var in critic_vars:
        if not 'bias' in var.name:
            critic_loss += L2_REG_CRITIC * 0.5 * tf.nn.l2_loss(var)

    # optimize critic
    critic_train_op = tf.train.AdamOptimizer(LEARNING_RATE_CRITIC * LR_DECAY **
                                             episodes).minimize(critic_loss)

    ''' Create a tf operation to train the actor networks
    - Calculate the Actor network's loss
    - Create the tf operation to train the actor network
    '''
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
            # env.render()
            # Reshape State
            # print("State initial" + str(state.shape)) (3,1)
            # state_to_feed = state.reshape(1, state.shape[0])
            # print("State to feed" + str(state_to_feed.shape)) (1, 3)
            state = np.squeeze(state)
            # print("State after " + str(state.shape)) #(3,)

            # Choose an action
            action = sess.run(actions, feed_dict={ \
                state_placeholder: state[None],
                is_training_placeholder: False})
            # print(action)

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
                state_batch, action_batch, reward_batch, done_batch, \
                    next_state_batch = \
                        replay_memory.sample_batch(MINI_BATCH_SIZE)

                _, _ = sess.run([critic_train_op, actor_train_op], \
                    feed_dict={
                        state_placeholder: state_batch,
                        action_placeholder: np.asarray( \
                            [a[0] for a in action_batch]),
                        reward_placeholder: reward_batch,
                        is_not_terminal_placeholder: done_batch,
                        next_state_placeholder: next_state_batch,
                        is_training_placeholder: True
                })

                _ = sess.run(update_targets_op)
                # _ = sess.run(update_targets_op)


            state = next_state
            num_steps += 1
            num_steps_in_episode += 1

            if done:
                _ = sess.run(episode_incr_op)
                break     

        print(str((episode, total_reward, num_steps_in_episode, noise_scale)))

    # write_to_file('info.json', json.dumps(info))
    env.close()
    # gym.upload(OUTPUT_DIR)

main()


