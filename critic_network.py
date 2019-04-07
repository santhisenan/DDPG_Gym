import tensorflow as tf
from tensorflow import keras

class CriticNetwork(object):
    def __init__(self, state_dim, action_dim, 
                 h1_critic, h2_critic, h3_critic, 
                 dropout):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = self.state_dim + self.action_dim
        self.h1_critic = h1_critic
        self.h2_critic = h2_critic
        self.h3_critic = h3_critic
        self.dropout = dropout

        self.model = self.create_network()

    def create_network(self):
        inputs = keras.Input(shape=self.input_dim)

        hidden_1 = keras.layers.Dense(self.h1_critic,
                                      activation=tf.nn.relu)(inputs)
        # TODO: Add dropout

        hidden_2 = keras.layers.Dense(self.h2_critic, activation=tf.nn.relu)\
            (hidden_1)

        hidden_3 = keras.layers.Dense(self.h3_critic, activation=tf.nn.relu)\
            (hidden_2)

        output = keras.layers.Dense(1, activation=tf.nn.tanh)(hidden_3)

        model = keras.Model(inputs=inputs, outputs=output,
                            name='critic_network')

        return model


