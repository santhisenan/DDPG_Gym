import tensorflow as tf
from tensorflow import keras 

class ActorNetwork(object):
    def __init__(self, state_dim, action_dim, 
                 h1_actor, h2_actor, h3_actor, dropout):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.h1_actor = h1_actor
        self.h2_actor = h2_actor
        self.h3_actor = h3_actor
        self.dropout = dropout
        self.model, self.input, self.actions = self.create_network()

    def predict_actions(self, state, steps):
        return self.model.predict(state, steps=steps)
        
    def create_network(self):
        input_layer = keras.Input(shape=(self.state_dim,))
        hidden_1 = keras.layers.Dense(self.h1_actor, 
                                      activation=tf.nn.relu)(input_layer)
        # TODO: Add dropout

        hidden_2 = keras.layers.Dense(self.h2_actor, activation=tf.nn.relu)\
            (hidden_1)

        hidden_3 = keras.layers.Dense(self.h3_actor, activation=tf.nn.relu)\
            (hidden_2)

        output = keras.layers.Dense(self.action_dim, 
                                    activation=tf.nn.tanh)(hidden_3)

        model = keras.Model(inputs=input_layer, outputs=output,
                            name='actor_network')

        return model, input_layer, output