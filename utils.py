import tensorflow as tf 

def scale_actions(actor_network_outputs, action_bound_low, action_bound_high):
    actions = action_bound_low + tf.nn.sigmoid(actor_network_outputs)* \
        (action_bound_high - action_bound_low)
    return actions
