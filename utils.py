import tensorflow as tf 

def scale_actions(actor_network_outputs, action_bound_low, action_bound_high):
    actions = action_bound_low + tf.nn.sigmoid(actor_network_outputs)* \
        (action_bound_high - action_bound_low)
    print("*")
    return actions

def update_target_networks(tau, target_actor_vars, actor_vars, 
                           target_critic_vars, critic_vars):
    update_targets_ops = []

    # TODO: print placeholders and find what goes into it
    for i, target_actor_var in enumerate(target_actor_vars):
        update_target_actor_op = target_actor_var.assign(tau*actor_vars[i] + \
            (1 - tau)*target_actor_var)
        update_targets_ops.append(update_target_actor_op)

    for i, target_critic_var in enumerate(target_critic_vars):
        update_target_critic_op = target_critic_var.assign(tau*critic_vars[i] + \
            (1 - tau)*target_critic_var)
        update_targets_ops.append(update_target_critic_op)

    update_targets_op = tf.group(*update_targets_ops, name='update_targets')
    
    return update_targets_op

