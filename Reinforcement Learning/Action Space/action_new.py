import random
import torch
import numpy as np


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select greedy action
    state = torch.from_numpy(state).float().unsqueeze(0)

    # Get the Q-values for all actions in the current state
    q_values = policy_net(state.reshape(1,3,96,96))

    # Select the action with the highest Q-value (i.e., the greedy action)
    _, action = q_values.max(1)

    # Convert the action tensor to an integer
    action = action[0].item()

    return action
    

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select exploratory action
    exploration_prob = exploration.value(t)
    if random.random() < exploration_prob:
        # Select a random action
        action_id = random.randrange(action_size)
    else:
        # Select the greedy action
       action_id=select_greedy_action(state,policy_net,action_size)
    return action_id

def get_action_set():
    """ Get the list of available actions
    Returns
    -------
    list
        list of available actions
    """
    return [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0],[0,0,0],[-0.7,0,1],[0.7,0,1],[-0.7,0.1,0],[0.7,0.1,0]]