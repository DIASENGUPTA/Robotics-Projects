import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """
     # Step 1: Sample transitions from replay_buffer
    state, action, reward,next_state, done = replay_buffer.sample(batch_size)
    

    # Convert batch to tensors
    state_batch = torch.tensor(state, device=device, dtype=torch.float32)
    action_batch = torch.tensor(action, device=device, dtype=torch.long)
    reward_batch = torch.tensor(reward, device=device, dtype=torch.float32)
    next_state_batch = torch.tensor(next_state, device=device, dtype=torch.float32)
    done_batch = torch.tensor(done, device=device, dtype=torch.uint8)
    #print(state_batch.shape)
    #print(next_state_batch.shape)

    # Step 2: Compute Q(s_t, a)
    q_values = policy_net(state_batch.reshape(batch_size,3,96,96)).gather(1, action_batch.unsqueeze(1))
    #print(q_values.shape,"Q")

    # Step 3: Compute max_a Q(s_{t+1}, a) for all next states.
    with torch.no_grad():
        #print(next_state_batch[0].shape)
        next_q_values = target_net(next_state_batch.reshape(batch_size,3,96,96)).max(1)[0].unsqueeze(1)
        # Step 4: Mask next state values where episodes have terminated
        masked_next_q_values = (1 - done_batch.reshape(32,1)) * next_q_values
        
    # Step 5: Compute the target
    targets = reward_batch.reshape(32,1) + gamma * masked_next_q_values
    #print(targets.shape)

    # Step 6: Compute the loss
    loss = F.mse_loss(q_values.to(device), targets.to(device))

    # Step 7: Calculate the gradients
    optimizer.zero_grad()
    loss.backward()

    # Step 8: Clip the gradients
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    # Step 9: Optimize the model
    optimizer.step()

    return loss.item()

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
    target_net.load_state_dict(policy_net.state_dict())