import random
import torch
from collections import deque
from typing import Tuple, List

class MultiAgentReplayBuffer:
    """
    Stores joint experiences for the MADDPG agents.
    """
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: List[float], action: List[int], reward: List[float], next_state: List[float]):
        """
        Stores a joint experience tuple (S, A, R, S').
        Lists should be of length D (number of agents).
        """
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random mini-batch of experiences.
        """
        batch = random.sample(self.buffer, batch_size)
        
        state_batch = torch.FloatTensor([exp[0] for exp in batch])
        action_batch = torch.FloatTensor([exp[1] for exp in batch])
        reward_batch = torch.FloatTensor([exp[2] for exp in batch])
        next_state_batch = torch.FloatTensor([exp[3] for exp in batch])
        
        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)