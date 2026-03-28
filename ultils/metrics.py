def compute_reward(t_max: float, e_max: float, m_total: int, 
                   t_im: float, e_im: float, 
                   t_accm: float, e_accm: float, 
                   p_out: float, 
                   lambdas: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)) -> float:
    """
    Calculates the reward for an offloading decision based on Equation 24.
    """
    l1, l2, l3, l4, l5 = lambdas
    
    # 1. Subtask Delay Reward: Encourages beating the average time allowance
    reward_delay_subtask = l1 * ((t_max / m_total) - t_im)
    
    # 2. Cumulative Delay Reward: Encourages keeping the overall task under T_max
    reward_delay_accm = l2 * (t_max - t_accm)
    
    # 3. Subtask Energy Reward: Encourages beating the average energy allowance
    reward_energy_subtask = l3 * ((e_max / m_total) - e_im)
    
    # 4. Cumulative Energy Reward: Encourages keeping overall task under E_max
    reward_energy_accm = l4 * (e_max - e_accm)
    
    # 5. Out-of-bounds Penalty: Penalizes broken connections (p_out should be <= 0)
    penalty = l5 * p_out
    
    # Total Reward
    total_reward = (reward_delay_subtask + 
                    reward_delay_accm + 
                    reward_energy_subtask + 
                    reward_energy_accm + 
                    penalty)
                    
    return total_reward