"""Centralized paper-reproduction parameters and status flags.

This module keeps one source of truth for experiment settings and makes it
explicit which values are confirmed from the paper text versus provisional
defaults that still need Table II confirmation.
"""

from typing import Dict


PAPER_PARAMS: Dict[str, Dict[str, float]] = {
    "confirmed": {
        # Topology / scenario
        "num_devices": 10,
        "num_servers": 3,
        "time_slots": 50,
        "slot_duration_s": 1.0,
        "device_speed_mps": 1.0,
        "coverage_radius_m": 12.0,
        # Communication / system
        "bandwidth_hz": 10e6,
        "noise_power_dbm": -43,
        "device_tx_power_w": 0.5,
        "server_tx_power_w": 1.2,
        "device_energy_coeff": 1e-28,
        "server_energy_coeff": 1e-27,
        # Learning settings confirmed in paper text
        "gcn_hidden_dim": 32,
        "gcn_lr": 0.01,
        "rl_lr": 0.0001,
        "train_episodes_full": 1000,
    },
    "provisional_table2_needed": {
        # Pending Table II verification (kept as current project defaults).
        "gamma": 0.99,
        "batch_size": 64,
        "replay_buffer_capacity": 100000,
        "epsilon_init": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.998,
        "tau_soft_update": 0.01,
        # Task constraints currently used as placeholders.
        "t_max": 1.0,
        "e_max": 1.0,
    },
}
