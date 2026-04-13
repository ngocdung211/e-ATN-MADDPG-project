from typing import Dict, List, Optional, Tuple

import numpy as np

from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice, TaskDAG


class DITENEnv:
    def __init__(
        self,
        devices: List[IndustrialDevice],
        servers: List[EdgeServer],
        network_env: NetworkEnvironment,
        slot_duration: float = 1.0,
        subslot_count: int = 100,
        lambda1: float = 1.0,
        lambda2: float = 0.2,
        lambda3: float = 1.0,
        lambda4: float = 0.2,
        lambda5: float = 0.5,
        p_out_value: float = -1.0,
        local_estimation_error: float = 0.05,
        edge_estimation_error: float = 0.05,
        connection_lookahead_slots: int = 5,
        edge_energy_scale: float = 0.2,
    ):
        self.devices = devices
        self.servers = servers
        self.network_env = network_env
        self.num_agents = len(devices)

        self.slot_duration = slot_duration
        self.subslot_count = subslot_count

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.p_out_value = p_out_value
        self.local_estimation_error = max(0.0, local_estimation_error)
        self.edge_estimation_error = max(0.0, edge_estimation_error)
        self.connection_lookahead_slots = max(1, connection_lookahead_slots)
        self.edge_energy_scale = max(0.0, edge_energy_scale)

        self.device_accumulated_delay: Dict[int, float] = {}
        self.device_accumulated_energy: Dict[int, float] = {}
        self.subtask_finish_times: Dict[int, Dict[int, float]] = {}
        self.subtask_locations: Dict[int, Dict[int, int]] = {}
        self.local_finish_time: Dict[int, float] = {}
        self.server_finish_time: Dict[int, float] = {}

        self.task_dags: Dict[int, TaskDAG] = {}
        self.priorities: Dict[int, List[int]] = {}
        self.current_step = 0
        self.current_slot = 0.0
        self.connection_windows: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.last_step_metrics: List[Dict[str, float]] = []
        self.device_estimated_power: Dict[int, float] = {}
        self.server_estimated_power: Dict[int, float] = {}

    def reset(self, task_dags: Dict[int, TaskDAG], priorities: Dict[int, List[int]]) -> np.ndarray:
        self.task_dags = task_dags
        self.priorities = priorities
        self.current_step = 0
        self.current_slot = 0.0

        self.device_accumulated_delay = {d.id: 0.0 for d in self.devices}
        self.device_accumulated_energy = {d.id: 0.0 for d in self.devices}
        self.subtask_finish_times = {d.id: {} for d in self.devices}
        self.subtask_locations = {d.id: {} for d in self.devices}
        self.local_finish_time = {d.id: 0.0 for d in self.devices}
        self.server_finish_time = {s.id: 0.0 for s in self.servers}
        self.last_step_metrics = []
        self.device_estimated_power = {
            d.id: self._sample_estimated_power(d.compute_power, self.local_estimation_error) for d in self.devices
        }
        self.server_estimated_power = {
            s.id: self._sample_estimated_power(s.compute_power, self.edge_estimation_error) for s in self.servers
        }

        for device in self.devices:
            direction = np.random.uniform(-1, 1, 2)
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = np.array([1.0, 0.0], dtype=float)
            device.set_direction(direction)

        self._update_connection_windows()
        return self._get_joint_state()

    def step(self, joint_actions: List[int]) -> Tuple[np.ndarray, List[float], bool, dict]:
        rewards = []
        step_durations = []
        self.last_step_metrics = []

        for idx, action in enumerate(joint_actions):
            device = self.devices[idx]
            task_dag = self.task_dags[device.id]
            current_subtask_id = self.priorities[device.id][self.current_step]
            subtask = task_dag.subtasks[current_subtask_id]

            t_im = 0.0
            e_im = 0.0
            tx_energy = 0.0
            p_out = 0.0

            predecessors = [pred for pred, succ in task_dag.edges if succ == current_subtask_id]
            predecessor_ready_time = 0.0
            for pred_id in predecessors:
                pred_finish = self.subtask_finish_times[device.id].get(pred_id, 0.0)
                pred_location = self.subtask_locations[device.id].get(pred_id, 0)
                if pred_location != action:
                    trans_delay = 0.0
                    if action == 0 and pred_location > 0:
                        pred_server = self.servers[pred_location - 1]
                        gain = self.network_env.calculate_channel_gain(device.location, pred_server.location)
                        down_rate = max(self.network_env.get_downlink_rate(pred_server.transmit_power, gain), 1e-9)
                        trans_delay = task_dag.subtasks[pred_id].result_size / down_rate
                    elif action > 0:
                        target_server = self.servers[action - 1]
                        gain = self.network_env.calculate_channel_gain(device.location, target_server.location)
                        up_rate = max(self.network_env.get_uplink_rate(device.transmit_power, gain), 1e-9)
                        trans_delay = task_dag.subtasks[pred_id].result_size / up_rate
                        tx_energy += device.transmit_power * trans_delay
                    pred_finish += trans_delay
                predecessor_ready_time = max(predecessor_ready_time, pred_finish)

            if action == 0:
                f_est = self.device_estimated_power[device.id]
                f_actual = device.compute_power
                t_comp, e_comp = self.network_env.calculate_local_computation(
                    subtask.cpu_cycles, device.energy_coeff, f_est, f_actual
                )
                w_d = self._compute_waiting_delay_local(device)
                start_time = max(self.current_slot + w_d, predecessor_ready_time)
                finish_time = start_time + t_comp
                self.local_finish_time[device.id] = finish_time
            else:
                server = self.servers[action - 1]
                f_est = self.server_estimated_power[server.id]
                f_actual = server.compute_power
                t_comp, e_comp = self.network_env.calculate_edge_computation(
                    subtask.cpu_cycles, server.energy_coeff, f_est, f_actual
                )
                e_comp *= self.edge_energy_scale
                w_s = self._compute_waiting_delay_server(server)
                l_start, l_end = self.connection_windows[(device.id, server.id)]
                start_time = max(self.current_slot + w_s, predecessor_ready_time, l_start + 1e-9)
                finish_time = start_time + t_comp
                if not (start_time > l_start and finish_time < l_end):
                    p_out = self.p_out_value
                self.server_finish_time[server.id] = finish_time

            t_im = max(0.0, finish_time - self.current_slot)
            e_im = tx_energy + e_comp

            self.subtask_finish_times[device.id][current_subtask_id] = finish_time
            self.subtask_locations[device.id][current_subtask_id] = action

            t_accm = self.device_accumulated_delay[device.id]
            e_accm = self.device_accumulated_energy[device.id]
            self.device_accumulated_delay[device.id] += t_im
            self.device_accumulated_energy[device.id] += e_im

            reward = self._calculate_reward(t_im, e_im, t_accm, e_accm, p_out, task_dag)
            rewards.append(reward)
            step_durations.append(t_im)
            self.last_step_metrics.append(
                {
                    "device_id": float(device.id),
                    "subtask_id": float(current_subtask_id),
                    "action": float(action),
                    "f_est": float(f_est),
                    "f_actual": float(f_actual),
                    "delay": float(t_im),
                    "energy": float(e_im),
                    "tx_energy": float(tx_energy),
                    "comp_energy": float(e_comp),
                    "reward": float(reward),
                    "p_out": float(p_out),
                }
            )

        elapsed = max(step_durations) if step_durations else self.slot_duration
        for device in self.devices:
            device.update_location(device.project_location(elapsed))

        self.current_slot += max(self.slot_duration, elapsed)
        self._update_connection_windows()
        self.current_step += 1
        done = self.current_step >= len(self.task_dags[self.devices[0].id].subtasks)
        return self._get_joint_state(), rewards, done, {}

    def get_state_dim(self) -> int:
        # Eq. (23): {f_loc, W_d, subtask features(C,D,R), priority sequence, F_edge, W_s, l_start, l_end}
        return 6 + 4 * len(self.servers)

    def _update_connection_windows(self):
        self.connection_windows = {}
        horizon = self.slot_duration * self.connection_lookahead_slots
        dt = horizon / float(self.subslot_count * self.connection_lookahead_slots)
        for device in self.devices:
            for server in self.servers:
                l_start: Optional[float] = None
                l_end: Optional[float] = None
                for k in range(self.subslot_count * self.connection_lookahead_slots + 1):
                    t_rel = k * dt
                    loc = device.location + device.direction * device.speed_mps * t_rel
                    inside = np.linalg.norm(loc - server.location) <= server.coverage_radius
                    if inside and l_start is None:
                        l_start = self.current_slot + t_rel
                    if (not inside) and l_start is not None:
                        l_end = self.current_slot + t_rel
                        break

                if l_start is None:
                    l_start = self.current_slot + horizon
                    l_end = self.current_slot + horizon
                elif l_end is None:
                    l_end = self.current_slot + horizon
                self.connection_windows[(device.id, server.id)] = (l_start, l_end)

    def _sample_estimated_power(self, actual_power: float, error_ratio: float) -> float:
        if error_ratio == 0.0:
            return actual_power
        noise = np.random.uniform(-error_ratio, error_ratio)
        estimated = actual_power * (1.0 + noise)
        return max(1e-9, estimated)

    def _compute_waiting_delay_local(self, device: IndustrialDevice) -> float:
        # Eq. (7): queueing delay at local node
        return max(0.0, self.local_finish_time[device.id] - self.current_slot)

    def _compute_waiting_delay_server(self, server: EdgeServer) -> float:
        # Eq. (12): queueing delay at edge server
        return max(0.0, self.server_finish_time[server.id] - self.current_slot)

    def _get_joint_state(self) -> np.ndarray:
        joint_state = []
        for device in self.devices:
            task_dag = self.task_dags[device.id]
            state_step = min(self.current_step, len(self.priorities[device.id]) - 1)
            current_subtask_id = self.priorities[device.id][state_step]
            subtask = task_dag.subtasks[current_subtask_id]

            f_loc = self.device_estimated_power[device.id] / 1e9
            w_d = self._compute_waiting_delay_local(device) / max(self.slot_duration, 1e-6)
            c_norm = subtask.cpu_cycles / 1e9
            d_norm = subtask.data_size / 1e6
            r_norm = subtask.result_size / 1e6
            priority_norm = (self.current_step + 1) / max(len(task_dag.subtasks), 1)
            edge_f = [self.server_estimated_power[server.id] / 1e9 for server in self.servers]
            edge_w = [
                self._compute_waiting_delay_server(server) / max(self.slot_duration, 1e-6)
                for server in self.servers
            ]
            l_starts = []
            l_ends = []
            for server in self.servers:
                l_start, l_end = self.connection_windows[(device.id, server.id)]
                horizon = self.slot_duration * self.connection_lookahead_slots
                l_starts.append((l_start - self.current_slot) / max(horizon, 1e-6))
                l_ends.append((l_end - self.current_slot) / max(horizon, 1e-6))

            state_vector = [f_loc, w_d, c_norm, d_norm, r_norm, priority_norm] + edge_f + edge_w + l_starts + l_ends
            joint_state.append(state_vector)
        return np.array(joint_state, dtype=np.float32)

    def _calculate_reward(self, t_im, e_im, t_accm, e_accm, p_out, task_dag) -> float:
        # Eq. (24)
        t_max = task_dag.t_max
        e_max = task_dag.e_max
        m_total = len(task_dag.subtasks)
        return (
            self.lambda1 * ((t_max / m_total) - t_im)
            + self.lambda2 * (t_max - t_accm)
            + self.lambda3 * ((e_max / m_total) - e_im)
            + self.lambda4 * (e_max - e_accm)
            + self.lambda5 * p_out
        )
