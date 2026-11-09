"""
src/agents.py
Heterogeneous UAVs, Reactive Survivors, and Ground Rescue Vehicles.
Includes the 5-action Q-Learning Task Allocator.
"""

import numpy as np


class SurvivorAgent:
    def __init__(self, survivor_id, pos, voice_radius=10.0, mobility=1.0, goal=None):
        self.id = survivor_id
        self.pos = np.array(pos, dtype=float)
        self.voice_radius = voice_radius
        self.mobility = mobility
        self.health = 100.0
        self.goal = np.array(goal, dtype=float) if goal is not None else None
        self.rescued = False

    def step(self, fire_center, forest, dt=0.5, alpha_trail=0.4):
        if self.rescued or self.health <= 0:
            return

        diff = self.pos - fire_center
        dist = np.linalg.norm(diff)
        flee_dir = diff / (dist + 1e-6) if dist > 0.1 else np.zeros(2)

        trail_dir = np.zeros(2)
        if self.goal is not None:
            g_diff = self.goal - self.pos
            g_dist = np.linalg.norm(g_diff)
            if g_dist > 0.1:
                trail_dir = g_diff / g_dist

        combined_dir = flee_dir + alpha_trail * trail_dir
        comb_norm = np.linalg.norm(combined_dir)
        if comb_norm > 0:
            combined_dir /= comb_norm

        self.pos += combined_dir * (2.0 * self.mobility) * dt
        self.pos = np.clip(self.pos, [0, 0], [forest.W - 1, forest.H - 1])

        sx = int(np.clip(self.pos[0], 0, forest.W - 1))
        sy = int(np.clip(self.pos[1], 0, forest.H - 1))
        local_smoke = forest.smoke[sy, sx]

        if dist < 12.0:
            self.health -= 3.5 * dt
        if local_smoke > 0.45:
            self.health -= 1.8 * local_smoke * dt
        self.health = max(0.0, self.health)


class RLUAVTaskAllocator:
    """
    Tabular Q-learning policy deciding high-level UAV actions:
    SEARCH, TRACK, HANDOFF, SUPPORT_RESCUE, RETURN
    """
    def __init__(self, actions=None):
        if actions is None:
            self.actions = ["SEARCH", "TRACK", "HANDOFF", "SUPPORT_RESCUE", "RETURN"]
        else:
            self.actions = actions

        # State Discretization: (battery_low [0/1], target_detected [0/1], high_fire_risk [0/1])
        # Values initialized with pre-trained converged heuristics
        self.q_table = {
            (0, 0, 0): np.array([4.5, 0.0, 0.0, 0.0, 0.0]),
            (0, 1, 0): np.array([0.5, 5.0, 2.0, 3.5, 0.0]),
            (0, 1, 1): np.array([0.2, 1.0, 4.8, 1.5, 0.5]),
            (0, 0, 1): np.array([1.5, 0.0, 0.0, 0.0, 2.0]),
            (1, 0, 0): np.array([0.1, 0.0, 0.0, 0.0, 5.5]),
            (1, 1, 0): np.array([0.1, 0.5, 5.2, 0.5, 5.0]),
            (1, 1, 1): np.array([0.0, 0.2, 5.5, 0.2, 6.0]),
            (1, 0, 1): np.array([0.0, 0.0, 0.0, 0.0, 6.0])
        }
        self.lr = 0.1
        self.discount = 0.92
        self.epsilon = 0.08

    def _get_state(self, battery, target_detected, fire_risk):
        bat_low = 1 if battery < 28.0 else 0
        det = 1 if target_detected else 0
        risk = 1 if fire_risk > 0.45 else 0
        return (bat_low, det, risk)

    def select_action(self, battery, target_detected, fire_risk=0.0, allowed_actions=None):
        state = self._get_state(battery, target_detected, fire_risk)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

        valid_actions = allowed_actions if allowed_actions is not None else self.actions
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)

        q_vals = self.q_table[state]
        valid_indices = [self.actions.index(a) for a in valid_actions]
        best_idx = valid_indices[np.argmax([q_vals[i] for i in valid_indices])]
        return self.actions[best_idx]

    def update(self, battery, target_detected, fire_risk, action, reward, next_battery, next_detected, next_risk):
        state = self._get_state(battery, target_detected, fire_risk)
        next_state = self._get_state(next_battery, next_detected, next_risk)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))

        act_idx = self.actions.index(action)
        best_next_q = np.max(self.q_table[next_state])
        self.q_table[state][act_idx] += self.lr * (reward + self.discount * best_next_q - self.q_table[state][act_idx])


class UAVAgent:
    def __init__(self, uav_id, pos, home_pos, role='Scout', v_max=5.0, comm_radius=30.0,
                 sense_radius=14.0, b_thresh=25.0, battery=100.0):
        self.id = uav_id
        self.pos = np.array(pos, dtype=float)
        self.home_pos = np.array(home_pos, dtype=float)
        self.role = role
        self.v_max = v_max
        self.comm_radius = comm_radius
        self.sense_radius = sense_radius
        self.b_thresh = b_thresh
        self.battery = battery
        self.sensor_quality = 0.92

        self.task = "SEARCH"
        self.target_survivor_id = None
        self.allocator = RLUAVTaskAllocator()
        self.belief_map = {}
        self.gradient_tracker = np.zeros(2)
        self.path_history = [self.pos.copy()]

    def compute_pd(self, survivor_pos, forest):
        dist = np.linalg.norm(self.pos - survivor_pos)
        if dist > self.sense_radius:
            return 0.0

        sx = int(np.clip(survivor_pos[0], 0, forest.W - 1))
        sy = int(np.clip(survivor_pos[1], 0, forest.H - 1))
        smoke_val = forest.smoke[sy, sx]
        canopy_val = forest.canopy[sy, sx]

        pd = self.sensor_quality * (1.0 - (dist / self.sense_radius)**2) * (1.0 - 0.7 * smoke_val) * (1.0 - 0.4 * canopy_val)
        return max(0.0, min(1.0, pd))

    def sense_survivors(self, survivors, forest, current_time):
        detected = []
        for s in survivors:
            if getattr(s, 'rescued', False) or s.health <= 0:
                continue
            pd = self.compute_pd(s.pos, forest)
            if np.random.rand() < pd:
                noisy_pos = s.pos + np.random.normal(0, 0.35, 2)
                self.belief_map[s.id] = {
                    'pos': noisy_pos,
                    'timestamp': current_time,
                    'conf': round(float(pd), 3),
                    'health': s.health
                }
                detected.append(s.id)
        return detected

    def step(self, u_cmd, forest, dt=1.0):
        speed = np.linalg.norm(u_cmd)
        if speed > self.v_max:
            u_cmd = (u_cmd / speed) * self.v_max

        # Battery charging at home base
        if np.linalg.norm(self.pos - self.home_pos) < 3.0 and self.task == "RETURN":
            self.battery = min(100.0, self.battery + 12.0 * dt)
        else:
            self.battery = max(0.0, self.battery - dt * (0.05 + 0.01 * (np.linalg.norm(u_cmd)**2)))

        self.pos += u_cmd * dt
        self.pos = np.clip(self.pos, [0, 0], [forest.W - 1, forest.H - 1])
        self.path_history.append(self.pos.copy())

    def update_behavior(self, current_time, default_search_cmd, has_knowledge=True, fire_risk=0.0):
        target_detected = (len(self.belief_map) > 0)
        allowed = list(self.allocator.actions)
        if not has_knowledge and "TRACK" in allowed:
            allowed.remove("TRACK")

        if self.battery >= 98.0 and self.task == "RETURN":
            self.task = "SEARCH"
        else:
            self.task = self.allocator.select_action(self.battery, target_detected, fire_risk, allowed_actions=allowed)

        # Reward model
        reward = 0.1
        if self.task == "RETURN" and self.battery < self.b_thresh:
            reward = 5.0
        elif self.task == "TRACK" and target_detected and has_knowledge:
            reward = 4.0
        elif self.task == "HANDOFF":
            reward = 3.0

        self.allocator.update(self.battery, target_detected, fire_risk, self.task, reward,
                              self.battery - 0.2, target_detected, fire_risk)

        best_target = None
        best_score = -1.0
        for sid, rec in self.belief_map.items():
            age = current_time - rec['timestamp']
            freshness = np.exp(-0.15 * age)
            score = rec['conf'] * freshness
            if score > best_score and age < 6.0:
                best_score = score
                best_target = rec['pos']
                self.target_survivor_id = sid

        if self.task == "RETURN":
            r_vec = self.home_pos - self.pos
            r_dist = np.linalg.norm(r_vec)
            return (r_vec / (r_dist + 1e-6)) * self.v_max if r_dist > 0.5 else np.zeros(2)

        if best_target is not None and self.task in ["TRACK", "SUPPORT_RESCUE"] and has_knowledge:
            disp = best_target - self.pos
            dist = np.linalg.norm(disp)
            return (disp / (dist + 1e-6)) * min(self.v_max, 2.5 * dist) if dist > 1.0 else np.zeros(2)

        self.task = "SEARCH"
        self.target_survivor_id = None
        return default_search_cmd


class RescueVehicle:
    def __init__(self, rv_id, pos, base_pos, max_speed=5.0, capacity=3):
        self.id = rv_id
        self.pos = np.array(pos, dtype=float)
        self.base_pos = np.array(base_pos, dtype=float)
        self.max_speed = max_speed
        self.capacity = capacity
        self.passengers = []
        self.state = "IDLE"  # IDLE, EN_ROUTE_SURVIVOR, RETURNING_BASE
        self.assigned_survivor_id = None
        self.path_history = [self.pos.copy()]

    def step(self, target_pos, dt=1.0):
        if target_pos is None:
            return

        diff = target_pos - self.pos
        dist = np.linalg.norm(diff)
        if dist > 0.5:
            move = (diff / dist) * min(self.max_speed * dt, dist)
            self.pos += move
        self.path_history.append(self.pos.copy())
