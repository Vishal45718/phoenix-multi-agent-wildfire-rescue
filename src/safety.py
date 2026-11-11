"""
src/safety.py
Signal Temporal Logic (STL) Safety Shields and Epistemic Logic (EL) Knowledge Guards.
"""

import numpy as np


class STLMonitor:
    def __init__(self, d_safe=5.5, d_fire=9.0, h_min=25.0):
        self.d_safe = d_safe
        self.d_fire = d_fire
        self.h_min = h_min

    def evaluate_safety_margins(self, uavs, forest, survivors):
        """
        Calculates STL robustness margins (rho).
        rho > 0 => Safe; rho < 0 => Safety Specification Violated.
        """
        # Spec 1: Box_{[0, T]} ( ||p_i - p_j|| >= d_safe )
        min_uav_dist = float('inf')
        N = len(uavs)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(uavs[i].pos - uavs[j].pos)
                if d < min_uav_dist:
                    min_uav_dist = d
        rho_uav = min_uav_dist - self.d_safe if N > 1 else 10.0

        # Spec 2: Box_{[0, T]} ( dist(UAV, Fire) >= d_fire )
        fc = forest.get_fire_centroid()
        min_fire_dist = min([np.linalg.norm(u.pos - fc) for u in uavs]) if uavs else 10.0
        rho_fire = min_fire_dist - self.d_fire

        # Spec 3: Diamond_{[0, T]} ( Health(Survivor) >= h_min )
        active = [s.health for s in survivors if not getattr(s, 'rescued', False)]
        rho_health = (min(active) - self.h_min) if active else 50.0

        return {
            'uav_spacing': rho_uav,
            'fire_clearance': rho_fire,
            'survivor_health': rho_health
        }


class ActiveSTLController(STLMonitor):
    def __init__(self, d_safe=5.5, d_fire=9.0, h_min=25.0, buffer_zone=5.8, k_repel=8.5, fire_clearance_trigger=9.5):
        super().__init__(d_safe, d_fire, h_min)
        self.buffer_zone = buffer_zone
        self.k_repel = k_repel
        self.fire_clearance_trigger = fire_clearance_trigger

    def apply_safety_shield(self, uav_idx, u_nominal, uavs, forest):
        """
        Projects nominal control into the safe control set using barrier potential gradients.
        """
        u_safe = u_nominal.copy().astype(float)
        active_uav = uavs[uav_idx]
        intervened = False

        # Inter-UAV collision avoidance barrier
        for j, other in enumerate(uavs):
            if j == uav_idx:
                continue
            disp = active_uav.pos - other.pos
            dist = np.linalg.norm(disp)
            if dist < self.buffer_zone:
                intervened = True
                u_safe += (disp / (dist + 1e-6)) * self.k_repel * (self.buffer_zone - dist)

        # Fire proximity barrier
        fc = forest.get_fire_centroid()
        f_disp = active_uav.pos - fc
        f_dist = np.linalg.norm(f_disp)
        fire_trigger = getattr(self, 'fire_clearance_trigger', 9.5)
        if f_dist < fire_trigger:
            intervened = True
            u_safe += (f_disp / (f_dist + 1e-6)) * (self.k_repel * 1.5) * (fire_trigger - f_dist)

        speed = np.linalg.norm(u_safe)
        if speed > active_uav.v_max:
            u_safe = (u_safe / speed) * active_uav.v_max

        return u_safe, intervened


class EpistemicLogicGuard:
    """
    K_i phi: Evaluates whether agent i 'Knows' proposition phi.
    Disallows action unless epistemic validity holds.
    """
    @staticmethod
    def satisfies_knowledge_check(uav, survivor_id, current_time, max_staleness=8.0, min_conf=0.30):
        if survivor_id not in uav.belief_map:
            return False
        rec = uav.belief_map[survivor_id]
        is_fresh = (current_time - rec['timestamp']) <= max_staleness
        is_confident = rec['conf'] >= min_conf
        return bool(is_fresh and is_confident)
