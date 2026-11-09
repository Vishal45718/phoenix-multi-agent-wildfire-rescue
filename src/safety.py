import numpy as np

class STLMonitor:
    def __init__(self, d_safe=6.0, d_fire=10.0, h_min=25.0):
        self.d_safe = d_safe
        self.d_fire = d_fire
        self.h_min = h_min

    def evaluate_safety_margins(self, uavs, forest, survivors):
        # 1. Pairwise UAV spacing robustness: Box( ||p_i - p_j|| >= d_safe )
        min_uav_dist = float('inf')
        N = len(uavs)
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(uavs[i].pos - uavs[j].pos)
                if dist < min_uav_dist:
                    min_uav_dist = dist
        uav_spacing_robustness = min_uav_dist - self.d_safe if N > 1 else 10.0

        # 2. Fire clearance robustness
        fire_coords = np.argwhere(forest.state == 1)
        if len(fire_coords) > 0:
            fire_xy = fire_coords[:, [1, 0]].astype(float)
            min_fire_dist = float('inf')
            for u in uavs:
                dists = np.linalg.norm(fire_xy - u.pos, axis=1)
                min_val = np.min(dists)
                if min_val < min_fire_dist:
                    min_fire_dist = min_val
            fire_clearance_robustness = min_fire_dist - self.d_fire
        else:
            fire_clearance_robustness = 50.0

        # 3. Survivor health satisfaction margin
        min_health = min([s.health for s in survivors if not getattr(s, 'rescued', False)]) if len(survivors) > 0 else 100.0
        health_robustness = min_health - self.h_min

        return {
            'uav_spacing': uav_spacing_robustness,
            'fire_clearance': fire_clearance_robustness,
            'survivor_health': health_robustness
        }


class ActiveSTLController(STLMonitor):
    """
    Extends STLMonitor to actively filter nominal UAV velocity
    commands u_nom whenever margins approach violation.
    """
    def __init__(self, d_safe=6.0, d_fire=10.0, h_min=25.0, buffer_zone=7.0, k_repel=9.0):
        super().__init__(d_safe=d_safe, d_fire=d_fire, h_min=h_min)
        self.buffer_zone = buffer_zone
        self.k_repel = k_repel

    def apply_safety_shield(self, uav_idx, u_nom, uavs, forest):
        u_safe = np.array(u_nom, dtype=float).copy()
        uav = uavs[uav_idx]
        repel_vec = np.zeros(2, dtype=float)
        shield_triggered = False

        # 1. Inter-UAV Collision Avoidance Barrier
        for j, other in enumerate(uavs):
            if j == uav_idx:
                continue
            diff = uav.pos - other.pos
            dist = np.linalg.norm(diff)
            rho_uav = dist - self.d_safe
            if rho_uav < self.buffer_zone and dist > 1e-3:
                weight = self.k_repel * (self.buffer_zone - rho_uav) / (self.buffer_zone + 1e-3)
                repel_vec += (diff / dist) * weight
                shield_triggered = True

        # 2. Fire Standoff Avoidance Barrier
        fire_cells = np.argwhere(forest.state == 1)
        if len(fire_cells) > 0:
            fire_xy = fire_cells[:, [1, 0]].astype(float)
            diffs = uav.pos - fire_xy
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            rho_fire = min_dist - self.d_fire

            if rho_fire < self.buffer_zone and min_dist > 1e-3:
                weight = (self.k_repel * 1.8) * (self.buffer_zone - rho_fire) / (self.buffer_zone + 1e-3)
                repel_vec += (diffs[min_idx] / min_dist) * weight
                shield_triggered = True

        if shield_triggered:
            u_safe = u_safe + repel_vec
            speed = np.linalg.norm(u_safe)
            if speed > uav.v_max:
                u_safe = (u_safe / speed) * uav.v_max

        return u_safe, shield_triggered


class EpistemicLogicGuard:
    @staticmethod
    def satisfies_knowledge_check(uav, survivor_id, confidence_threshold=0.6, freshness_threshold=8.0, current_time=0.0):
        """
        Evaluates K_i(∃ survivor) using strict criteria:
        Requires high confidence and bounded age of the observation.
        """
        if survivor_id not in uav.belief_map:
            return False
        record = uav.belief_map[survivor_id]
        age = current_time - record['timestamp']
        if record['conf'] >= confidence_threshold and age <= freshness_threshold:
            return True
        return False
