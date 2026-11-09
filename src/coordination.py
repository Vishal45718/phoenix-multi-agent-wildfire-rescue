"""
src/coordination.py
Dynamic Communication Graph, Consensus Belief Tracking, and ADMM Optimization.
"""

import numpy as np
from .safety import EpistemicLogicGuard


class GraphManager:
    @staticmethod
    def compute_uav_matrices(uavs):
        N = len(uavs)
        A = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(uavs[i].pos - uavs[j].pos)
                if dist <= min(uavs[i].comm_radius, uavs[j].comm_radius):
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        eigenvalues = np.sort(np.linalg.eigvals(L).real)
        lambda_2 = eigenvalues[1] if N > 1 else 0.0
        return A, D, L, lambda_2

    @staticmethod
    def synchronize_and_track_gradients(uavs, A_u, current_time):
        N = len(uavs)
        # 1. Decentralized consensus exchange
        for i in range(N):
            for j in range(i + 1, N):
                if A_u[i, j] == 1.0:
                    for sid, record in uavs[j].belief_map.items():
                        if sid not in uavs[i].belief_map or record['timestamp'] > uavs[i].belief_map[sid]['timestamp']:
                            uavs[i].belief_map[sid] = dict(record)
                    for sid, record in uavs[i].belief_map.items():
                        if sid not in uavs[j].belief_map or record['timestamp'] > uavs[j].belief_map[sid]['timestamp']:
                            uavs[j].belief_map[sid] = dict(record)

        # 2. Gradient tracking step on fire centroid consensus
        degrees = np.sum(A_u, axis=1)
        W = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j and A_u[i, j] == 1.0:
                    W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
            W[i, i] = 1.0 - np.sum(W[i, :])

        positions = np.array([u.pos for u in uavs])
        mixed_positions = W @ positions
        for i in range(N):
            uavs[i].gradient_tracker = mixed_positions[i] - uavs[i].pos

    @staticmethod
    def handle_tracking_handoff(uavs):
        """Hands off tracking task when an agent battery depletes or another UAV is closer."""
        for u in uavs:
            if u.task == "TRACK" and u.battery < u.b_thresh:
                u.task = "HANDOFF"
                candidates = [other for other in uavs if other.id != u.id and other.battery > u.b_thresh]
                if candidates and u.target_survivor_id in u.belief_map:
                    target_p = u.belief_map[u.target_survivor_id]['pos']
                    best_cand = min(candidates, key=lambda c: np.linalg.norm(c.pos - target_p))
                    best_cand.task = "TRACK"
                    best_cand.target_survivor_id = u.target_survivor_id


class ADMMConsensusOptimizer:
    """
    Alternating Direction Method of Multipliers (ADMM)
    Solving multi-agent rescue vehicle allocation:
    min sum cost(x_i, z) s.t. x_i = z
    """
    @staticmethod
    def run_assignment(uavs, rvs, survivors, rho=1.0, max_iters=15):
        if not survivors or not rvs:
            return {}

        N_surv = len(survivors)
        N_rv = len(rvs)
        # Cost Matrix: C[i, j] = ETA + Risk + Capacity Penalty
        C = np.zeros((N_surv, N_rv))
        for i, s in enumerate(survivors):
            for j, rv in enumerate(rvs):
                eta = np.linalg.norm(s.pos - rv.pos) / rv.max_speed
                risk_penalty = 2.0 if s.health < 40.0 else 0.5
                cap_penalty = 10.0 if len(rv.passengers) >= rv.capacity else 0.0
                C[i, j] = eta + risk_penalty + cap_penalty

        # Scaled ADMM Iterates
        x = np.zeros((N_surv, N_rv))
        z = np.zeros((N_surv, N_rv))
        u = np.zeros((N_surv, N_rv))

        for _ in range(max_iters):
            # Primal Update: x-minimization
            x = (rho * z - u - C) / rho
            # Consensus Update: z-averaging with 1-to-1 matching constraint
            z = np.zeros_like(x)
            for i in range(N_surv):
                best_j = np.argmax(x[i, :] + u[i, :])
                z[i, best_j] = 1.0
            # Dual Multiplier Update
            u += (x - z)

        allocations = {}
        for i, s in enumerate(survivors):
            assigned_rv_idx = int(np.argmax(z[i, :]))
            allocations[s.id] = assigned_rv_idx

        return allocations


class RescueDispatchManager:
    @staticmethod
    def step_rescue_vehicles(rvs, survivors, uavs, allocations, env, current_time, dt=1.0, require_epistemic=True):
        surv_dict = {s.id: s for s in survivors}

        for rv in rvs:
            # 1. Base Unload Cycle
            if rv.state == "RETURNING_BASE":
                rv.step(rv.base_pos, dt)
                if np.linalg.norm(rv.pos - rv.base_pos) < 3.0:
                    rv.passengers.clear()
                    rv.state = "IDLE"
                    rv.assigned_survivor_id = None
                continue

            # 2. Check Capacity Saturation
            if len(rv.passengers) >= rv.capacity:
                rv.state = "RETURNING_BASE"
                rv.step(rv.base_pos, dt)
                continue

            # 3. Target Acquisition
            if rv.assigned_survivor_id is None:
                assigned_sids = [sid for sid, rv_idx in allocations.items() if rv_idx == rv.id]
                for sid in assigned_sids:
                    s_cand = surv_dict.get(sid)
                    if s_cand and not s_cand.rescued and s_cand.health > 0:
                        if require_epistemic:
                            knows = any(EpistemicLogicGuard.satisfies_knowledge_check(u, sid, current_time) for u in uavs)
                            if not knows:
                                continue
                        rv.assigned_survivor_id = sid
                        rv.state = "EN_ROUTE_SURVIVOR"
                        break

            # 4. En-Route Steer
            if rv.assigned_survivor_id is not None:
                s_target = surv_dict.get(rv.assigned_survivor_id)
                if s_target and not s_target.rescued and s_target.health > 0:
                    rv.step(s_target.pos, dt)
                    if np.linalg.norm(rv.pos - s_target.pos) < 2.5:
                        s_target.rescued = True
                        rv.passengers.append(s_target.id)
                        rv.assigned_survivor_id = None
                        if len(rv.passengers) >= rv.capacity:
                            rv.state = "RETURNING_BASE"
                        else:
                            rv.state = "IDLE"
                else:
                    rv.assigned_survivor_id = None
                    rv.state = "IDLE"
