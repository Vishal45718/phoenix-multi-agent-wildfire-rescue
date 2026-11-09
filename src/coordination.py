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
    def get_metropolis_weights(A):
        """
        Calculates Metropolis-Hastings weights for decentralized
        gradient tracking over the communication graph.
        """
        N = A.shape[0]
        W = np.zeros((N, N))
        degrees = np.sum(A, axis=1)
        for i in range(N):
            for j in range(N):
                if i != j and A[i, j] == 1.0:
                    W[i, j] = 1.0 / (max(degrees[i], degrees[j]) + 1.0)
                elif i == j:
                    sum_neighbors = 0.0
                    for k in range(N):
                        if i != k and A[i, k] == 1.0:
                            sum_neighbors += 1.0 / (max(degrees[i], degrees[k]) + 1.0)
                    W[i, i] = 1.0 - sum_neighbors
        return W

    @staticmethod
    def synchronize_and_track_gradients(uavs, A, current_time):
        """
        Decentralized Gossip & Gradient Tracking.
        Synchronizes the belief maps between connected UAVs using timestamps,
        and executes gradient tracking to align local optimization states.
        """
        N = len(uavs)
        W = GraphManager.get_metropolis_weights(A)

        # Consensus Belief Synchronization
        for i in range(N):
            for j in range(N):
                if i != j and A[i, j] == 1.0:
                    for sid, record_j in uavs[j].belief_map.items():
                        age_j = current_time - record_j['timestamp']
                        conf_j_decayed = record_j['conf'] * np.exp(-0.08 * age_j)

                        if sid not in uavs[i].belief_map:
                            uavs[i].belief_map[sid] = dict(record_j)
                        else:
                            record_i = uavs[i].belief_map[sid]
                            age_i = current_time - record_i['timestamp']
                            conf_i_decayed = record_i['conf'] * np.exp(-0.08 * age_i)
                            if conf_j_decayed > conf_i_decayed:
                                uavs[i].belief_map[sid] = dict(record_j)

        # Gradient Tracking Updates
        new_grads = []
        for i in range(N):
            curr_grad_f = (uavs[i].pos - uavs[i].home_pos) * 0.05
            neighbor_mix = np.zeros(2)
            for j in range(N):
                neighbor_mix += W[i, j] * uavs[j].grad_state

            s_next = neighbor_mix + (curr_grad_f - uavs[i].prev_grad_f)
            uavs[i].prev_grad_f = curr_grad_f.copy()
            new_grads.append(s_next)

        for i in range(N):
            uavs[i].grad_state = new_grads[i]

    @staticmethod
    def handle_tracking_handoff(uavs):
        """
        Resolves conflicts where multiple UAVs attempt to track the same target.
        Only the closest UAV remains in TRACK mode; others revert to SEARCH.
        """
        targets = {}
        for u in uavs:
            if u.task == "TRACK" and u.target_survivor_id is not None:
                targets.setdefault(u.target_survivor_id, []).append(u)

        for sid, assigned_uavs in targets.items():
            if len(assigned_uavs) > 1:
                target_pos = assigned_uavs[0].belief_map[sid]['pos']
                assigned_uavs.sort(key=lambda u: np.linalg.norm(u.pos - target_pos))
                for redundant_uav in assigned_uavs[1:]:
                    redundant_uav.task = "SEARCH"
                    redundant_uav.target_survivor_id = None


class ADMMConsensusOptimizer:
    @staticmethod
    def project_simplex(v):
        """
        Closed-form Euclidean projection onto the probability simplex.
        Ensures assignment variables sum to 1.
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_indices = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
        if len(rho_indices) == 0:
            theta = (np.sum(v) - 1.0) / n
        else:
            k = rho_indices[-1]
            theta = (cssv[k] - 1.0) / (k + 1)
        return np.maximum(v - theta, 0.0)

    @staticmethod
    def run_assignment(uavs, rescue_vehicles, survivors, max_iters=25, rho=2.0, return_history=False):
        """
        Executes ADMM to solve the decentralized task allocation problem.
        Filters out Rescue Vehicles that are full and returning to base.
        """
        # Only assign tasks to vehicles that are IDLE/RESCUING and have capacity
        active_rvs = [rv for rv in rescue_vehicles if rv.state != "RETURNING" and rv.rescued_count < rv.capacity]
        
        num_rv = len(active_rvs)
        num_s = len(survivors)
        if num_rv == 0 or num_s == 0:
            if return_history: return {}, [], None, None
            return {}

        N = len(uavs)
        X = [np.ones((num_rv, num_s)) / num_rv for _ in range(N)]
        Z = np.ones((num_rv, num_s)) / num_rv
        U = [np.zeros((num_rv, num_s)) for _ in range(N)]

        # Build local cost matrices for each UAV estimator
        C = []
        for u in uavs:
            C_i = np.zeros((num_rv, num_s))
            for r_idx, rv in enumerate(active_rvs):
                for s_idx, s in enumerate(survivors):
                    if hasattr(u, 'belief_map') and s.id in u.belief_map:
                        s_pos = u.belief_map[s.id]['pos']
                        s_health = u.belief_map[s.id].get('health', s.health)
                    else:
                        s_pos = s.pos
                        s_health = s.health
                    
                    dist = np.linalg.norm(rv.pos - s_pos)
                    urgency_multiplier = (120.0 - s_health) / 100.0
                    C_i[r_idx, s_idx] = dist * urgency_multiplier
            C.append(C_i)

        res_history = []
        for _ in range(max_iters):
            # 1. Primal local update: proximal minimization
            for i in range(N):
                target = Z - U[i] - (C[i] / rho)
                for s_idx in range(num_s):
                    X[i][:, s_idx] = ADMMConsensusOptimizer.project_simplex(target[:, s_idx])

            # 2. Consensus update over all estimators (Z step)
            avg_X = np.mean(X, axis=0)
            avg_U = np.mean(U, axis=0)
            for s_idx in range(num_s):
                Z[:, s_idx] = ADMMConsensusOptimizer.project_simplex(avg_X[:, s_idx] + avg_U[:, s_idx])

            # 3. Dual update (U step)
            for i in range(N):
                U[i] += X[i] - Z

            res = sum(np.linalg.norm(X[i] - Z) for i in range(N))
            res_history.append(res)

        # Decode discrete assignments mapped to the ORIGINAL rv.id
        allocations = {}
        for s_idx in range(num_s):
            best_active_rv_idx = np.argmax(Z[:, s_idx])
            actual_rv_id = active_rvs[best_active_rv_idx].id
            allocations[survivors[s_idx].id] = actual_rv_id

        if return_history:
            return allocations, res_history, Z, X
        return allocations


class RescueDispatchManager:
    @staticmethod
    def step_rescue_vehicles(rescue_vehicles, survivors, uavs, allocations, forest, current_time, dt=1.0, pickup_radius=4.5, require_epistemic=True):
        """
        Translates ADMM allocations into physical movement for ground vehicles.
        Manages round-trip logistics (IDLE -> RESCUING -> RETURNING -> IDLE).
        Applies Epistemic Logic checks to prevent premature/unsafe dispatch.
        """
        rescued_ids = []
        survivor_map = {s.id: s for s in survivors}

        # Clear targets for all vehicles first
        for rv in rescue_vehicles:
            rv.assigned_targets = []
            
            # State transitions based on capacity
            if rv.rescued_count >= rv.capacity:
                rv.state = "RETURNING"
            
            # --- LOGISTICS: RETURNING TO BASE (HOSPITAL) ---
            if rv.state == "RETURNING":
                diff = rv.base_pos - rv.pos
                dist = np.linalg.norm(diff)
                if dist <= pickup_radius:
                    rv.rescued_count = 0  # Drop off successful, reset capacity
                    rv.state = "IDLE"
                else:
                    move_dir = diff / dist
                    step_vec = move_dir * min(rv.max_speed, dist / dt) * dt
                    rv.pos = np.clip(rv.pos + step_vec, [0, 0], [forest.W - 1, forest.H - 1])
                
                rv.path_history.append(rv.pos.copy())
                continue  # Skip ADMM assignments while returning

        # Map ADMM allocations to active rescue vehicles
        for s_id, rv_id in allocations.items():
            if s_id in survivor_map:
                s = survivor_map[s_id]
                if getattr(s, 'rescued', False) or s.health <= 0.0:
                    continue

                # Epistemic Logic Verification Guard
                if require_epistemic:
                    verified = any(EpistemicLogicGuard.satisfies_knowledge_check(
                        u, s_id, confidence_threshold=0.55, freshness_threshold=8.0, current_time=current_time
                    ) for u in uavs)
                else:
                    verified = any(s_id in u.belief_map for u in uavs)

                if verified:
                    for rv in rescue_vehicles:
                        if rv.id == rv_id and rv.state != "RETURNING":
                            rv.assigned_targets.append(s)

        # Execute kinematic motion for rescue vehicles
        for rv in rescue_vehicles:
            if rv.state == "RETURNING":
                continue # Handled above

            if len(rv.assigned_targets) > 0:
                rv.state = "RESCUING"
                rv.assigned_targets.sort(key=lambda x: x.health)  # Prioritize lowest health
                target_s = rv.assigned_targets[0]
                diff = target_s.pos - rv.pos
                dist = np.linalg.norm(diff)

                if dist <= pickup_radius:
                    target_s.rescued = True
                    rv.rescued_count += 1
                    rescued_ids.append(target_s.id)
                else:
                    move_dir = diff / dist
                    step_vec = move_dir * min(rv.max_speed, dist / dt) * dt
                    rv.pos = np.clip(rv.pos + step_vec, [0, 0], [forest.W - 1, forest.H - 1])

            rv.path_history.append(rv.pos.copy())

        return rescued_ids
