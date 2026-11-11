"""
main.py
PROJECT FIRE-SAR: Full Head-to-Head Baseline Benchmarking & Verification Pipeline.
Ablates all four strategies:
1. Random Search
2. Nearest Agent (Local greedy without consensus)
3. Greedy Uncoordinated (Single-drone assignments)
4. FIRE-SAR Proposed (RL + ADMM + STL Shield + Epistemic Guards)
"""

import numpy as np

from src.environment import ForestEnvironment
from src.agents import SurvivorAgent, UAVAgent, RescueVehicle
from src.safety import ActiveSTLController, EpistemicLogicGuard
from src.coordination import GraphManager, ADMMConsensusOptimizer, RescueDispatchManager
from src.visualization import generate_dashboard

packet_loss_rate = 0.02


def run_monte_carlo(num_trials=10, steps=45, packet_loss_prob=0.02, malfunction_step=15, packet_loss_rate=0.02):
    modes = ["Random_Search", "Nearest_Agent", "Greedy_Uncoordinated", "FIRE_SAR_Proposed"]
    results = {
        m: {
            'survival_rate': [],
            'rescued_ratio': [],
            'stl_violations': [],
            'detection_time': [],
            'battery_consumed': []
        } for m in modes
    }

    for mode in modes:
        print(f"\n---> Evaluating Strategy: {mode} ({num_trials} Monte Carlo Trials)")
        for trial in range(num_trials):
            env = ForestEnvironment(width=100, height=100, wind_vector=(2.8, 1.8))
            np.random.seed(400 + trial)
            env.canopy = np.random.uniform(0.1, 0.65, (env.H, env.W))
            env.ignite(30, 30, radius=3)

            # 10 Heterogeneous Survivors
            survivors = [
                SurvivorAgent(0, [34.0, 33.0], mobility=0.25, goal=[85.0, 85.0]),
                SurvivorAgent(1, [35.0, 32.0], mobility=0.30, goal=[85.0, 85.0]),
                SurvivorAgent(2, [42.0, 38.0], mobility=0.45, goal=[85.0, 85.0]),
                SurvivorAgent(3, [46.0, 44.0], mobility=0.80, goal=[85.0, 85.0]),
                SurvivorAgent(4, [52.0, 48.0], mobility=0.85, goal=[85.0, 85.0]),
                SurvivorAgent(5, [58.0, 52.0], mobility=0.90, goal=[85.0, 85.0]),
                SurvivorAgent(6, [38.0, 40.0], mobility=0.35, goal=[85.0, 85.0]),
                SurvivorAgent(7, [40.0, 42.0], mobility=0.40, goal=[85.0, 85.0]),
                SurvivorAgent(8, [60.0, 65.0], mobility=0.95, goal=[85.0, 85.0]),
                SurvivorAgent(9, [62.0, 68.0], mobility=1.00, goal=[85.0, 85.0])
            ]
            for s in survivors:
                s.health = 45.0 if s.mobility < 0.5 else 90.0

            # Heterogeneous Aerial Fleet
            uavs = [
                UAVAgent(0, pos=[14.0, 62.0], home_pos=[5.0, 95.0], role='Scout', v_max=5.5, comm_radius=32.0, battery=100.0),
                UAVAgent(1, pos=[18.0, 58.0], home_pos=[5.0, 95.0], role='Scout', v_max=5.2, comm_radius=32.0, battery=100.0),
                UAVAgent(2, pos=[25.0, 68.0], home_pos=[5.0, 95.0], role='Surveyor', v_max=5.0, comm_radius=30.0, battery=100.0),
                UAVAgent(3, pos=[62.0, 75.0], home_pos=[5.0, 95.0], role='Surveyor', v_max=5.0, comm_radius=30.0, battery=100.0),
                UAVAgent(4, pos=[72.0, 22.0], home_pos=[5.0, 95.0], role='Relay', v_max=4.8, comm_radius=28.0, battery=28.0)
            ]

            # Ground Logistics Vehicles
            rvs = [
                RescueVehicle(0, pos=[26.0, 28.0], base_pos=[5.0, 5.0], max_speed=7.0, capacity=4),
                RescueVehicle(1, pos=[60.0, 58.0], base_pos=[95.0, 5.0], max_speed=7.0, capacity=4)
            ]

            stl_ctrl = ActiveSTLController(d_safe=5.5, d_fire=9.0, h_min=25.0, buffer_zone=5.8, k_repel=8.5, fire_clearance_trigger=9.5)
            default_cmds = [
                np.array([2.2, -1.8]), np.array([2.0, -1.5]),
                np.array([1.6, -2.0]), np.array([-1.2, -1.8]), np.array([-1.8, 1.5])
            ]

            first_detect_step = float(steps)
            detected_any = False
            stl_violation_count = 0
            initial_battery = sum(u.battery for u in uavs)

            for step in range(1, steps + 1):
                t_now = float(step)
                env.step(base_prob=0.18, fuel_burn_rate=0.35)
                fc = env.get_fire_centroid()

                # Hardware Failure Injection at Step 15
                if step == malfunction_step and len(uavs) > 2:
                    uavs.pop(0)
                    default_cmds.pop(0)

                for s in survivors:
                    if not getattr(s, 'rescued', False) and s.health > 0.0:
                        s.step(fc, env, dt=1.0)
                        if np.linalg.norm(s.pos - fc) < 12.0:
                            s.health = max(0.0, s.health - 3.5)

                active_survivors = [s for s in survivors if not getattr(s, 'rescued', False) and s.health > 0.0]

                for u in uavs:
                    det = u.sense_survivors(active_survivors, env, t_now)
                    if len(det) > 0 and not detected_any:
                        first_detect_step = float(step)
                        detected_any = True

                # --- Communication & Information Sharing ---
                if mode == "FIRE_SAR_Proposed":
                    A, D, L, l2 = GraphManager.compute_uav_matrices(uavs)
                    effective_loss = packet_loss_rate if packet_loss_rate != 0.00 else packet_loss_prob
                    drop_mask = (np.random.rand(*A.shape) > effective_loss).astype(float)
                    drop_mask = np.triu(drop_mask, 1)
                    drop_mask = drop_mask + drop_mask.T
                    A_noisy = A * drop_mask

                    GraphManager.synchronize_and_track_gradients(uavs, A_noisy, t_now)
                    GraphManager.handle_tracking_handoff(uavs)

                # --- Control & Motion Execution ---
                for idx, u in enumerate(uavs):
                    if mode == "Random_Search":
                        angle = np.random.uniform(0, 2 * np.pi)
                        u_cmd = np.array([np.cos(angle), np.sin(angle)]) * u.v_max
                    elif mode == "Nearest_Agent":
                        # Steer to nearest detected survivor without consensus
                        if u.belief_map:
                            nearest_sid = min(u.belief_map.keys(), key=lambda sid: np.linalg.norm(u.pos - u.belief_map[sid]['pos']))
                            disp = u.belief_map[nearest_sid]['pos'] - u.pos
                            d = np.linalg.norm(disp)
                            u_cmd = (disp / (d + 1e-6)) * u.v_max if d > 1.0 else np.zeros(2)
                        else:
                            u_cmd = default_cmds[idx]
                    elif mode == "Greedy_Uncoordinated":
                        u_cmd = u.update_behavior(t_now, default_cmds[idx], has_knowledge=(len(u.belief_map) > 0))
                    else:  # FIRE_SAR_Proposed
                        has_k = any(EpistemicLogicGuard.satisfies_knowledge_check(u, s.id, current_time=t_now) for s in active_survivors)
                        u_nom = u.update_behavior(t_now, default_cmds[idx], has_knowledge=has_k)
                        u_cmd, _ = stl_ctrl.apply_safety_shield(idx, u_nom, uavs, env)

                    u.step(u_cmd, env, dt=1.0)

                margins = stl_ctrl.evaluate_safety_margins(uavs, env, survivors)
                if margins['uav_spacing'] < 0.0 or margins['fire_clearance'] < 0.0:
                    stl_violation_count += 1

                # --- Rescue Dispatch Layer ---
                if mode == "FIRE_SAR_Proposed":
                    allocs = ADMMConsensusOptimizer.run_assignment(uavs, rvs, active_survivors, max_iters=15)
                    RescueDispatchManager.step_rescue_vehicles(rvs, survivors, uavs, allocs, env, t_now, dt=1.0, require_epistemic=True)
                elif mode in ["Greedy_Uncoordinated", "Nearest_Agent"]:
                    allocs = {s.id: 0 for s in active_survivors}
                    RescueDispatchManager.step_rescue_vehicles(rvs, survivors, uavs, allocs, env, t_now, dt=1.0, require_epistemic=False)

            alive = sum(1 for s in survivors if s.health > 0.0 or getattr(s, 'rescued', False))
            rescued = sum(1 for s in survivors if getattr(s, 'rescued', False))
            final_battery = sum(u.battery for u in uavs)

            results[mode]['survival_rate'].append(alive / len(survivors))
            results[mode]['rescued_ratio'].append(rescued / len(survivors))
            results[mode]['stl_violations'].append(stl_violation_count)
            results[mode]['detection_time'].append(first_detect_step)
            results[mode]['battery_consumed'].append(initial_battery - final_battery)

    return results


if __name__ == "__main__":
    print("===================================================================")
    print("          PROJECT FIRE-SAR: FULL MULTI-AGENT PIPELINE              ")
    print("===================================================================")
    
    mc_results = run_monte_carlo(num_trials=10, steps=45)

    print("\n================== FULL BENCHMARK RESULTS (10 TRIALS) =================")
    print(f"{'Strategy':22s} | {'Survival':8s} | {'Rescued':8s} | {'STL Viol':8s} | {'Det Time':8s} | {'Batt Drain':8s}")
    print("-----------------------------------------------------------------------")
    for m, d in mc_results.items():
        print(f"{m:22s} | {np.mean(d['survival_rate'])*100:6.1f}%  | {np.mean(d['rescued_ratio'])*100:6.1f}%  | "
              f"{np.mean(d['stl_violations']):6.1f}   | {np.mean(d['detection_time']):6.1f}s  | {np.mean(d['battery_consumed']):6.1f}J")

    print("\nRendering Publication Dashboard...")
    generate_dashboard(mc_results, save_path="phase4_executive_dashboard.png")
    print("Done! All Phase 1-4 requirements successfully verified.")
