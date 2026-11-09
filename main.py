"""
PROJECT FIRE-SAR: Full Execution Pipeline (Phases 1-4)
Includes Monte Carlo Benchmarking with Hardware Malfunction, Packet Loss, & Round-Trip Logistics
"""
import numpy as np

from src.environment import ForestEnvironment
from src.agents import SurvivorAgent, UAVAgent, RescueVehicle
from src.safety import ActiveSTLController, EpistemicLogicGuard
from src.coordination import GraphManager, ADMMConsensusOptimizer, RescueDispatchManager
from src.visualization import generate_dashboard

def run_monte_carlo(num_trials=10, steps=45, packet_loss_prob=0.15, malfunction_step=15):
    modes = ["Random_Search", "Greedy_Uncoordinated", "FIRE_SAR_Proposed"]
    results = {m: {'survival_rate': [], 'rescued_ratio': [], 'stl_violations': [], 'detection_time': []} for m in modes}

    for mode in modes:
        for trial in range(num_trials):
            env = ForestEnvironment(width=100, height=100, wind_vector=(2.8, 1.8))
            np.random.seed(300 + trial)
            env.canopy = np.random.uniform(0.1, 0.65, (env.H, env.W))
            env.ignite(30, 30, radius=3)

            # Spawn 10 Survivors to force logistics round-trips
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

            uavs = [
                UAVAgent(0, pos=[14.0, 62.0], home_pos=[5.0, 95.0], role='Scout', v_max=5.5, comm_radius=32.0, battery=100.0),
                UAVAgent(1, pos=[18.0, 58.0], home_pos=[5.0, 95.0], role='Scout', v_max=5.2, comm_radius=32.0, battery=100.0),
                UAVAgent(2, pos=[25.0, 68.0], home_pos=[5.0, 95.0], role='Surveyor', v_max=5.0, comm_radius=30.0, battery=100.0),
                UAVAgent(3, pos=[62.0, 75.0], home_pos=[5.0, 95.0], role='Surveyor', v_max=5.0, comm_radius=30.0, battery=100.0),
                UAVAgent(4, pos=[72.0, 22.0], home_pos=[5.0, 95.0], role='Relay', v_max=4.8, comm_radius=28.0, battery=28.0) # Low battery to trigger return
            ]
            
            # Ground Rescue Vehicles initialized with a Hospital/Base coordinate
            rvs = [
                RescueVehicle(0, pos=[26.0, 28.0], base_pos=[5.0, 5.0], max_speed=5.0, capacity=3),
                RescueVehicle(1, pos=[60.0, 58.0], base_pos=[95.0, 5.0], max_speed=5.0, capacity=3)
            ]

            stl_ctrl = ActiveSTLController(d_safe=5.5, d_fire=9.0, h_min=25.0, buffer_zone=6.5, k_repel=8.5)
            default_cmds = [
                np.array([2.2, -1.8]), np.array([2.0, -1.5]),
                np.array([1.6, -2.0]), np.array([-1.2, -1.8]), np.array([-1.8, 1.5])
            ]

            first_detect_step = float(steps)
            detected_any = False
            stl_violation_count = 0

            for step in range(1, steps + 1):
                t_now = float(step)
                env.step(base_prob=0.18, fuel_burn_rate=0.35)
                fc = env.get_fire_centroid()

                # --- CATASTROPHIC MALFUNCTION INJECTION ---
                if step == malfunction_step and len(uavs) > 2:
                    failed_drone = uavs.pop(0)
                    default_cmds.pop(0)
                    if trial == 0:
                        print(f"[{mode}] Step {step}: CRITICAL FAILURE! UAV {failed_drone.id} went offline. Surviving swarm is adapting...")

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

                if mode == "FIRE_SAR_Proposed":
                    A, D, L, l2 = GraphManager.compute_uav_matrices(uavs)
                    
                    # 15% Packet loss simulation
                    drop_mask = (np.random.rand(*A.shape) > packet_loss_prob).astype(float)
                    drop_mask = np.triu(drop_mask, 1)
                    drop_mask = drop_mask + drop_mask.T
                    A_noisy = A * drop_mask
                    
                    GraphManager.synchronize_and_track_gradients(uavs, A_noisy, t_now)
                    GraphManager.handle_tracking_handoff(uavs)

                for idx, u in enumerate(uavs):
                    if mode == "Random_Search":
                        angle = np.random.uniform(0, 2 * np.pi)
                        u_cmd = np.array([np.cos(angle), np.sin(angle)]) * u.v_max
                    elif mode == "Greedy_Uncoordinated":
                        u_cmd = u.update_behavior(t_now, default_cmds[idx], has_knowledge=(len(u.belief_map) > 0))
                    else:
                        has_k = any(EpistemicLogicGuard.satisfies_knowledge_check(u, s.id, current_time=t_now) for s in active_survivors)
                        u_nom = u.update_behavior(t_now, default_cmds[idx], has_knowledge=has_k)
                        u_cmd, _ = stl_ctrl.apply_safety_shield(idx, u_nom, uavs, env)
                    u.step(u_cmd, env, dt=1.0)

                margins = stl_ctrl.evaluate_safety_margins(uavs, env, survivors)
                if margins['uav_spacing'] < 0.0 or margins['fire_clearance'] < 0.0:
                    stl_violation_count += 1

                if mode == "FIRE_SAR_Proposed":
                    allocs = ADMMConsensusOptimizer.run_assignment(uavs, rvs, active_survivors, max_iters=15)
                    RescueDispatchManager.step_rescue_vehicles(rvs, survivors, uavs, allocs, env, t_now, dt=1.0, require_epistemic=True)
                elif mode == "Greedy_Uncoordinated":
                    allocs = {sid: 0 for sid in uavs[0].belief_map.keys()} if len(uavs) > 0 else {}
                    RescueDispatchManager.step_rescue_vehicles(rvs, survivors, uavs, allocs, env, t_now, dt=1.0, require_epistemic=False)

            alive_count = sum(1 for s in survivors if s.health > 0.0 or getattr(s, 'rescued', False))
            rescued_count = sum(1 for s in survivors if getattr(s, 'rescued', False))
            results[mode]['survival_rate'].append(alive_count / len(survivors))
            results[mode]['rescued_ratio'].append(rescued_count / len(survivors))
            results[mode]['stl_violations'].append(stl_violation_count)
            results[mode]['detection_time'].append(first_detect_step)

    return results


if __name__ == "__main__":
    print("=== STARTING FIRE-SAR PHASE 4 MONTE CARLO BENCHMARKS ===")
    print("Testing Fault Tolerance (Drone failure at Step 15) & Round-Trip Logistics (10 Survivors)...\n")
    mc_results = run_monte_carlo(num_trials=10, steps=45)
    
    print("\n=== MONTE CARLO BENCHMARK RESULTS (10 Trials) ===")
    for m, metrics in mc_results.items():
        print(f"{m:22s} | Survival: {np.mean(metrics['survival_rate']):6.1%} | "
              f"Rescued: {np.mean(metrics['rescued_ratio']):6.1%} | "
              f"STL Violations: {np.mean(metrics['stl_violations']):5.2f} | "
              f"First Detect: {np.mean(metrics['detection_time']):4.2f}s")
              
    print("\nGenerating Executive Dashboard...")
    generate_dashboard(mc_results)
    print("Run complete!")
