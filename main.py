"""
PROJECT FIRE-SAR: Full Execution Pipeline (Phases 1-4)
Orchestrates Environment, UAVs, Survivors, ADMM Rescue Dispatch, and Formal Safety.
"""
import numpy as np
import os
import matplotlib.pyplot as plt

# Import from source modules
from src.environment import ForestEnvironment
from src.agents import SurvivorAgent, UAVAgent, RescueVehicle
from src.coordination import GraphManager, ADMMConsensusOptimizer, RescueDispatchManager
from src.safety import ActiveSTLController, STLMonitor

def run_full_mission():
    np.random.seed(42)
    current_time = 0.0
    dt = 0.5
    max_steps = 60

    print("=== INITIALIZING FIRE-SAR MISSION ENVIRONMENT ===")
    
    # 1. Environment (Phase 1 & 2)
    forest = ForestEnvironment(width=60, height=60, wind_vector=(3.0, 1.8), c_slope=0.15)
    forest.ignite(20, 20, radius=2)

    # 2. Agents (Phase 1 & 2)
    survivors = [
        SurvivorAgent(0, [26.0, 26.0], mobility=1.0),
        SurvivorAgent(1, [27.0, 25.0], mobility=0.8),
        SurvivorAgent(2, [28.0, 28.0], mobility=0.3),  # Injured
        SurvivorAgent(3, [24.0, 28.0], mobility=0.9)
    ]

    uavs = [
        UAVAgent(0, [10.0, 50.0], role='Scout', v_max=5.0, comm_radius=22.0, sense_radius=15.0),
        UAVAgent(1, [15.0, 45.0], role='Scout', v_max=5.0, comm_radius=22.0, sense_radius=15.0),
        UAVAgent(2, [30.0, 55.0], role='Relay', v_max=4.0, comm_radius=25.0, sense_radius=18.0)
    ]

    # Ground Rescue Vehicles for Phase 3 ADMM Dispatch
    rescue_vehicles = [
        RescueVehicle(0, [2.0, 2.0], max_speed=2.5, capacity=2),
        RescueVehicle(1, [58.0, 2.0], max_speed=3.0, capacity=2)
    ]

    # 3. Formal Safety Controllers (Phase 4)
    stl_controller = ActiveSTLController(d_safe=4.5, d_fire=6.0, buffer_zone=8.0, k_repel=12.0)
    stl_monitor = STLMonitor(d_safe=4.5, d_fire=6.0)

    # Search pattern default vectors
    default_commands = [
        np.array([1.2, -1.5]),
        np.array([1.0, -1.8]),
        np.array([-0.5, -0.8])
    ]

    print("=== COMMENCING MISSION LOOP ===")
    
    for step in range(1, max_steps + 1):
        current_time += dt
        
        # Step Environment
        forest.step()
        fc = forest.get_fire_centroid()

        # Step Survivors
        for s in survivors:
            s.step(fc, forest, dt)

        # 1. UAV Sensing (P_D Model)
        for u in uavs:
            u.sense_survivors(survivors, forest, current_time)

        # 2. Dynamic Communication Graph & Belief Synchronization
        A_u, D_u, L_u, l2 = GraphManager.compute_uav_matrices(uavs)
        GraphManager.synchronize_and_track_gradients(uavs, A_u, current_time)

        # 3. UAV Behavioral Switch (SEARCH -> TRACK)
        for idx, u in enumerate(uavs):
            u_nom = u.update_behavior(current_time, default_commands[idx])
            
            # Phase 4: Apply STL Safety Shield (Collision / Fire Avoidance)
            u_safe, shield_active = stl_controller.apply_safety_shield(idx, u_nom, uavs, forest)
            if shield_active:
                u.task = "AVOID"
                
            u.step(u_safe, dt)

        # 4. Handle Redundant Tracking Handoffs
        GraphManager.handle_tracking_handoff(uavs)

        # 5. Rescue Team Allocation via ADMM (Phase 3)
        # Execute ADMM optimizer every 10 steps to assign trucks to targets
        if step % 10 == 0:
            allocations = ADMMConsensusOptimizer.run_assignment(uavs, rescue_vehicles, survivors, max_iters=20, rho=2.5)
            if allocations:
                print(f"[t={current_time:.1f}s] ADMM Allocations Computed: {allocations}")

            # 6. Epistemic Guarded Dispatch (Phase 4)
            rescued_this_step = RescueDispatchManager.step_rescue_vehicles(
                rescue_vehicles, survivors, uavs, allocations, forest, current_time, dt, pickup_radius=5.0, require_epistemic=True
            )
            for rid in rescued_this_step:
                print(f">>> CRITICAL: Survivor {rid} RESCUED by Ground Team at t={current_time:.1f}s!")

        # 7. STL Robustness Monitoring Logging
        if step % 15 == 0:
            margins = stl_monitor.evaluate_safety_margins(uavs, forest, survivors)
            print(f"[t={current_time:.1f}s] STL Robustness | UAV Spacing: {margins['uav_spacing']:.2f} | Fire Standoff: {margins['fire_clearance']:.2f}")

    print("=== MISSION COMPLETE ===")
    
    # Final Metric Outputs
    total_rescued = sum([1 for s in survivors if s.rescued])
    print(f"Total Survivors Rescued: {total_rescued} / {len(survivors)}")
    for i, u in enumerate(uavs):
        print(f"UAV {i} Battery Remaining: {u.battery:.1f}%")

if __name__ == "__main__":
    run_full_mission()
