"""
train_rl.py
Offline Reinforcement Learning Training Loop for UAV Task Allocation.
Runs 500 fast-forwarded episodes to converge the Q-table weights.
"""

import numpy as np
import json
from src.environment import ForestEnvironment
from src.agents import UAVAgent, SurvivorAgent, RLUAVTaskAllocator

def train_uav_policy(episodes=500, steps_per_episode=40):
    print(f"Starting offline RL training for {episodes} episodes...")
    allocator = RLUAVTaskAllocator()
    # Increase exploration for training
    allocator.epsilon = 0.5  
    
    for ep in range(episodes):
        env = ForestEnvironment(width=100, height=100)
        env.ignite(30, 30, radius=3)
        
        survivors = [SurvivorAgent(i, [np.random.uniform(20, 80), np.random.uniform(20, 80)]) for i in range(5)]
        uav = UAVAgent(0, [10.0, 10.0], [5.0, 5.0])
        uav.allocator = allocator
        
        # Decay exploration rate over time
        allocator.epsilon = max(0.05, allocator.epsilon * 0.99)
        
        total_reward = 0
        for step in range(steps_per_episode):
            env.step()
            fc = env.get_fire_centroid()
            
            # UAV senses the environment
            det = uav.sense_survivors(survivors, env, float(step))
            has_knowledge = len(uav.belief_map) > 0
            
            # Fire risk proxy: distance to fire
            f_dist = np.linalg.norm(uav.pos - fc)
            fire_risk = 1.0 if f_dist < 10.0 else 0.0
            
            # Step the behavior (this triggers allocator.update internally)
            default_cmd = np.array([1.5, 1.5])
            u_cmd = uav.update_behavior(float(step), default_cmd, has_knowledge, fire_risk)
            uav.step(u_cmd, env, dt=1.0)
            
            # Accumulate reward just for tracking progress
            total_reward += 1.0 if uav.task == "TRACK" and has_knowledge else 0.0
            
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{episodes} completed. Epsilon: {allocator.epsilon:.3f}")

    # Format the Q-table for easy copy-pasting into agents.py
    print("\n=== TRAINING COMPLETE: OPTIMAL Q-TABLE WEIGHTS ===")
    formatted_q = "{\n"
    for state, q_vals in allocator.q_table.items():
        formatted_q += f"    {state}: np.array([{', '.join([f'{q:.2f}' for q in q_vals])}]),\n"
    formatted_q += "}"
    print(formatted_q)

if __name__ == "__main__":
    train_uav_policy()
