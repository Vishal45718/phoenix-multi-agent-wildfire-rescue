"""
export_metrics.py
Generates the final CSV data logs for the project report.
"""

import pandas as pd
import numpy as np
import datetime
from main import run_monte_carlo

def export_benchmark_to_csv(num_trials=30):
    print(f"Running benchmarking suite for {num_trials} trials...")
    # Run the Monte Carlo simulation from main.py
    results = run_monte_carlo(num_trials=num_trials, steps=45)
    
    # Flatten the results into a dataset
    data = []
    for mode, metrics in results.items():
        for trial in range(num_trials):
            data.append({
                "Strategy": mode,
                "Trial_Number": trial + 1,
                "Survival_Rate_%": round(metrics['survival_rate'][trial] * 100, 2),
                "Rescued_Ratio_%": round(metrics['rescued_ratio'][trial] * 100, 2),
                "STL_Safety_Violations": metrics['stl_violations'][trial],
                "Detection_Time_sec": metrics['detection_time'][trial],
                "Battery_Consumed_Joules": round(metrics['battery_consumed'][trial], 2)
            })
            
    df = pd.DataFrame(data)
    
    # Calculate Summary Statistics
    summary = df.groupby("Strategy").agg({
        "Survival_Rate_%": "mean",
        "Rescued_Ratio_%": "mean",
        "STL_Safety_Violations": "mean",
        "Detection_Time_sec": "mean",
        "Battery_Consumed_Joules": "mean"
    }).round(2)
    
    # Save to files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    raw_filename = f"FIRE_SAR_Raw_Trials_{timestamp}.csv"
    summary_filename = f"FIRE_SAR_Summary_{timestamp}.csv"
    
    df.to_csv(raw_filename, index=False)
    summary.to_csv(summary_filename)
    
    print(f"\nSUCCESS: Data exported!")
    print(f"1. Raw data saved to: {raw_filename}")
    print(f"2. Averages saved to: {summary_filename}")
    print("\nSummary Table for your Report:")
    print(summary.to_string())

if __name__ == "__main__":
    export_benchmark_to_csv(num_trials=30)
