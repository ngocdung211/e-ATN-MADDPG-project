"""Plotting utilities for comparison experiments."""

import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

class DITENPlotter2:
    """Plot comparison curves with paper-aligned styles."""

    def __init__(self, save_dir: str = "plots"):
        """Initialize the plotter.

        Args:
            save_dir: Output directory for saved plots.
        """
        self.save_dir: str = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
            
        # Hardcoded styles to match the distinct lines in the paper's figures
        self.styles = {
            "MAAC":         {"color": "#1f77b4", "marker": "v"},  # Blue, Triangle Down
            "MAPPO":        {"color": "#ff7f0e", "marker": "^"},  # Orange, Triangle Up
            "MADDPG":       {"color": "#2ca02c", "marker": "o"},  # Green, Circle
            "GR-MADDPG":    {"color": "#00ffff", "marker": "s"},  # Cyan, Square
            "ATN-MADDPG":   {"color": "#800080", "marker": "+"},  # Purple, Plus
            "GAT-Mask MAPPO": {"color": "#d62728", "marker": "*"}   # Red, Star (Proposed)
        }

    def plot_training_curve(
        self, data_dict: Dict[str, List[float]], title: str, ylabel: str, filename: str
    ) -> None:
        """Plot multiple training curves on the same chart.

        Args:
            data_dict: Mapping of algorithm names to metric histories.
            title: Plot title.
            ylabel: Y-axis label.
            filename: Output filename.
        """
        plt.figure(figsize=(10, 6))
        
        for algo_name, data in data_dict.items():
            episodes = np.arange(len(data))
            
            # Fetch style if defined, otherwise use matplotlib defaults
            style = self.styles.get(algo_name, {"color": None, "marker": None})
            
            # Space out the markers so they are legible (e.g., plot a marker every 20 episodes)
            markevery = max(1, len(data) // 100)
            
            plt.plot(
                episodes, 
                data, 
                label=algo_name, 
                color=style["color"], 
                marker=style["marker"], 
                markevery=markevery,
                linewidth=1.4,
                markersize=6,
                alpha=0.9 # Slight transparency to see overlapping lines
            )

        # Formatting matching standard IEEE plots
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        
        # Place legend in a standard spot (or outside the plot if it gets crowded)
        plt.legend(loc='best', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the plot
        
        date_string = time.strftime("%Y-%m-%d_%H", time.localtime())
        # folder = os.join('plots',date_string)
        os.makedirs(f"plots/{date_string}", exist_ok=True)
        self.save_dir = f"plots/{date_string}"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filepath}")
