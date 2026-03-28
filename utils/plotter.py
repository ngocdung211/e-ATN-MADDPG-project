import matplotlib.pyplot as plt
import numpy as np
import os

class DITENPlotter:
    """
    Generates comparison charts formatted to match the paper's visual style.
    """
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set a clean style resembling the paper's formatting
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_training_curve(self, data_dict: dict, title: str, ylabel: str, filename: str):
        """
        Generates line charts like Figure 6 (GCN vs Baselines) or Figure 8 (Reward Comparison).
        
        data_dict: Dictionary where keys are algorithm names and values are lists of metrics over episodes.
        """
        plt.figure(figsize=(10, 6))
        
        # Define specific markers and colors to mimic the paper's style
        markers = ['*', '^', 's', 'o', 'd', 'v']
        colors = ['red', 'orange', 'blue', 'green', 'purple', 'cyan']
        
        for i, (algo_name, values) in enumerate(data_dict.items()):
            # Plot every 10th marker to avoid cluttering the graph, just like the paper
            plt.plot(
                values, 
                label=algo_name, 
                marker=markers[i % len(markers)], 
                markevery=20, 
                color=colors[i % len(colors)],
                linewidth=1.5,
                markersize=6
            )
            
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {save_path}")
        plt.close()

    def plot_device_scaling_bar_chart(self, devices_list: list, data_dict: dict, ylabel: str, filename: str):
        """
        Generates grouped bar charts like Figure 7 (Average rewards vs number of devices).
        
        devices_list: e.g., [5, 10, 15]
        data_dict: Dictionary mapping algorithms to a list of average rewards.
        """
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(devices_list))
        num_algos = len(data_dict)
        width = 0.8 / num_algos # Dynamic bar width
        
        colors = ['blue', 'yellow', 'purple', 'green', 'orange', 'red']
        
        for i, (algo_name, values) in enumerate(data_dict.items()):
            # Offset each algorithm's bar so they group together nicely
            offset = (i - num_algos / 2) * width + width / 2
            plt.bar(x + offset, values, width, label=algo_name, color=colors[i % len(colors)])
            
        plt.xlabel('The number of industrial devices', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(x, devices_list)
        plt.legend(loc='best', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved successfully to {save_path}")
        plt.close()