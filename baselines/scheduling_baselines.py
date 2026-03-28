import random
from typing import List, Dict
# Assuming we import our previously defined TaskDAG and Subtask classes
# from environment.system_models import TaskDAG, Subtask

class BaselineSchedulers:
    """
    Implements the baseline scheduling algorithms to compare against the GCN.
    """
    
    @staticmethod
    def random_scheduling(task_dag) -> List[int]:
        """
        Random Scheduling: Completely ignores dependencies and randomly 
        shuffles the subtasks to generate an execution priority.
        """
        # Extract all subtask IDs from the DAG
        subtask_ids = list(task_dag.subtasks.keys())
        
        # Shuffle them randomly
        random.shuffle(subtask_ids)
        
        return subtask_ids

    @staticmethod
    def greedy_scheduling(task_dag) -> List[int]:
        """
        Greedy Scheduling: Prioritizes the subtask with the highest 
        cumulative computation volume of its successor subtasks.
        """
        subtask_ids = list(task_dag.subtasks.keys())
        cumulative_computations = {}
        
        # Helper function to recursively calculate computation volume of successors
        def get_successor_computation(current_id: int) -> float:
            # Find direct successors from the DAG edges
            successors = [succ for pred, succ in task_dag.edges if pred == current_id]
            
            if not successors:
                return 0.0
            
            total_volume = 0.0
            for succ_id in successors:
                # Add the successor's CPU cycles
                total_volume += task_dag.subtasks[succ_id].cpu_cycles
                # Recursively add the volume of its subsequent successors
                total_volume += get_successor_computation(succ_id)
                
            return total_volume

        # Calculate the cumulative successor computation for each subtask
        for s_id in subtask_ids:
            cumulative_computations[s_id] = get_successor_computation(s_id)
            
        # Sort subtask IDs based on the calculated volume in descending order
        # Subtasks with the largest successor computation volume get highest priority
        greedy_priority = sorted(
            subtask_ids, 
            key=lambda x: cumulative_computations[x], 
            reverse=True
        )
        
        return greedy_priority