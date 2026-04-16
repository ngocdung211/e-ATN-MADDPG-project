"""Baseline scheduling heuristics."""

import random
from typing import Dict, List

from environment.system_model import TaskDAG

class BaselineSchedulers:
    """Implement baseline scheduling algorithms to compare against the GCN."""
    
    @staticmethod
    def random_scheduling(task_dag: TaskDAG) -> List[int]:
        """Generate a random subtask execution order.

        Args:
            task_dag: Task DAG containing subtasks.

        Returns:
            Randomized list of subtask IDs.
        """
        # Extract all subtask IDs from the DAG
        subtask_ids = list(task_dag.subtasks.keys())
        
        # Shuffle them randomly
        random.shuffle(subtask_ids)
        
        return subtask_ids

    @staticmethod
    def greedy_scheduling(task_dag: TaskDAG) -> List[int]:
        """Prioritize subtasks by successor computation volume.

        Args:
            task_dag: Task DAG containing subtasks and dependencies.

        Returns:
            List of subtask IDs ordered by descending successor computation.
        """
        subtask_ids = list(task_dag.subtasks.keys())
        cumulative_computations: Dict[int, float] = {}
        
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
        for subtask_id in subtask_ids:
            cumulative_computations[subtask_id] = get_successor_computation(subtask_id)
            
        # Sort subtask IDs based on the calculated volume in descending order
        # Subtasks with the largest successor computation volume get highest priority
        greedy_priority = sorted(
            subtask_ids,
            key=lambda subtask_id: cumulative_computations[subtask_id],
            reverse=True,
        )
        
        return greedy_priority
