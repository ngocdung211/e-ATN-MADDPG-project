import numpy as np
import torch

def extract_gcn_inputs(task_dag):
    """
    Trích xuất Ma trận kề (A) và Ma trận đặc trưng (X) từ TaskDAG.
    Đặc trưng X gồm 3 chiều: [hierarchy_level, out_degree, computation_volume]
    """
    num_nodes = len(task_dag.subtasks)
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    X = np.zeros((num_nodes, 3), dtype=np.float32)

    # 1. Xây dựng ma trận kề A dựa trên dependencies (edges)
    for pred, succ in task_dag.edges:
        A[pred - 1, succ - 1] = 1.0  # Giả sử id bắt đầu từ 1

    # 2. Xây dựng ma trận đặc trưng X
    for sub_id, subtask in task_dag.subtasks.items():
        idx = sub_id - 1
        # Out-degree: Số lượng subtask con phụ thuộc trực tiếp vào nó
        out_degree = sum(1 for p, s in task_dag.edges if p == sub_id)
        
        # Gán vào ma trận X (chuẩn hóa computation_volume để tránh tràn số)
        X[idx, 0] = 1.0                     # Tạm gán hierarchy level tĩnh
        X[idx, 1] = float(out_degree)
        X[idx, 2] = subtask.cpu_cycles / 1e6 # Scale down CPU cycles

    return torch.FloatTensor(X), torch.FloatTensor(A)