import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Computes D^{-1/2} * (A + I) * D^{-1/2} as defined in the paper.
    """
    # Add identity matrix to adjacency matrix (A + I)
    I = torch.eye(A.size(0), device=A.device)
    A_hat = A + I
    
    # Calculate degree matrix D
    D = torch.sum(A_hat, dim=1)
    
    # D^{-1/2}
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0 # Handle division by zero
    
    # Create diagonal matrix for D^{-1/2}
    D_inv_sqrt_mat = torch.diag(D_inv_sqrt)
    
    # D^{-1/2} * (A + I) * D^{-1/2}
    normalized_A = torch.mm(torch.mm(D_inv_sqrt_mat, A_hat), D_inv_sqrt_mat)
    return normalized_A

class GCNLayer(nn.Module):
    """
    A single GCN layer implementing X^{(l+1)} = \sigma(D^{-1/2}(A+I)D^{-1/2} X^{(l)} W^{(l)})
    """
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        # Trainable weight matrix W^{(l)}
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X: torch.Tensor, normalized_A: torch.Tensor) -> torch.Tensor:
        # X * W
        support = torch.mm(X, self.weight)
        # normalized_A * (X * W)
        output = torch.mm(normalized_A, support)
        return output

class TaskPriorityGCN(nn.Module):
    """
    The 3-layer GCN architecture for extracting subtask execution priorities.
    """
    def __init__(self, num_features: int, hidden_dim: int = 32):
        super(TaskPriorityGCN, self).__init__()
        # Layer 1: Input to Hidden
        self.gcn1 = GCNLayer(num_features, hidden_dim)
        # Layer 2: Hidden to Hidden
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        # Layer 3: Hidden to Output (1 scalar priority value per subtask)
        self.gcn3 = GCNLayer(hidden_dim, 1)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X: Node feature matrix (e.g., subtask hierarchy, out-degree, computation volume)
        A: Adjacency matrix of the DAG
        """
        # Precompute the normalized adjacency matrix
        normalized_A = normalize_adjacency(A)
        
        # Layer 1 + ReLU
        x = self.gcn1(X, normalized_A)
        x = F.relu(x)
        
        # Layer 2 + ReLU
        x = self.gcn2(x, normalized_A)
        x = F.relu(x)
        
        # Layer 3 (Output layer to predict priority score)
        x = self.gcn3(x, normalized_A)
        
        return x

# --- Training Setup Example ---
# If you are testing this isolated from the rest of the code:
# model = TaskPriorityGCN(num_features=3, hidden_dim=32)
# optimizer = optim.Adam(model.parameters(), lr=0.01) # Learning rate from paper
# criterion = nn.MSELoss() # MSE Loss from paper