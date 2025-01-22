import torch

# -------------------
#  Define row/col norms for 2D input
# -------------------
def row_norm_2D(x):
    """
    x: shape [N, M]. We subtract logsumexp over dimension=1
       so that each row sums to 1 (in normal space).
    """
    return x - torch.logsumexp(x, dim=1, keepdim=True)

def col_norm_2D(x):
    """
    x: shape [N, M]. We subtract logsumexp over dimension=0
       so that each column sums to 1 (in normal space).
    """
    return x - torch.logsumexp(x, dim=0, keepdim=True)

# -------------------
#  Your sample input
# -------------------
x = torch.tensor([
    [1., 2., 3.],
    [2., 3., 4.],
    [3., 4., 5.]
])

# Optional "temperature" tau
tau = 0.05

# Number of Sinkhorn iterations
sinkhorn_iters = 10

# Move into log-space (also incorporate tau if desired)
x = x / tau

# -------------------
#  Iterative Sinkhorn
# -------------------
for _ in range(sinkhorn_iters):
    x = row_norm_2D(x)  # make each row sum to 1
    x = col_norm_2D(x)  # make each column sum to 1

# Map back to normal space & optionally add small eps
eps = 1e-9
prob = torch.exp(x) + eps

# -------------------
#  Check row/col sums
# -------------------
row_sum = prob.sum(dim=1)   # sums over columns => each row's total
col_sum = prob.sum(dim=0)   # sums over rows => each column's total

print("Final doubly-stochastic matrix (approx):\n", prob)
print("\nRow sums:\n", row_sum)
print("Column sums:\n", col_sum)
