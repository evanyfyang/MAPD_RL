import torch
import torch.nn as nn


class SPMPNNGridEncoder(nn.Module):
    """
    Grid-only SP-MPNN encoder:
    - Nodes: grid cells only
    - Edges: shortest-path distance buckets (dist_1..dist_k)
    """

    def __init__(self, input_dim=4, hidden_dim=64, num_layers=3, max_distance=3, dropout=0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_distance = int(max_distance)
        self.dropout = float(dropout)

        self.node_init = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

        self.pre_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.num_layers)])

        self.message_mlps = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        for _ in range(self.num_layers):
            dist_mlp = nn.ModuleDict()
            for d in range(1, self.max_distance + 1):
                dist_mlp[f"dist_{d}"] = nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
            self.message_mlps.append(dist_mlp)

            # self + dist_1..dist_k
            message_types = 1 + self.max_distance
            self.update_mlps.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim * message_types, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
            )

    def _aggregate_messages(self, x, edge_index, msg_mlp):
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros_like(x)

        src, dst = edge_index
        m_in = torch.cat([x[dst], x[src]], dim=-1)
        m = msg_mlp(m_in)

        out = torch.zeros_like(x)
        out = out.index_add(0, dst, m)

        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg = deg.index_add(0, dst, torch.ones(dst.size(0), device=x.device, dtype=x.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(-1)
        return out / deg

    def forward(self, node_features, distance_edges):
        """
        node_features: [N, input_dim]
        distance_edges: dict with keys dist_1..dist_k, value [2, E_d]
        """
        x = self.node_init(node_features)
        for layer_idx in range(self.num_layers):
            x_norm = self.pre_norms[layer_idx](x)

            parts = [x_norm]  # self message
            for d in range(1, self.max_distance + 1):
                key = f"dist_{d}"
                edge_index = distance_edges.get(key, None)
                part = self._aggregate_messages(
                    x_norm,
                    edge_index,
                    self.message_mlps[layer_idx][key],
                )
                parts.append(part)

            merged = torch.cat(parts, dim=-1)
            delta = self.update_mlps[layer_idx](merged)
            delta = self.dropout_layers[layer_idx](delta)
            x = x + delta
        return x


class RingRegressionHead(nn.Module):
    """
    Predict ring labels from center embedding.
    Output layout per center:
      [num_rings, 4] where channels are
      (agent_ratio, pickup_ratio, delivery_ratio, heat_mean)
    """

    def __init__(self, hidden_dim=64, num_rings=4, head_hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_rings = int(num_rings)
        self.out_dim = self.num_rings * 4
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden_dim, self.out_dim),
        )

    def forward(self, z_center):
        return self.mlp(z_center).view(z_center.size(0), self.num_rings, 4)
