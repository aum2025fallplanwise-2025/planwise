from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from django.conf import settings
from django.db.models import Avg

from scheduling_app.models import SubtaskRelationship, CriticalPathPrediction, NormalizedEdge
from task_app.models import SubtaskVector


class DynamicGraphProvider:
    def __init__(self, node_list, graph_dict):
        self.node_list = list(node_list)
        self.graph_dict = graph_dict
        self.node2idx = {n: i for i, n in enumerate(self.node_list)}
        self.N = len(self.node_list)

        self.edge_list = []
        self.costs = []
        self.durations = []
        self.risks = []

        self._prepare_edges()
        self._normalize()
        self.in_deg, self.out_deg = self._degrees()
        self.max_costs     = self._all_pairs_longest_paths(self.norm_costs)
        self.max_durations = self._all_pairs_longest_paths(self.norm_durations)
        self.max_risks     = self._all_pairs_longest_paths(self.norm_risks)
        self.node_features = self._build_node_features()

    def _prepare_edges(self):
        for u in self.node_list:
            for v, c, d, r in self.graph_dict.get(u, []):
                if u not in self.node2idx or v not in self.node2idx:
                    continue
                self.edge_list.append((self.node2idx[u], self.node2idx[v]))
                self.costs.append(float(c))
                self.durations.append(float(d))
                self.risks.append(float(r))

    def _normalize(self):
        def norm(values):
            if not values:
                return np.array([], dtype=np.float32)
            mn, mx = min(values), max(values)
            if mx > mn:
                return np.array([(v - mn) / (mx - mn) for v in values], dtype=np.float32)
            return np.zeros(len(values), dtype=np.float32)
        self.norm_costs     = norm(self.costs)
        self.norm_durations = norm(self.durations)
        self.norm_risks     = norm(self.risks)

    def _degrees(self):
        in_d  = np.zeros(self.N, dtype=np.float32)
        out_d = np.zeros(self.N, dtype=np.float32)
        for u, v in self.edge_list:
            out_d[u] += 1.0
            in_d[v]  += 1.0
        return in_d, out_d

    def _bellman_ford_longest(self, start, adj):
        dist = np.full(self.N, -np.inf, dtype=np.float32)
        dist[start] = 0.0
        for _ in range(self.N - 1):
            for u in range(self.N):
                if dist[u] == -np.inf:
                    continue
                for v, w in adj[u]:
                    if dist[v] < dist[u] + w:
                        dist[v] = dist[u] + w
        dist[np.isinf(dist)] = 0.0
        return dist

    def _all_pairs_longest_paths(self, edge_weights):
        if self.N == 0:
            return np.zeros((0, 0), dtype=np.float32)
        adj = [[] for _ in range(self.N)]
        for i, (u, v) in enumerate(self.edge_list):
            w = float(edge_weights[i]) if len(edge_weights) > i else 0.0
            adj[u].append((v, w))
        M = np.zeros((self.N, self.N), dtype=np.float32)
        for src in range(self.N):
            M[src] = self._bellman_ford_longest(src, adj)
        return M

    def _build_node_features(self):
        feats = np.zeros((self.N, 9), dtype=np.float32)
        reachability    = np.sum(self.max_costs > 0, axis=1).astype(np.float32)
        in_reachability = np.sum(self.max_costs > 0, axis=0).astype(np.float32)
        max_cost_per_node     = self.max_costs.max(axis=1)     if self.N > 0 else np.zeros(self.N)
        max_duration_per_node = self.max_durations.max(axis=1) if self.N > 0 else np.zeros(self.N)

        for i in range(self.N):
            total       = self.in_deg[i] + self.out_deg[i] + 1e-9
            feats[i, 0] = self.in_deg[i]          / self.N
            feats[i, 1] = self.out_deg[i]         / self.N
            feats[i, 2] = self.in_deg[i]          / total
            feats[i, 3] = self.out_deg[i]         / total
            feats[i, 4] = total                   / (2 * self.N)
            feats[i, 5] = reachability[i]         / self.N
            feats[i, 6] = in_reachability[i]      / self.N
            feats[i, 7] = max_cost_per_node[i]
            feats[i, 8] = max_duration_per_node[i]

        for col in range(feats.shape[1]):
            mn, mx = feats[:, col].min(), feats[:, col].max()
            if mx > mn:
                feats[:, col] = (feats[:, col] - mn) / (mx - mn)

        return feats.astype(np.float32)

    def generate_edge_index(self):
        if not self.edge_list:
            return np.zeros((2, 0), dtype=np.int64)
        return np.array(self.edge_list, dtype=np.int64).T

    def get_graph(self, destination_node):
        edge_idx = self.generate_edge_index()
        dst_idx  = self.node2idx[destination_node]
        y        = self.max_costs[:, dst_idx].reshape(-1, 1).astype(np.float32)
        return self.node_features, edge_idx, y


class GATLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, num_heads=1, concat=True, negative_slope=0.2, **kwargs):
        super().__init__(**kwargs)
        self.out_dim        = out_dim
        self.num_heads      = num_heads
        self.concat         = concat
        self.negative_slope = negative_slope

    def build(self, input_shape):
        f_in   = input_shape[0][-1]
        self.W = self.add_weight(shape=(self.num_heads, f_in, self.out_dim),
                                 initializer="glorot_uniform", trainable=True)
        self.a = self.add_weight(shape=(self.num_heads, 2 * self.out_dim),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        X        = tf.cast(inputs[0], tf.float32)
        edge_idx = tf.cast(inputs[1], tf.int64)
        N        = tf.shape(X)[0]
        src      = edge_idx[0]
        dst      = edge_idx[1]
        outputs  = []

        for h in range(self.num_heads):
            Wh        = tf.matmul(X, self.W[h])
            num_edges = tf.shape(edge_idx)[1]

            def with_edges():
                Wh_src     = tf.gather(Wh, src)
                Wh_dst     = tf.gather(Wh, dst)
                e          = tf.reduce_sum(tf.concat([Wh_dst, Wh_src], axis=1) * self.a[h], axis=1)
                e          = tf.nn.leaky_relu(e, alpha=self.negative_slope)
                alpha      = tf.exp(e)
                denom      = tf.math.unsorted_segment_sum(alpha, dst, num_segments=N)
                alpha_norm = alpha / (tf.gather(denom, dst) + 1e-9)
                return tf.math.unsorted_segment_sum(
                    tf.expand_dims(alpha_norm, 1) * Wh_src, dst, num_segments=N)

            def without_edges():
                return Wh

            outputs.append(tf.cond(num_edges > 0, with_edges, without_edges))

        if self.concat:
            return tf.concat(outputs, axis=1)
        return tf.reduce_mean(tf.stack(outputs, axis=0), axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_dim": self.out_dim, "num_heads": self.num_heads,
            "concat": self.concat, "negative_slope": self.negative_slope
        })
        return config


class GATModel(tf.keras.Model):
    def __init__(self, hidden_dim=64, num_heads=4, out_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.gat1   = GATLayer(hidden_dim, num_heads=num_heads, concat=True,  name="gat_layer_1")
        self.gat2   = GATLayer(out_dim,    num_heads=2,         concat=True,  name="gat_layer_2")
        self.linear = tf.keras.layers.Dense(1, activation=None)

    def call(self, inp):
        X, E = inp
        h1   = tf.nn.elu(self.gat1([X, E]))
        h2   = tf.nn.elu(self.gat2([h1, E]))
        return self.linear(h2)


class GAT2Service:
    def __init__(self):
        self.model_path = Path(settings.BASE_DIR) / "pretrained_models" / "model4" / "GAT2_model.keras"

    def _subtask_stats_from_db(self, subtask_id):
        qs = NormalizedEdge.objects.filter(subtask_id=subtask_id)
        if not qs.exists():
            return {"cost": 1.0, "duration": 1.0, "risk": 1.0}
        agg = qs.aggregate(
            cost_avg=Avg("avg_defects"),
            duration_avg=Avg("avg_duration"),
            risk_avg=Avg("total_defects")
        )
        return {
            "cost":     float(agg["cost_avg"]     or 1.0),
            "duration": float(agg["duration_avg"] or 1.0),
            "risk":     float(agg["risk_avg"]     or 1.0),
        }

    def _build_graph_dict_from_db(self, br_id, node_list):
        qs               = SubtaskRelationship.objects.filter(br_id=br_id).order_by("id")
        graph_dict       = {n: [] for n in node_list}
        node_set         = set(node_list)
        name_to_id       = {
            str(r.subtask_name).strip(): int(r.subtask_id)
            for r in SubtaskVector.objects.all()
        }
        has_runtime_edges = False

        for row in qs:
            pred = str(row.predicted_relationship).strip().upper()
            if not pred or pred == "NO_RELATION":
                continue
            u = str(row.subtask_a).strip()
            v = str(row.subtask_b).strip()
            if u not in node_set or v not in node_set:
                continue
            v_id  = name_to_id.get(v)
            stats = self._subtask_stats_from_db(v_id) if v_id is not None else {"cost": 1.0, "duration": 1.0, "risk": 1.0}
            conf  = float(row.confidence or 0.0)
            graph_dict[u].append((v, stats["cost"], stats["duration"], stats["risk"] * (1.0 - conf + 1e-6)))
            has_runtime_edges = True

        if not has_runtime_edges:
            raise ValueError("No predicted subtask relationships found for this BR.")

        return graph_dict

    def execute_from_db(self, br_id):
        rows      = list(SubtaskVector.objects.all().order_by("subtask_id"))
        node_list = [str(r.subtask_name).strip() for r in rows]

        graph_dict = self._build_graph_dict_from_db(br_id, node_list)
        provider   = DynamicGraphProvider(node_list=node_list, graph_dict=graph_dict)

        model = tf.keras.models.load_model(
            str(self.model_path),
            custom_objects={"GATLayer": GATLayer, "GATModel": GATModel}
        )

        X, edge_idx, _ = provider.get_graph(node_list[0])

        y_pred = model([
            tf.constant(X,        dtype=tf.float32),
            tf.constant(edge_idx, dtype=tf.int64)
        ]).numpy().reshape(-1)

        min_val = y_pred.min()
        if min_val < 0:
            y_pred = y_pred + abs(min_val)

        runtime_nodes = [
            n for n in node_list
            if SubtaskRelationship.objects.filter(br_id=br_id, subtask_a=n).exists()
            or SubtaskRelationship.objects.filter(br_id=br_id, subtask_b=n).exists()
        ]
        if not runtime_nodes:
            runtime_nodes = [node_list[0]]

        export_df = pd.DataFrame({
            "subtask":         provider.node_list,
            "predicted_score": y_pred,
            "in_degree":       provider.in_deg,
            "out_degree":      provider.out_deg,
        })

        export_df = export_df[export_df["subtask"].isin(runtime_nodes)].copy()
        export_df = export_df.sort_values(by="predicted_score", ascending=False)

        CriticalPathPrediction.objects.filter(br_id=br_id).delete()

        for _, row in export_df.iterrows():
            CriticalPathPrediction.objects.create(
                br_id=br_id,
                subtask_name=str(row["subtask"]),
                predicted_score=float(row["predicted_score"]),
                in_degree=float(row["in_degree"]),
                out_degree=float(row["out_degree"]),
            )

        return {
            "row_count": int(len(export_df)),
            "table":     export_df.to_dict(orient="records"),
        }