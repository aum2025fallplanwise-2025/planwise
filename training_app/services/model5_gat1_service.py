from pathlib import Path

import ahpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from django.conf import settings
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn import GATConv

from employee_app.models import Employee, EmployeeVector
from task_app.models import SubtaskVector
from project_app.models import SystemRequirement
from scheduling_app.models import NormalizedEdge, OptimalAssignment, CriticalPathPrediction


class BipartiteGAT_EdgeFeat(nn.Module):
    def __init__(self, emp_in, sub_in, edge_in=3, hid=64, heads=4, dropout=0.2):
        super().__init__()
        self.emp_proj = nn.Linear(emp_in, hid)
        self.sub_proj = nn.Linear(sub_in, hid)
        self.gat1     = GATConv(hid, hid, heads=heads, concat=True,  dropout=dropout)
        self.gat2     = GATConv(hid * heads, hid, heads=1, concat=False, dropout=dropout)
        self.scorer   = nn.Sequential(
            nn.Linear(hid * 2 + edge_in, hid),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hid, 3)
        )

    def forward(self, X_emp, X_sub, edge_index, u, v, edge_attr_std):
        E = X_emp.size(0)
        x = torch.zeros((E + X_sub.size(0), self.emp_proj.out_features), device=X_emp.device)
        x[:E] = self.emp_proj(X_emp)
        x[E:] = self.sub_proj(X_sub)
        x  = self.gat1(x, edge_index)
        x  = torch.relu(x)
        x  = self.gat2(x, edge_index)
        zu = x[u]
        zv = x[v + E]
        return self.scorer(torch.cat([zu, zv, edge_attr_std], dim=1))


class GAT1Service:
    def __init__(self):
        self.model_path = Path(settings.BASE_DIR) / "pretrained_models" / "model5" / "GAT_cost_model.pt"
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _runtime_subtask_names(self, br_id):
        rows  = SystemRequirement.objects.filter(
            business_requirement__br_id=br_id
        ).order_by("id")
        names = []
        seen  = set()
        for row in rows:
            name = str(row.task_subtype).strip()
            if not name:
                continue
            key = name.lower()
            if key not in seen:
                names.append(name)
                seen.add(key)
        if not names:
            raise ValueError("No runtime subtasks found for this BR.")
        return names

    def _load_db_frames(self, br_id):
        runtime_names     = self._runtime_subtask_names(br_id)
        runtime_name_keys = {x.lower(): x for x in runtime_names}

        edge_rows = []
        for row in NormalizedEdge.objects.all().order_by("id"):
            edge_rows.append({
                "Employee ID":                 row.employee_id,
                "Subtask ID":                  row.subtask_id,
                "Avg Duration (Hours)":        row.avg_duration,
                "Avg Defects":                 row.avg_defects,
                "Total Tasks Done (Sub Task)": row.total_tasks_done,
                "Total Defects (Sub Task)":    row.total_defects,
            })
        edges_df = pd.DataFrame(edge_rows)

        emp_rows = []
        for row in EmployeeVector.objects.select_related("employee").all().order_by("employee__employee_id"):
            payload = {"EmployeeID": row.employee.employee_id}
            for i, v in enumerate(row.vector_data):
                payload[f"e{i}"] = float(v)
            emp_rows.append(payload)
        emp_df = pd.DataFrame(emp_rows)

        gat2_rows = {}
        for row in CriticalPathPrediction.objects.all():
            key = str(row.subtask_name).strip().lower()
            gat2_rows[key] = {
                "GAT_Predicted_Critical_Score": float(row.predicted_score),
                "In_Degree":                    float(row.in_degree),
                "Out_Degree":                   float(row.out_degree),
            }

        if gat2_rows:
            all_scores = [v["GAT_Predicted_Critical_Score"] for v in gat2_rows.values()]
            min_score  = min(all_scores)
            if min_score < 0:
                for key in gat2_rows:
                    gat2_rows[key]["GAT_Predicted_Critical_Score"] += abs(min_score)

        sub_rows = []
        for row in SubtaskVector.objects.all().order_by("subtask_id"):
            sub_name = str(row.subtask_name).strip()
            if sub_name.lower() not in runtime_name_keys:
                continue
            payload = {"SubtaskID_v8": row.subtask_id, "SubtaskName": sub_name}
            for i, v in enumerate(row.vector_data):
                payload[f"s{i}"] = float(v)

            gat2_feat = gat2_rows.get(sub_name.lower(), {
                "GAT_Predicted_Critical_Score": 0.0,
                "In_Degree":                    0.0,
                "Out_Degree":                   0.0,
            })
            payload["GAT_Predicted_Critical_Score"] = gat2_feat["GAT_Predicted_Critical_Score"]
            payload["In_Degree"]                    = gat2_feat["In_Degree"]
            payload["Out_Degree"]                   = gat2_feat["Out_Degree"]
            sub_rows.append(payload)

        sub_df = pd.DataFrame(sub_rows)

        if edges_df.empty:
            raise ValueError("No NormalizedEdge rows found in database.")
        if emp_df.empty:
            raise ValueError("No EmployeeVector rows found in database.")
        if sub_df.empty:
            raise ValueError("No matching SubtaskVector rows found for this BR subtasks.")

        for col in ["GAT_Predicted_Critical_Score", "In_Degree", "Out_Degree"]:
            mu_col  = sub_df[col].mean()
            std_col = sub_df[col].std()
            sub_df[col] = (sub_df[col] - mu_col) / (std_col + 1e-8)

        valid_sub_ids = set(sub_df["SubtaskID_v8"].tolist())
        edges_df = edges_df[edges_df["Subtask ID"].isin(valid_sub_ids)].copy().reset_index(drop=True)

        if edges_df.empty:
            raise ValueError("No NormalizedEdge rows matched the runtime subtasks for this BR.")

        zero_mask = (
            (edges_df["Avg Duration (Hours)"] == 0)
            & (edges_df["Avg Defects"] == 0)
            & (edges_df["Total Defects (Sub Task)"] == 0)
        )
        edges_df = edges_df[~zero_mask].copy().reset_index(drop=True)

        if edges_df.empty:
            raise ValueError("All matched edges were zero rows.")

        return edges_df, emp_df, sub_df

    def _prepare_graph(self, br_id):
        edges_df, emp_df, sub_df = self._load_db_frames(br_id)

        emp_ids = sorted(emp_df["EmployeeID"].unique())
        emp2idx = {e: i for i, e in enumerate(emp_ids)}
        sub_ids = sorted(sub_df["SubtaskID_v8"].unique())
        sub2idx = {s: i for i, s in enumerate(sub_ids)}

        sub_name_map = {
            int(row["SubtaskID_v8"]): str(row["SubtaskName"]).strip()
            for _, row in sub_df.iterrows()
        }

        emp_feat_cols = [c for c in emp_df.columns if c != "EmployeeID"]

        sub_feat_cols = [c for c in sub_df.columns if c.startswith("s")] + \
                        ["GAT_Predicted_Critical_Score", "In_Degree", "Out_Degree"]

        X_emp = torch.tensor(
            emp_df.set_index("EmployeeID").loc[emp_ids][emp_feat_cols].values,
            dtype=torch.float32, device=self.device
        )
        X_sub = torch.tensor(
            sub_df.set_index("SubtaskID_v8").loc[sub_ids][sub_feat_cols].values,
            dtype=torch.float32, device=self.device
        )

        edges_df["u"] = edges_df["Employee ID"].map(emp2idx)
        edges_df["v"] = edges_df["Subtask ID"].map(sub2idx)
        edges_df = edges_df.dropna(subset=["u", "v"]).copy()
        edges_df["u"] = edges_df["u"].astype(int)
        edges_df["v"] = edges_df["v"].astype(int)

        u_all = torch.tensor(edges_df["u"].values, dtype=torch.long, device=self.device)
        v_all = torch.tensor(edges_df["v"].values, dtype=torch.long, device=self.device)

        edge_feat_cols = ["Avg Duration (Hours)", "Avg Defects", "Total Defects (Sub Task)"]
        edge_attr_all  = torch.tensor(
            edges_df[edge_feat_cols].values,
            dtype=torch.float32, device=self.device
        )

        train_idx_t         = torch.arange(len(edges_df), dtype=torch.long, device=self.device)
        mu                  = edge_attr_all[train_idx_t].mean(dim=0, keepdim=True)
        std                 = edge_attr_all[train_idx_t].std(dim=0,  keepdim=True).clamp_min(1e-6)
        edge_attr_train_std = (edge_attr_all - mu) / std

        E          = X_emp.size(0)
        edge_index = torch.stack([
            torch.cat([u_all, v_all + E]),
            torch.cat([v_all + E, u_all])
        ], dim=0)

        return {
            "edges_df":            edges_df,
            "emp_ids":             emp_ids,
            "sub_ids":             sub_ids,
            "sub_name_map":        sub_name_map,
            "X_emp":               X_emp,
            "X_sub":               X_sub,
            "u_all":               u_all,
            "v_all":               v_all,
            "edge_attr_train_std": edge_attr_train_std,
            "edge_index":          edge_index,
        }

    def execute_from_db(self, br_id):
        data = self._prepare_graph(br_id)

        ckpt  = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model = BipartiteGAT_EdgeFeat(
            emp_in=data["X_emp"].size(1),
            sub_in=data["X_sub"].size(1),
            edge_in=3
        ).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        comparisons = {
            ('duration', 'avg_defects'):      3,
            ('duration', 'total_defects'):    5,
            ('avg_defects', 'total_defects'): 3,
        }
        criteria = ahpy.Compare('Criteria', comparisons, precision=4)
        beta_1   = criteria.target_weights['duration']
        beta_2   = criteria.target_weights['avg_defects']
        beta_3   = criteria.target_weights['total_defects']
        b        = torch.tensor([beta_1, beta_2, beta_3], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_all = model(
                data["X_emp"], data["X_sub"], data["edge_index"],
                data["u_all"], data["v_all"],
                data["edge_attr_train_std"],
            )
            cost_all = (pred_all * b).sum(dim=1)

        n_emp       = len(data["emp_ids"])
        n_sub       = len(data["sub_ids"])
        cost_matrix = np.full((n_emp, n_sub), np.inf)

        u_np    = data["u_all"].cpu().numpy()
        v_np    = data["v_all"].cpu().numpy()
        cost_np = cost_all.cpu().numpy()

        for i in range(len(u_np)):
            emp_idx = u_np[i]
            sub_idx = v_np[i]
            if cost_np[i] < cost_matrix[emp_idx, sub_idx]:
                cost_matrix[emp_idx, sub_idx] = cost_np[i]

        cost_matrix = np.where(np.isinf(cost_matrix), 1e9, cost_matrix)
        min_val     = cost_matrix[cost_matrix != 1e9].min()
        cost_matrix = np.where(cost_matrix != 1e9, cost_matrix + abs(min_val), cost_matrix)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        emp_name_map = {}
        for emp in Employee.objects.filter(employee_id__in=data["emp_ids"]):
            emp_name_map[emp.employee_id] = emp.name

        OptimalAssignment.objects.filter(br_id=br_id).delete()

        rows = []
        for r, c in zip(row_ind, col_ind):
            emp_id   = data["emp_ids"][r]
            sub_id   = data["sub_ids"][c]
            sub_name = data["sub_name_map"].get(sub_id, "")
            emp_name = emp_name_map.get(emp_id, "")
            cost     = float(cost_matrix[r, c])

            OptimalAssignment.objects.create(
                br_id=br_id,
                employee_id=emp_id,
                employee_name=emp_name,
                subtask_id=sub_id,
                subtask_name=sub_name,
                cost=cost,
            )
            rows.append({
                "employee_id":   emp_id,
                "employee_name": emp_name,
                "subtask_id":    sub_id,
                "subtask_name":  sub_name,
                "cost":          cost,
            })

        inf_assignments = sum(1 for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] == 1e9)

        return {
            "row_count":         len(rows),
            "total_cost":        float(cost_matrix[row_ind, col_ind].sum()),
            "inf_assignments":   int(inf_assignments),
            "valid_assignments": int(len(row_ind) - inf_assignments),
            "consistency_ratio": float(criteria.consistency_ratio),
            "table":             rows,
        }