from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
from django.conf import settings

from employee_app.models import Employee, EmployeeVector
from task_app.models import EmployeeTask


class DNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, num_classes)

    def forward(self, x, return_embedding=False):
        x = torch.relu(self.fc1(x))
        emb = torch.relu(self.fc2(x))
        if return_embedding:
            return emb
        return self.out(emb)


class ExperienceDNNService:
    def __init__(self):
        self.model_dir = Path(settings.BASE_DIR) / "pretrained_models" / "model2"

        self.columns_path       = self.model_dir / "dnn_columns.joblib"
        self.scaler_path        = self.model_dir / "dnn_scaler.joblib"
        self.label_encoder_path = self.model_dir / "dnn_label_encoder.joblib"
        self.weights_path       = self.model_dir / "experience_dnn_weights.pth"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_dataframe_from_db(self):
        rows = []

        qs = EmployeeTask.objects.select_related(
            "employee",
            "employee__experience",
            "task"
        ).all().order_by("id")

        for row in qs:
            emp  = row.employee
            task = row.task

            rows.append({
                "employee_id":              emp.employee_id,
                "first_name":               emp.first_name,
                "last_name":                emp.last_name,
                "task":                     task.task_type,
                "sub_task":                 task.name,
                "experience_level":         emp.experience.name,
                "avg_duration_hours":       row.working_hours,
                "avg_defects":              row.defects,
                "total_tasks_done_subtask": row.total_tasks_done,
                "total_hours_subtask":      row.total_hours,
                "total_defects_subtask":    row.total_defects,
                "task_id":                  task.task_id,
                "subtask_id":               task.subtask_id,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No EmployeeTask data found in the database.")

        return df

    def _load_model(self):
        dnn_columns = joblib.load(self.columns_path)

        payload = torch.load(self.weights_path, map_location=self.device)

        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload

        input_dim   = len(dnn_columns)
        num_classes = state_dict["out.weight"].shape[0]

        model = DNN(input_dim, num_classes).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def _prepare_features(self, df):
        target_col = "experience_level"
        X_all = df.drop(columns=[target_col])
        X_all = pd.get_dummies(X_all, drop_first=False)

        dnn_columns = joblib.load(self.columns_path)
        scaler      = joblib.load(self.scaler_path)

        X_all = X_all.reindex(columns=dnn_columns, fill_value=0)
        X_all_scaled = scaler.transform(X_all)

        return X_all_scaled, df["employee_id"].values

    def predict_from_db(self):
        df    = self._build_dataframe_from_db()
        model = self._load_model()
        le    = joblib.load(self.label_encoder_path)

        X_all_scaled, emp_ids = self._prepare_features(df)
        X_all_t = torch.tensor(X_all_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = model(X_all_t).cpu()
            preds  = torch.argmax(logits, dim=1).numpy()

        all_classes   = list(le.classes_)
        valid_classes = all_classes[:model.out.out_features]
        predicted_levels = [valid_classes[p] for p in preds]

        output = pd.DataFrame({
            "employee_id":               df["employee_id"],
            "task":                      df["task"],
            "sub_task":                  df["sub_task"],
            "predicted_experience_level": predicted_levels,
        })

        return {
            "table":     output.to_dict(orient="records"),
            "row_count": int(len(output)),
        }

    def save_embeddings_to_db(self):
        """
        Runs the DNN on all EmployeeTask rows, extracts the 16-dim
        embedding from fc2 for each employee, averages across all their
        rows, then saves/updates EmployeeVector in the DB.
        """
        df    = self._build_dataframe_from_db()
        model = self._load_model()

        X_all_scaled, emp_ids = self._prepare_features(df)
        X_all_t = torch.tensor(X_all_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            embeddings = model(X_all_t, return_embedding=True).cpu().numpy()

        emb_df = pd.DataFrame(embeddings, columns=[f"e{i}" for i in range(embeddings.shape[1])])
        emb_df["employee_id"] = emp_ids

        emb_avg = emb_df.groupby("employee_id").mean().reset_index()

        saved = 0
        skipped = 0

        for _, row in emb_avg.iterrows():
            emp_id = int(row["employee_id"])
            vector = [float(row[f"e{i}"]) for i in range(embeddings.shape[1])]

            try:
                employee = Employee.objects.get(employee_id=emp_id)
            except Employee.DoesNotExist:
                skipped += 1
                continue

            EmployeeVector.objects.update_or_create(
                employee=employee,
                defaults={"vector_data": vector},
            )
            saved += 1

        return {
            "saved":   saved,
            "skipped": skipped,
            "message": f"Embeddings saved for {saved} employees, skipped {skipped}.",
        }
