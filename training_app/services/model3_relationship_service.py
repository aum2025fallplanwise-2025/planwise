from pathlib import Path
from django.conf import settings
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from scheduling_app.models import RuntimeSubtaskPair, SubtaskRelationship


class RelationshipClassifierService:
    def __init__(self):
        self.model_dir = Path(settings.BASE_DIR) / "pretrained_models" / "model3"
        self.model_path = self.model_dir / "subtask_relation_dnn.keras"
        self.vectorizer_path = self.model_dir / "tfidf_vectorizer.joblib"
        self.label_map_path = self.model_dir / "label_map.json"

    def _make_pair_text(self, a, b):
        return f"SUBTASK_A: {a} [SEP] SUBTASK_B: {b}"

    def predict_from_runtime_pairs(self, br_id):
        qs = RuntimeSubtaskPair.objects.filter(br_id=br_id).order_by("id")

        rows = []
        for row in qs:
            rows.append({
                "br_id": row.br_id,
                "subtask_a": str(row.subtask_a).strip(),
                "subtask_b": str(row.subtask_b).strip(),
            })

        df_pred = pd.DataFrame(rows)

        if df_pred.empty:
            return {
                "row_count": 0,
                "table": [],
                "message": "No runtime subtask pairs found for this BR."
            }

        vectorizer = joblib.load(self.vectorizer_path)
        model = tf.keras.models.load_model(self.model_path)

        with open(self.label_map_path, "r") as f:
            label_map = json.load(f)

        id_to_class = {int(k): v for k, v in label_map["id_to_class"].items()}

        df_pred["pair_text"] = df_pred.apply(
            lambda r: self._make_pair_text(r["subtask_a"], r["subtask_b"]),
            axis=1
        )

        X_pred = vectorizer.transform(df_pred["pair_text"].values).toarray().astype("float32")

        pred_probs = model.predict(X_pred, verbose=0)
        pred_ids = np.argmax(pred_probs, axis=1)

        df_pred["predicted_relationship"] = [id_to_class[int(i)] for i in pred_ids]
        df_pred["confidence"] = np.max(pred_probs, axis=1)

        SubtaskRelationship.objects.filter(br_id=br_id).delete()

        for _, row in df_pred.iterrows():
            SubtaskRelationship.objects.create(
                br_id=int(row["br_id"]),
                subtask_a=row["subtask_a"],
                subtask_b=row["subtask_b"],
                pair_text=row["pair_text"],
                relationship_label="NO_RELATION",
                predicted_relationship=row["predicted_relationship"],
                confidence=float(row["confidence"]),
            )

        return {
            "row_count": int(len(df_pred)),
            "table": df_pred[
                ["br_id", "subtask_a", "subtask_b", "predicted_relationship", "confidence"]
            ].to_dict(orient="records"),
            "message": "Relationship prediction completed successfully."
        }
