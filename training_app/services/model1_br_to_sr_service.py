import json
import re
import difflib
from pathlib import Path

import torch
from django.conf import settings
from transformers import T5ForConditionalGeneration, T5Tokenizer

from project_app.models import SystemRequirement
from task_app.models import SubtaskVector


class BRToSRService:
    def __init__(self):
        self.model_dir = Path(settings.BASE_DIR) / "pretrained_models" / "model1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_trained_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(str(self.model_dir))
        self.model = T5ForConditionalGeneration.from_pretrained(str(self.model_dir)).to(self.device)
        self.model.eval()

    def parse_model_output(self, raw_text: str):
        raw = str(raw_text).strip()

        def grab_str(key):
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', raw)
            return m.group(1).strip() if m else ""

        br = {
            "br_id":        grab_str("br_id"),
            "text":         grab_str("text"),
            "task_type":    grab_str("task_type"),
            "task_subtype": grab_str("task_subtype"),
        }

        sr_ids   = re.findall(r'"sr_id"\s*:\s*"([^"]+)"', raw)
        sr_texts = re.findall(r'"text"\s*:\s*"([^"]+)"', raw)
        req_exps = re.findall(r'"required_experience"\s*:\s*"([^"]+)"', raw)
        uncs     = re.findall(r'"uncertainty"\s*:\s*([0-9]*\.?[0-9]+)', raw)

        srs = []
        for i, sr_id in enumerate(sr_ids):
            if i >= 2:
                break
            text = sr_texts[i + 1] if (i + 1) < len(sr_texts) else (sr_texts[i] if i < len(sr_texts) else "")
            srs.append({
                "sr_id":               f"{sr_id}-{i+1}",
                "text":                text,
                "required_experience": req_exps[i] if i < len(req_exps) else "Junior Developer",
                "uncertainty":         float(uncs[i]) if i < len(uncs) else 0.1,
                "task_type":           br["task_type"],
                "task_subtype":        br["task_subtype"],
            })

        return {"business_requirement": br, "system_requirements": srs}

    def generate_raw(self, br_json_dict, max_new_tokens=600):
        if self.model is None or self.tokenizer is None:
            self.load_trained_model()

        prompt = (
            "Return ONLY valid JSON with EXACTLY 2 system requirements.\n"
            "Output must match this schema exactly:\n"
            '{"business_requirement":{"br_id":"","text":"","task_type":"","task_subtype":""},'
            '"system_requirements":['
            '{"sr_id":"","text":"","required_experience":"","uncertainty":0.0},'
            '{"sr_id":"","text":"","required_experience":"","uncertainty":0.0}'
            ']}\n\n'
            "INPUT:\n"
            + json.dumps({"business_requirement": br_json_dict}, ensure_ascii=False)
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=600,
                num_beams=4
            )

        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _tokenize(self, text: str):
        words = re.findall(r"[a-zA-Z]+", str(text).lower())
        stop = {
            "the", "and", "for", "with", "shall", "system", "support", "provide",
            "existing", "future", "core", "needs", "need", "company", "platform",
            "solution", "critical", "processes"
        }
        return [w for w in words if len(w) > 2 and w not in stop]

    def _candidate_subtasks(self):
        rows = []
        for row in SubtaskVector.objects.all().order_by("subtask_id"):
            rows.append({
                "subtask_id":   row.subtask_id,
                "task_name":    str(row.task_name).strip(),
                "subtask_name": str(row.subtask_name).strip(),
                "search_text":  " ".join([
                    str(row.task_name).strip(),
                    str(row.subtask_name).strip(),
                    str(getattr(row, "task_norm", "")).strip(),
                    str(getattr(row, "sub_norm",  "")).strip(),
                ]).lower()
            })
        return rows

    def _score_candidate(self, query_text: str, candidate_text: str):
        query_text     = str(query_text).lower().strip()
        candidate_text = str(candidate_text).lower().strip()

        ratio    = difflib.SequenceMatcher(None, query_text, candidate_text).ratio()
        q_tokens = set(self._tokenize(query_text))
        c_tokens = set(self._tokenize(candidate_text))
        overlap  = len(q_tokens & c_tokens)

        important = {
            "login", "auth", "authentication", "password",
            "monitor", "monitoring", "logging", "log",
            "error", "errors", "backend", "api", "database",
            "report", "dashboard", "notification", "trigger",
            "automation", "maintenance", "maintainability",
            "reliability", "stability"
        }
        boost = len((q_tokens & c_tokens) & important) * 0.15
        return ratio + (overlap * 0.08) + boost

    def map_to_existing_subtasks(self, parsed_output):
        candidates = self._candidate_subtasks()
        if not candidates:
            return parsed_output

        used_subtask_ids = set()
        sr_list = parsed_output.get("system_requirements", [])
        br_text = parsed_output.get("business_requirement", {}).get("text", "")

        if len(sr_list) >= 2 and sr_list[0].get("text") == sr_list[1].get("text"):
            sr_list[1]["text"] = f"non-functional reliability performance scalability {br_text}"

        for sr in sr_list:
            query_text = " ".join([
                str(sr.get("text", "")),
                str(sr.get("required_experience", "")),
            ]).strip()

            scored = sorted(
                [(self._score_candidate(query_text, c["search_text"]), c) for c in candidates],
                key=lambda x: x[0], reverse=True
            )

            chosen = next(
                (c for _, c in scored if c["subtask_id"] not in used_subtask_ids),
                scored[0][1] if scored else None
            )

            if chosen:
                sr["task_type"]    = chosen["task_name"] or sr.get("task_type", "General Development")
                sr["task_subtype"] = chosen["subtask_name"]
                used_subtask_ids.add(chosen["subtask_id"])

        parsed_output["system_requirements"] = sr_list
        return parsed_output

    def predict_from_text(self, business_requirement_text, br_id="", task_type="", task_subtype=""):
        br_json = {
            "br_id":        str(br_id),
            "text":         str(business_requirement_text).strip(),
            "task_type":    str(task_type).strip(),
            "task_subtype": str(task_subtype).strip(),
        }

        raw    = self.generate_raw(br_json)
        parsed = self.parse_model_output(raw)

        if not parsed["business_requirement"].get("text"):
            parsed["business_requirement"]["text"] = br_json["text"]
        if not parsed["business_requirement"].get("br_id"):
            parsed["business_requirement"]["br_id"] = br_json["br_id"]
        if not parsed["business_requirement"].get("task_type"):
            parsed["business_requirement"]["task_type"] = br_json["task_type"] or "General Development"
        if not parsed["business_requirement"].get("task_subtype"):
            parsed["business_requirement"]["task_subtype"] = br_json["task_subtype"] or "General Backend Enhancement"

        sr_list = parsed.get("system_requirements", [])

        for idx, sr in enumerate(sr_list, start=1):
            if not sr.get("task_type"):
                sr["task_type"]    = parsed["business_requirement"].get("task_type", "General Development")
            if not sr.get("task_subtype"):
                sr["task_subtype"] = f"Generated Subtask {idx}"
            if not sr.get("sr_id"):
                sr["sr_id"]        = f"AUTO-{br_json['br_id']}-{idx}"
            if not sr.get("text"):
                sr["text"]         = f"The system shall support {br_json['task_subtype']} as part of {br_json['task_type']}."
            if not sr.get("required_experience"):
                sr["required_experience"] = "Junior Developer"

        if len(sr_list) == 0:
            sr_list = [
                {
                    "sr_id":               f"AUTO-{br_json['br_id']}-1",
                    "text":                f"The system shall support {br_json['task_subtype']} as part of {br_json['task_type']}.",
                    "required_experience": "Junior Developer",
                    "uncertainty":         0.1,
                    "task_type":           parsed["business_requirement"].get("task_type", "General Development"),
                    "task_subtype":        br_json.get("task_subtype", "Generated Subtask 1"),
                },
                {
                    "sr_id":               f"AUTO-{br_json['br_id']}-2",
                    "text":                f"non-functional reliability performance scalability {br_json['text']}",
                    "required_experience": "Junior Developer",
                    "uncertainty":         0.1,
                    "task_type":           parsed["business_requirement"].get("task_type", "General Development"),
                    "task_subtype":        br_json.get("task_subtype", "Generated Subtask 2"),
                },
            ]

        parsed["system_requirements"] = sr_list
        parsed = self.map_to_existing_subtasks(parsed)

        return {"raw_text": raw, "parsed": parsed}

    def save_prediction_to_db(self, br_obj, parsed_output):
        sr_list = parsed_output.get("system_requirements", [])
        SystemRequirement.objects.filter(business_requirement=br_obj).delete()

        created_rows = []
        for idx, sr in enumerate(sr_list, start=1):
            created = SystemRequirement.objects.create(
                business_requirement=br_obj,
                system_req_id=str(sr.get("sr_id", f"AUTO-{br_obj.br_id}-{idx}")),
                task_type=str(sr.get("task_type", "General Development")).strip(),
                task_subtype=str(sr.get("task_subtype", f"Generated Subtask {idx}")).strip(),
                required_experience_level=str(sr.get("required_experience", "Junior Developer")).strip(),
                uncertainty=float(sr.get("uncertainty", 0.0) or 0.0),
            )
            created_rows.append(created)

        avg_uncertainty = (
            sum(float(sr.get("uncertainty", 0.0) or 0.0) for sr in sr_list) / len(sr_list)
            if sr_list else 0.0
        )
        br_obj.num_system_requirements = len(created_rows)
        br_obj.uncertainty = avg_uncertainty
        br_obj.save(update_fields=["num_system_requirements", "uncertainty"])

        return created_rows
