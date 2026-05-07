from project_app.models import SystemRequirement
from scheduling_app.models import RuntimeSubtaskPair


class RuntimePairBuilderService:
    def __init__(self, br_id):
        self.br_id = br_id

    def build_runtime_subtasks(self):
        sr_rows = SystemRequirement.objects.filter(
            business_requirement__br_id=self.br_id
        ).order_by("id")

        subtasks = []
        seen = set()

        for sr in sr_rows:
            sub_name = str(getattr(sr, "task_subtype", "")).strip()
            if not sub_name:
                continue

            key = sub_name.lower()
            if key not in seen:
                subtasks.append(sub_name)
                seen.add(key)

        return subtasks

    def save_runtime_pairs(self):
        subtasks = self.build_runtime_subtasks()

        RuntimeSubtaskPair.objects.filter(br_id=self.br_id).delete()

        if len(subtasks) < 2:
            return {
                "subtasks": subtasks,
                "pair_count": 0,
                "message": "Not enough subtasks to build runtime pairs."
            }

        created = []

        for i in range(len(subtasks)):
            for j in range(len(subtasks)):
                if i == j:
                    continue

                obj = RuntimeSubtaskPair.objects.create(
                    br_id=self.br_id,
                    subtask_a=subtasks[i],
                    subtask_b=subtasks[j],
                )
                created.append(obj)

        return {
            "subtasks": subtasks,
            "pair_count": len(created),
            "message": "Runtime pairs created successfully."
        }