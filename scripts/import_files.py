import os
import sys
import csv
import json
import math
import django
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planwise.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
django.setup()

from employee_app.models import ExperienceLevel, Employee, EmployeeVector
from task_app.models import Task, EmployeeTask, SubtaskVector
from project_app.models import ProjectType, BusinessRequirement, SystemRequirement
from scheduling_app.models import NormalizedEdge, OptimalAssignment, CriticalPathPrediction, SubtaskRelationship

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Clean Data")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def safe_str(val):
    if pd.isna(val):
        return ""
    return str(val).strip()


def safe_int(val):
    if pd.isna(val):
        return 0
    return int(val)


def safe_float(val):
    if pd.isna(val):
        return 0.0
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def import_experience_levels():
    print("Importing experience levels...")
    ExperienceLevel.objects.all().delete()

    df = pd.read_excel(os.path.join(DATA_DIR, "Software_Experience_Levels.xlsx"))

    for idx, row in df.iterrows():
        ExperienceLevel.objects.create(
            name=safe_str(row["Rank"]),
            rank=idx + 1,
            min_years=safe_int(row["Min Years"]),
            max_years=safe_int(row["Max Years"]),
            primary_focus=safe_str(row["Primary Focus"]),
            autonomy=safe_str(row["Autonomy"]),
            decision_scope=safe_str(row["Decision Scope"]),
            leadership=safe_str(row["Leadership"]),
            system_impact=safe_str(row["System Impact"]),
        )

    print(f"  -> {ExperienceLevel.objects.count()} experience levels imported")


def import_project_types():
    print("Importing project types...")
    ProjectType.objects.all().delete()

    df = pd.read_excel(os.path.join(DATA_DIR, "Software_Project_Types.xlsx"))

    for _, row in df.iterrows():
        ProjectType.objects.create(
            name=safe_str(row["Project Type"]),
            innovation=safe_str(row["Innovation"]),
            risk=safe_str(row["Risk"]),
            roi_horizon=safe_str(row["ROI Horizon"]),
        )

    print(f"  -> {ProjectType.objects.count()} project types imported")


def import_tasks_taxonomy():
    print("Importing task taxonomy...")
    Task.objects.all().delete()

    df = pd.read_excel(os.path.join(DATA_DIR, "Software_Task_Taxonomy.xlsx"))

    for _, row in df.iterrows():
        task_type = safe_str(row["Task Type"])
        subtask_name = safe_str(row["Task Subtype"])
        Task.objects.create(
            name=subtask_name,
            task_type=task_type,
        )

    print(f"  -> {Task.objects.count()} tasks imported")


def import_employees_and_assignments():
    print("Importing employees and task assignments...")
    EmployeeTask.objects.all().delete()
    Employee.objects.all().delete()

    df = pd.read_excel(os.path.join(DATA_DIR, "Generated_Employee_Task_Assignments_v8_FIXED_ALL_EXPERIENCE.xlsx"))

    exp_cache = {}
    for exp in ExperienceLevel.objects.all():
        exp_cache[exp.name] = exp

    emp_cache = {}
    task_cache = {}

    task_qs = Task.objects.all()
    for t in task_qs:
        key = (t.task_type, t.name)
        task_cache[key] = t

    batch = []
    batch_size = 5000
    total = 0

    for _, row in df.iterrows():
        emp_id = safe_int(row["Employee ID"])
        first = safe_str(row["First Name"])
        last = safe_str(row["Last Name"])
        full_name = f"{first} {last}"
        exp_name = safe_str(row["Experience Level"])
        task_type = safe_str(row["Task"])
        subtask_name = safe_str(row["Sub Task"])
        hours = safe_float(row["Avg Duration (Hours)"])
        defects = safe_int(row["Avg Defects"])
        total_tasks = safe_int(row["Total Tasks Done (Sub Task)"])
        total_hrs = safe_float(row["Total Hours (Sub Task)"])
        total_def = safe_int(row["Total Defects (Sub Task)"])
        t_id = safe_int(row["Task ID"])
        s_id = safe_int(row["Subtask ID"])

        if emp_id not in emp_cache:
            exp_obj = exp_cache.get(exp_name)
            if not exp_obj:
                exp_obj, _ = ExperienceLevel.objects.get_or_create(name=exp_name)
                exp_cache[exp_name] = exp_obj
            emp_obj = Employee.objects.create(
                employee_id=emp_id,
                first_name=first,
                last_name=last,
                name=full_name,
                experience=exp_obj,
            )
            emp_cache[emp_id] = emp_obj
        else:
            emp_obj = emp_cache[emp_id]

        task_key = (task_type, subtask_name)
        if task_key not in task_cache:
            task_obj = Task.objects.create(
                name=subtask_name,
                task_type=task_type,
                task_id=t_id,
                subtask_id=s_id,
            )
            task_cache[task_key] = task_obj
        else:
            task_obj = task_cache[task_key]
            if task_obj.task_id == 0 and t_id != 0:
                task_obj.task_id = t_id
                task_obj.subtask_id = s_id
                task_obj.save()

        exp_level = exp_cache.get(exp_name)

        if hours == 0 and defects == 0 and total_tasks == 0:
            continue

        batch.append(EmployeeTask(
            employee=emp_obj,
            task=task_obj,
            experience_level=exp_level,
            working_hours=hours,
            defects=defects,
            total_tasks_done=total_tasks,
            total_hours=total_hrs,
            total_defects=total_def,
        ))

        if len(batch) >= batch_size:
            EmployeeTask.objects.bulk_create(batch)
            total += len(batch)
            batch = []
            print(f"  -> {total} assignments so far...")

    if batch:
        EmployeeTask.objects.bulk_create(batch)
        total += len(batch)

    print(f"  -> {Employee.objects.count()} employees imported")
    print(f"  -> {total} employee-task assignments imported")


def import_employee_vectors():
    print("Importing employee vectors...")
    EmployeeVector.objects.all().delete()

    emp_lookup = {}
    for emp in Employee.objects.all():
        emp_lookup[emp.employee_id] = emp

    vectors = []
    filepath = os.path.join(DATA_DIR, "emp_vectors_clean.csv")

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = int(row["EmployeeID"])
            emp = emp_lookup.get(eid)
            if not emp:
                continue
            vec = []
            for i in range(16):
                vec.append(float(row[f"e{i}"]))
            vectors.append(EmployeeVector(
                employee=emp,
                vector_data=vec,
            ))

    EmployeeVector.objects.bulk_create(vectors)
    print(f"  -> {len(vectors)} employee vectors imported")


def import_subtask_vectors():
    print("Importing subtask vectors...")
    SubtaskVector.objects.all().delete()

    vectors = []
    filepath = os.path.join(DATA_DIR, "subtask_vectors_Final_by_v8_id.csv")

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["SubtaskID_v8"])
            vec = []
            for i in range(384):
                vec.append(float(row[f"s{i}"]))
            vectors.append(SubtaskVector(
                subtask_id=sid,
                task_name=row["Task"],
                subtask_name=row["Sub Task"],
                task_norm=row.get("Task_norm", ""),
                sub_norm=row.get("Sub_norm", ""),
                vector_data=vec,
            ))

    SubtaskVector.objects.bulk_create(vectors)
    print(f"  -> {len(vectors)} subtask vectors imported")


def import_business_requirements():
    print("Importing business requirements...")
    SystemRequirement.objects.all().delete()
    BusinessRequirement.objects.all().delete()

    filepath = os.path.join(DATA_DIR, "BUSINESS Requirement.xlsx")

    br_df = pd.read_excel(filepath, sheet_name="Business_Requirements")
    br_objects = {}

    for _, row in br_df.iterrows():
        br = BusinessRequirement.objects.create(
            br_id=safe_int(row["BR_ID"]),
            text=safe_str(row["Business_Requirement_Text"]),
            tokens=safe_str(row["BR_Tokens"]),
            primary_task_type=safe_str(row["Primary_Task_Type"]),
            primary_task_subtype=safe_str(row["Primary_Task_Subtype"]),
            num_system_requirements=safe_int(row["Num_System_Requirements"]),
            uncertainty=safe_float(row["Uncertainty_Overall_0_1"]),
        )
        br_objects[br.br_id] = br

    print(f"  -> {len(br_objects)} business requirements imported")

    sr_df = pd.read_excel(filepath, sheet_name="System_Requirements")
    sr_batch = []

    for _, row in sr_df.iterrows():
        br_id = safe_int(row["BR_ID"])
        br_obj = br_objects.get(br_id)
        if not br_obj:
            continue
        sr_batch.append(SystemRequirement(
            business_requirement=br_obj,
            system_req_id=safe_str(row["SystemReq_ID"]),
            task_type=safe_str(row["Task_Type"]),
            task_subtype=safe_str(row["Task_Subtype"]),
            required_experience_level=safe_str(row["Required_Experience_Level"]),
            uncertainty=safe_float(row["Uncertainty_PerReq_0_1"]),
        ))

    SystemRequirement.objects.bulk_create(sr_batch)
    print(f"  -> {len(sr_batch)} system requirements imported")


def import_normalized_edges():
    print("Importing normalized edges...")
    NormalizedEdge.objects.all().delete()

    filepath = os.path.join(DATA_DIR, "Normalized_edges.csv")
    batch = []
    batch_size = 10000
    total = 0

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            avg_dur = float(row["Avg Duration (Hours)"])
            avg_def = float(row["Avg Defects"])
            ttd = float(row["Total Tasks Done (Sub Task)"])
            td = float(row["Total Defects (Sub Task)"])

            if avg_dur == 0 and avg_def == 0 and ttd == 0 and td == 0:
                continue

            batch.append(NormalizedEdge(
                employee_id=int(row["Employee ID"]),
                subtask_id=int(row["Subtask ID"]),
                avg_duration=avg_dur,
                avg_defects=avg_def,
                total_tasks_done=ttd,
                total_defects=td,
            ))

            if len(batch) >= batch_size:
                NormalizedEdge.objects.bulk_create(batch)
                total += len(batch)
                batch = []
                print(f"  -> {total} edges so far...")

    if batch:
        NormalizedEdge.objects.bulk_create(batch)
        total += len(batch)

    print(f"  -> {total} normalized edges imported")


def import_gat1_assignments():
    print("Importing GAT1 optimal assignments...")
    OptimalAssignment.objects.all().delete()

    from task_app.models import SubtaskVector
    from employee_app.models import Employee

    sub_name_map = {
        row.subtask_id: row.subtask_name
        for row in SubtaskVector.objects.all()
    }

    emp_name_map = {
        emp.employee_id: emp.name
        for emp in Employee.objects.all()
    }

    filepath = os.path.join(DATA_DIR, "optimal_assignments.csv")
    batch = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub_id = int(row["SubtaskID"])
            emp_id = int(row["EmployeeID"])
            batch.append(OptimalAssignment(
                employee_id=emp_id,
                employee_name=emp_name_map.get(emp_id, ""),
                subtask_id=sub_id,
                subtask_name=sub_name_map.get(sub_id, ""),
                cost=float(row["Cost"]),
            ))

    OptimalAssignment.objects.bulk_create(batch)
    print(f"  -> {len(batch)} optimal assignments imported")


def import_gat2_predictions():
    print("Importing GAT2 critical path predictions...")
    CriticalPathPrediction.objects.all().delete()

    filepath = os.path.join(DATA_DIR, "GAT2_CriticalPath_Predictions_Shifted.csv")
    batch = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append(CriticalPathPrediction(
                subtask_name=row["Subtask"],
                predicted_score=float(row["GAT_Predicted_Critical_Score_Shifted"]),
                in_degree=float(row["In_Degree"]),
                out_degree=float(row["Out_Degree"]),
            ))

    CriticalPathPrediction.objects.bulk_create(batch)
    print(f"  -> {len(batch)} critical path predictions imported")

def import_subtask_relationships():
    print("Importing subtask relationships...")
    SubtaskRelationship.objects.all().delete()

    labels_df = pd.read_excel(os.path.join(DATA_DIR, "Subtask_Relationship_Labels.xlsx"))
    preds_df = pd.read_csv(os.path.join(DATA_DIR, "Subtask_Direct_Relationships_WITH_PREDICTIONS.csv"))

    merged = labels_df.merge(
        preds_df[["BR_ID", "Subtask_A", "Subtask_B", "pair_text", "Predicted_Relationship", "Confidence"]],
        on=["BR_ID", "Subtask_A", "Subtask_B"],
        how="left",
    )

    batch = []
    batch_size = 5000
    total = 0

    for _, row in merged.iterrows():
        batch.append(SubtaskRelationship(
            br_id=safe_int(row["BR_ID"]),
            subtask_a=safe_str(row["Subtask_A"]),
            subtask_b=safe_str(row["Subtask_B"]),
            pair_text=safe_str(row["pair_text"]) if not pd.isna(row["pair_text"]) else "",
            relationship_label=safe_str(row["Relationship"]),
            predicted_relationship=safe_str(row["Predicted_Relationship"]) if not pd.isna(row["Predicted_Relationship"]) else "",
            confidence=safe_float(row["Confidence"]),
        ))

        if len(batch) >= batch_size:
            SubtaskRelationship.objects.bulk_create(batch)
            total += len(batch)
            batch = []
            print(f"  -> {total} relationships so far...")

    if batch:
        SubtaskRelationship.objects.bulk_create(batch)
        total += len(batch)

    print(f"  -> {total} subtask relationships imported")


if __name__ == "__main__":
    import_experience_levels()
    import_project_types()
    import_tasks_taxonomy()
    import_employees_and_assignments()
    import_employee_vectors()
    import_subtask_vectors()
    import_business_requirements()
    import_normalized_edges()
    import_gat1_assignments()
    import_gat2_predictions()
    import_subtask_relationships()