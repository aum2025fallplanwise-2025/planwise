import csv
import os

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render

from project_app.models import BusinessRequirement, SystemRequirement
from task_app.models import EmployeeTask
from scheduling_app.models import (
    SubtaskRelationship, CriticalPathPrediction,
    OptimalAssignment, NormalizedEdge
)

from .services.model1_br_to_sr_service import BRToSRService
from .services.model2_experience_service import ExperienceDNNService
from .services.model3_relationship_service import RelationshipClassifierService
from .services.model4_gat2_service import GAT2Service
from .services.model5_gat1_service import GAT1Service
from .services.runtime_pair_builder import RuntimePairBuilderService


@login_required
def training_home_view(request):
    return render(request, "training_home.html")


@login_required
def model1_train_view(request):
    sample_input  = BusinessRequirement.objects.order_by("br_id")[:4]
    sample_output = SystemRequirement.objects.order_by("id")[:4]

    context = {
        "loss_plot_url": "/media/ai_reports/model1_loss.png",
        "sample_input":  sample_input,
        "sample_output": sample_output,
    }
    return render(request, "model1_train.html", context)


@login_required
def model1_runtime_input_view(request):
    context = {}

    if request.method == "POST":
        br_text = request.POST.get("business_requirement", "").strip()

        if br_text:
            last_br = BusinessRequirement.objects.order_by("-br_id").first()
            next_br_id = 1 if not last_br else last_br.br_id + 1

            br_obj = BusinessRequirement.objects.create(
                br_id=next_br_id,
                text=br_text,
                tokens="",
                primary_task_type="",
                primary_task_subtype="",
                num_system_requirements=0,
                uncertainty=0.0,
            )

            return redirect("model1_execute", br_id=br_obj.br_id)

    return render(request, "model1_input.html", context)


@login_required
def model1_execute_view(request, br_id):
    br_obj  = get_object_or_404(BusinessRequirement, br_id=br_id)
    service = BRToSRService()

    context = {
        "br":                br_obj,
        "prediction":        None,
        "saved_rows":        None,
        "pair_build_result": None,
    }

    if request.method == "POST":
        prediction = service.predict_from_text(
            business_requirement_text=br_obj.text,
            br_id=br_obj.br_id,
            task_type=br_obj.primary_task_type or "",
            task_subtype=br_obj.primary_task_subtype or "",
        )
        context["prediction"] = prediction

        if "save_to_db" in request.POST:
            saved_rows = service.save_prediction_to_db(br_obj, prediction["parsed"])
            context["saved_rows"] = saved_rows

        if "save_and_build_pairs" in request.POST:
            saved_rows = service.save_prediction_to_db(br_obj, prediction["parsed"])
            context["saved_rows"] = saved_rows
            pair_builder = RuntimePairBuilderService(br_obj.br_id)
            context["pair_build_result"] = pair_builder.save_runtime_pairs()

    return render(request, "model1_execute.html", context)


@login_required
def model2_train_view(request):
    sample_input = EmployeeTask.objects.select_related("employee", "task").order_by("id")[:4]

    sample_output = []
    csv_path = os.path.join(settings.MEDIA_ROOT, "ai_reports", "model2_output.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 4:
                    break
                sample_output.append({
                    "employee_id":                row.get("Employee ID", ""),
                    "task":                       row.get("Task", ""),
                    "sub_task":                   row.get("Sub Task", ""),
                    "predicted_experience_level": row.get("Predicted_ExperienceLevel", ""),
                })

    context = {
        "loss_plot_url":     "/media/ai_reports/model2_loss.png",
        "test_accuracy_url": "/media/ai_reports/model2_test_accuracy.png",
        "sample_input":      sample_input,
        "sample_output":     sample_output,
    }
    return render(request, "model2_train.html", context)


@login_required
def model2_execute_view(request):
    context = {}
    if request.method == "POST":
        service = ExperienceDNNService()
        context["result"] = service.predict_from_db()
    return render(request, "model2_execute.html", context)


@login_required
def model3_train_view(request):
    sample_input  = SubtaskRelationship.objects.order_by("id")[:4]
    sample_output = SubtaskRelationship.objects.exclude(predicted_relationship="").order_by("id")[:4]

    context = {
        "loss_plot_url":             "/media/ai_reports/model3_loss.png",
        "acc_plot_url":              "/media/ai_reports/model3_accuracy.png",
        "classification_report_url": "/media/ai_reports/model3_classification_report.png",
        "sample_input":              sample_input,
        "sample_output":             sample_output,
    }
    return render(request, "model3_train.html", context)


@login_required
def model3_execute_view(request, br_id):
    context = {"br_id": br_id}
    if request.method == "POST":
        service = RelationshipClassifierService()
        context["result"] = service.predict_from_runtime_pairs(br_id=br_id)
    return render(request, "model3_execute.html", context)


@login_required
def model4_train_view(request):
    sample_input  = SubtaskRelationship.objects.exclude(predicted_relationship="").order_by("id")[:4]

    sample_output = []
    csv_path = os.path.join(settings.MEDIA_ROOT, "ai_reports", "model4_output.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 4:
                    break
                sample_output.append({
                    "subtask":         row.get("Subtask", ""),
                    "predicted_score": row.get("GAT_Predicted_Critical_Score", ""),
                    "in_degree":       row.get("In_Degree", ""),
                    "out_degree":      row.get("Out_Degree", ""),
                })

    context = {
        "loss_plot_url":    "/media/ai_reports/model4_loss.png",
        "test_metrics_url": "/media/ai_reports/model4_test_metrics.png",
        "sample_input":     sample_input,
        "sample_output":    sample_output,
    }
    return render(request, "model4_train.html", context)


@login_required
def model4_execute_view(request, br_id):
    context = {"br_id": br_id}
    if request.method == "POST":
        service = GAT2Service()
        context["result"] = service.execute_from_db(br_id=br_id)
    return render(request, "model4_execute.html", context)


@login_required
def model5_train_view(request):
    from employee_app.models import Employee
    from task_app.models import SubtaskVector
    import numpy as np

    sample_input = NormalizedEdge.objects.order_by("id")[:4]

    sub_name_map = {
        row.subtask_id: row.subtask_name
        for row in SubtaskVector.objects.all()
    }
    emp_name_map = {
        emp.employee_id: emp.name
        for emp in Employee.objects.all()
    }

    raw_rows = []
    csv_path = os.path.join(settings.MEDIA_ROOT, "ai_reports", "model5_output.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 4:
                    break
                raw_rows.append(row)

    sample_output = []
    if raw_rows:
        costs = [float(r.get("Cost", 0)) for r in raw_rows]
        min_val = min(costs)
        shift = abs(min_val) if min_val < 0 else 0.0

        for row in raw_rows:
            emp_id = int(row.get("EmployeeID", 0))
            sub_id = int(row.get("SubtaskID", 0))
            cost   = float(row.get("Cost", 0)) + shift
            sample_output.append({
                "employee_id":   emp_id,
                "employee_name": emp_name_map.get(emp_id, ""),
                "subtask_id":    sub_id,
                "subtask_name":  sub_name_map.get(sub_id, ""),
                "cost":          cost,
            })

    context = {
        "loss_plot_url": "/media/ai_reports/model5_train_val_loss.png",
        "sample_input":  sample_input,
        "sample_output": sample_output,
    }
    return render(request, "model5_train.html", context)


@login_required
def model5_execute_view(request, br_id):
    context = {"br_id": br_id}
    if request.method == "POST":
        service = GAT1Service()
        context["result"] = service.execute_from_db(br_id=br_id)
    return render(request, "model5_execute.html", context)