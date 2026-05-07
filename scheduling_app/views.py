from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from project_app.models import BusinessRequirement
from scheduling_app.models import (
    OptimalAssignment,
    CriticalPathPrediction,
    SubtaskRelationship,
)


@login_required
def scheduling_home_view(request):
    latest_br = BusinessRequirement.objects.order_by("-id").first()

    if not latest_br:
        return render(request, "scheduling_home.html", {"latest_br": None})

    recent_brs = BusinessRequirement.objects.filter(num_system_requirements__gt=0).order_by("br_id")[:10]

    return render(request, "scheduling_home.html", {
        "latest_br": latest_br,
        "recent_brs": recent_brs,
    })


@login_required
def combined_schedule_view(request, br_id):
    assignments = list(
        OptimalAssignment.objects.filter(br_id=br_id).order_by("subtask_name", "employee_id")
    )

    assigned_subtask_names = [str(row.subtask_name).strip() for row in assignments]

    critical_rows = list(
        CriticalPathPrediction.objects.filter(
            br_id=0,
            subtask_name__in=assigned_subtask_names
        ).order_by("-predicted_score")
    )

    assigned_by_name = {
        str(row.subtask_name).strip(): row
        for row in assignments
    }

    critical_mapping = []
    for row in critical_rows:
        subtask_name = str(row.subtask_name).strip()
        assigned_row = assigned_by_name.get(subtask_name)

        critical_mapping.append({
            "subtask_name":         subtask_name,
            "predicted_score":      row.predicted_score,
            "in_degree":            row.in_degree,
            "out_degree":           row.out_degree,
            "assigned_employee_id": assigned_row.employee_id if assigned_row else None,
            "assignment_cost":      assigned_row.cost if assigned_row else None,
        })

    critical_order = {row["subtask_name"]: i for i, row in enumerate(critical_mapping)}
    assignments = sorted(
        assignments,
        key=lambda r: critical_order.get(str(r.subtask_name).strip(), 999)
    )

    relationship_rows = []
    qs = SubtaskRelationship.objects.filter(br_id=br_id).order_by("id")

    for row in qs:
        if str(row.predicted_relationship).upper() == "NO_RELATION":
            continue

        a_name = str(row.subtask_a).strip()
        b_name = str(row.subtask_b).strip()

        a_assignment = assigned_by_name.get(a_name)
        b_assignment = assigned_by_name.get(b_name)

        relationship_rows.append({
            "subtask_a":              a_name,
            "subtask_b":              b_name,
            "predicted_relationship": row.predicted_relationship,
            "confidence":             row.confidence,
            "employee_a":             a_assignment.employee_id if a_assignment else None,
            "employee_b":             b_assignment.employee_id if b_assignment else None,
        })

    context = {
        "br_id":             br_id,
        "assignment_table":  assignments,
        "critical_mapping":  critical_mapping,
        "relationship_rows": relationship_rows,
    }
    return render(request, "combined_schedule.html", context)
