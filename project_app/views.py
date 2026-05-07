from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render

from .models import Project, ProjectType, BusinessRequirement
from training_app.services.model1_br_to_sr_service import BRToSRService
from training_app.services.model2_experience_service import ExperienceDNNService
from training_app.services.runtime_pair_builder import RuntimePairBuilderService
from training_app.services.model3_relationship_service import RelationshipClassifierService
from training_app.services.model4_gat2_service import GAT2Service
from training_app.services.model5_gat1_service import GAT1Service


def projects_list(request):
    projects = Project.objects.all().select_related("project_type")
    project_types = ProjectType.objects.all()

    return render(request, "projects.html", {
        "projects": projects,
        "project_types": project_types,
    })


def add_project(request):
    if request.method == "POST":
        name = request.POST.get("name")
        project_type_id = request.POST.get("project_type")
        business_requirements = request.POST.get("business_requirements")

        if name and project_type_id and business_requirements:
            Project.objects.create(
                name=name,
                project_type_id=project_type_id,
                business_requirements=business_requirements,
            )

    return redirect("projects")


def edit_project(request, project_id):
    project = get_object_or_404(Project, id=project_id)

    if request.method == "POST":
        name = request.POST.get("name")
        project_type_id = request.POST.get("project_type")
        business_requirements = request.POST.get("business_requirements")

        if name and project_type_id and business_requirements:
            project.name = name
            project.project_type_id = project_type_id
            project.business_requirements = business_requirements
            project.save()

    return redirect("projects")


def delete_project(request, project_id):
    project = get_object_or_404(Project, id=project_id)

    if request.method == "POST":
        project.delete()

    return redirect("projects")


def project_detail(request, project_id):
    project = get_object_or_404(Project, id=project_id)

    latest_br = BusinessRequirement.objects.filter(project=project).order_by("-id").first()

    return render(request, "project_detail.html", {
        "project": project,
        "latest_br": latest_br,
    })


@login_required
def run_pipeline_for_project_view(request, project_id):
    project = get_object_or_404(Project, id=project_id)

    if request.method != "POST":
        return redirect("project_detail", project_id=project.id)

    br_text = str(project.business_requirements).strip()

    if not br_text:
        return redirect("project_detail", project_id=project.id)

    last_br = BusinessRequirement.objects.order_by("-br_id").first()
    next_br_id = 1 if not last_br else last_br.br_id + 1

    br_obj = BusinessRequirement.objects.create(
        project=project,
        br_id=next_br_id,
        text=br_text,
        tokens="",
        primary_task_type="",
        primary_task_subtype="",
        num_system_requirements=0,
        uncertainty=0.0,
    )

    model1 = BRToSRService()
    pred1 = model1.predict_from_text(
        business_requirement_text=br_text,
        br_id=br_obj.br_id,
        task_type="",
        task_subtype="",
    )
    model1.save_prediction_to_db(br_obj, pred1["parsed"])

    parsed_br = pred1["parsed"].get("business_requirement", {})
    br_obj.primary_task_type    = parsed_br.get("task_type", "") or ""
    br_obj.primary_task_subtype = parsed_br.get("task_subtype", "") or ""
    br_obj.save(update_fields=["primary_task_type", "primary_task_subtype"])


    pair_builder = RuntimePairBuilderService(br_obj.br_id)
    pair_builder.save_runtime_pairs()

    model3 = RelationshipClassifierService()
    model3.predict_from_runtime_pairs(br_obj.br_id)

    model4 = GAT2Service()
    try:
        model4.execute_from_db(br_id=br_obj.br_id)
    except Exception as e:
        print("MODEL 4 ERROR:", str(e))

    try:
        model2 = ExperienceDNNService()
        model2.save_embeddings_to_db()
    except Exception as e:
        print("MODEL 2 ERROR:", str(e))

    model5 = GAT1Service()
    try:
        model5.execute_from_db(br_id=br_obj.br_id)
    except Exception as e:
        print("MODEL 5 ERROR:", str(e))

    return redirect("combined_schedule", br_id=br_obj.br_id)