from django.urls import path
from . import views

urlpatterns = [
    path("", views.projects_list, name="projects"),
    path("new/", views.add_project, name="new_project"),
    path("<int:project_id>/", views.project_detail, name="project_detail"),
    path("<int:project_id>/edit/", views.edit_project, name="edit_project"),
    path("<int:project_id>/delete/", views.delete_project, name="delete_project"),
    path("<int:project_id>/run-pipeline/", views.run_pipeline_for_project_view, name="run_pipeline_for_project"),
]