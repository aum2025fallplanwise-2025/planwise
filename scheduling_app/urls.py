from django.urls import path
from . import views

urlpatterns = [
    path("", views.scheduling_home_view, name="scheduling_home"),
    path("combined/<int:br_id>/", views.combined_schedule_view, name="combined_schedule"),
]