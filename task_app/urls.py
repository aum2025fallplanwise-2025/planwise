from django.urls import path
from . import views

urlpatterns = [

    path('', views.tasks_list, name="tasks"),

    path('new/', views.add_task, name="new_task"),

    path('<int:task_id>/', views.task_detail, name='task_detail'),

]