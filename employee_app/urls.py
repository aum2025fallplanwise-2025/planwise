from django.urls import path
from . import views

urlpatterns = [

    path('', views.employees_view, name="employees"),
    path('employees/<int:id>/', views.employee_detail, name='employee_detail'),

]