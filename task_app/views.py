from django.shortcuts import render, redirect, get_object_or_404
from .models import Task, EmployeeTask

def tasks_list(request):
    tasks = Task.objects.all()

    return render(request, "tasks.html", {
        "tasks": tasks
    })


def add_task(request):
    if request.method == "POST":
        name = request.POST.get("name")
        task_type = request.POST.get("task_type")

        Task.objects.create(
            name=name,
            task_type=task_type
        )

        return redirect("tasks")

    return render(request, "new_task.html")

def task_detail(request, task_id):
    task = get_object_or_404(Task, id=task_id)

    employee_tasks = EmployeeTask.objects.filter(task=task).select_related('employee', 'experience_level')

    total_hours = sum(et.working_hours for et in employee_tasks)
    total_defects = sum(et.defects for et in employee_tasks)
    count = employee_tasks.count()

    avg_hours = total_hours / count if count > 0 else 0
    avg_defects = total_defects / count if count > 0 else 0

    return render(request, "task_detail.html", {
        "task": task,
        "employee_tasks": employee_tasks,
        "avg_hours": avg_hours,
        "avg_defects": avg_defects
    })
