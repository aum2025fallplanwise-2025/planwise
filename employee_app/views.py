from django.shortcuts import render, get_object_or_404
from django.db.models import Avg
from task_app.models import EmployeeTask
from employee_app.models import Employee


def employees_view(request):
    query = request.GET.get('q', '')
    sort = request.GET.get('sort', '')

    employees = Employee.objects.select_related('experience').all()

    if query:
        employees = employees.filter(name__icontains=query)

    data = []

    for emp in employees:
        tasks = EmployeeTask.objects.filter(employee=emp)

        avg_hours = tasks.aggregate(Avg('working_hours'))['working_hours__avg']
        avg_defects = tasks.aggregate(Avg('defects'))['defects__avg']

        data.append({
            'id': emp.id,
            'employee_id': emp.employee_id,
            'name': emp.name,
            'experience': emp.experience.name,
            'hours': round(avg_hours, 2) if avg_hours else 0,
            'defects': round(avg_defects, 2) if avg_defects else 0,
        })

    if sort == 'name':
        data.sort(key=lambda x: x['name'])

    elif sort == 'defects':
        data.sort(key=lambda x: x['defects'])

    return render(request, 'employees.html', {
        'employees': data,
        'query': query,
        'sort': sort,
    })


def employee_detail(request, id):
    employee = get_object_or_404(Employee, id=id)
    employee_tasks = EmployeeTask.objects.filter(employee=employee).select_related('task', 'experience_level')

    avg_hours = employee_tasks.aggregate(Avg('working_hours'))['working_hours__avg']
    avg_defects = employee_tasks.aggregate(Avg('defects'))['defects__avg']

    return render(request, 'employee_detail.html', {
        'employee': employee,
        'employee_tasks': employee_tasks,
        'avg_hours': round(avg_hours, 2) if avg_hours else 0,
        'avg_defects': round(avg_defects, 2) if avg_defects else 0,
    })
