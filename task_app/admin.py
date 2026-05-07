from django.contrib import admin
from .models import Task, EmployeeTask, SubtaskVector


class TaskAdmin(admin.ModelAdmin):
    list_display = ('name', 'task_type', 'task_id', 'subtask_id')
    search_fields = ('name', 'task_type')
    list_filter = ('task_type',)


class EmployeeTaskAdmin(admin.ModelAdmin):
    list_display = ('employee', 'task', 'experience_level', 'working_hours', 'defects', 'total_tasks_done', 'total_hours', 'total_defects')
    search_fields = ('employee__name', 'task__name')
    list_filter = ('experience_level',)


class SubtaskVectorAdmin(admin.ModelAdmin):
    list_display = ('subtask_id', 'task_name', 'subtask_name', 'task_norm', 'sub_norm')
    search_fields = ('task_name', 'subtask_name')


admin.site.register(Task, TaskAdmin)
admin.site.register(EmployeeTask, EmployeeTaskAdmin)
admin.site.register(SubtaskVector, SubtaskVectorAdmin)
