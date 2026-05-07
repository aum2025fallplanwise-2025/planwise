from django.db import models
from project_app.models import Project
from employee_app.models import Employee, ExperienceLevel


class Task(models.Model):
    task_id = models.IntegerField(default=0)
    subtask_id = models.IntegerField(default=0)
    name = models.CharField(max_length=200)
    task_type = models.CharField(max_length=255, default="General")
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return self.name


class EmployeeTask(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    experience_level = models.ForeignKey(ExperienceLevel, on_delete=models.SET_NULL, null=True, blank=True)
    working_hours = models.FloatField(default=0)
    defects = models.IntegerField(default=0)
    total_tasks_done = models.IntegerField(default=0)
    total_hours = models.FloatField(default=0)
    total_defects = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.employee} - {self.task}"


class SubtaskVector(models.Model):
    subtask_id = models.IntegerField(unique=True)
    task_name = models.CharField(max_length=200)
    subtask_name = models.CharField(max_length=200)
    task_norm = models.CharField(max_length=200, blank=True, default="")
    sub_norm = models.CharField(max_length=200, blank=True, default="")
    vector_data = models.JSONField()

    def __str__(self):
        return f"{self.task_name} - {self.subtask_name}"
