from django.db import models


class ExperienceLevel(models.Model):
    name = models.CharField(max_length=100)
    rank = models.IntegerField(default=0)
    min_years = models.IntegerField(default=0)
    max_years = models.IntegerField(default=0)
    primary_focus = models.CharField(max_length=200, blank=True, default="")
    autonomy = models.CharField(max_length=100, blank=True, default="")
    decision_scope = models.CharField(max_length=100, blank=True, default="")
    leadership = models.CharField(max_length=100, blank=True, default="")
    system_impact = models.CharField(max_length=100, blank=True, default="")

    def __str__(self):
        return self.name


class Employee(models.Model):
    employee_id = models.IntegerField(unique=True, default=0)
    first_name = models.CharField(max_length=100, default="")
    last_name = models.CharField(max_length=100, default="")
    name = models.CharField(max_length=200)
    experience = models.ForeignKey(ExperienceLevel, on_delete=models.CASCADE)

    def __str__(self):
        return self.name


class EmployeeVector(models.Model):
    employee = models.OneToOneField(Employee, on_delete=models.CASCADE, related_name="vector")
    vector_data = models.JSONField()

    def __str__(self):
        return f"Vector for {self.employee.name}"
