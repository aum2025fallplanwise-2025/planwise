from django.db import models


class ProjectType(models.Model):
    name = models.CharField(max_length=200, unique=True)
    innovation = models.CharField(max_length=100, blank=True, default="")
    risk = models.CharField(max_length=100, blank=True, default="")
    roi_horizon = models.CharField(max_length=100, blank=True, default="")

    def __str__(self):
        return self.name


class Project(models.Model):
    name = models.CharField(max_length=200)
    project_type = models.ForeignKey(ProjectType, on_delete=models.CASCADE)
    business_requirements = models.TextField()

    def __str__(self):
        return self.name


class BusinessRequirement(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True, blank=True, related_name="br_items")
    br_id = models.IntegerField(unique=True)
    text = models.TextField()
    tokens = models.TextField(blank=True, default="")
    primary_task_type = models.CharField(max_length=200)
    primary_task_subtype = models.CharField(max_length=200)
    num_system_requirements = models.IntegerField(default=0)
    uncertainty = models.FloatField(default=0)

    def __str__(self):
        return f"BR-{self.br_id}"


class SystemRequirement(models.Model):
    business_requirement = models.ForeignKey(BusinessRequirement, on_delete=models.CASCADE, related_name="system_requirements")
    system_req_id = models.CharField(max_length=50)
    task_type = models.CharField(max_length=200)
    task_subtype = models.CharField(max_length=200)
    required_experience_level = models.CharField(max_length=100)
    uncertainty = models.FloatField(default=0)

    def __str__(self):
        return self.system_req_id
