from django.db import models


class NormalizedEdge(models.Model):
    employee_id = models.IntegerField()
    subtask_id = models.IntegerField()
    avg_duration = models.FloatField(default=0)
    avg_defects = models.FloatField(default=0)
    total_tasks_done = models.FloatField(default=0)
    total_defects = models.FloatField(default=0)

    class Meta:
        unique_together = ('employee_id', 'subtask_id')

    def __str__(self):
        return f"Edge {self.employee_id} -> {self.subtask_id}"


class OptimalAssignment(models.Model):
    br_id = models.IntegerField(default=0)
    employee_id = models.IntegerField()
    employee_name = models.CharField(max_length=200, default="")
    subtask_id = models.IntegerField()
    subtask_name = models.CharField(max_length=255, default="")
    cost = models.FloatField()

    def __str__(self):
        return f"BR{self.br_id}: Assignment {self.employee_id} -> {self.subtask_name}"


class CriticalPathPrediction(models.Model):
    br_id = models.IntegerField(default=0)
    subtask_name = models.CharField(max_length=200)
    predicted_score = models.FloatField()
    in_degree = models.FloatField()
    out_degree = models.FloatField()

    def __str__(self):
        return f"BR{self.br_id}: {self.subtask_name}"


class SubtaskRelationship(models.Model):
    RELATIONSHIP_CHOICES = [
        ('FS', 'Finish-Start'),
        ('SF', 'Start-Finish'),
        ('FF', 'Finish-Finish'),
        ('SS', 'Start-Start'),
        ('NO_RELATION', 'No Relation'),
    ]

    br_id = models.IntegerField()
    subtask_a = models.CharField(max_length=255)
    subtask_b = models.CharField(max_length=255)
    pair_text = models.TextField(blank=True)
    relationship_label = models.CharField(max_length=20, choices=RELATIONSHIP_CHOICES)
    predicted_relationship = models.CharField(max_length=20, choices=RELATIONSHIP_CHOICES, blank=True)
    confidence = models.FloatField(default=0.0)

    class Meta:
        unique_together = ('br_id', 'subtask_a', 'subtask_b')

    def __str__(self):
        return f"BR{self.br_id}: {self.subtask_a} -> {self.subtask_b} [{self.predicted_relationship}]"


class RuntimeSubtaskPair(models.Model):
    br_id = models.IntegerField()
    subtask_a = models.CharField(max_length=200)
    subtask_b = models.CharField(max_length=200)

    class Meta:
        unique_together = ('br_id', 'subtask_a', 'subtask_b')

    def __str__(self):
        return f"BR{self.br_id}: {self.subtask_a} -> {self.subtask_b}"