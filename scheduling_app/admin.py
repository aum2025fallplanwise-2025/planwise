from django.contrib import admin
from .models import NormalizedEdge, OptimalAssignment, CriticalPathPrediction


class NormalizedEdgeAdmin(admin.ModelAdmin):
    list_display = ('employee_id', 'subtask_id', 'avg_duration', 'avg_defects', 'total_tasks_done', 'total_defects')
    search_fields = ('employee_id', 'subtask_id')


class OptimalAssignmentAdmin(admin.ModelAdmin):
    list_display = ('employee_id', 'subtask_id', 'cost')
    search_fields = ('employee_id', 'subtask_id')


class CriticalPathPredictionAdmin(admin.ModelAdmin):
    list_display = ('subtask_name', 'predicted_score', 'in_degree', 'out_degree')
    search_fields = ('subtask_name',)


admin.site.register(NormalizedEdge, NormalizedEdgeAdmin)
admin.site.register(OptimalAssignment, OptimalAssignmentAdmin)
admin.site.register(CriticalPathPrediction, CriticalPathPredictionAdmin)
