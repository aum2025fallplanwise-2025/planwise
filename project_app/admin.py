from django.contrib import admin
from .models import Project, ProjectType, BusinessRequirement, SystemRequirement


class ProjectTypeAdmin(admin.ModelAdmin):
    list_display = ('name', 'innovation', 'risk', 'roi_horizon')


class ProjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'project_type')
    search_fields = ('name',)
    list_filter = ('project_type',)


class BusinessRequirementAdmin(admin.ModelAdmin):
    list_display = ('br_id', 'primary_task_type', 'primary_task_subtype', 'num_system_requirements', 'uncertainty')
    search_fields = ('text', 'primary_task_type', 'primary_task_subtype')
    list_filter = ('primary_task_type',)


class SystemRequirementAdmin(admin.ModelAdmin):
    list_display = ('system_req_id', 'business_requirement', 'task_type', 'task_subtype', 'required_experience_level', 'uncertainty')
    search_fields = ('system_req_id', 'task_type', 'task_subtype')
    list_filter = ('task_type', 'required_experience_level')


admin.site.register(Project, ProjectAdmin)
admin.site.register(ProjectType, ProjectTypeAdmin)
admin.site.register(BusinessRequirement, BusinessRequirementAdmin)
admin.site.register(SystemRequirement, SystemRequirementAdmin)
