from django.contrib import admin
from .models import Employee, ExperienceLevel, EmployeeVector


class ExperienceLevelAdmin(admin.ModelAdmin):
    list_display = ('name', 'rank', 'min_years', 'max_years', 'primary_focus', 'autonomy', 'decision_scope', 'leadership', 'system_impact')
    search_fields = ('name',)


class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('employee_id', 'first_name', 'last_name', 'name', 'experience')
    search_fields = ('name', 'first_name', 'last_name')
    list_filter = ('experience',)


class EmployeeVectorAdmin(admin.ModelAdmin):
    list_display = ('employee',)
    search_fields = ('employee__name',)


admin.site.register(ExperienceLevel, ExperienceLevelAdmin)
admin.site.register(Employee, EmployeeAdmin)
admin.site.register(EmployeeVector, EmployeeVectorAdmin)
