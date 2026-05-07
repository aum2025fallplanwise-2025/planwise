from django.urls import path
from . import views

urlpatterns = [
    path("", views.training_home_view, name="training"),

    path("model1/train/", views.model1_train_view, name="model1_train"),

    path("model2/train/", views.model2_train_view, name="model2_train"),

    path("model3/train/", views.model3_train_view, name="model3_train"),


    path("model4/train/", views.model4_train_view, name="model4_train"),

    path("model5/train/", views.model5_train_view, name="model5_train"),
]