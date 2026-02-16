from rest_framework.routers import DefaultRouter
from django.urls import path, include
from .views import DiabetesViewSet

router = DefaultRouter()

router.register(r"diabetes", DiabetesViewSet, basename="diabetes")

urlpatterns = [path("", include(router.urls))]
