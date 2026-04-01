from .views import DiabetesViewSet, HeartDiseaseViewSet, ChronicKidneyDiseaseViewSet
from rest_framework.routers import DefaultRouter
from django.urls import path, include

router = DefaultRouter()

router.register(r"diabetes", DiabetesViewSet, basename="diabetes")
router.register(r"heart-disease", HeartDiseaseViewSet, basename="heart-disease")
router.register(
    r"chronic-kidney-disease",
    ChronicKidneyDiseaseViewSet,
    basename="chronic-kidney-disease",
)

urlpatterns = [path("", include(router.urls))]
