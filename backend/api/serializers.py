from rest_framework import serializers

# !Project serializers


class DiabetesPredictionSerializer(serializers.Serializer):
    # Float fields
    age = serializers.FloatField()
    alcohol_consumption_per_week = serializers.FloatField()
    physical_activity_minutes_per_week = serializers.FloatField()
    diet_score = serializers.FloatField()
    sleep_hours_per_day = serializers.FloatField()
    screen_time_hours_per_day = serializers.FloatField()
    bmi = serializers.FloatField()
    waist_to_hip_ratio = serializers.FloatField()
    systolic_bp = serializers.FloatField()
    diastolic_bp = serializers.FloatField()
    heart_rate = serializers.FloatField()
    cholesterol_total = serializers.FloatField()
    hdl_cholesterol = serializers.FloatField()
    ldl_cholesterol = serializers.FloatField()
    triglycerides = serializers.FloatField()

    # Integer fields (0 or 1)
    family_history_diabetes = serializers.IntegerField()
    hypertension_history = serializers.IntegerField()
    cardiovascular_history = serializers.IntegerField()
