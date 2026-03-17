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


class HeartDiseasePredictionSerializer(serializers.Serializer):
    age = serializers.FloatField()
    sex = serializers.IntegerField()  # 0=Female, 1=Male
    chest_pain_type = serializers.IntegerField()  # 1-4
    bp = serializers.FloatField()
    cholesterol = serializers.FloatField()
    fbs_over_120 = serializers.IntegerField()  # 0 or 1
    ekg_results = serializers.IntegerField()  # 0, 1, 2
    max_hr = serializers.FloatField()
    exercise_angina = serializers.IntegerField()  # 0 or 1
    st_depression = serializers.FloatField()
    slope_of_st = serializers.IntegerField()  # 1, 2, 3
    num_vessels_fluro = serializers.IntegerField()  # 0-3
    thallium = serializers.IntegerField()  # 3, 6, 7


class ChronicKidneyDiseasePredictionSerializer(serializers.Serializer):
    # All features are label-encoded integers (as produced during training)
    bp_diastolic = serializers.IntegerField()
    bp_limit = serializers.IntegerField()
    sg = serializers.IntegerField()
    al = serializers.IntegerField()
    rbc = serializers.IntegerField()
    su = serializers.IntegerField()
    pc = serializers.IntegerField()
    pcc = serializers.IntegerField()
    ba = serializers.IntegerField()
    bgr = serializers.IntegerField()
    bu = serializers.IntegerField()
    sod = serializers.IntegerField()
    sc = serializers.IntegerField()
    pot = serializers.IntegerField()
    hemo = serializers.IntegerField()
    pcv = serializers.IntegerField()
    rbcc = serializers.IntegerField()
    wbcc = serializers.IntegerField()
    htn = serializers.IntegerField()  # 0 or 1
    dm = serializers.IntegerField()  # 0 or 1
    cad = serializers.IntegerField()  # 0 or 1
    appet = serializers.IntegerField()  # 0 or 1
    pe = serializers.IntegerField()  # 0 or 1
    ane = serializers.IntegerField()  # 0 or 1
    grf = serializers.IntegerField()
    stage = serializers.IntegerField()
    age = serializers.IntegerField()
