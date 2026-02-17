# SEDI App Backend API

## Diabetes Prediction API

### Endpoint
`POST /api/diabetes/`

### Description
Predicts diabetes risk based on health and lifestyle factors using a machine learning model (LightGBM with StandardScaler pipeline).

### Request Body
```json
{
    "age": 70,
    "alcohol_consumption_per_week": 10,
    "physical_activity_minutes_per_week": 100,
    "diet_score": 6.5,
    "sleep_hours_per_day": 7.2,
    "screen_time_hours_per_day": 5,
    "bmi": 25.0,
    "waist_to_hip_ratio": 0.9,
    "systolic_bp": 130,
    "diastolic_bp": 85,
    "heart_rate": 75,
    "cholesterol_total": 100,
    "hdl_cholesterol": 50,
    "ldl_cholesterol": 120,
    "triglycerides": 150,
    "family_history_diabetes": 1,
    "hypertension_history": 1,
    "cardiovascular_history": 1
}
```

### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| age | int | Age in years |
| alcohol_consumption_per_week | int | Alcohol units per week |
| physical_activity_minutes_per_week | int | Minutes of physical activity per week |
| diet_score | float | Diet quality score (0-10) |
| sleep_hours_per_day | float | Average sleep hours per day |
| screen_time_hours_per_day | int | Screen time hours per day |
| bmi | float | Body Mass Index |
| waist_to_hip_ratio | float | Waist to hip ratio |
| systolic_bp | int | Systolic blood pressure (mmHg) |
| diastolic_bp | int | Diastolic blood pressure (mmHg) |
| heart_rate | int | Resting heart rate (bpm) |
| cholesterol_total | int | Total cholesterol (mg/dL) |
| hdl_cholesterol | int | HDL cholesterol (mg/dL) |
| ldl_cholesterol | int | LDL cholesterol (mg/dL) |
| triglycerides | int | Triglycerides (mg/dL) |
| family_history_diabetes | int | Family history of diabetes (0=No, 1=Yes) |
| hypertension_history | int | History of hypertension (0=No, 1=Yes) |
| cardiovascular_history | int | History of cardiovascular disease (0=No, 1=Yes) |

### Response
```json
{
    "prediction": 1,
    "label": "Diabetic",
    "probability": {
        "non_diabetic": 0.089,
        "diabetic": 0.911
    },
    "confidence": 91.1
}
```

### Health Check
`GET /api/diabetes/`

Returns model status:
```json
{
    "status": "Diabetes model live",
    "model_loaded": true
}
```