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

| Parameter                          | Type  | Description                                     |
| ---------------------------------- | ----- | ----------------------------------------------- |
| age                                | int   | Age in years                                    |
| alcohol_consumption_per_week       | int   | Alcohol units per week                          |
| physical_activity_minutes_per_week | int   | Minutes of physical activity per week           |
| diet_score                         | float | Diet quality score (0-10)                       |
| sleep_hours_per_day                | float | Average sleep hours per day                     |
| screen_time_hours_per_day          | int   | Screen time hours per day                       |
| bmi                                | float | Body Mass Index                                 |
| waist_to_hip_ratio                 | float | Waist to hip ratio                              |
| systolic_bp                        | int   | Systolic blood pressure (mmHg)                  |
| diastolic_bp                       | int   | Diastolic blood pressure (mmHg)                 |
| heart_rate                         | int   | Resting heart rate (bpm)                        |
| cholesterol_total                  | int   | Total cholesterol (mg/dL)                       |
| hdl_cholesterol                    | int   | HDL cholesterol (mg/dL)                         |
| ldl_cholesterol                    | int   | LDL cholesterol (mg/dL)                         |
| triglycerides                      | int   | Triglycerides (mg/dL)                           |
| family_history_diabetes            | int   | Family history of diabetes (0=No, 1=Yes)        |
| hypertension_history               | int   | History of hypertension (0=No, 1=Yes)           |
| cardiovascular_history             | int   | History of cardiovascular disease (0=No, 1=Yes) |

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

---

## Heart Disease Prediction API

### Endpoint

`POST /api/heart-disease/`

### Description

Predicts heart disease presence based on clinical measurements using a machine learning model (best of LR, RF, XGBoost, LightGBM, MLP with StandardScaler pipeline).

### Request Body

```json
{
  "age": 58,
  "sex": 1,
  "chest_pain_type": 4,
  "bp": 152,
  "cholesterol": 239,
  "fbs_over_120": 0,
  "ekg_results": 0,
  "max_hr": 158,
  "exercise_angina": 1,
  "st_depression": 3.6,
  "slope_of_st": 2,
  "num_vessels_fluro": 2,
  "thallium": 7
}
```

### Parameters

| Parameter         | Type  | Description                                                                               |
| ----------------- | ----- | ----------------------------------------------------------------------------------------- |
| age               | float | Age in years                                                                              |
| sex               | int   | Sex (0=Female, 1=Male)                                                                    |
| chest_pain_type   | int   | Chest pain type (1=Typical angina, 2=Atypical angina, 3=Non-anginal pain, 4=Asymptomatic) |
| bp                | float | Resting blood pressure (mmHg)                                                             |
| cholesterol       | float | Serum cholesterol (mg/dL)                                                                 |
| fbs_over_120      | int   | Fasting blood sugar > 120 mg/dL (0=No, 1=Yes)                                             |
| ekg_results       | int   | Resting EKG results (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy)                      |
| max_hr            | float | Maximum heart rate achieved (bpm)                                                         |
| exercise_angina   | int   | Exercise-induced angina (0=No, 1=Yes)                                                     |
| st_depression     | float | ST depression induced by exercise                                                         |
| slope_of_st       | int   | Slope of peak exercise ST segment (1=Upsloping, 2=Flat, 3=Downsloping)                    |
| num_vessels_fluro | int   | Number of major vessels coloured by fluoroscopy (0-3)                                     |
| thallium          | int   | Thallium stress test result (3=Normal, 6=Fixed defect, 7=Reversable defect)               |

### Response

```json
{
  "prediction": 1,
  "label": "Presence",
  "probability": {
    "no_disease": 0.12,
    "heart_disease": 0.88
  },
  "confidence": 88.0
}
```

### Health Check

`GET /api/heart-disease/`

```json
{
  "status": "Heart Disease model live",
  "model_loaded": true
}
```

---

## Chronic Kidney Disease Prediction API

### Endpoint

`POST /api/chronic-kidney-disease/`

### Description

Predicts chronic kidney disease based on clinical and lab measurements using a machine learning model (best of LR, RF, XGBoost, LightGBM, MLP with StandardScaler pipeline).

> **Note:** All feature values are label-encoded integers as produced during model training. Categorical columns (e.g. sg, al, bgr ranges) must be passed as their encoded integer representations.

### Request Body

```json
{
  "bp_diastolic": 0,
  "bp_limit": 0,
  "sg": 2,
  "al": 1,
  "rbc": 0,
  "su": 0,
  "pc": 0,
  "pcc": 0,
  "ba": 0,
  "bgr": 1,
  "bu": 0,
  "sod": 3,
  "sc": 0,
  "pot": 0,
  "hemo": 2,
  "pcv": 2,
  "rbcc": 2,
  "wbcc": 3,
  "htn": 0,
  "dm": 0,
  "cad": 0,
  "appet": 0,
  "pe": 0,
  "ane": 0,
  "grf": 1,
  "stage": 0,
  "age": 1
}
```

### Parameters

| Parameter    | Type | Description                                         |
| ------------ | ---- | --------------------------------------------------- |
| bp_diastolic | int  | Diastolic blood pressure (label-encoded)            |
| bp_limit     | int  | BP limit category (label-encoded)                   |
| sg           | int  | Specific gravity (label-encoded)                    |
| al           | int  | Albumin level (label-encoded)                       |
| rbc          | int  | Red blood cell count (label-encoded)                |
| su           | int  | Sugar level (label-encoded)                         |
| pc           | int  | Pus cell (label-encoded)                            |
| pcc          | int  | Pus cell clumps (label-encoded)                     |
| ba           | int  | Bacteria (label-encoded)                            |
| bgr          | int  | Blood glucose random (label-encoded)                |
| bu           | int  | Blood urea (label-encoded)                          |
| sod          | int  | Sodium (label-encoded)                              |
| sc           | int  | Serum creatinine (label-encoded)                    |
| pot          | int  | Potassium (label-encoded)                           |
| hemo         | int  | Hemoglobin (label-encoded)                          |
| pcv          | int  | Packed cell volume (label-encoded)                  |
| rbcc         | int  | RBC count (label-encoded)                           |
| wbcc         | int  | WBC count (label-encoded)                           |
| htn          | int  | Hypertension (0=No, 1=Yes)                          |
| dm           | int  | Diabetes mellitus (0=No, 1=Yes)                     |
| cad          | int  | Coronary artery disease (0=No, 1=Yes)               |
| appet        | int  | Appetite (0=Poor, 1=Good)                           |
| pe           | int  | Pedal edema (0=No, 1=Yes)                           |
| ane          | int  | Anemia (0=No, 1=Yes)                                |
| grf          | int  | Glomerular filtration rate category (label-encoded) |
| stage        | int  | Disease stage (label-encoded)                       |
| age          | int  | Age group (label-encoded)                           |

### Response

```json
{
  "prediction": 1,
  "label": "Affected",
  "probability": {
    "not_affected": 0.05,
    "affected": 0.95
  },
  "confidence": 95.0
}
```

### Health Check

`GET /api/chronic-kidney-disease/`

```json
{
  "status": "Chronic Kidney Disease model live",
  "model_loaded": true
}
```
