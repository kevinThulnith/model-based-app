from .serializers import (
    DiabetesPredictionSerializer,
    HeartDiseasePredictionSerializer,
    ChronicKidneyDiseasePredictionSerializer,
)
from rest_framework.response import Response
from rest_framework import status, viewsets
from django.conf import settings
import pandas as pd
import joblib
import os

# !Project ViewSets

# ── Diabetes ──────────────────────────────────────────────────────────────────

DIABETES_MODEL_PATH = os.path.join(
    settings.BASE_DIR, "diabetes", "models", "diabetes.pkl"
)

try:
    print(f"Loading AI artifact from {DIABETES_MODEL_PATH}...")
    diabetes_model = joblib.load(DIABETES_MODEL_PATH)
    print("✓ Diabetes Model Loaded Successfully")
except Exception as e:
    print(f"!! ERROR loading diabetes model: {e}")
    diabetes_model = None


class DiabetesViewSet(viewsets.ViewSet):
    """
    GET  -> Checks if model is live
    POST -> Predicts diabetes risk
    """

    serializer_class = DiabetesPredictionSerializer

    def list(self, request):
        status_code = (
            status.HTTP_200_OK
            if diabetes_model
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return Response(
            {
                "status": (
                    "Diabetes model live" if diabetes_model else "Model unavailable"
                ),
                "model_loaded": diabetes_model is not None,
            },
            status=status_code,
        )

    def create(self, request):
        if not diabetes_model:
            return Response(
                {"error": "Model not loaded."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        serializer = DiabetesPredictionSerializer(data=request.data)

        if serializer.is_valid():
            try:
                data = serializer.validated_data

                feature_names = [
                    "age",
                    "alcohol_consumption_per_week",
                    "physical_activity_minutes_per_week",
                    "diet_score",
                    "sleep_hours_per_day",
                    "screen_time_hours_per_day",
                    "bmi",
                    "waist_to_hip_ratio",
                    "systolic_bp",
                    "diastolic_bp",
                    "heart_rate",
                    "cholesterol_total",
                    "hdl_cholesterol",
                    "ldl_cholesterol",
                    "triglycerides",
                    "family_history_diabetes",
                    "hypertension_history",
                    "cardiovascular_history",
                ]

                final_features = pd.DataFrame(
                    [[data.get(name) for name in feature_names]],
                    columns=feature_names,
                )

                prediction_class = int(diabetes_model.predict(final_features)[0])

                if hasattr(diabetes_model, "predict_proba"):
                    probabilities = diabetes_model.predict_proba(final_features)[0]
                    prob_non_diabetic = probabilities[0]
                    prob_diabetic = probabilities[1]
                    confidence = max(prob_non_diabetic, prob_diabetic) * 100
                else:
                    prob_non_diabetic = 0
                    prob_diabetic = 1.0 if prediction_class == 1 else 0.0
                    confidence = 100.0

                return Response(
                    {
                        "prediction": prediction_class,
                        "label": (
                            "Diabetic" if prediction_class == 1 else "Non-Diabetic"
                        ),
                        "probability": {
                            "non_diabetic": prob_non_diabetic,
                            "diabetic": prob_diabetic,
                        },
                        "confidence": confidence,
                    },
                    status=status.HTTP_200_OK,
                )

            except Exception as e:
                return Response(
                    {"error": f"Prediction failed: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ── Heart Disease ─────────────────────────────────────────────────────────────

HEART_MODEL_PATH = os.path.join(
    settings.BASE_DIR, "heart disease", "models", "heart_disease.pkl"
)

try:
    print(f"Loading AI artifact from {HEART_MODEL_PATH}...")
    heart_model = joblib.load(HEART_MODEL_PATH)
    print("✓ Heart Disease Model Loaded Successfully")
except Exception as e:
    print(f"!! ERROR loading heart disease model: {e}")
    heart_model = None


class HeartDiseaseViewSet(viewsets.ViewSet):
    """
    GET  -> Checks if model is live
    POST -> Predicts heart disease risk
    """

    serializer_class = HeartDiseasePredictionSerializer

    def list(self, request):
        status_code = (
            status.HTTP_200_OK if heart_model else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return Response(
            {
                "status": (
                    "Heart Disease model live" if heart_model else "Model unavailable"
                ),
                "model_loaded": heart_model is not None,
            },
            status=status_code,
        )

    def create(self, request):
        if not heart_model:
            return Response(
                {"error": "Model not loaded."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        serializer = HeartDiseasePredictionSerializer(data=request.data)

        if serializer.is_valid():
            try:
                data = serializer.validated_data

                # Maps serializer field names -> original training column names
                field_to_col = {
                    "age": "Age",
                    "sex": "Sex",
                    "chest_pain_type": "Chest pain type",
                    "bp": "BP",
                    "cholesterol": "Cholesterol",
                    "fbs_over_120": "FBS over 120",
                    "ekg_results": "EKG results",
                    "max_hr": "Max HR",
                    "exercise_angina": "Exercise angina",
                    "st_depression": "ST depression",
                    "slope_of_st": "Slope of ST",
                    "num_vessels_fluro": "Number of vessels fluro",
                    "thallium": "Thallium",
                }

                col_names = list(field_to_col.values())
                values = [data[field] for field in field_to_col]
                final_features = pd.DataFrame([values], columns=col_names)

                prediction_class = int(heart_model.predict(final_features)[0])

                if hasattr(heart_model, "predict_proba"):
                    probabilities = heart_model.predict_proba(final_features)[0]
                    prob_no_disease = probabilities[0]
                    prob_disease = probabilities[1]
                    confidence = max(prob_no_disease, prob_disease) * 100
                else:
                    prob_no_disease = 0
                    prob_disease = 1.0 if prediction_class == 1 else 0.0
                    confidence = 100.0

                return Response(
                    {
                        "prediction": prediction_class,
                        "label": "Presence" if prediction_class == 1 else "Absence",
                        "probability": {
                            "no_disease": prob_no_disease,
                            "heart_disease": prob_disease,
                        },
                        "confidence": confidence,
                    },
                    status=status.HTTP_200_OK,
                )

            except Exception as e:
                return Response(
                    {"error": f"Prediction failed: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ── Chronic Kidney Disease ────────────────────────────────────────────────────

CKD_MODEL_PATH = os.path.join(
    settings.BASE_DIR, "chronic kidney disease", "models", "chronic_kidney_disease.pkl"
)

try:
    print(f"Loading AI artifact from {CKD_MODEL_PATH}...")
    ckd_model = joblib.load(CKD_MODEL_PATH)
    print("✓ Chronic Kidney Disease Model Loaded Successfully")
except Exception as e:
    print(f"!! ERROR loading CKD model: {e}")
    ckd_model = None


class ChronicKidneyDiseaseViewSet(viewsets.ViewSet):
    """
    GET  -> Checks if model is live
    POST -> Predicts chronic kidney disease risk
    """

    serializer_class = ChronicKidneyDiseasePredictionSerializer

    def list(self, request):
        status_code = (
            status.HTTP_200_OK if ckd_model else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return Response(
            {
                "status": (
                    "Chronic Kidney Disease model live"
                    if ckd_model
                    else "Model unavailable"
                ),
                "model_loaded": ckd_model is not None,
            },
            status=status_code,
        )

    def create(self, request):
        if not ckd_model:
            return Response(
                {"error": "Model not loaded."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        serializer = ChronicKidneyDiseasePredictionSerializer(data=request.data)

        if serializer.is_valid():
            try:
                data = serializer.validated_data

                # Maps serializer field names -> original training column names
                field_to_col = {
                    "bp_diastolic": "bp (Diastolic)",
                    "bp_limit": "bp limit",
                    "sg": "sg",
                    "al": "al",
                    "rbc": "rbc",
                    "su": "su",
                    "pc": "pc",
                    "pcc": "pcc",
                    "ba": "ba",
                    "bgr": "bgr",
                    "bu": "bu",
                    "sod": "sod",
                    "sc": "sc",
                    "pot": "pot",
                    "hemo": "hemo",
                    "pcv": "pcv",
                    "rbcc": "rbcc",
                    "wbcc": "wbcc",
                    "htn": "htn",
                    "dm": "dm",
                    "cad": "cad",
                    "appet": "appet",
                    "pe": "pe",
                    "ane": "ane",
                    "grf": "grf",
                    "stage": "stage",
                    "age": "age",
                }

                col_names = list(field_to_col.values())
                values = [data[field] for field in field_to_col]
                final_features = pd.DataFrame([values], columns=col_names)

                prediction_class = int(ckd_model.predict(final_features)[0])

                if hasattr(ckd_model, "predict_proba"):
                    probabilities = ckd_model.predict_proba(final_features)[0]
                    prob_not_affected = probabilities[0]
                    prob_affected = probabilities[1]
                    confidence = max(prob_not_affected, prob_affected) * 100
                else:
                    prob_not_affected = 0
                    prob_affected = 1.0 if prediction_class == 1 else 0.0
                    confidence = 100.0

                return Response(
                    {
                        "prediction": prediction_class,
                        "label": (
                            "Affected" if prediction_class == 1 else "Not Affected"
                        ),
                        "probability": {
                            "not_affected": prob_not_affected,
                            "affected": prob_affected,
                        },
                        "confidence": confidence,
                    },
                    status=status.HTTP_200_OK,
                )

            except Exception as e:
                return Response(
                    {"error": f"Prediction failed: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
