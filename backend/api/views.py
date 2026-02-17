from .serializers import DiabetesPredictionSerializer
from rest_framework.response import Response
from rest_framework import status, viewsets
from django.conf import settings
import pandas as pd
import joblib
import os

# !Project ViewSets

MODELS_DIR = os.path.join(settings.BASE_DIR, "diabetes", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "diabetes.pkl")

try:
    print(f"Loading AI artifact from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("✓ Diabetes Model Loaded Successfully")
except Exception as e:
    print(f"!! ERROR loading model: {e}")
    model = None


class DiabetesViewSet(viewsets.ViewSet):
    """
    GET  -> Checks if model is live
    POST -> Predicts diabetes risk
    """

    serializer_class = DiabetesPredictionSerializer

    def list(self, request):
        status_code = (
            status.HTTP_200_OK if model else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return Response(
            {
                "status": "Diabetes model live" if model else "Model unavailable",
                "model_loaded": model is not None,
            },
            status=status_code,
        )

    def create(self, request):
        if not model:
            return Response(
                {"error": "Model not loaded."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        serializer = DiabetesPredictionSerializer(data=request.data)

        if serializer.is_valid():
            try:
                # 1. Prepare Features
                # Extract validated data into a flat list or array in CORRECT order
                data = serializer.validated_data

                # Ensure features match training order!
                # IMPORTANT: Update this list if model training features change
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

                # Build DataFrame with feature names to match training format
                final_features = pd.DataFrame(
                    [[data.get(name) for name in feature_names]],
                    columns=feature_names
                )

                # 2. Predict
                prediction_class = int(model.predict(final_features)[0])

                # Check if model supports probability
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(final_features)[0]
                    prob_non_diabetic = probabilities[0]
                    prob_diabetic = probabilities[1]
                    confidence = max(prob_non_diabetic, prob_diabetic) * 100
                else:
                    # Fallback if model behaves differently (e.g. SVM without probability)
                    prob_non_diabetic = 0
                    prob_diabetic = 1.0 if prediction_class == 1 else 0.0
                    confidence = 100.0

                # 3. Construct Response
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
