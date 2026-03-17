class DiabetesRequest {
  final int age;
  final int alcoholConsumptionPerWeek;
  final int physicalActivityMinutesPerWeek;
  final double dietScore;
  final double sleepHoursPerDay;
  final int screenTimeHoursPerDay;
  final double bmi;
  final double waistToHipRatio;
  final int systolicBp;
  final int diastolicBp;
  final int heartRate;
  final int cholesterolTotal;
  final int hdlCholesterol;
  final int ldlCholesterol;
  final int triglycerides;
  final int familyHistoryDiabetes;
  final int hypertensionHistory;
  final int cardiovascularHistory;

  DiabetesRequest({
    required this.age,
    required this.alcoholConsumptionPerWeek,
    required this.physicalActivityMinutesPerWeek,
    required this.dietScore,
    required this.sleepHoursPerDay,
    required this.screenTimeHoursPerDay,
    required this.bmi,
    required this.waistToHipRatio,
    required this.systolicBp,
    required this.diastolicBp,
    required this.heartRate,
    required this.cholesterolTotal,
    required this.hdlCholesterol,
    required this.ldlCholesterol,
    required this.triglycerides,
    required this.familyHistoryDiabetes,
    required this.hypertensionHistory,
    required this.cardiovascularHistory,
  });

  Map<String, dynamic> toJson() => {
    "age": age,
    "alcohol_consumption_per_week": alcoholConsumptionPerWeek,
    "physical_activity_minutes_per_week": physicalActivityMinutesPerWeek,
    "diet_score": dietScore,
    "sleep_hours_per_day": sleepHoursPerDay,
    "screen_time_hours_per_day": screenTimeHoursPerDay,
    "bmi": bmi,
    "waist_to_hip_ratio": waistToHipRatio,
    "systolic_bp": systolicBp,
    "diastolic_bp": diastolicBp,
    "heart_rate": heartRate,
    "cholesterol_total": cholesterolTotal,
    "hdl_cholesterol": hdlCholesterol,
    "ldl_cholesterol": ldlCholesterol,
    "triglycerides": triglycerides,
    "family_history_diabetes": familyHistoryDiabetes,
    "hypertension_history": hypertensionHistory,
    "cardiovascular_history": cardiovascularHistory,
  };
}

class DiabetesResponse {
  final int prediction;
  final String label;
  final double confidence;
  final double nonDiabeticProbability;
  final double diabeticProbability;

  DiabetesResponse({
    required this.prediction,
    required this.label,
    required this.confidence,
    required this.nonDiabeticProbability,
    required this.diabeticProbability,
  });

  factory DiabetesResponse.fromJson(Map<String, dynamic> json) {
    final probability = (json['probability'] ?? <String, dynamic>{}) as Map<String, dynamic>;

    return DiabetesResponse(
      prediction: (json['prediction'] ?? 0) as int,
      label: (json['label'] ?? 'Unknown').toString(),
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
      nonDiabeticProbability:
          (probability['non_diabetic'] as num?)?.toDouble() ?? 0,
      diabeticProbability: (probability['diabetic'] as num?)?.toDouble() ?? 0,
    );
  }
}