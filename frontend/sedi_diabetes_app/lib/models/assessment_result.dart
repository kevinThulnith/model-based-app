class AssessmentResult {
  final String id;
  final String type; // 'diabetes', 'heart_disease', 'ckd'
  final String userId;
  final DateTime createdAt;
  final int prediction;
  final String label;
  final double confidence;
  final Map<String, dynamic> probability;
  final Map<String, dynamic> inputs;

  AssessmentResult({
    required this.id,
    required this.type,
    required this.userId,
    required this.createdAt,
    required this.prediction,
    required this.label,
    required this.confidence,
    required this.probability,
    required this.inputs,
  });

  factory AssessmentResult.fromMap(Map<String, dynamic> map, String id) {
    return AssessmentResult(
      id: id,
      type: map['type'] ?? '',
      userId: map['userId'] ?? '',
      createdAt: map['createdAt'] != null
          ? (map['createdAt'] as dynamic).toDate()
          : DateTime.now(),
      prediction: map['prediction'] ?? 0,
      label: map['label'] ?? '',
      confidence: (map['confidence'] ?? 0).toDouble(),
      probability: Map<String, dynamic>.from(map['probability'] ?? {}),
      inputs: Map<String, dynamic>.from(map['inputs'] ?? {}),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'type': type,
      'userId': userId,
      'createdAt': createdAt,
      'prediction': prediction,
      'label': label,
      'confidence': confidence,
      'probability': probability,
      'inputs': inputs,
    };
  }

  String get typeDisplayName {
    switch (type) {
      case 'diabetes':
        return 'Diabetes Screening';
      case 'heart_disease':
        return 'Heart Disease Sync';
      case 'ckd':
        return 'CKD Stage Monitor';
      default:
        return 'Assessment';
    }
  }

  bool get isPositive => prediction == 1;
}
