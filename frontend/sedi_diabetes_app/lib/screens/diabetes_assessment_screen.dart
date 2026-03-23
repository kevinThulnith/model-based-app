import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import '../services/firestore_service.dart';
import '../models/assessment_result.dart';
import '../theme/app_theme.dart';
import '../widgets/common_widgets.dart';
import 'result_screen.dart';

class DiabetesAssessmentScreen extends StatefulWidget {
  const DiabetesAssessmentScreen({super.key});

  @override
  State<DiabetesAssessmentScreen> createState() =>
      _DiabetesAssessmentScreenState();
}

class _DiabetesAssessmentScreenState extends State<DiabetesAssessmentScreen> {
  final _formKey = GlobalKey<FormState>();

  final _age = TextEditingController(text: '45');
  final _alcohol = TextEditingController(text: '2');
  final _physicalActivity = TextEditingController(text: '150');
  final _sleepHours = TextEditingController(text: '7.0');
  final _screenTime = TextEditingController(text: '4');
  final _bmi = TextEditingController(text: '24.0');
  final _waistHip = TextEditingController(text: '0.85');
  final _systolicBp = TextEditingController(text: '120');
  final _diastolicBp = TextEditingController(text: '80');
  final _heartRate = TextEditingController(text: '72');
  final _cholTotal = TextEditingController(text: '180');
  final _hdl = TextEditingController(text: '55');
  final _ldl = TextEditingController(text: '110');
  final _triglycerides = TextEditingController(text: '140');

  double _dietScore = 5.0;
  int _familyHistory = 0;
  int _hypertensionHistory = 0;
  int _cardiovascularHistory = 0;
  bool _isLoading = false;

  @override
  void dispose() {
    for (final c in [
      _age, _alcohol, _physicalActivity, _sleepHours, _screenTime,
      _bmi, _waistHip, _systolicBp, _diastolicBp, _heartRate,
      _cholTotal, _hdl, _ldl, _triglycerides,
    ]) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _runAssessment() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _isLoading = true);

    try {
      final payload = {
        'age': int.parse(_age.text),
        'alcohol_consumption_per_week': int.parse(_alcohol.text),
        'physical_activity_minutes_per_week': int.parse(_physicalActivity.text),
        'diet_score': _dietScore,
        'sleep_hours_per_day': double.parse(_sleepHours.text),
        'screen_time_hours_per_day': int.parse(_screenTime.text),
        'bmi': double.parse(_bmi.text),
        'waist_to_hip_ratio': double.parse(_waistHip.text),
        'systolic_bp': int.parse(_systolicBp.text),
        'diastolic_bp': int.parse(_diastolicBp.text),
        'heart_rate': int.parse(_heartRate.text),
        'cholesterol_total': int.parse(_cholTotal.text),
        'hdl_cholesterol': int.parse(_hdl.text),
        'ldl_cholesterol': int.parse(_ldl.text),
        'triglycerides': int.parse(_triglycerides.text),
        'family_history_diabetes': _familyHistory,
        'hypertension_history': _hypertensionHistory,
        'cardiovascular_history': _cardiovascularHistory,
      };

      final response = await ApiService().predictDiabetes(payload);
      final userId = AuthService().currentUser?.uid ?? '';

      final result = AssessmentResult(
        id: '',
        type: 'diabetes',
        userId: userId,
        createdAt: DateTime.now(),
        prediction: response['prediction'],
        label: response['label'],
        confidence: (response['confidence'] as num).toDouble(),
        probability: Map<String, dynamic>.from(response['probability']),
        inputs: payload,
      );

      if (userId.isNotEmpty) {
        await FirestoreService().saveAssessment(result);
      }

      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => ResultScreen(result: result)),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Assessment failed: $e'),
          backgroundColor: AppColors.tertiary,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16)),
        ),
      );
    }
  }

  String? _required(String? v) =>
      (v == null || v.trim().isEmpty) ? 'Required' : null;

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Scaffold(
          backgroundColor: AppColors.surface,
          body: SafeArea(
            child: Column(
              children: [
                AssessmentHeader(
                  badge: 'CLINICAL BRIEFING',
                  title: 'Diabetes',
                  titleAccent: 'Assessment',
                  subtitle:
                      'Comprehensive metabolic and lifestyle risk evaluation across 18 clinical biomarkers.',
                  onBack: () => Navigator.pop(context),
                ),
                Expanded(
                  child: Form(
                    key: _formKey,
                    child: ListView(
                      padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
                      children: [
                        FormSection(
                          icon: Icons.person_outline,
                          label: 'DEMOGRAPHICS',
                          title: 'Lifestyle & Vitals',
                          children: [
                            PillInputField(label: 'Age', hint: 'Years', controller: _age, validator: _required),
                            PillInputField(label: 'Alcohol Consumption (units/week)', hint: 'e.g. 2', controller: _alcohol, validator: _required),
                            PillInputField(label: 'Physical Activity (min/week)', hint: 'e.g. 150', controller: _physicalActivity, validator: _required),
                            PillInputField(label: 'Sleep Hours Per Day', hint: 'e.g. 7.5', controller: _sleepHours, keyboardType: const TextInputType.numberWithOptions(decimal: true), validator: _required),
                            PillInputField(label: 'Screen Time (hours/day)', hint: 'e.g. 4', controller: _screenTime, validator: _required),
                            PillInputField(label: 'Diet Score (0-10)', isSlider: true, sliderMin: 0, sliderMax: 10, sliderValue: _dietScore, onSliderChanged: (v) => setState(() => _dietScore = v)),
                          ],
                        ),
                        const SizedBox(height: 16),
                        FormSection(
                          icon: Icons.monitor_weight_outlined,
                          label: 'BIOMETRICS',
                          title: 'Body Composition',
                          children: [
                            PillInputField(label: 'BMI', hint: 'e.g. 24.5', controller: _bmi, keyboardType: const TextInputType.numberWithOptions(decimal: true), validator: _required),
                            PillInputField(label: 'Waist-to-Hip Ratio', hint: 'e.g. 0.85', controller: _waistHip, keyboardType: const TextInputType.numberWithOptions(decimal: true), validator: _required),
                          ],
                        ),
                        const SizedBox(height: 16),
                        FormSection(
                          icon: Icons.favorite_border,
                          label: 'CARDIOVASCULAR',
                          title: 'Vascular & BP',
                          children: [
                            PillInputField(label: 'Systolic BP (mmHg)', hint: 'e.g. 120', controller: _systolicBp, validator: _required),
                            PillInputField(label: 'Diastolic BP (mmHg)', hint: 'e.g. 80', controller: _diastolicBp, validator: _required),
                            PillInputField(label: 'Heart Rate (bpm)', hint: 'e.g. 72', controller: _heartRate, validator: _required),
                          ],
                        ),
                        const SizedBox(height: 16),
                        FormSection(
                          icon: Icons.biotech_outlined,
                          label: 'BLOOD PANEL',
                          title: 'Cholesterol & Lipids',
                          children: [
                            PillInputField(label: 'Total Cholesterol (mg/dL)', hint: 'e.g. 180', controller: _cholTotal, validator: _required),
                            PillInputField(label: 'HDL Cholesterol (mg/dL)', hint: 'e.g. 55', controller: _hdl, validator: _required),
                            PillInputField(label: 'LDL Cholesterol (mg/dL)', hint: 'e.g. 110', controller: _ldl, validator: _required),
                            PillInputField(label: 'Triglycerides (mg/dL)', hint: 'e.g. 140', controller: _triglycerides, validator: _required),
                          ],
                        ),
                        const SizedBox(height: 16),
                        FormSection(
                          icon: Icons.history_edu_outlined,
                          label: 'CLINICAL HISTORY',
                          title: 'Medical Background',
                          children: [
                            BinaryToggle(label: 'Family History of Diabetes', value: _familyHistory, onChanged: (v) => setState(() => _familyHistory = v)),
                            BinaryToggle(label: 'History of Hypertension', value: _hypertensionHistory, onChanged: (v) => setState(() => _hypertensionHistory = v)),
                            BinaryToggle(label: 'History of Cardiovascular Disease', value: _cardiovascularHistory, onChanged: (v) => setState(() => _cardiovascularHistory = v)),
                          ],
                        ),
                        const SizedBox(height: 32),
                        GradientButton(
                          label: 'Run Assessment',
                          icon: Icons.play_arrow_rounded,
                          onPressed: _isLoading ? null : _runAssessment,
                          isLoading: _isLoading,
                        ),
                        const SizedBox(height: 16),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
        // Public LoadingOverlay from common_widgets.dart
        if (_isLoading) const LoadingOverlay(label: 'Diabetes'),
      ],
    );
  }
}
