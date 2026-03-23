import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import '../services/firestore_service.dart';
import '../models/assessment_result.dart';
import '../theme/app_theme.dart';
import '../widgets/common_widgets.dart';
import 'result_screen.dart';

class HeartDiseaseAssessmentScreen extends StatefulWidget {
  const HeartDiseaseAssessmentScreen({super.key});

  @override
  State<HeartDiseaseAssessmentScreen> createState() =>
      _HeartDiseaseAssessmentScreenState();
}

class _HeartDiseaseAssessmentScreenState
    extends State<HeartDiseaseAssessmentScreen> {
  final _formKey = GlobalKey<FormState>();

  final _age = TextEditingController(text: '55');
  final _bp = TextEditingController(text: '130');
  final _cholesterol = TextEditingController(text: '220');
  final _maxHr = TextEditingController(text: '150');
  final _stDepression = TextEditingController(text: '1.5');
  final _numVessels = TextEditingController(text: '1');

  int _sex = 1;
  int _chestPainType = 1;
  int _fbsOver120 = 0;
  int _ekgResults = 0;
  int _exerciseAngina = 0;
  int _slopeOfSt = 1;
  int _thallium = 3;
  bool _isLoading = false;

  @override
  void dispose() {
    for (final c in [
      _age, _bp, _cholesterol, _maxHr, _stDepression, _numVessels,
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
        'age': double.parse(_age.text),
        'sex': _sex,
        'chest_pain_type': _chestPainType,
        'bp': double.parse(_bp.text),
        'cholesterol': double.parse(_cholesterol.text),
        'fbs_over_120': _fbsOver120,
        'ekg_results': _ekgResults,
        'max_hr': double.parse(_maxHr.text),
        'exercise_angina': _exerciseAngina,
        'st_depression': double.parse(_stDepression.text),
        'slope_of_st': _slopeOfSt,
        'num_vessels_fluro': int.parse(_numVessels.text),
        'thallium': _thallium,
      };

      final response = await ApiService().predictHeartDisease(payload);
      final userId = AuthService().currentUser?.uid ?? '';

      final result = AssessmentResult(
        id: '',
        type: 'heart_disease',
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
                  title: 'Heart Disease',
                  titleAccent: 'CV Assessment',
                  subtitle:
                      'Predictive cardiovascular risk scoring across 13 clinical and diagnostic markers.',
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
                          title: 'Patient Profile',
                          children: [
                            PillInputField(label: 'Age', hint: 'Years', controller: _age, keyboardType: const TextInputType.numberWithOptions(decimal: true), validator: _required),
                            BinaryToggle(label: 'Biological Sex', value: _sex, option0Label: 'Female', option1Label: 'Male', onChanged: (v) => setState(() => _sex = v)),
                          ],
                        ),
                        const SizedBox(height: 16),
                        FormSection(
                          icon: Icons.favorite_border,
                          label: 'CARDIOVASCULAR',
                          title: 'Cardiac Markers',
                          children: [
                            SegmentedSelector(label: 'Chest Pain Type', value: _chestPainType, options: ['Typical', 'Atypical', 'Non-anginal', 'Asymptom.'], onChanged: (v) => setState(() => _chestPainType = v)),
                            PillInputField(label: 'Blood Pressure (mmHg)', hint: 'e.g. 130', controller: _bp, keyboardType: const TextInputType.numberWithOptions(decimal: true), validator: _required),
                            PillInputField(label: 'Serum Cholesterol (mg/dL)', hint: 'e.g. 220', controller: _cholesterol, keyboardType: const TextInputType.numberWithOptions(decimal: true), validator: _required),
                            BinaryToggle(label: 'Fasting Blood Sugar > 120 mg/dL', value: _fbsOver120, onChanged: (v) => setState(() => _fbsOver120 = v)),
                            SegmentedSelector(label: 'Resting EKG Results', value: _ekgResults + 1, options: ['Normal', 'ST-T Abnorm.', 'LV Hypert.'], onChanged: (v) => setState(() => _ekgResults = v - 1)),
                            PillInputField(label: 'Maximum Heart Rate (bpm)', hint: 'e.g. 150', controller: _maxHr, keyboardType: const TextInputType.numberWithOptions(decimal: true), validator: _required),
                            BinaryToggle(label: 'Exercise-Induced Angina', value: _exerciseAngina, onChanged: (v) => setState(() => _exerciseAngina = v)),
                            PillInputField(label: 'ST Depression (exercise)', hint: 'e.g. 1.5', controller: _stDepression, keyboardType: const TextInputType.numberWithOptions(decimal: true), validator: _required),
                            SegmentedSelector(label: 'Slope of Peak Exercise ST', value: _slopeOfSt, options: ['Upsloping', 'Flat', 'Downsloping'], onChanged: (v) => setState(() => _slopeOfSt = v)),
                            PillInputField(label: 'Major Vessels (Fluoroscopy: 0-3)', hint: 'e.g. 1', controller: _numVessels, validator: _required),
                            SegmentedSelector(
                              label: 'Thallium Stress Test',
                              value: _thallium == 3 ? 1 : _thallium == 6 ? 2 : 3,
                              options: ['Normal', 'Fixed', 'Reversible'],
                              onChanged: (v) => setState(() {
                                _thallium = v == 1 ? 3 : v == 2 ? 6 : 7;
                              }),
                            ),
                          ],
                        ),
                        const SizedBox(height: 32),
                        GradientButton(
                          label: 'Run CV Assessment',
                          icon: Icons.favorite_outline,
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
        if (_isLoading) const LoadingOverlay(label: 'Heart Disease'),
      ],
    );
  }
}
