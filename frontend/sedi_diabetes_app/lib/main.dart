import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'models/diabetes_model.dart';
import 'services/api_service.dart';

void main() => runApp(const SEDIApp());

class SEDIApp extends StatelessWidget {
  const SEDIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0F766E),
          primary: const Color(0xFF0F766E),
          secondary: const Color(0xFF10B981),
          surface: const Color(0xFFF8FAFC),
        ),
        scaffoldBackgroundColor: const Color(0xFFF1F5F9),
        textTheme: GoogleFonts.outfitTextTheme(),
      ),
      home: const PredictionForm(),
    );
  }
}

class PredictionForm extends StatefulWidget {
  const PredictionForm({super.key});

  @override
  State<PredictionForm> createState() => _PredictionFormState();
}

class _PredictionFormState extends State<PredictionForm> {
  final _formKey = GlobalKey<FormState>();
  final ApiService _apiService = ApiService();
  bool _isLoading = false;
  bool _isCheckingHealth = false;

  final TextEditingController _ageCtrl = TextEditingController();
  final TextEditingController _alcoholCtrl = TextEditingController();
  final TextEditingController _activityCtrl = TextEditingController();
  final TextEditingController _dietCtrl = TextEditingController();
  final TextEditingController _sleepCtrl = TextEditingController();
  final TextEditingController _screenTimeCtrl = TextEditingController();
  final TextEditingController _bmiCtrl = TextEditingController();
  final TextEditingController _whrCtrl = TextEditingController();
  final TextEditingController _sysBpCtrl = TextEditingController();
  final TextEditingController _diaBpCtrl = TextEditingController();
  final TextEditingController _heartRateCtrl = TextEditingController();
  final TextEditingController _cholTotalCtrl = TextEditingController();
  final TextEditingController _hdlCtrl = TextEditingController();
  final TextEditingController _ldlCtrl = TextEditingController();
  final TextEditingController _triCtrl = TextEditingController();

  int _familyHist = 0;
  int _hyperHist = 0;
  int _cardioHist = 0;

  @override
  void initState() {
    super.initState();
    _ageCtrl.text = '70';
    _alcoholCtrl.text = '10';
    _activityCtrl.text = '100';
    _dietCtrl.text = '6.5';
    _sleepCtrl.text = '7.2';
    _screenTimeCtrl.text = '5';
    _bmiCtrl.text = '25.0';
    _whrCtrl.text = '0.9';
    _sysBpCtrl.text = '130';
    _diaBpCtrl.text = '85';
    _heartRateCtrl.text = '75';
    _cholTotalCtrl.text = '100';
    _hdlCtrl.text = '50';
    _ldlCtrl.text = '120';
    _triCtrl.text = '150';
    _familyHist = 1;
    _hyperHist = 1;
    _cardioHist = 1;
  }

  @override
  void dispose() {
    _ageCtrl.dispose();
    _alcoholCtrl.dispose();
    _activityCtrl.dispose();
    _dietCtrl.dispose();
    _sleepCtrl.dispose();
    _screenTimeCtrl.dispose();
    _bmiCtrl.dispose();
    _whrCtrl.dispose();
    _sysBpCtrl.dispose();
    _diaBpCtrl.dispose();
    _heartRateCtrl.dispose();
    _cholTotalCtrl.dispose();
    _hdlCtrl.dispose();
    _ldlCtrl.dispose();
    _triCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);

    try {
      final request = DiabetesRequest(
        age: int.parse(_ageCtrl.text.trim()),
        alcoholConsumptionPerWeek: int.parse(_alcoholCtrl.text.trim()),
        physicalActivityMinutesPerWeek: int.parse(_activityCtrl.text.trim()),
        dietScore: double.parse(_dietCtrl.text.trim()),
        sleepHoursPerDay: double.parse(_sleepCtrl.text.trim()),
        screenTimeHoursPerDay: int.parse(_screenTimeCtrl.text.trim()),
        bmi: double.parse(_bmiCtrl.text.trim()),
        waistToHipRatio: double.parse(_whrCtrl.text.trim()),
        systolicBp: int.parse(_sysBpCtrl.text.trim()),
        diastolicBp: int.parse(_diaBpCtrl.text.trim()),
        heartRate: int.parse(_heartRateCtrl.text.trim()),
        cholesterolTotal: int.parse(_cholTotalCtrl.text.trim()),
        hdlCholesterol: int.parse(_hdlCtrl.text.trim()),
        ldlCholesterol: int.parse(_ldlCtrl.text.trim()),
        triglycerides: int.parse(_triCtrl.text.trim()),
        familyHistoryDiabetes: _familyHist,
        hypertensionHistory: _hyperHist,
        cardiovascularHistory: _cardioHist,
      );

      final result = await _apiService.predict(request);
      if (!mounted) return;
      _showResultDialog(result);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Prediction request failed: $e')),
      );
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Future<void> _runHealthCheck() async {
    setState(() => _isCheckingHealth = true);

    try {
      final health = await _apiService.checkHealth();
      if (!mounted) return;

      final status = health['status']?.toString() ?? 'Unknown';
      final modelLoaded = health['model_loaded']?.toString() ?? 'Unknown';

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Status: $status | model_loaded: $modelLoaded')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Health check failed: $e')),
      );
    } finally {
      if (mounted) {
        setState(() => _isCheckingHealth = false);
      }
    }
  }

  void _showResultDialog(DiabetesResponse res) {
    final color =
        res.prediction == 1 ? const Color(0xFFDC2626) : const Color(0xFF059669);

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        title: const Text('Prediction Result'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              res.label,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
            const SizedBox(height: 12),
            Text('Confidence: ${res.confidence.toStringAsFixed(1)}%'),
            Text(
              'P(Non-diabetic): ${(res.nonDiabeticProbability * 100).toStringAsFixed(1)}%',
            ),
            Text('P(Diabetic): ${(res.diabeticProbability * 100).toStringAsFixed(1)}%'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFFECFEFF), Color(0xFFF8FAFC)],
          ),
        ),
        child: SafeArea(
          child: _isLoading
              ? const Center(child: CircularProgressIndicator())
              : Center(
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 920),
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.all(16),
                      child: Form(
                        key: _formKey,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Diabetes Risk Predictor',
                              style: textTheme.headlineMedium?.copyWith(
                                fontWeight: FontWeight.w800,
                                color: const Color(0xFF0F172A),
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'POST /api/diabetes/ with clinical and lifestyle factors',
                              style: textTheme.bodyMedium?.copyWith(
                                color: const Color(0xFF334155),
                              ),
                            ),
                            const SizedBox(height: 16),
                            Wrap(
                              spacing: 12,
                              runSpacing: 12,
                              children: [
                                FilledButton.icon(
                                  onPressed: _submit,
                                  icon: const Icon(Icons.auto_graph),
                                  label: const Text('Predict Risk'),
                                ),
                                OutlinedButton.icon(
                                  onPressed:
                                      _isCheckingHealth ? null : _runHealthCheck,
                                  icon: const Icon(Icons.health_and_safety),
                                  label: Text(
                                    _isCheckingHealth
                                        ? 'Checking...'
                                        : 'Check Model Health',
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 20),
                            _buildSection('Personal & Lifestyle', [
                              _buildTextField(
                                controller: _ageCtrl,
                                label: 'Age (years)',
                                icon: Icons.badge_outlined,
                                expectsInt: true,
                              ),
                              _buildTextField(
                                controller: _alcoholCtrl,
                                label: 'Alcohol consumption/week (units)',
                                icon: Icons.wine_bar_outlined,
                                expectsInt: true,
                              ),
                              _buildTextField(
                                controller: _activityCtrl,
                                label: 'Physical activity/week (minutes)',
                                icon: Icons.directions_run,
                                expectsInt: true,
                              ),
                              _buildTextField(
                                controller: _dietCtrl,
                                label: 'Diet score (0-10)',
                                icon: Icons.restaurant_menu,
                                expectsInt: false,
                              ),
                              _buildTextField(
                                controller: _sleepCtrl,
                                label: 'Sleep (hours/day)',
                                icon: Icons.bedtime_outlined,
                                expectsInt: false,
                              ),
                              _buildTextField(
                                controller: _screenTimeCtrl,
                                label: 'Screen time (hours/day)',
                                icon: Icons.screenshot_monitor,
                                expectsInt: true,
                              ),
                            ]),
                            _buildSection('Body & Vitals', [
                              _buildTextField(
                                controller: _bmiCtrl,
                                label: 'BMI',
                                icon: Icons.monitor_weight_outlined,
                                expectsInt: false,
                              ),
                              _buildTextField(
                                controller: _whrCtrl,
                                label: 'Waist-to-hip ratio',
                                icon: Icons.straighten,
                                expectsInt: false,
                              ),
                              _buildTextField(
                                controller: _sysBpCtrl,
                                label: 'Systolic BP (mmHg)',
                                icon: Icons.favorite_border,
                                expectsInt: true,
                              ),
                              _buildTextField(
                                controller: _diaBpCtrl,
                                label: 'Diastolic BP (mmHg)',
                                icon: Icons.favorite_border,
                                expectsInt: true,
                              ),
                              _buildTextField(
                                controller: _heartRateCtrl,
                                label: 'Resting heart rate (bpm)',
                                icon: Icons.monitor_heart_outlined,
                                expectsInt: true,
                              ),
                            ]),
                            _buildSection('Lab Profile', [
                              _buildTextField(
                                controller: _cholTotalCtrl,
                                label: 'Total cholesterol (mg/dL)',
                                icon: Icons.science_outlined,
                                expectsInt: true,
                              ),
                              _buildTextField(
                                controller: _hdlCtrl,
                                label: 'HDL cholesterol (mg/dL)',
                                icon: Icons.biotech_outlined,
                                expectsInt: true,
                              ),
                              _buildTextField(
                                controller: _ldlCtrl,
                                label: 'LDL cholesterol (mg/dL)',
                                icon: Icons.biotech_outlined,
                                expectsInt: true,
                              ),
                              _buildTextField(
                                controller: _triCtrl,
                                label: 'Triglycerides (mg/dL)',
                                icon: Icons.science,
                                expectsInt: true,
                              ),
                            ]),
                            _buildSection('Medical History', [
                              _buildBinarySwitch(
                                title: 'Family history of diabetes',
                                value: _familyHist,
                                onChanged: (v) => setState(() => _familyHist = v),
                              ),
                              _buildBinarySwitch(
                                title: 'History of hypertension',
                                value: _hyperHist,
                                onChanged: (v) => setState(() => _hyperHist = v),
                              ),
                              _buildBinarySwitch(
                                title: 'Cardiovascular disease history',
                                value: _cardioHist,
                                onChanged: (v) => setState(() => _cardioHist = v),
                              ),
                            ]),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
        ),
      ),
    );
  }

  Widget _buildSection(String title, List<Widget> children) {
    return Card(
      elevation: 0.8,
      color: Colors.white,
      margin: const EdgeInsets.only(bottom: 16),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: Color(0xFF0F766E),
              ),
            ),
            const Divider(),
            ...children,
          ],
        ),
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    required bool expectsInt,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: TextFormField(
        controller: controller,
        keyboardType: TextInputType.number,
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: Icon(icon, color: const Color(0xFF0F766E)),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
          filled: true,
          fillColor: const Color(0xFFF8FAFC),
        ),
        validator: (value) {
          final text = value?.trim() ?? '';
          if (text.isEmpty) return 'Required';
          if (expectsInt && int.tryParse(text) == null) {
            return 'Enter a valid integer';
          }
          if (!expectsInt && double.tryParse(text) == null) {
            return 'Enter a valid number';
          }
          return null;
        },
      ),
    );
  }

  Widget _buildBinarySwitch({
    required String title,
    required int value,
    required ValueChanged<int> onChanged,
  }) {
    return SwitchListTile.adaptive(
      contentPadding: EdgeInsets.zero,
      title: Text(title),
      value: value == 1,
      onChanged: (enabled) => onChanged(enabled ? 1 : 0),
    );
  }
}
