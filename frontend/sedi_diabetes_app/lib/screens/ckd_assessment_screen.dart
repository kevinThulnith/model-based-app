import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/api_service.dart';
import '../services/auth_service.dart';
import '../services/firestore_service.dart';
import '../models/assessment_result.dart';
import '../theme/app_theme.dart';
import '../widgets/common_widgets.dart';
import 'result_screen.dart';

class CKDAssessmentScreen extends StatefulWidget {
  const CKDAssessmentScreen({super.key});

  @override
  State<CKDAssessmentScreen> createState() => _CKDAssessmentScreenState();
}

class _CKDAssessmentScreenState extends State<CKDAssessmentScreen> {
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;

  int _bpDiastolic = 0, _bpLimit = 0, _sg = 2, _al = 1, _rbc = 0;
  int _su = 0, _pc = 0, _pcc = 0, _ba = 0, _bgr = 1;
  int _bu = 0, _sod = 3, _sc = 0, _pot = 0, _hemo = 2;
  int _pcv = 2, _rbcc = 2, _wbcc = 3, _htn = 0, _dm = 0;
  int _cad = 0, _appet = 1, _pe = 0, _ane = 0, _grf = 1;
  int _stage = 0, _ageEncoded = 1;

  Future<void> _runAssessment() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _isLoading = true);

    try {
      final payload = {
        'bp_diastolic': _bpDiastolic, 'bp_limit': _bpLimit, 'sg': _sg,
        'al': _al, 'rbc': _rbc, 'su': _su, 'pc': _pc, 'pcc': _pcc,
        'ba': _ba, 'bgr': _bgr, 'bu': _bu, 'sod': _sod, 'sc': _sc,
        'pot': _pot, 'hemo': _hemo, 'pcv': _pcv, 'rbcc': _rbcc,
        'wbcc': _wbcc, 'htn': _htn, 'dm': _dm, 'cad': _cad,
        'appet': _appet, 'pe': _pe, 'ane': _ane, 'grf': _grf,
        'stage': _stage, 'age': _ageEncoded,
      };

      final response = await ApiService().predictCKD(payload);
      final userId = AuthService().currentUser?.uid ?? '';

      final result = AssessmentResult(
        id: '',
        type: 'ckd',
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
                  title: 'CKD Assessment',
                  titleAccent: 'Clinical Factors',
                  subtitle:
                      'A comprehensive evaluation of 27 critical biomarkers required for accurate CKD staging.',
                  onBack: () => Navigator.pop(context),
                ),
                Expanded(
                  child: Form(
                    key: _formKey,
                    child: ListView(
                      padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
                      children: [
                        _CKDSection(
                          icon: Icons.water_drop_outlined,
                          label: 'RENAL FUNCTION',
                          title: 'Renal Function Markers',
                          subtitle: 'Essential diagnostic inputs (5 factors)',
                          chips: const ['Serum Creatinine (sCr)', 'Estimated GFR (eGFR)', 'Albumin-to-Creatinine Ratio (UACR)', 'Blood Urea Nitrogen (BUN)', 'Cystatin C'],
                          children: [
                            _LabelSlider(label: 'Serum Creatinine (sc)', value: _sc, max: 5, onChanged: (v) => setState(() => _sc = v)),
                            _LabelSlider(label: 'GFR Category (grf)', value: _grf, max: 5, onChanged: (v) => setState(() => _grf = v)),
                            _LabelSlider(label: 'Albumin Level (al)', value: _al, max: 5, onChanged: (v) => setState(() => _al = v)),
                            _LabelSlider(label: 'Blood Urea (bu)', value: _bu, max: 5, onChanged: (v) => setState(() => _bu = v)),
                            _LabelSlider(label: 'Specific Gravity (sg)', value: _sg, max: 5, onChanged: (v) => setState(() => _sg = v)),
                          ],
                        ),
                        const SizedBox(height: 16),
                        _CKDSection(
                          icon: Icons.science_outlined,
                          label: 'METABOLIC',
                          title: 'Metabolic & Electrolytes',
                          subtitle: 'Systemic balance indicators (8 factors)',
                          chips: const ['Potassium (K+)', 'Sodium (Na+)', 'Bicarbonate', 'Calcium', 'Phosphorus', 'PTH Levels', 'Hemoglobin', 'Serum Iron / Ferritin / TSAT'],
                          children: [
                            _LabelSlider(label: 'Potassium (pot)', value: _pot, max: 5, onChanged: (v) => setState(() => _pot = v)),
                            _LabelSlider(label: 'Sodium (sod)', value: _sod, max: 5, onChanged: (v) => setState(() => _sod = v)),
                            _LabelSlider(label: 'Blood Glucose Random (bgr)', value: _bgr, max: 5, onChanged: (v) => setState(() => _bgr = v)),
                            _LabelSlider(label: 'Hemoglobin (hemo)', value: _hemo, max: 5, onChanged: (v) => setState(() => _hemo = v)),
                            _LabelSlider(label: 'Packed Cell Volume (pcv)', value: _pcv, max: 5, onChanged: (v) => setState(() => _pcv = v)),
                            _LabelSlider(label: 'RBC Count (rbcc)', value: _rbcc, max: 5, onChanged: (v) => setState(() => _rbcc = v)),
                            _LabelSlider(label: 'WBC Count (wbcc)', value: _wbcc, max: 5, onChanged: (v) => setState(() => _wbcc = v)),
                            _LabelSlider(label: 'Red Blood Cells (rbc)', value: _rbc, max: 3, onChanged: (v) => setState(() => _rbc = v)),
                          ],
                        ),
                        const SizedBox(height: 16),
                        _CKDSection(
                          icon: Icons.favorite_border,
                          label: 'VASCULAR & RISK',
                          title: 'Vascular & Risk',
                          subtitle: '',
                          isCritical: true,
                          chips: const [],
                          children: [
                            _LabelSlider(label: 'Diastolic BP (bp_diastolic)', value: _bpDiastolic, max: 5, onChanged: (v) => setState(() => _bpDiastolic = v), isCritical: true),
                            _LabelSlider(label: 'BP Limit (bp_limit)', value: _bpLimit, max: 5, onChanged: (v) => setState(() => _bpLimit = v), isCritical: true),
                            BinaryToggle(label: 'Hypertension (htn)', value: _htn, onChanged: (v) => setState(() => _htn = v)),
                            BinaryToggle(label: 'Diabetes Mellitus (dm)', value: _dm, onChanged: (v) => setState(() => _dm = v)),
                            BinaryToggle(label: 'Coronary Artery Disease (cad)', value: _cad, onChanged: (v) => setState(() => _cad = v)),
                            _LabelSlider(label: 'Sugar (su)', value: _su, max: 5, onChanged: (v) => setState(() => _su = v)),
                          ],
                        ),
                        const SizedBox(height: 16),
                        _CKDSection(
                          icon: Icons.person_outline,
                          label: 'LIFESTYLE',
                          title: 'Lifestyle & Patient Data',
                          subtitle: '',
                          chips: const [],
                          children: [
                            _LabelSlider(label: 'Age Group (age)', value: _ageEncoded, max: 5, onChanged: (v) => setState(() => _ageEncoded = v)),
                            BinaryToggle(label: 'Appetite (appet)', value: _appet, option0Label: 'Poor', option1Label: 'Good', onChanged: (v) => setState(() => _appet = v)),
                            BinaryToggle(label: 'Pedal Edema (pe)', value: _pe, onChanged: (v) => setState(() => _pe = v)),
                            BinaryToggle(label: 'Anemia (ane)', value: _ane, onChanged: (v) => setState(() => _ane = v)),
                            _LabelSlider(label: 'Pus Cell (pc)', value: _pc, max: 3, onChanged: (v) => setState(() => _pc = v)),
                            _LabelSlider(label: 'Pus Cell Clumps (pcc)', value: _pcc, max: 3, onChanged: (v) => setState(() => _pcc = v)),
                            _LabelSlider(label: 'Bacteria (ba)', value: _ba, max: 3, onChanged: (v) => setState(() => _ba = v)),
                            _LabelSlider(label: 'CKD Stage (stage)', value: _stage, max: 5, onChanged: (v) => setState(() => _stage = v)),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Container(
                          padding: const EdgeInsets.all(20),
                          decoration: BoxDecoration(
                            color: AppColors.surfaceContainerLowest,
                            borderRadius: BorderRadius.circular(24),
                          ),
                          child: Column(children: [
                            Text('TOTAL ASSESSMENT SCOPE', style: GoogleFonts.inter(fontSize: 10, fontWeight: FontWeight.w600, color: AppColors.onSurfaceMuted, letterSpacing: 1.2)),
                            const SizedBox(height: 4),
                            RichText(text: TextSpan(children: [
                              TextSpan(text: '27', style: GoogleFonts.plusJakartaSans(fontSize: 48, fontWeight: FontWeight.w800, color: AppColors.primary)),
                              TextSpan(text: '/27', style: GoogleFonts.plusJakartaSans(fontSize: 20, fontWeight: FontWeight.w500, color: AppColors.onSurfaceMuted)),
                            ])),
                            Text('Clinical factors successfully identified', style: Theme.of(context).textTheme.bodySmall),
                          ]),
                        ),
                        const SizedBox(height: 32),
                        GradientButton(
                          label: 'Start Assessment',
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
        if (_isLoading) const LoadingOverlay(label: 'CKD'),
      ],
    );
  }
}

// ─── CKD SECTION ─────────────────────────────────────────────────────────────

class _CKDSection extends StatefulWidget {
  final IconData icon;
  final String label, title, subtitle;
  final List<String> chips;
  final List<Widget> children;
  final bool isCritical;

  const _CKDSection({
    required this.icon, required this.label, required this.title,
    required this.subtitle, required this.chips, required this.children,
    this.isCritical = false,
  });

  @override
  State<_CKDSection> createState() => _CKDSectionState();
}

class _CKDSectionState extends State<_CKDSection> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final color = widget.isCritical ? AppColors.tertiary : AppColors.primary;
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [BoxShadow(color: AppColors.primary.withOpacity(0.05), blurRadius: 16, offset: const Offset(0, 4))],
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        GestureDetector(
          onTap: () => setState(() => _expanded = !_expanded),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Row(children: [
              Container(width: 44, height: 44, decoration: BoxDecoration(color: color.withOpacity(0.12), borderRadius: BorderRadius.circular(14)), child: Icon(widget.icon, color: color, size: 22)),
              const SizedBox(width: 14),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(widget.title, style: GoogleFonts.plusJakartaSans(fontSize: 16, fontWeight: FontWeight.w700, color: AppColors.onSurface)),
                if (widget.subtitle.isNotEmpty) Text(widget.subtitle, style: GoogleFonts.inter(fontSize: 11, color: AppColors.onSurfaceMuted)),
              ])),
              AnimatedRotation(turns: _expanded ? 0.5 : 0, duration: const Duration(milliseconds: 200), child: const Icon(Icons.keyboard_arrow_down, color: AppColors.onSurfaceMuted)),
            ]),
          ),
        ),
        if (widget.chips.isNotEmpty)
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 12),
            child: Column(children: widget.chips.map((chip) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: _FactorChip(label: chip, isCritical: widget.isCritical),
            )).toList()),
          ),
        if (_expanded)
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
            child: Column(children: widget.children.map((child) => Padding(padding: const EdgeInsets.only(bottom: 14), child: child)).toList()),
          ),
      ]),
    );
  }
}

class _FactorChip extends StatelessWidget {
  final String label;
  final bool isCritical;
  const _FactorChip({required this.label, this.isCritical = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 11),
      decoration: BoxDecoration(color: AppColors.surface, borderRadius: BorderRadius.circular(9999)),
      child: Row(children: [
        Container(width: 8, height: 8, decoration: BoxDecoration(color: isCritical ? AppColors.tertiary : AppColors.primary, shape: BoxShape.circle)),
        const SizedBox(width: 10),
        Expanded(child: Text(label, style: GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.w500, color: AppColors.onSurface))),
        if (isCritical)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(color: AppColors.tertiary, borderRadius: BorderRadius.circular(9999)),
            child: Text('CRITICAL', style: GoogleFonts.inter(fontSize: 9, fontWeight: FontWeight.w700, color: Colors.white, letterSpacing: 0.5)),
          ),
      ]),
    );
  }
}

class _LabelSlider extends StatelessWidget {
  final String label;
  final int value, max;
  final void Function(int) onChanged;
  final bool isCritical;

  const _LabelSlider({required this.label, required this.value, required this.max, required this.onChanged, this.isCritical = false});

  @override
  Widget build(BuildContext context) {
    final color = isCritical ? AppColors.tertiary : AppColors.primary;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        Expanded(child: Text(label, style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: AppColors.onSurfaceMuted))),
        Text('$value', style: GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.w700, color: color)),
      ]),
      const SizedBox(height: 6),
      SliderTheme(
        data: SliderTheme.of(context).copyWith(activeTrackColor: color, inactiveTrackColor: AppColors.surfaceContainer, thumbColor: color, overlayColor: color.withOpacity(0.1), trackHeight: 4),
        child: Slider(value: value.toDouble(), min: 0, max: max.toDouble(), divisions: max, onChanged: (v) => onChanged(v.round())),
      ),
    ]);
  }
}
