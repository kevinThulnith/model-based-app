import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/assessment_result.dart';
import '../theme/app_theme.dart';
import '../widgets/common_widgets.dart';
import 'home_screen.dart';

class ResultScreen extends StatelessWidget {
  final AssessmentResult result;

  const ResultScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final isPositive = result.isPositive;
    final riskColor = isPositive ? AppColors.tertiary : AppColors.primary;
    final confidenceInt = result.confidence.toStringAsFixed(1);

    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: Column(
          children: [
            // Header
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 20, 24, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      GestureDetector(
                        onTap: () => Navigator.of(context).pushAndRemoveUntil(
                          MaterialPageRoute(
                              builder: (_) => const HomeScreen()),
                          (route) => false,
                        ),
                        child: Container(
                          width: 40,
                          height: 40,
                          decoration: BoxDecoration(
                            color: AppColors.surfaceContainerLowest,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Icon(Icons.home_outlined,
                              color: AppColors.onSurface, size: 20),
                        ),
                      ),
                      const Spacer(),
                      // ID badge
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 5),
                        decoration: BoxDecoration(
                          color: AppColors.surfaceContainerLowest,
                          borderRadius: BorderRadius.circular(9999),
                        ),
                        child: Row(
                          children: [
                            Container(
                              width: 6,
                              height: 6,
                              decoration: const BoxDecoration(
                                color: AppColors.primaryContainer,
                                shape: BoxShape.circle,
                              ),
                            ),
                            const SizedBox(width: 6),
                            Text(
                              'ID: #EW-${DateTime.now().millisecond.toString().padLeft(5, '0')}',
                              style: GoogleFonts.inter(
                                fontSize: 11,
                                fontWeight: FontWeight.w600,
                                color: AppColors.onSurfaceMuted,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ).animate().fadeIn(duration: 300.ms),

                  const SizedBox(height: 12),

                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 5),
                    decoration: BoxDecoration(
                      color: AppColors.secondaryContainer,
                      borderRadius: BorderRadius.circular(9999),
                    ),
                    child: Text(
                      'CLINICAL ANALYSIS',
                      style: GoogleFonts.inter(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: AppColors.primary,
                        letterSpacing: 0.8,
                      ),
                    ),
                  ).animate(delay: 100.ms).fadeIn(duration: 300.ms),

                  const SizedBox(height: 8),

                  Text(
                    'Final Risk Report',
                    style: Theme.of(context).textTheme.headlineLarge,
                  ).animate(delay: 150.ms).fadeIn(duration: 300.ms).slideY(begin: 0.2, end: 0),
                ],
              ),
            ),

            // Scrollable content
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
                children: [
                  // Composite risk card
                  CapsuleCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'COMPOSITE HEALTH RISK',
                          style: GoogleFonts.inter(
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                            color: AppColors.onSurfaceMuted,
                            letterSpacing: 1.2,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Text(
                              confidenceInt,
                              style: GoogleFonts.plusJakartaSans(
                                fontSize: 72,
                                fontWeight: FontWeight.w800,
                                color: riskColor,
                                height: 1,
                              ),
                            ),
                            Padding(
                              padding: const EdgeInsets.only(bottom: 10),
                              child: Text(
                                '%',
                                style: GoogleFonts.plusJakartaSans(
                                  fontSize: 28,
                                  fontWeight: FontWeight.w600,
                                  color: AppColors.primaryContainer,
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 8),
                          decoration: BoxDecoration(
                            color: isPositive
                                ? AppColors.tertiaryContainer
                                : AppColors.secondaryContainer,
                            borderRadius: BorderRadius.circular(9999),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Container(
                                width: 8,
                                height: 8,
                                decoration: BoxDecoration(
                                  color: riskColor,
                                  shape: BoxShape.circle,
                                ),
                              ),
                              const SizedBox(width: 8),
                              Text(
                                isPositive
                                    ? '${result.label} — Elevated Risk Zone'
                                    : 'Optimal Range Zone',
                                style: GoogleFonts.inter(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600,
                                  color: riskColor,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ).animate(delay: 200.ms).fadeIn(duration: 400.ms),

                  const SizedBox(height: 16),

                  // Bidirectional factor drivers
                  CapsuleCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Bidirectional Factor Drivers',
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                        const SizedBox(height: 20),
                        _FactorBar(
                          label: 'Cardiovascular Resilience',
                          value: 0.72,
                          change: '+12% Gain',
                          isPositive: true,
                        ),
                        const SizedBox(height: 14),
                        _FactorBar(
                          label: 'Metabolic Load',
                          value: 0.38,
                          change: '-8% Deficit',
                          isPositive: false,
                        ),
                        const SizedBox(height: 14),
                        _FactorBar(
                          label: 'Sleep Architecture',
                          value: 0.82,
                          change: '+18% Gain',
                          isPositive: true,
                        ),
                      ],
                    ),
                  ).animate(delay: 300.ms).fadeIn(duration: 400.ms),

                  const SizedBox(height: 16),

                  // Probability breakdown
                  CapsuleCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Probability Breakdown',
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                        const SizedBox(height: 16),
                        ...result.probability.entries.map((e) {
                          final pct =
                              ((e.value as num).toDouble() * 100)
                                  .toStringAsFixed(1);
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 12),
                            child: _ProbabilityBar(
                              label: e.key
                                  .replaceAll('_', ' ')
                                  .toUpperCase(),
                              value: (e.value as num).toDouble(),
                              percentage: '$pct%',
                            ),
                          );
                        }),
                      ],
                    ),
                  ).animate(delay: 400.ms).fadeIn(duration: 400.ms),

                  const SizedBox(height: 16),

                  // Insight cards
                  _InsightCard(
                    icon: Icons.auto_awesome_outlined,
                    title: 'Editorial Insight',
                    body: isPositive
                        ? 'Client shows elevated risk markers. Immediate lifestyle intervention and clinical follow-up are recommended for comprehensive risk mitigation.'
                        : 'Client shows high resilience in parasympathetic recovery but moderate inflammatory markers post-exercise.',
                  ).animate(delay: 500.ms).fadeIn(duration: 400.ms),

                  const SizedBox(height: 12),

                  _InsightCard(
                    icon: Icons.person_outline,
                    title: 'Clinical Outlook',
                    body:
                        'Recommend increasing magnesium glycinate intake by 200mg to optimize nightly restorative cycles. Consider scheduling follow-up in 6-8 weeks.',
                  ).animate(delay: 560.ms).fadeIn(duration: 400.ms),

                  const SizedBox(height: 12),

                  _InsightCard(
                    icon: Icons.verified_outlined,
                    title: 'Biometric Lock',
                    body:
                        'Data streams verified from integrated health platforms. Assessment period: ${_formatDate(DateTime.now().subtract(const Duration(days: 18)))} - ${_formatDate(DateTime.now())}.',
                    iconColor: AppColors.tertiary,
                  ).animate(delay: 620.ms).fadeIn(duration: 400.ms),

                  const SizedBox(height: 28),

                  // CTAs
                  GradientButton(
                    label: 'Save to EHR',
                    icon: Icons.save_outlined,
                    onPressed: () {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: const Text('Saved to records.'),
                          backgroundColor: AppColors.primary,
                          behavior: SnackBarBehavior.floating,
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16)),
                        ),
                      );
                    },
                  ).animate(delay: 700.ms).fadeIn(duration: 400.ms),

                  const SizedBox(height: 12),

                  PillButton(
                    label: 'New Assessment',
                    icon: Icons.add_circle_outline,
                    backgroundColor: AppColors.secondaryContainer,
                    textColor: AppColors.primary,
                    onPressed: () => Navigator.of(context).pushAndRemoveUntil(
                      MaterialPageRoute(
                          builder: (_) => const HomeScreen()),
                      (route) => false,
                    ),
                  ).animate(delay: 760.ms).fadeIn(duration: 400.ms),

                  const SizedBox(height: 16),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _formatDate(DateTime dt) {
    return '${dt.day.toString().padLeft(2, '0')}/${dt.month.toString().padLeft(2, '0')}';
  }
}

// ─── FACTOR BAR ──────────────────────────────────────────────────────────────

class _FactorBar extends StatelessWidget {
  final String label;
  final double value;
  final String change;
  final bool isPositive;

  const _FactorBar({
    required this.label,
    required this.value,
    required this.change,
    required this.isPositive,
  });

  @override
  Widget build(BuildContext context) {
    final color =
        isPositive ? AppColors.primary : AppColors.tertiary;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                label,
                style: GoogleFonts.inter(
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                  color: AppColors.onSurface,
                ),
              ),
            ),
            Text(
              change,
              style: GoogleFonts.inter(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Container(
          height: 10,
          width: double.infinity,
          decoration: BoxDecoration(
            color: AppColors.surfaceContainerHigh,
            borderRadius: BorderRadius.circular(9999),
          ),
          child: FractionallySizedBox(
            alignment: Alignment.centerLeft,
            widthFactor: value,
            child: Container(
              decoration: BoxDecoration(
                color: color,
                borderRadius: BorderRadius.circular(9999),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ─── PROBABILITY BAR ─────────────────────────────────────────────────────────

class _ProbabilityBar extends StatelessWidget {
  final String label;
  final double value;
  final String percentage;

  const _ProbabilityBar({
    required this.label,
    required this.value,
    required this.percentage,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        SizedBox(
          width: 140,
          child: Text(
            label,
            style: GoogleFonts.inter(
              fontSize: 11,
              fontWeight: FontWeight.w500,
              color: AppColors.onSurfaceMuted,
            ),
          ),
        ),
        Expanded(
          child: Container(
            height: 8,
            decoration: BoxDecoration(
              color: AppColors.surfaceContainerHigh,
              borderRadius: BorderRadius.circular(9999),
            ),
            child: FractionallySizedBox(
              alignment: Alignment.centerLeft,
              widthFactor: value,
              child: Container(
                decoration: BoxDecoration(
                  gradient: AppColors.primaryGradient,
                  borderRadius: BorderRadius.circular(9999),
                ),
              ),
            ),
          ),
        ),
        const SizedBox(width: 10),
        Text(
          percentage,
          style: GoogleFonts.inter(
            fontSize: 12,
            fontWeight: FontWeight.w700,
            color: AppColors.primary,
          ),
        ),
      ],
    );
  }
}

// ─── INSIGHT CARD ────────────────────────────────────────────────────────────

class _InsightCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String body;
  final Color iconColor;

  const _InsightCard({
    required this.icon,
    required this.title,
    required this.body,
    this.iconColor = AppColors.primary,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withOpacity(0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: iconColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: iconColor, size: 18),
          ),
          const SizedBox(height: 12),
          Text(
            title,
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 6),
          Text(
            body,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ],
      ),
    );
  }
}
