import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

class LoadingScreen extends StatefulWidget {
  final String assessmentType;

  const LoadingScreen({super.key, required this.assessmentType});

  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen>
    with TickerProviderStateMixin {
  late AnimationController _rotationCtrl;
  int _currentStep = 0;

  final List<_ProcessStep> _steps = [
    _ProcessStep(
      icon: Icons.analytics_outlined,
      title: 'Data Ingestion',
      subtitle: 'Securely pulling insights from your connected wearables and medical records.',
    ),
    _ProcessStep(
      icon: Icons.hub_outlined,
      title: 'Pattern Recognition',
      subtitle: 'Identifying non-linear health trends across 42 different data variables.',
    ),
    _ProcessStep(
      icon: Icons.auto_awesome_outlined,
      title: 'Editorial Synthesis',
      subtitle: 'Transforming complex numbers into a beautiful, readable wellness narrative.',
    ),
    _ProcessStep(
      icon: Icons.verified_outlined,
      title: 'Result Verification',
      subtitle: 'Cross-referencing outputs with curated clinical research databases.',
    ),
  ];

  @override
  void initState() {
    super.initState();
    _rotationCtrl = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();

    _animateSteps();
  }

  void _animateSteps() async {
    for (int i = 0; i < _steps.length; i++) {
      await Future.delayed(const Duration(milliseconds: 700));
      if (mounted) setState(() => _currentStep = i);
    }
  }

  @override
  void dispose() {
    _rotationCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 24, 24, 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // App bar
              Row(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      gradient: AppColors.primaryGradient,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.spa_outlined,
                        color: Colors.white, size: 20),
                  ),
                  const SizedBox(width: 10),
                  Text(
                    'Editorial Wellness',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ],
              ),

              const SizedBox(height: 36),

              // Title
              Text(
                'Curating Your\nWellness Profile',
                style: Theme.of(context).textTheme.headlineLarge,
              ).animate().fadeIn(duration: 400.ms).slideY(begin: 0.2, end: 0),

              const SizedBox(height: 36),

              // Animated circle
              Center(
                child: AnimatedBuilder(
                  animation: _rotationCtrl,
                  builder: (context, child) {
                    return Stack(
                      alignment: Alignment.center,
                      children: [
                        // Outer rotating ring
                        Transform.rotate(
                          angle: _rotationCtrl.value * 6.28,
                          child: Container(
                            width: 120,
                            height: 120,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(
                                color: AppColors.surfaceContainer,
                                width: 2,
                              ),
                            ),
                            child: Align(
                              alignment: Alignment.topCenter,
                              child: Container(
                                width: 10,
                                height: 10,
                                decoration: const BoxDecoration(
                                  color: AppColors.primary,
                                  shape: BoxShape.circle,
                                ),
                              ),
                            ),
                          ),
                        ),
                        // Inner circle
                        Container(
                          width: 90,
                          height: 90,
                          decoration: BoxDecoration(
                            color: AppColors.surfaceContainerLowest,
                            shape: BoxShape.circle,
                            boxShadow: [
                              BoxShadow(
                                color: AppColors.primary.withOpacity(0.1),
                                blurRadius: 20,
                              ),
                            ],
                          ),
                          child: Icon(
                            _steps[_currentStep].icon,
                            color: AppColors.primary,
                            size: 36,
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ).animate().fadeIn(delay: 200.ms, duration: 400.ms),

              const SizedBox(height: 28),

              // Processing sequence text
              Center(
                child: Column(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 6),
                      decoration: BoxDecoration(
                        color: AppColors.secondaryContainer,
                        borderRadius: BorderRadius.circular(9999),
                      ),
                      child: Text(
                        'PROCESSING SEQUENCE 0${_currentStep + 1}',
                        style: GoogleFonts.inter(
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          color: AppColors.primary,
                          letterSpacing: 0.8,
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    AnimatedSwitcher(
                      duration: const Duration(milliseconds: 400),
                      child: Text(
                        '${_steps[_currentStep].title}...',
                        key: ValueKey(_currentStep),
                        style: Theme.of(context).textTheme.headlineSmall,
                        textAlign: TextAlign.center,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Our AI engine is currently aligning your historical biomarkers with curated editorial research.',
                      style: Theme.of(context).textTheme.bodyMedium,
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 36),

              // Step cards
              Expanded(
                child: ListView.separated(
                  itemCount: _steps.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 12),
                  itemBuilder: (context, i) {
                    final isActive = i == _currentStep;
                    final isDone = i < _currentStep;

                    return AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      padding: const EdgeInsets.all(18),
                      decoration: BoxDecoration(
                        color: isActive
                            ? AppColors.primary
                            : AppColors.surfaceContainerLowest,
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: isActive
                            ? [
                                BoxShadow(
                                  color: AppColors.primary.withOpacity(0.3),
                                  blurRadius: 20,
                                  offset: const Offset(0, 8),
                                ),
                              ]
                            : [],
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 40,
                            height: 40,
                            decoration: BoxDecoration(
                              color: isActive
                                  ? Colors.white.withOpacity(0.2)
                                  : AppColors.surfaceContainerHigh,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Icon(
                              isDone
                                  ? Icons.check
                                  : _steps[i].icon,
                              color: isActive
                                  ? Colors.white
                                  : isDone
                                      ? AppColors.primary
                                      : AppColors.onSurfaceMuted,
                              size: 20,
                            ),
                          ),
                          const SizedBox(width: 14),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  _steps[i].title,
                                  style: GoogleFonts.plusJakartaSans(
                                    fontSize: 15,
                                    fontWeight: FontWeight.w700,
                                    color: isActive
                                        ? Colors.white
                                        : AppColors.onSurface,
                                  ),
                                ),
                                Text(
                                  _steps[i].subtitle,
                                  style: GoogleFonts.inter(
                                    fontSize: 12,
                                    color: isActive
                                        ? Colors.white.withOpacity(0.8)
                                        : AppColors.onSurfaceMuted,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    )
                        .animate(delay: (i * 100).ms)
                        .fadeIn(duration: 350.ms)
                        .slideY(begin: 0.1, end: 0);
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ProcessStep {
  final IconData icon;
  final String title;
  final String subtitle;

  const _ProcessStep({
    required this.icon,
    required this.title,
    required this.subtitle,
  });
}
