import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/auth_service.dart';
import '../services/firestore_service.dart';
import '../models/assessment_result.dart';
import '../theme/app_theme.dart';
import '../widgets/common_widgets.dart';
import 'diabetes_assessment_screen.dart';
import 'heart_disease_assessment_screen.dart';
import 'ckd_assessment_screen.dart';
import 'history_screen.dart';
import 'profile_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _auth = AuthService();
  final _firestore = FirestoreService();
  int _selectedTab = 0;

  List<Widget> get _tabs => [
        _HomeTab(
          auth: _auth,
          firestore: _firestore,
        ),
        HistoryScreen(isEmbedded: true),
        ProfileScreen(isEmbedded: true),
      ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      body: IndexedStack(
        index: _selectedTab,
        children: _tabs,
      ),
      bottomNavigationBar: _BottomNav(
        selectedIndex: _selectedTab,
        onItemTapped: (i) => setState(() => _selectedTab = i),
      ),
    );
  }
}

// ─── HOME TAB ────────────────────────────────────────────────────────────────

class _HomeTab extends StatelessWidget {
  final AuthService auth;
  final FirestoreService firestore;

  const _HomeTab({required this.auth, required this.firestore});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(24, 24, 24, 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // App bar row
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
                const Spacer(),
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: AppColors.surfaceContainerLowest,
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [
                      BoxShadow(
                        color: AppColors.primary.withOpacity(0.06),
                        blurRadius: 10,
                      ),
                    ],
                  ),
                  child: const Icon(Icons.notifications_outlined,
                      color: AppColors.onSurface, size: 20),
                ),
              ],
            ).animate().fadeIn(duration: 400.ms),

            const SizedBox(height: 32),

            // Welcome heading
            RichText(
              text: TextSpan(
                children: [
                  TextSpan(
                    text: 'Welcome, ',
                    style: Theme.of(context).textTheme.headlineLarge,
                  ),
                  TextSpan(
                    text: auth.userDisplayName.split(' ').first,
                    style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                          color: AppColors.primary,
                        ),
                  ),
                ],
              ),
            ).animate(delay: 100.ms).fadeIn(duration: 400.ms),

            const SizedBox(height: 6),

            Text(
              'Predictive diagnostics powered by SEDI AI.',
              style: Theme.of(context).textTheme.bodyMedium,
            ).animate(delay: 150.ms).fadeIn(duration: 400.ms),

            const SizedBox(height: 32),

            // Predictive Models heading
            Row(
              children: [
                Text(
                  'Predictive Models',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 6),
                  decoration: BoxDecoration(
                    color: AppColors.secondaryContainer,
                    borderRadius: BorderRadius.circular(9999),
                  ),
                  child: Text(
                    '3 Active Engines',
                    style: GoogleFonts.inter(
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      color: AppColors.primary,
                    ),
                  ),
                ),
              ],
            ).animate(delay: 200.ms).fadeIn(duration: 400.ms),

            const SizedBox(height: 16),

            // Model cards
            _ModelCard(
              title: 'Diabetes',
              subtitle:
                  'Real-time glucose pattern analysis and risk stratification.',
              icon: Icons.back_hand_outlined,
              gradient: AppColors.cardGradient,
              buttonLabel: 'Start Assessment',
              onPressed: () => Navigator.push(
                context,
                _slideRoute(const DiabetesAssessmentScreen()),
              ),
              delay: 300,
            ),

            const SizedBox(height: 14),

            _ModelCard(
              title: 'Heart Disease',
              subtitle:
                  'Predictive CV risk scoring and arrhythmia detection.',
              icon: Icons.favorite_border,
              gradient: AppColors.heartGradient,
              buttonLabel: 'Launch Suite',
              onPressed: () => Navigator.push(
                context,
                _slideRoute(const HeartDiseaseAssessmentScreen()),
              ),
              delay: 400,
            ),

            const SizedBox(height: 14),

            _ModelCard(
              title: 'CKD',
              subtitle:
                  'Early chronic kidney disease stage identification.',
              icon: Icons.water_drop_outlined,
              gradient: AppColors.ckdGradient,
              buttonLabel: 'Screen Now',
              onPressed: () => Navigator.push(
                context,
                _slideRoute(const CKDAssessmentScreen()),
              ),
              delay: 500,
            ),

            const SizedBox(height: 36),

            // Recent assessments
            Row(
              children: [
                Text(
                  'Recent Assessments',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const Spacer(),
                Text(
                  'View All',
                  style: Theme.of(context).textTheme.labelLarge,
                ),
              ],
            ).animate(delay: 600.ms).fadeIn(duration: 400.ms),

            const SizedBox(height: 14),

            StreamBuilder<List<AssessmentResult>>(
              stream: firestore.watchAssessments(
                  AuthService().currentUser?.uid ?? ''),
              builder: (context, snapshot) {
                if (!snapshot.hasData || snapshot.data!.isEmpty) {
                  return _EmptyHistory();
                }
                final recent = snapshot.data!.take(3).toList();
                return Column(
                  children: recent
                      .asMap()
                      .entries
                      .map((e) => Padding(
                            padding: const EdgeInsets.only(bottom: 10),
                            child: _RecentAssessmentTile(
                              result: e.value,
                            )
                                .animate(delay: (650 + e.key * 80).ms)
                                .fadeIn(duration: 350.ms)
                                .slideX(begin: 0.1, end: 0),
                          ))
                      .toList(),
                );
              },
            ),

            const SizedBox(height: 36),

            // Vitality pulse section
            Text(
              'Vitality Pulse',
              style: Theme.of(context).textTheme.titleLarge,
            ).animate(delay: 850.ms).fadeIn(duration: 400.ms),

            const SizedBox(height: 14),

            _VitalityPulseCard()
                .animate(delay: 900.ms)
                .fadeIn(duration: 400.ms),
          ],
        ),
      ),
    );
  }

  PageRouteBuilder _slideRoute(Widget page) {
    return PageRouteBuilder(
      pageBuilder: (_, __, ___) => page,
      transitionsBuilder: (_, animation, __, child) {
        return SlideTransition(
          position:
              Tween<Offset>(begin: const Offset(1, 0), end: Offset.zero)
                  .animate(CurvedAnimation(
                      parent: animation, curve: Curves.easeOutCubic)),
          child: child,
        );
      },
      transitionDuration: const Duration(milliseconds: 400),
    );
  }
}

// ─── MODEL CARD ──────────────────────────────────────────────────────────────

class _ModelCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final LinearGradient gradient;
  final String buttonLabel;
  final VoidCallback onPressed;
  final int delay;

  const _ModelCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.gradient,
    required this.buttonLabel,
    required this.onPressed,
    required this.delay,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: gradient,
        borderRadius: BorderRadius.circular(28),
        boxShadow: [
          BoxShadow(
            color: gradient.colors.first.withOpacity(0.3),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(icon, color: Colors.white, size: 22),
          ),
          const SizedBox(height: 16),
          Text(
            title,
            style: GoogleFonts.plusJakartaSans(
              fontSize: 24,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            subtitle,
            style: GoogleFonts.inter(
              fontSize: 13,
              color: Colors.white.withOpacity(0.8),
              fontWeight: FontWeight.w400,
            ),
          ),
          const SizedBox(height: 20),
          GestureDetector(
            onTap: onPressed,
            child: Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 22, vertical: 12),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(9999),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    buttonLabel,
                    style: GoogleFonts.inter(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: AppColors.onSurface,
                    ),
                  ),
                  const SizedBox(width: 6),
                  const Icon(Icons.arrow_forward,
                      size: 16, color: AppColors.onSurface),
                ],
              ),
            ),
          ),
        ],
      ),
    ).animate(delay: Duration(milliseconds: delay)).fadeIn(duration: 400.ms).slideY(begin: 0.15, end: 0);
  }
}

// ─── RECENT ASSESSMENT TILE ──────────────────────────────────────────────────

class _RecentAssessmentTile extends StatelessWidget {
  final AssessmentResult result;

  const _RecentAssessmentTile({required this.result});

  @override
  Widget build(BuildContext context) {
    final color = result.isPositive ? AppColors.tertiary : AppColors.primary;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withOpacity(0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
            child: Icon(Icons.person_outline, color: color, size: 22),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  result.typeDisplayName,
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const SizedBox(height: 2),
                Text(
                  '${result.label.toUpperCase()} • ${_timeAgo(result.createdAt)}',
                  style: GoogleFonts.inter(
                    fontSize: 10,
                    fontWeight: FontWeight.w500,
                    color: AppColors.onSurfaceMuted,
                    letterSpacing: 0.5,
                  ),
                ),
              ],
            ),
          ),
          const Icon(Icons.chevron_right,
              color: AppColors.onSurfaceMuted, size: 20),
        ],
      ),
    );
  }

  String _timeAgo(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inMinutes < 60) return '${diff.inMinutes}M AGO';
    if (diff.inHours < 24) return '${diff.inHours}H AGO';
    if (diff.inDays == 1) return 'YESTERDAY';
    return '${diff.inDays}D AGO';
  }
}

class _EmptyHistory extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(28),
      decoration: BoxDecoration(
        color: AppColors.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        children: [
          Icon(Icons.history_outlined,
              size: 36, color: AppColors.onSurfaceMuted.withOpacity(0.5)),
          const SizedBox(height: 10),
          Text(
            'No assessments yet.',
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          const SizedBox(height: 4),
          Text(
            'Run your first predictive scan above.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    );
  }
}

// ─── VITALITY PULSE CARD ─────────────────────────────────────────────────────

class _VitalityPulseCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: AppColors.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(28),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withOpacity(0.06),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'SYSTEM LOAD',
            style: GoogleFonts.inter(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              letterSpacing: 1.5,
              color: AppColors.onSurfaceMuted,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Optimal',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  color: AppColors.primary,
                ),
          ),
          const SizedBox(height: 20),
          // Bar chart
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [40.0, 60.0, 55.0, 75.0, 65.0, 80.0, 55.0, 45.0]
                .asMap()
                .entries
                .map((e) => Expanded(
                      child: Padding(
                        padding:
                            const EdgeInsets.symmetric(horizontal: 2),
                        child: Container(
                          height: e.value,
                          decoration: BoxDecoration(
                            color: e.key == 5
                                ? AppColors.primary
                                : AppColors.surfaceContainer,
                            borderRadius: BorderRadius.circular(6),
                          ),
                        ),
                      ),
                    ))
                .toList(),
          ),
          const SizedBox(height: 18),
          Row(
            children: [
              Text(
                'Assessments Today',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const Spacer(),
              Text(
                '124',
                style: Theme.of(context).textTheme.titleSmall,
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Text(
                'Cloud Sync',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const Spacer(),
              Container(
                width: 8,
                height: 8,
                decoration: const BoxDecoration(
                  color: Color(0xFF22C55E),
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 6),
              Text(
                'Live',
                style: GoogleFonts.inter(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: AppColors.onSurface,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ─── BOTTOM NAV ──────────────────────────────────────────────────────────────

class _BottomNav extends StatelessWidget {
  final int selectedIndex;
  final void Function(int) onItemTapped;

  const _BottomNav({
    required this.selectedIndex,
    required this.onItemTapped,
  });

  @override
  Widget build(BuildContext context) {
    final items = [
      (Icons.home_outlined, Icons.home, 'Home'),
      (Icons.history_outlined, Icons.history, 'History'),
      (Icons.person_outline, Icons.person, 'Profile'),
    ];

    final bottomPadding = MediaQuery.of(context).padding.bottom;

    return Container(
      decoration: BoxDecoration(
        color: AppColors.surfaceContainerLowest,
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withOpacity(0.08),
            blurRadius: 20,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: Padding(
        padding: EdgeInsets.only(
          top: 10,
          bottom: bottomPadding > 0 ? bottomPadding : 12,
        ),
        child: Row(
          children: items.asMap().entries.map((entry) {
            final i = entry.key;
            final item = entry.value;
            final isSelected = selectedIndex == i;

            return Expanded(
              child: GestureDetector(
                onTap: () => onItemTapped(i),
                behavior: HitTestBehavior.opaque,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 250),
                      width: isSelected ? 52 : 40,
                      height: 34,
                      decoration: BoxDecoration(
                        color: isSelected
                            ? AppColors.primary
                            : Colors.transparent,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Icon(
                        isSelected ? item.$2 : item.$1,
                        color: isSelected
                            ? Colors.white
                            : AppColors.onSurfaceMuted,
                        size: 20,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      item.$3,
                      style: GoogleFonts.inter(
                        fontSize: 10,
                        fontWeight: isSelected
                            ? FontWeight.w600
                            : FontWeight.w400,
                        color: isSelected
                            ? AppColors.primary
                            : AppColors.onSurfaceMuted,
                      ),
                    ),
                  ],
                ),
              ),
            );
          }).toList(),
        ),
      ),
    );
  }
}
