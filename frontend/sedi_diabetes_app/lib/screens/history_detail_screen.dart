import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/assessment_result.dart';
import '../theme/app_theme.dart';

class HistoryDetailScreen extends StatelessWidget {
  final AssessmentResult result;

  const HistoryDetailScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final bool isPositive = result.isPositive;
    final Color accent = isPositive ? AppColors.tertiary : AppColors.primary;

    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(
        title: Text(result.typeDisplayName),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(20, 12, 20, 24),
        children: [
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: AppColors.surfaceContainerLowest,
              borderRadius: BorderRadius.circular(22),
              boxShadow: [
                BoxShadow(
                  color: AppColors.primary.withValues(alpha: 0.05),
                  blurRadius: 14,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  result.label.toUpperCase(),
                  style: GoogleFonts.inter(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.1,
                    color: accent,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  '${result.confidence.toStringAsFixed(1)}% confidence',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 6),
                Text(
                  _formatDate(result.createdAt),
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          _SectionCard(
            title: 'Prediction Summary',
            child: Column(
              children: [
                _KeyValueRow(label: 'Assessment Type', value: result.typeDisplayName),
                const SizedBox(height: 10),
                _KeyValueRow(label: 'Prediction', value: result.label),
                const SizedBox(height: 10),
                _KeyValueRow(label: 'Confidence', value: '${result.confidence.toStringAsFixed(1)}%'),
              ],
            ),
          ),
          const SizedBox(height: 14),
          _SectionCard(
            title: 'Probabilities',
            child: result.probability.isEmpty
                ? Text('No probability breakdown found.', style: Theme.of(context).textTheme.bodyMedium)
                : Column(
                    children: result.probability.entries
                        .map((e) => Padding(
                              padding: const EdgeInsets.only(bottom: 10),
                              child: _KeyValueRow(
                                label: _humanizeKey(e.key),
                                value: _formatValue(e.value),
                              ),
                            ))
                        .toList(),
                  ),
          ),
          const SizedBox(height: 14),
          _SectionCard(
            title: 'Submitted Inputs',
            child: result.inputs.isEmpty
                ? Text('No input details saved.', style: Theme.of(context).textTheme.bodyMedium)
                : Column(
                    children: result.inputs.entries
                        .map((e) => Padding(
                              padding: const EdgeInsets.only(bottom: 10),
                              child: _KeyValueRow(
                                label: _humanizeKey(e.key),
                                value: _formatValue(e.value),
                              ),
                            ))
                        .toList(),
                  ),
          ),
        ],
      ),
    );
  }

  String _formatDate(DateTime dt) {
    final months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    return '${months[dt.month - 1]} ${dt.day}, ${dt.year} at ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }

  String _humanizeKey(String key) {
    return key
        .replaceAll('_', ' ')
        .split(' ')
        .where((part) => part.isNotEmpty)
        .map((part) => part[0].toUpperCase() + part.substring(1))
        .join(' ');
  }

  String _formatValue(dynamic value) {
    if (value is double) {
      return value.toStringAsFixed(3);
    }
    return value.toString();
  }
}

class _SectionCard extends StatelessWidget {
  final String title;
  final Widget child;

  const _SectionCard({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }
}

class _KeyValueRow extends StatelessWidget {
  final String label;
  final String value;

  const _KeyValueRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Text(
            label,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            value,
            textAlign: TextAlign.right,
            style: Theme.of(context).textTheme.titleSmall,
          ),
        ),
      ],
    );
  }
}
