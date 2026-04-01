import 'package:flutter/material.dart';
import '../services/auth_service.dart';
import '../services/firestore_service.dart';
import '../theme/app_theme.dart';
import '../widgets/common_widgets.dart';

class AccountDetailsScreen extends StatefulWidget {
  const AccountDetailsScreen({super.key});

  @override
  State<AccountDetailsScreen> createState() => _AccountDetailsScreenState();
}

class _AccountDetailsScreenState extends State<AccountDetailsScreen> {
  final _formKey = GlobalKey<FormState>();
  final _fullNameController = TextEditingController();
  final _emailController = TextEditingController();
  final _phoneController = TextEditingController();
  final _specializationController = TextEditingController();
  final _institutionController = TextEditingController();

  bool _loading = true;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _loadProfile();
  }

  @override
  void dispose() {
    _fullNameController.dispose();
    _emailController.dispose();
    _phoneController.dispose();
    _specializationController.dispose();
    _institutionController.dispose();
    super.dispose();
  }

  Future<void> _loadProfile() async {
    final auth = AuthService();
    final user = auth.currentUser;

    _fullNameController.text = auth.userDisplayName;
    _emailController.text = auth.userEmail;

    if (user == null) {
      if (mounted) {
        setState(() => _loading = false);
      }
      return;
    }

    final profile = await FirestoreService().getUserProfile(user.uid);

    if (!mounted) return;

    _phoneController.text = (profile?['phone'] ?? '').toString();
    _specializationController.text = (profile?['specialization'] ?? '').toString();
    _institutionController.text = (profile?['institution'] ?? '').toString();

    setState(() => _loading = false);
  }

  Future<void> _saveProfile() async {
    if (!_formKey.currentState!.validate()) return;

    final user = AuthService().currentUser;
    if (user == null) return;

    setState(() => _saving = true);

    await FirestoreService().saveUserProfile(user.uid, {
      'displayName': _fullNameController.text.trim(),
      'email': _emailController.text.trim(),
      'phone': _phoneController.text.trim(),
      'specialization': _specializationController.text.trim(),
      'institution': _institutionController.text.trim(),
      'updatedAt': DateTime.now(),
    });

    if (!mounted) return;

    setState(() => _saving = false);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Account details updated.')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(title: const Text('Account Details')),
      body: _loading
          ? const Center(
              child: CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(AppColors.primary),
              ),
            )
          : SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
              child: Form(
                key: _formKey,
                child: Column(
                  children: [
                    PillInputField(
                      label: 'Full Name',
                      hint: 'Enter your full name',
                      keyboardType: TextInputType.name,
                      controller: _fullNameController,
                      validator: (v) => (v == null || v.trim().isEmpty) ? 'Name is required' : null,
                    ),
                    const SizedBox(height: 14),
                    PillInputField(
                      label: 'Email',
                      hint: 'Enter your email address',
                      keyboardType: TextInputType.emailAddress,
                      controller: _emailController,
                      validator: (v) => (v == null || !v.contains('@')) ? 'Enter a valid email' : null,
                    ),
                    const SizedBox(height: 14),
                    PillInputField(
                      label: 'Phone',
                      hint: 'Enter your phone number',
                      keyboardType: TextInputType.phone,
                      controller: _phoneController,
                    ),
                    const SizedBox(height: 14),
                    PillInputField(
                      label: 'Specialization',
                      hint: 'e.g. Endocrinology',
                      keyboardType: TextInputType.text,
                      controller: _specializationController,
                    ),
                    const SizedBox(height: 14),
                    PillInputField(
                      label: 'Institution',
                      hint: 'Hospital or clinic name',
                      keyboardType: TextInputType.text,
                      controller: _institutionController,
                    ),
                    const SizedBox(height: 24),
                    GradientButton(
                      label: _saving ? 'Saving...' : 'Save Changes',
                      isLoading: _saving,
                      onPressed: _saving ? null : _saveProfile,
                    ),
                  ],
                ),
              ),
            ),
    );
  }
}
