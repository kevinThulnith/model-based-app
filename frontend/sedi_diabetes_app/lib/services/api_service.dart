import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../models/diabetes_model.dart';

class ApiService {
  static const String _apiBaseUrl = String.fromEnvironment('API_BASE_URL');

  String get diabetesUrl {
    if (_apiBaseUrl.isNotEmpty) {
      final base = _apiBaseUrl.endsWith('/') ? _apiBaseUrl : '$_apiBaseUrl/';
      return '${base}api/diabetes/';
    }

    // Android emulator cannot reach host loopback via 127.0.0.1.
    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8000/api/diabetes/';
    }

    return 'http://127.0.0.1:8000/api/diabetes/';
  }

  Future<DiabetesResponse> predict(DiabetesRequest data) async {
    final response = await http.post(
      Uri.parse(diabetesUrl),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode(data.toJson()),
    );

    if (response.statusCode == 200) {
      return DiabetesResponse.fromJson(jsonDecode(response.body));
    }

    throw Exception(
      'Prediction failed (${response.statusCode}): ${response.body}',
    );
  }

  Future<Map<String, dynamic>> checkHealth() async {
    final response = await http.get(Uri.parse(diabetesUrl));

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }

    throw Exception(
      'Health check failed (${response.statusCode}): ${response.body}',
    );
  }
}