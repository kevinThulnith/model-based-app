import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  // Replace with your deployed Django backend URL
  static const String _baseUrl = 'http://10.0.2.2:8000';

  Future<Map<String, dynamic>> predictDiabetes(
      Map<String, dynamic> data) async {
    return await _post('/api/diabetes/', data);
  }

  Future<Map<String, dynamic>> predictHeartDisease(
      Map<String, dynamic> data) async {
    return await _post('/api/heart-disease/', data);
  }

  Future<Map<String, dynamic>> predictCKD(
      Map<String, dynamic> data) async {
    return await _post('/api/chronic-kidney-disease/', data);
  }

  Future<Map<String, dynamic>> _post(
      String path, Map<String, dynamic> body) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl$path'),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        throw ApiException(
          statusCode: response.statusCode,
          message: 'Server error: ${response.statusCode}',
        );
      }
    } on http.ClientException catch (e) {
      throw ApiException(message: 'Network error: ${e.message}');
    } catch (e) {
      rethrow;
    }
  }
}

class ApiException implements Exception {
  final int? statusCode;
  final String message;
  ApiException({this.statusCode, required this.message});

  @override
  String toString() => message;
}
