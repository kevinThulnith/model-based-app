import 'package:cloud_firestore/cloud_firestore.dart';
import '../models/assessment_result.dart';

class FirestoreService {
  static final FirestoreService _instance = FirestoreService._internal();
  factory FirestoreService() => _instance;
  FirestoreService._internal();

  final FirebaseFirestore _db = FirebaseFirestore.instance;

  CollectionReference<Map<String, dynamic>> _assessments(String userId) =>
      _db.collection('users').doc(userId).collection('assessments');

  Future<String> saveAssessment(AssessmentResult result) async {
    final docRef = await _assessments(result.userId).add(result.toMap());
    return docRef.id;
  }

  Stream<List<AssessmentResult>> watchAssessments(String userId) {
    return _assessments(userId)
        .orderBy('createdAt', descending: true)
        .limit(50)
        .snapshots()
        .map((snapshot) => snapshot.docs
            .map((doc) => AssessmentResult.fromMap(doc.data(), doc.id))
            .toList());
  }

  Future<List<AssessmentResult>> getRecentAssessments(
      String userId, {int limit = 5}) async {
    final snapshot = await _assessments(userId)
        .orderBy('createdAt', descending: true)
        .limit(limit)
        .get();
    return snapshot.docs
        .map((doc) => AssessmentResult.fromMap(doc.data(), doc.id))
        .toList();
  }

  Future<AssessmentResult?> getAssessment(
      String userId, String assessmentId) async {
    final doc = await _assessments(userId).doc(assessmentId).get();
    if (!doc.exists) return null;
    return AssessmentResult.fromMap(doc.data()!, doc.id);
  }

  Future<void> deleteAssessment(String userId, String assessmentId) async {
    await _assessments(userId).doc(assessmentId).delete();
  }

  Future<void> saveUserProfile(String userId, Map<String, dynamic> data) async {
    await _db
        .collection('users')
        .doc(userId)
        .set(data, SetOptions(merge: true));
  }

  Future<Map<String, dynamic>?> getUserProfile(String userId) async {
    final doc = await _db.collection('users').doc(userId).get();
    return doc.data();
  }
}
