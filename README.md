# SEDI - Sedentary Editorial Wellness Predictive Diagnostics

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.13+-green.svg)
![Flutter](https://img.shields.io/badge/Flutter-3.0+-blue.svg)
![Django](https://img.shields.io/badge/Django-6.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A comprehensive health prediction mobile application powered by machine learning models for diabetes, heart disease, and chronic kidney disease risk assessment.

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [API Documentation](#api-documentation)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

SEDI is a mobile health application that leverages advanced machine learning models to predict three major health conditions:

- **Diabetes** - Risk assessment based on lifestyle and metabolic factors
- **Heart Disease** - Prediction using clinical measurements and cardiovascular indicators
- **Chronic Kidney Disease** - Detection based on lab results and clinical parameters

The application combines a Flutter-based cross-platform mobile frontend with a Django REST API backend that serves trained ML models for real-time health predictions.

## ✨ Features

### Mobile App (Flutter)

- 🔐 Firebase Authentication with Google Sign-In
- 📊 Interactive health assessment forms
- 🎯 Real-time prediction results with confidence scores
- 📱 Cross-platform support (Android, iOS, Web)
- 🎨 Modern Material Design UI with animations
- 💾 Firebase Realtime Database for data persistence
- 📈 Historical predictions tracking
- 🔄 Real-time data synchronization

### Backend API (Django)

- 🤖 Three trained ML models (Diabetes, Heart Disease, CKD)
- 🚀 RESTful API with Django REST Framework
- 🔄 CORS-enabled for cross-origin requests
- 📦 Pipeline-based preprocessing (StandardScaler)
- 🎲 Multiple algorithms (LightGBM, XGBoost, Random Forest, MLP)
- ✅ Health check endpoints for model status
- 🔒 Production-ready with Gunicorn support

## 🏗️ Architecture

```
┌─────────────────────────────┐
│      Flutter App            │
│      (Mobile/Web)           │
└──────┬────────────┬─────────┘
       │            │
       │ HTTP/REST  │ Firebase SDK
       │            │
       │            ▼
       │      ┌──────────────────┐
       │      │  Firebase        │
       │      │  - Google Auth   │
       │      │  - Realtime DB   │
       │      └──────────────────┘
       │
       ▼
┌──────────────┐      ┌──────────────────┐
│  Django API  │──────│  ML Models       │
│  (Backend)   │      │  (.pkl files)    │
└──────────────┘      └──────────────────┘
```

## 🛠️ Technologies

### Backend

- **Framework**: Django 6.0+, Django REST Framework 3.16+
- **ML Libraries**: scikit-learn, LightGBM, XGBoost, pandas
- **Server**: Gunicorn (production)
- **Python**: 3.13+

### Frontend

- **Framework**: Flutter 3.0+
- **Language**: Dart
- **Authentication**: Firebase Auth with Google Sign-In
- **Database**: Firebase Realtime Database
- **HTTP Client**: http package for REST API calls
- **UI Libraries**: Google Fonts, Flutter Animate, Shimmer

## 📁 Project Structure

```
sedi-app/
├── backend/                    # Django REST API
│   ├── api/                   # API app
│   │   ├── views.py          # API endpoints
│   │   ├── serializers.py    # Request/response serializers
│   │   └── urls.py           # URL routing
│   ├── backend/               # Django project settings
│   │   ├── settings.py       # Configuration
│   │   └── urls.py           # Main URL config
│   ├── diabetes/              # Diabetes model files
│   ├── heart disease/         # Heart disease model files
│   ├── chronic kidney disease/ # CKD model files
│   ├── manage.py             # Django management
│   ├── pyproject.toml        # Dependencies
│   └── README.md             # API documentation
│
└── frontend/                  # Flutter mobile app
    └── sedi_diabetes_app/
        ├── lib/
        │   ├── main.dart     # App entry point
        │   ├── models/       # Data models
        │   ├── screens/      # UI screens
        │   ├── services/     # API & Firebase services
        │   ├── theme/        # App theming
        │   └── widgets/      # Reusable widgets
        ├── assets/           # Images, fonts, etc.
        ├── pubspec.yaml      # Flutter dependencies
        └── README.md         # Setup guide
```

## 🚀 Getting Started

### Prerequisites

- **Backend**:
  - Python 3.13 or higher
  - pip or uv package manager
- **Frontend**:
  - Flutter SDK 3.0 or higher
  - Dart SDK
  - Android Studio / Xcode (for mobile development)
  - Firebase account

### Backend Setup

1. **Navigate to backend directory**

   ```bash
   cd backend
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   # Using pip
   pip install -e .

   # Using uv (faster)
   uv pip install -e .
   ```

4. **Set up environment variables**

   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run migrations**

   ```bash
   python manage.py migrate
   ```

6. **Start development server**

   ```bash
   python manage.py runserver
   # API will be available at http://localhost:8000
   ```

7. **Test the API**
   ```bash
   python test_api_endpoints.py
   ```

### Frontend Setup

1. **Navigate to frontend directory**

   ```bash
   cd frontend/sedi_diabetes_app
   ```

2. **Install Flutter dependencies**

   ```bash
   flutter pub get
   ```

3. **Configure Firebase**
   - Create a Firebase project at [Firebase Console](https://console.firebase.google.com)
   - Enable **Firebase Authentication** and add Google Sign-In provider
   - Enable **Firebase Realtime Database** and set up security rules
   - Add Android/iOS/Web apps to your project
   - Download and place configuration files:
     - `google-services.json` (Android) → `android/app/`
     - `GoogleService-Info.plist` (iOS) → `ios/Runner/`
   - Update `firebase.json` with your project settings
   - Configure `firestore.rules` for database security

4. **Update API endpoint**
   - Open `lib/services/api_service.dart`
   - Update the base URL to your backend URL

5. **Run the app**

   ```bash
   # Android
   flutter run

   # iOS (Mac only)
   flutter run -d ios

   # Web
   flutter run -d chrome
   ```

## 📚 API Documentation

Complete API documentation is available in [`backend/README.md`](backend/README.md).

### Base URL

```
http://localhost:8000/api/
```

### Available Endpoints

| Endpoint                       | Method | Description                          |
| ------------------------------ | ------ | ------------------------------------ |
| `/api/diabetes/`               | GET    | Health check for diabetes model      |
| `/api/diabetes/`               | POST   | Predict diabetes risk                |
| `/api/heart-disease/`          | GET    | Health check for heart disease model |
| `/api/heart-disease/`          | POST   | Predict heart disease                |
| `/api/chronic-kidney-disease/` | GET    | Health check for CKD model           |
| `/api/chronic-kidney-disease/` | POST   | Predict chronic kidney disease       |

### Example Request

```bash
curl -X POST http://localhost:8000/api/diabetes/ \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "bmi": 28.5,
    "systolic_bp": 130,
    "diastolic_bp": 85,
    ...
  }'
```

### Example Response

```json
{
  "prediction": 1,
  "label": "Diabetic",
  "probability": {
    "non_diabetic": 0.089,
    "diabetic": 0.911
  },
  "confidence": 91.1
}
```

## 🤖 Models

### 1. Diabetes Prediction Model

- **Algorithm**: LightGBM with StandardScaler
- **Features**: 18 health and lifestyle factors
- **Output**: Binary classification (Diabetic/Non-diabetic)

### 2. Heart Disease Prediction Model

- **Algorithm**: Ensemble (LR, RF, XGBoost, LightGBM, MLP)
- **Features**: 13 clinical measurements
- **Output**: Binary classification (Presence/Absence)

### 3. Chronic Kidney Disease Model

- **Algorithm**: Ensemble (LR, RF, XGBoost, LightGBM, MLP)
- **Features**: 27 lab and clinical parameters
- **Output**: Binary classification (Affected/Not Affected)

All models use StandardScaler for feature preprocessing and provide probability scores with confidence levels.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Your Name** - Initial work

## 🙏 Acknowledgments

- Machine learning models trained on public health datasets
- Firebase for authentication and realtime database services
- Google Sign-In for seamless authentication
- Flutter community for excellent packages and support
- Django REST Framework for robust API development

---

<div align="center">
Made with ❤️ using Python & Flutter
</div>
