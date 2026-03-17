import urllib.request
import json

BASE_URL = "http://127.0.0.1:8000/api"


def test_endpoint(endpoint, payload):
    url = f"{BASE_URL}/{endpoint}/"
    print(f"\n--- Testing POST {url} ---")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        print(f"Status: {response.status}")
        print(f"Response: {json.dumps(result, indent=2)}")
    except urllib.error.HTTPError as e:
        print(f"Error Status: {e.code}")
        print(f"Error Response: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    # 1. Test Diabetes Model
    diabetes_data = {
        "age": 45.0,
        "alcohol_consumption_per_week": 2.0,
        "physical_activity_minutes_per_week": 150.0,
        "diet_score": 5.0,
        "sleep_hours_per_day": 7.0,
        "screen_time_hours_per_day": 4.0,
        "bmi": 28.5,
        "waist_to_hip_ratio": 0.9,
        "systolic_bp": 130.0,
        "diastolic_bp": 85.0,
        "heart_rate": 72.0,
        "cholesterol_total": 210.0,
        "hdl_cholesterol": 45.0,
        "ldl_cholesterol": 130.0,
        "triglycerides": 150.0,
        "family_history_diabetes": 1,
        "hypertension_history": 1,
        "cardiovascular_history": 0,
    }
    test_endpoint("diabetes", diabetes_data)

    # 2. Test Heart Disease Model
    heart_data = {
        "age": 50.0,
        "sex": 1,
        "chest_pain_type": 2,
        "bp": 120.0,
        "cholesterol": 200.0,
        "fbs_over_120": 0,
        "ekg_results": 1,
        "max_hr": 150.0,
        "exercise_angina": 0,
        "st_depression": 1.5,
        "slope_of_st": 2,
        "num_vessels_fluro": 0,
        "thallium": 3,
    }
    test_endpoint("heart-disease", heart_data)

    # 3. Test Chronic Kidney Disease Model
    ckd_data = {
        "bp_diastolic": 80,
        "bp_limit": 1,
        "sg": 2,
        "al": 1,
        "rbc": 0,
        "su": 0,
        "pc": 1,
        "pcc": 0,
        "ba": 0,
        "bgr": 121,
        "bu": 36,
        "sod": 138,
        "sc": 1,
        "pot": 4,
        "hemo": 15,
        "pcv": 44,
        "rbcc": 4,
        "wbcc": 7800,
        "htn": 0,
        "dm": 0,
        "cad": 0,
        "appet": 1,
        "pe": 0,
        "ane": 0,
        "grf": 0,
        "stage": 1,
        "age": 55,
    }
    test_endpoint("chronic-kidney-disease", ckd_data)
