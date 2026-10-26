"""
test_app.py
-----------
Automated test suite verifying the Customer Conversion Prediction System:
1. Successful admin login
2. Invalid login credentials
3. Protected dashboard route access without login
4. Logout and session protection
5. Customer conversion prediction using the trained ML model pipeline
"""

import os
import sys
import unittest
import json

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from server import app, ADMIN_USERNAME, ADMIN_EMAIL, RAW_ADMIN_PASSWORD


class CustomerConversionAdminTests(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

    def test_01_login_page_renders_cleanly(self):
        """Verify the Login page renders with proper titles and elements."""
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Customer Conversion", html)
        self.assertIn("Prediction System", html)
        self.assertIn("Powered by Machine Learning", html)
        self.assertIn("username", html)
        self.assertIn("password", html)
        self.assertIn("btn-toggle-password", html)

    def test_02_dashboard_access_blocked_without_login(self):
        """Verify unauthenticated requests to /dashboard redirect to /login."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_03_api_access_blocked_without_login(self):
        """Verify unauthenticated API requests return HTTP 401 Unauthorized."""
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 401)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "Unauthorized")

    def test_04_invalid_login_credentials(self):
        """Verify invalid credentials return 401 and an error message."""
        response = self.client.post(
            "/login",
            json={"username": "wrong_user", "password": "wrong_password"},
        )
        self.assertEqual(response.status_code, 401)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["message"], "Invalid username or password.")

    def test_05_empty_login_credentials(self):
        """Verify empty username or password returns 400."""
        response = self.client.post(
            "/login",
            json={"username": "", "password": ""},
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])

    def test_06_successful_admin_login(self):
        """Verify valid admin credentials successfully authenticate and set session."""
        response = self.client.post(
            "/login",
            json={"username": ADMIN_USERNAME, "password": RAW_ADMIN_PASSWORD},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("/dashboard", data["redirect_url"])

    def test_07_dashboard_accessible_with_active_session(self):
        """Verify dashboard is accessible when logged in."""
        with self.client.session_transaction() as sess:
            sess["is_admin"] = True
            sess["username"] = ADMIN_USERNAME

        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Total Customers Analyzed", html)
        self.assertIn("Predicted Conversions", html)
        self.assertIn("Predicted Non-Conversions", html)
        self.assertIn("Conversion Percentage", html)
        self.assertIn("Predict Prospective Conversion", html)

    def test_08_customer_conversion_prediction_ml_pipeline(self):
        """Verify ML prediction using the pre-trained Scikit-learn Pipeline."""
        with self.client.session_transaction() as sess:
            sess["is_admin"] = True
            sess["username"] = ADMIN_USERNAME

        # Test High-Intent Referral lead
        lead_payload = {
            "source": "Referral",
            "service_type": "Enterprise",
            "budget": 28000.0,
            "response_time_hours": 1.5,
            "previous_interactions_follow_up_count": 4,
        }

        response = self.client.post("/api/predict", json=lead_payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("prediction", data)
        self.assertIn(data["prediction"], [0, 1])
        self.assertIn("probability", data)
        self.assertTrue(0.0 <= data["probability"] <= 100.0)
        self.assertIn("class_label", data)
        self.assertIn("recommendation", data)
        self.assertIn("enquiry_id", data)

    def test_09_logout_clears_session_and_protects_routes(self):
        """Verify logout terminates session and blocks dashboard access."""
        with self.client.session_transaction() as sess:
            sess["is_admin"] = True
            sess["username"] = ADMIN_USERNAME

        # Perform logout
        logout_res = self.client.get("/logout")
        self.assertEqual(logout_res.status_code, 302)
        self.assertIn("/login", logout_res.headers["Location"])

        # Try to access dashboard after logout
        post_logout_res = self.client.get("/dashboard")
        self.assertEqual(post_logout_res.status_code, 302)
        self.assertIn("/login", post_logout_res.headers["Location"])


if __name__ == "__main__":
    unittest.main()
