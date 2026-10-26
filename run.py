"""
run.py
------
Convenience entrypoint to launch the Customer Conversion Prediction System Admin Portal.
Usage:
    python run.py
"""

import os
from server import app, ADMIN_USERNAME, ADMIN_EMAIL, RAW_ADMIN_PASSWORD

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "127.0.0.1")
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")

    print("\n" + "=" * 70)
    print("CUSTOMER CONVERSION PREDICTION SYSTEM - ADMIN PORTAL")
    print("=" * 70)
    print(f"-> Access the Admin Portal: http://{host}:{port}")
    print(f"-> Admin Username: {ADMIN_USERNAME}")
    print(f"-> Admin Email:    {ADMIN_EMAIL}")
    print(f"-> Admin Password: {RAW_ADMIN_PASSWORD}")
    print("=" * 70 + "\n")

    app.run(host=host, port=port, debug=debug)
