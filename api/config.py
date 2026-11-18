"""API configuration (environment variables, no secrets).

PATHSNAP_API_DATA : folder that directly contains the trip sub-folders the API may serve
                    (each <trip_id>/<trip_id>.track|.nodes|.arcs|.route).
                    Default: ../Dataset/student_test (development). At the final evaluation point it at the
                    evaluator trips, e.g.  $env:PATHSNAP_API_DATA="C:\...\Hackathon\evaluator_test"
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "api" / "artifacts"
DATA_ROOT = Path(os.environ.get("PATHSNAP_API_DATA", ROOT.parent / "Dataset" / "student_test")).resolve()
HOST = os.environ.get("PATHSNAP_API_HOST", "0.0.0.0")
PORT = int(os.environ.get("PATHSNAP_API_PORT", "8000"))
