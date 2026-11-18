"""End-to-end API test over every trip in PATHSNAP_API_DATA (default: student_test).
Run: python -m api.test_api
Checks: all trips return 200, the response carries no reference data, metrics are present, bad ids are refused."""
import json

from fastapi.testclient import TestClient

from api.main import app

ALLOWED_KEYS = {"trip_id", "n_points", "model", "decoder", "gps", "predicted_arcs", "predicted_route",
                "timings_ms", "metrics"}
FORBIDDEN_SUBSTR = ("reference", "label", "ground", "truth", "route_arcs")


def main():
    with TestClient(app) as c:
        h = c.get("/health").json(); print("health:", h)
        trips = c.get("/trips").json()["trips"]; print(len(trips), "trips:", [t["trip_id"] for t in trips])
        rows = []
        for t in trips:
            r = c.post("/predict", json={"trip_id": t["trip_id"]})
            assert r.status_code == 200, (t, r.text[:200])
            j = r.json()
            assert set(j) <= ALLOWED_KEYS, f"unexpected keys {set(j) - ALLOWED_KEYS}"
            assert not any(s in k.lower() for k in j for s in FORBIDDEN_SUBSTR if k != "predicted_route")
            assert len(j["gps"]) == j["n_points"] == len(j["predicted_arcs"])
            assert len(j["predicted_route"]) >= 2
            m = j["metrics"]
            rows.append((t["trip_id"], j["n_points"], m["top1_accuracy"], m["top1_accuracy_classifier_only"],
                         m["route_continuity"], m["candidate_miss_rate"], j["timings_ms"]["total_ms"]))
        print(f"{'trip':>10} {'points':>7} {'top1':>7} {'clf-only':>9} {'contin.':>8} {'miss':>7} {'ms':>8}")
        for r in rows:
            print(f"{r[0]:>10} {r[1]:>7} {r[2]:>7.4f} {r[3]:>9.4f} {r[4]:>8.4f} {r[5]:>7.4f} {r[6]:>8.0f}")
        assert c.post("/predict", json={"trip_id": "../../etc"}).status_code == 404
        assert c.post("/predict", json={"trip_id": "99999999"}).status_code == 404
        print("bad ids refused: OK; all", len(rows), "trips OK")


if __name__ == "__main__":
    main()
