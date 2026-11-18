# PathSnap API (Laptop 1, Windows)

Hosts the trained classifier, the HMM/Viterbi decoder, the road-network handling and the server-side evaluation.
The UI (Laptop 2) only sends a trip ID and shows what comes back.

## Install (once)
```powershell
git clone https://github.com/utkarsh-debug/pathsnap-map-matching.git
cd pathsnap-map-matching
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

## Choose which trips the API serves
`PATHSNAP_API_DATA` = a folder that directly contains the trip sub-folders (`<id>/<id>.track|.nodes|.arcs|.route`).

| Situation | Setting |
|---|---|
| Development / dry runs (default) | `..\Dataset\student_test` (nothing to set) |
| Final evaluation | `$env:PATHSNAP_API_DATA="C:\path\to\evaluator_test"` |

The `.route` files in that folder are the reference answers. They are read only by `api/scoring.py` after a prediction is finished and are never returned.
The evaluator trips must not be used for development, tuning or testing.

## Run
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Keep this console visible: every request is logged here (this is what the evaluator is shown):
```
>>> POST /predict from 192.168.1.20
trip 00000052: 262 points | decoder=hmm_viterbi | Top-1 0.9580 (classifier-only 0.9580), continuity 0.954, candidate miss 0.0 | pipeline 73 ms
<<< POST /predict -> 200 in 106 ms
```

## Let Laptop 2 connect
1. Find this laptop's address: `ipconfig` (the IPv4 address on the shared Wi-Fi/hotspot, e.g. `192.168.1.10`).
2. Allow the port (admin PowerShell, once):
   `New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow`
3. Make sure the Wi-Fi network is set to **Private**.
4. From Laptop 2: `curl http://<laptop1-ip>:8000/health`

## Endpoints
| | |
|---|---|
| `GET /health` | `{"status":"ok","model":...,"decoder":"hmm_viterbi"\|"classifier_only"}` |
| `GET /trips` | `{"trips":[{"trip_id":"00000015","n_points":1234}, ...]}` |
| `POST /predict` `{"trip_id":"..."}` | `trip_id, n_points, model, decoder, gps [[lat,lon]], predicted_arcs, predicted_route [[lat,lon]], timings_ms, metrics` |

`metrics`: Top-1 accuracy (final and classifier-only), route continuity, route arc precision/recall/F1, candidate miss rate.
No reference route or labels are ever returned.

## Self-test (development trips only)
```powershell
python -m api.test_api
```
Calls every served trip, checks the response contains no reference data, and checks that bad IDs get 404.

## Artifacts (`api/artifacts/`)
- `model.joblib` and `pipeline_config.json`: the exported model, feature list, candidate settings (`python -m api.export_artifacts`).
- `hmm_params.json`: tuned HMM/Viterbi parameters (same layout as `results/hmm_params.json`). If this file or `hmm/viterbi.py` is missing, the API runs classifier-only and every response says `"decoder": "classifier_only"`.
