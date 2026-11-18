# PathSnap UI (Laptop 2)

This is a small browser UI for the PathSnap FastAPI backend. It uses only browser APIs and Python's standard library; it does not need a JavaScript build step or additional Python packages.

## Start the UI on Linux

From the repository root, run:

```bash
python3 -m http.server --bind 0.0.0.0 --directory ui 8500
```

Open `http://localhost:8500` in a browser on Laptop 2. Serving the page over HTTP lets the browser call the API; opening `index.html` directly from the file system can block those requests.

## Connect to Laptop 1

1. Start the API on Laptop 1 using the instructions in `api/README.md`.
2. Find Laptop 1's IPv4 address on the shared Wi-Fi or hotspot.
3. Enter `http://<laptop-1-ip>:8000` in the API address field and choose **Connect**.
4. Select one of the trips returned by the API, then choose **Run prediction**.

The address is saved in the browser's local storage. Update it if Laptop 1 receives a different address. The API must allow Laptop 2 to reach port `8000`; its CORS configuration already permits browser requests from the UI origin.

## What the page displays

- API health, model name, and decoder.
- The trip IDs and point counts returned by `GET /trips`.
- The GPS observations and predicted route returned by `POST /predict`.
- The prediction metrics and timing returned by the server.

The page makes requests only to `GET /health`, `GET /trips`, and `POST /predict`. It does not read trip files or contain ground-truth labels. Reference-route metrics are displayed only when the API returns them.

## Troubleshooting

- **Could not reach the API:** check that both laptops are on the same network and the entered address is Laptop 1's current IPv4 address.
- **Connection times out:** confirm the API is running on port `8000` and that Laptop 1's firewall permits incoming connections on that port.
- **The API says classifier-only:** the UI is showing the decoder reported by the API. The HMM decoder and parameters must be installed on Laptop 1 for the API to report `hmm_viterbi`.
- **No trips are listed:** check `PATHSNAP_API_DATA` on Laptop 1 and confirm the folder contains the trip files.
