# Report screenshots

## Laptop 1: API console

`laptop1_api_console.png` is a capture of a Windows console displaying an
excerpt from the actual running API log. It is a log-excerpt display,
not a capture of a fresh prediction being executed in that console.
The original extracted lines are in `laptop1_api_log_excerpt.txt`.

The requests came from Laptop 2 at `10.20.205.142` and used student-test
trip `00000009`. Both returned HTTP 200. The tuned XGBoost classifier and
HMM/Viterbi decoder were active.

Suggested report caption:

> Laptop 1 (Windows): console display of actual API log entries for requests
> from Laptop 2. Student-test trip 00000009 contains 1,049 GPS observations;
> HMM Top-1 accuracy is 0.8684 versus 0.8646 for classifier-only predictions.
> The API returned HTTP 200 for both requests.

## Laptop 2: UI screenshot still needed

The UI screenshot supplied in chat has not been saved in this repository.
For a directly comparable pair, capture the UI after running student-test
trip `00000009`, with the dropdown closed and the trip, route, and matching
metrics visible. Save the original screenshot here without altering metrics.

## Folium screenshot still needed

The report also requires a representative geographical Folium map screenshot
or exported view. These API logs do not replace that requirement.

No evaluator-trip data was used for this evidence.
