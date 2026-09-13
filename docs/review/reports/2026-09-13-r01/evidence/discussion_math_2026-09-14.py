"""Pure arithmetic for the review discussion; no project imports or training."""
import json
from statistics import mean, pvariance
from math import sqrt, isclose

def corr(x, y):
    a = [v - mean(x) for v in x]
    b = [v - mean(y) for v in y]
    return sum(u*v for u, v in zip(a, b)) / sqrt(sum(u*u for u in a)*sum(v*v for v in b))

levels = [1000.0 * i for i in range(5) for _ in range(20)]
changes = [float(j) for _ in range(5) for j in range(20)]
truth = [v+d for v, d in zip(levels, changes)]
pred = [v-d for v, d in zip(levels, changes)]
within = [corr(truth[i:i+20], pred[i:i+20]) for i in range(0, 100, 20)]
pooled = corr(truth, pred)
derived = (pvariance(levels)-pvariance(changes))/(pvariance(levels)+pvariance(changes))
assert all(isclose(v, -1) for v in within)
assert isclose(pooled, 0.999966750552772)
assert isclose(pooled, derived)
print(json.dumps({"each_event_corr": within, "pooled_corr": pooled,
                  "var_level": pvariance(levels), "var_change": pvariance(changes),
                  "formula_result": derived}, ensure_ascii=False))

# A constant translation of one forecast cannot change its within-forecast corr.
forecast = [5.0, 3.0, 2.0, 0.0]
future_truth = [10.0, 11.0, 12.0, 13.0]
anchored = [v-forecast[0]+10.0 for v in forecast]
assert isclose(corr(forecast, future_truth), corr(anchored, future_truth))
assert anchored[0] == 10.0
print(json.dumps({"within_forecast_raw_corr": corr(forecast, future_truth),
                  "within_forecast_anchored_corr": corr(anchored, future_truth),
                  "anchored_first_step_equals_current_observation": True}))
print("PASS: arithmetic only; synthetic examples are not measured model performance.")
