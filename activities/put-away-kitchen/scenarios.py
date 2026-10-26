"""The put-away activity's scenarios (activities/put-away/scenarios.py), shared."""
import importlib.util
import os

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "put-away", "scenarios.py")
_spec = importlib.util.spec_from_file_location("put_away_scenarios", _p)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
globals().update({k: v for k, v in vars(_m).items() if k.startswith("scenario_")})
