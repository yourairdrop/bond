"""
City-level weather probability model for temperature bond markets.

This is intentionally lightweight for dry-run experimentation:
  1. Parse the Polymarket temperature threshold from the question.
  2. Fetch Open-Meteo daily max-temperature forecast for the city/date.
  3. Convert the point forecast into a probability with a conservative
     city/lead-time residual distribution.

It is not a replacement for proper historical calibration. It gives the
new city-bond dry-run strategy a real data source and records enough
metadata to audit where each p_fair came from.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

from alpha_suite.strategies.bond_buyer import (
    _parse_question_date_cst,
    weather_city_key,
)


OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

HTTP_TIMEOUT_SEC = float(os.environ.get("CITY_BOND_HTTP_TIMEOUT", "8"))
MODEL_WEIGHT = float(os.environ.get("CITY_BOND_MODEL_WEIGHT", "0.75"))
DEFAULT_SIGMA_C = float(os.environ.get("CITY_BOND_DEFAULT_SIGMA_C", "1.7"))
MAX_CACHE_AGE_SEC = float(os.environ.get("CITY_BOND_CACHE_SEC", "1800"))


@dataclass
class TempMarketSpec:
    city: str
    target_date: date
    unit: str
    kind: str
    lo: float
    hi: float


@dataclass
class CityProbability:
    p_yes: float
    p_outcome: float
    forecast_temp: float
    forecast_unit: str
    sigma: float
    model_weight: float
    city: str
    target_date: str
    kind: str
    lo: float
    hi: float
    source: str
    note: str = ""


_TEMP_RE = re.compile(
    r"highest\s+temperature\s+in\s+(.+?)\s+be\s+(.+?)\s+on\s+",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"between\s+(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CF])",
    re.IGNORECASE,
)
_NUM_UNIT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CF])", re.IGNORECASE)


def _http_json(url: str) -> Optional[object]:
    req = urllib.request.Request(url, headers={"User-Agent": "AlphaSuite-CityBond/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _norm_unit(unit: str) -> str:
    return "F" if str(unit).upper().startswith("F") else "C"


def _c_to_f(value: float) -> float:
    return value * 9.0 / 5.0 + 32.0


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _clamp_prob(value: float) -> float:
    return max(0.001, min(0.999, float(value)))


def parse_temperature_market(question: str, fallback: datetime | None = None) -> Optional[TempMarketSpec]:
    """Parse the supported highest-temperature Polymarket question forms."""
    q = question or ""
    match = _TEMP_RE.search(q)
    if not match:
        return None

    city = weather_city_key(q) or re.sub(r"[^a-z0-9 ]+", "", match.group(1).lower()).strip()
    target_date = _parse_question_date_cst(q, fallback)
    if not city or not target_date:
        return None

    clause = match.group(2).strip()
    range_match = _RANGE_RE.search(clause)
    if range_match:
        unit = _norm_unit(range_match.group(3))
        lo = float(range_match.group(1))
        hi = float(range_match.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return TempMarketSpec(city, target_date, unit, "between", lo - 0.5, hi + 0.5)

    numbers = _NUM_UNIT_RE.findall(clause)
    if not numbers:
        return None
    unit = _norm_unit(numbers[-1][1])

    lowered = clause.lower()
    if "between" in lowered and len(numbers) >= 2:
        lo = float(numbers[0][0])
        hi = float(numbers[1][0])
        if lo > hi:
            lo, hi = hi, lo
        return TempMarketSpec(city, target_date, unit, "between", lo - 0.5, hi + 0.5)

    value = float(numbers[0][0])
    if "or higher" in lowered or "or above" in lowered:
        return TempMarketSpec(city, target_date, unit, "gte", value - 0.5, math.inf)
    if "or below" in lowered or "or lower" in lowered:
        return TempMarketSpec(city, target_date, unit, "lte", -math.inf, value + 0.5)

    # Exact whole-degree bucket. Polymarket weather resolution text says the
    # source rounds to whole degrees, so exact 34C means [33.5, 34.5).
    return TempMarketSpec(city, target_date, unit, "exact", value - 0.5, value + 0.5)


class CityWeatherModel:
    """Small Open-Meteo backed probability model with in-memory caching."""

    def __init__(self):
        self._geo_cache: dict[str, tuple[float, float, str]] = {}
        self._forecast_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def geocode(self, city: str) -> Optional[tuple[float, float, str]]:
        key = city.strip().lower()
        if key in self._geo_cache:
            return self._geo_cache[key]

        qs = urllib.parse.urlencode({"name": city, "count": "1", "language": "en", "format": "json"})
        data = _http_json(f"{OPEN_METEO_GEOCODE}?{qs}")
        if not isinstance(data, dict):
            return None
        rows = data.get("results") or []
        if not rows:
            return None
        row = rows[0]
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            return None
        name = str(row.get("name") or city)
        country = str(row.get("country_code") or row.get("country") or "")
        label = f"{name}, {country}".strip().strip(",")
        self._geo_cache[key] = (lat, lon, label)
        return self._geo_cache[key]

    def forecast_max_c(self, city: str, target_date: date) -> Optional[tuple[float, str]]:
        cache_key = (city.strip().lower(), target_date.isoformat())
        cached = self._forecast_cache.get(cache_key)
        now = time.time()
        if cached and now - cached[1] <= MAX_CACHE_AGE_SEC:
            return cached[0], "open-meteo-cache"

        geo = self.geocode(city)
        if not geo:
            return None
        lat, lon, _label = geo
        qs = urllib.parse.urlencode({
            "latitude": f"{lat:.5f}",
            "longitude": f"{lon:.5f}",
            "daily": "temperature_2m_max",
            "timezone": "auto",
            "forecast_days": "7",
        })
        data = _http_json(f"{OPEN_METEO_FORECAST}?{qs}")
        if not isinstance(data, dict):
            return None
        daily = data.get("daily") or {}
        times = daily.get("time") or []
        temps = daily.get("temperature_2m_max") or []
        target = target_date.isoformat()
        for idx, day in enumerate(times):
            if day != target:
                continue
            try:
                value = float(temps[idx])
            except (IndexError, TypeError, ValueError):
                return None
            self._forecast_cache[cache_key] = (value, now)
            return value, "open-meteo"
        return None

    def sigma_for(self, city: str, days_ahead: int, unit: str) -> float:
        # Start conservative. Later backtests can replace this with learned
        # city residuals. A few historically noisy cities get extra width.
        sigma_c = DEFAULT_SIGMA_C
        if city in {"denver", "beijing", "tokyo", "chongqing", "manila"}:
            sigma_c += 0.35
        if days_ahead >= 2:
            sigma_c += 0.3
        return sigma_c * (9.0 / 5.0) if _norm_unit(unit) == "F" else sigma_c

    def estimate(self, question: str, outcome: str) -> Optional[CityProbability]:
        spec = parse_temperature_market(question, datetime.now(timezone.utc))
        if not spec:
            return None
        forecast = self.forecast_max_c(spec.city, spec.target_date)
        if not forecast:
            return None
        forecast_c, source = forecast
        forecast_temp = _c_to_f(forecast_c) if spec.unit == "F" else forecast_c
        days_ahead = (spec.target_date - datetime.now(timezone.utc).date()).days
        sigma = self.sigma_for(spec.city, days_ahead, spec.unit)

        p_lo = 0.0 if math.isinf(spec.lo) and spec.lo < 0 else _normal_cdf(spec.lo, forecast_temp, sigma)
        p_hi = 1.0 if math.isinf(spec.hi) and spec.hi > 0 else _normal_cdf(spec.hi, forecast_temp, sigma)
        p_yes = _clamp_prob(p_hi - p_lo)
        if spec.kind == "gte":
            p_yes = _clamp_prob(1.0 - _normal_cdf(spec.lo, forecast_temp, sigma))
        elif spec.kind == "lte":
            p_yes = _clamp_prob(_normal_cdf(spec.hi, forecast_temp, sigma))

        selected = p_yes if str(outcome).strip().lower() == "yes" else 1.0 - p_yes
        return CityProbability(
            p_yes=round(_clamp_prob(p_yes), 4),
            p_outcome=round(_clamp_prob(selected), 4),
            forecast_temp=round(forecast_temp, 2),
            forecast_unit=spec.unit,
            sigma=round(sigma, 3),
            model_weight=MODEL_WEIGHT,
            city=spec.city,
            target_date=spec.target_date.isoformat(),
            kind=spec.kind,
            lo=round(spec.lo, 3) if math.isfinite(spec.lo) else spec.lo,
            hi=round(spec.hi, 3) if math.isfinite(spec.hi) else spec.hi,
            source=source,
        )


_MODEL = CityWeatherModel()


def estimate_city_weather_probability(question: str, outcome: str) -> Optional[CityProbability]:
    return _MODEL.estimate(question, outcome)
