"""Fixed-length baseline spectral estimates and independent RMS noise synthesis.

This module deliberately has no dependency on ``audit`` or dataset loading.
Receipts describe a train-only *spectral shape*, not an electronics transfer
function or a new noise amplitude. Frequency units are MHz throughout.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np


SPECTRUM_CALIBRATION_TYPE = "raw_train_pretrigger_noise_spectrum"
SPECTRUM_SCHEMA_VERSION = 1
SPECTRUM_INTERPOLATION = (
    "one_sided_interior_half_nyquist_unchanged; linear_two_sided_psd; "
    "below_first_positive_bin_hold; dc_zero"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_payload_sha256(payload: dict) -> str:
    """Hash JSON without its self-hash; reject NaN rather than serializing it."""
    fields = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_spectrum_json(text: str) -> dict:
    """Reject ambiguous duplicate keys and the nonstandard NaN/Infinity tokens."""
    def pairs_to_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate noise spectrum JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite noise spectrum JSON constant: {value}")

    result = json.loads(text, object_pairs_hook=pairs_to_object,
                        parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("noise spectrum receipt must be a JSON object")
    return result


def _parameters(sample_rate_mhz, segment_samples):
    if (not isinstance(segment_samples, int) or isinstance(segment_samples, bool)
            or segment_samples < 8 or segment_samples % 2):
        raise ValueError("spectrum segment_samples must be an even integer >= 8")
    if (not isinstance(sample_rate_mhz, (int, float)) or isinstance(sample_rate_mhz, bool)
            or not np.isfinite(sample_rate_mhz) or float(sample_rate_mhz) != 1000.0):
        raise ValueError("noise spectrum requires prepared 1 GS/s (1000 MHz) data")
    return float(sample_rate_mhz), segment_samples


def fixed_welch_periodogram(baseline_volts, sample_rate_mhz=1000.0,
                           segment_samples=64):
    """Mean one-sided density of complete periodic-Hann, 50% overlap segments.

    Each segment is demeaned in raw volts before windowing. Segments are never
    padded; their common length fixes the frequency resolution. DC is removed
    because generated noise is zero-mean. Return (MHz, V**2/MHz, segment count).
    """
    rate, count = _parameters(sample_rate_mhz, segment_samples)
    baseline = np.asarray(baseline_volts, dtype=np.float64)
    if baseline.ndim != 1 or baseline.size < count or not np.isfinite(baseline).all():
        raise ValueError("baseline must contain a complete finite fixed-length segment")
    starts = np.arange(0, baseline.size - count + 1, count // 2)
    segments = np.stack([baseline[start:start + count] for start in starts])
    segments -= segments.mean(axis=1, keepdims=True)
    window = np.hanning(count + 1)[:-1]  # Periodic Hann, same as fftbins=True.
    transform = np.fft.rfft(segments * window[None, :], axis=1)
    power = np.mean(np.abs(transform) ** 2, axis=0) / (rate * np.sum(window ** 2))
    power[1:-1] *= 2.0
    power[0] = 0.0
    return np.fft.rfftfreq(count, d=1.0 / rate), power, len(starts)


def _spectral_arrays(frequencies_mhz, shape_psd_per_mhz, sample_rate_mhz=1000.0):
    frequency = np.asarray(frequencies_mhz, dtype=np.float64)
    power = np.asarray(shape_psd_per_mhz, dtype=np.float64)
    if (frequency.ndim != 1 or power.ndim != 1 or frequency.size < 5
            or power.shape != frequency.shape or not np.isfinite(frequency).all()
            or not np.isfinite(power).all() or np.any(power < 0)
            or frequency[0] != 0.0 or power[0] != 0.0
            or np.any(np.diff(frequency) <= 0)
            or not math.isclose(float(frequency[-1]), float(sample_rate_mhz) / 2,
                                rel_tol=0, abs_tol=1e-10)
            or np.sum(power[1:]) <= 0):
        raise ValueError("spectrum requires finite non-negative PSD on increasing 0..Nyquist bins")
    return frequency, power


def spectrum_noise(n: int, sigma: float, rng, frequencies_mhz, shape_psd_per_mhz,
                   sample_rate_mhz=1000.0) -> np.ndarray:
    """Draw the same n white samples as pink_noise, then shape and RMS-normalize.

    Convert the measured one-sided density to coefficient/two-sided density
    before piecewise linear interpolation: positive interior bins /2, Nyquist
    unchanged. This compensates the one-sided periodogram's paired interior
    bins without imposing an artificial Nyquist roll-off. Hold the first
    positive density below its center; those low frequencies remain assumed.
    DC is zero; there is no calibration above Nyquist.
    """
    _parameters(sample_rate_mhz, 8)
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("noise window must contain at least two samples")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("noise sigma must be finite and non-negative")
    frequency, power = _spectral_arrays(frequencies_mhz, shape_psd_per_mhz,
                                      sample_rate_mhz)
    two_sided = power.copy()
    two_sided[1:-1] *= 0.5
    target_f = np.fft.rfftfreq(n, d=1.0 / sample_rate_mhz)
    target_p = np.interp(target_f, frequency[1:], two_sided[1:])
    target_p[0] = 0.0
    if not np.any(target_p > 0):
        raise ValueError("spectrum has no nonzero power on the requested FFT grid")
    transform = np.fft.rfft(rng.standard_normal(n))
    out = np.fft.irfft(transform * np.sqrt(target_p), n=n)
    observed = float(out.std())
    if not np.isfinite(observed) or observed <= 0:
        raise ValueError("generated spectral noise has zero or invalid RMS")
    return out / observed * sigma


def validate_noise_spectrum_receipt(receipt, sample_rate_mhz=1000.0, *,
                                    event_id_scheme: str,
                                    signal_contract_version: str,
                                    source_contract_validator) -> dict:
    """Validate a portable, self-hashed train-only spectrum receipt; fail closed.

    The caller supplies the existing event/source contract validators so this
    numerical module can also be called by audit without an import cycle.
    A self-hash detects accidental/unsigned edits, not a forged recalibration.
    """
    if not isinstance(receipt, dict):
        raise ValueError("noise spectrum receipt must be a JSON object")
    if receipt.get("payload_sha256") != canonical_payload_sha256(receipt):
        raise ValueError("noise spectrum receipt payload hash mismatch")
    if (receipt.get("schema_version") != SPECTRUM_SCHEMA_VERSION
            or receipt.get("calibration_type") != SPECTRUM_CALIBRATION_TYPE
            or receipt.get("partition") != "train"
            or receipt.get("event_identity_method") != "event_hash_v1"
            or receipt.get("event_id_scheme") != event_id_scheme
            or receipt.get("signal_contract_version") != signal_contract_version):
        raise ValueError("noise spectrum receipt has no matching train/event/signal contract")
    source = receipt.get("source_contract")
    required = {"scintcpunet/noise_spectrum.py", "scripts/calibrate_noise_spectrum.py",
                "scripts/prepare_sim_data.py"}
    if (not source_contract_validator(source)
            or not required.issubset(source.get("files", {}))):
        raise ValueError("noise spectrum source contract is not frozen")
    if any(not isinstance(receipt.get(key), str) or not _SHA256.fullmatch(receipt[key])
           for key in ("input_sha256", "config_sha256")):
        raise ValueError("noise spectrum input/config SHA256 is missing or invalid")
    split = receipt.get("split_config")
    if not isinstance(split, dict) or split.get("method") != "event_hash_v1":
        raise ValueError("noise spectrum stable split configuration is missing")
    seed = split.get("seed")
    test, validation = split.get("test_fraction"), split.get("validation_fraction")
    if (not isinstance(seed, int) or isinstance(seed, bool)
            or not isinstance(test, (int, float)) or isinstance(test, bool)
            or not isinstance(validation, (int, float)) or isinstance(validation, bool)
            or not math.isfinite(test) or not math.isfinite(validation)
            or min(test, validation) < 0 or test + validation >= 1
            or receipt.get("split_seed") != seed):
        raise ValueError("noise spectrum split configuration is invalid")
    targets = receipt.get("target_real_event_ids")
    if (not isinstance(targets, list) or not targets
            or any(not isinstance(value, str) or not value.startswith("data:")
                   for value in targets)
            or targets != sorted(set(targets))):
        raise ValueError("noise spectrum train event cohort is missing or invalid")
    for event_id in targets:
        key = f"event_hash_v1|{seed}|data|{event_id}".encode()
        fraction = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        if fraction < test + validation:
            raise ValueError("noise spectrum receipt contains a held-out event ID")
    expected_ids_hash = hashlib.sha256("\n".join(targets).encode("utf-8")).hexdigest()
    if (receipt.get("used_event_ids_sha256") != expected_ids_hash
            or receipt.get("used_train_records") != len(targets)
            or not isinstance(receipt.get("selected_train_records"), int)
            or receipt["selected_train_records"] < len(targets)):
        raise ValueError("noise spectrum event cohort counts/hash do not agree")
    rate = receipt.get("sample_rate_mhz")
    estimator = receipt.get("estimator")
    if not isinstance(estimator, dict):
        raise ValueError("noise spectrum fixed-length estimator is missing")
    _, count = _parameters(rate, estimator.get("segment_samples"))
    if float(rate) != float(sample_rate_mhz):
        raise ValueError("noise spectrum sample rate mismatches generation rate")
    baseline = receipt.get("baseline_contract")
    if (not isinstance(baseline, dict) or baseline.get("voltage_units") != "V"
            or baseline.get("start_sample") != 0 or baseline.get("pre_trigger_ns") != 20.0
            or baseline.get("timing") != "main_pulse_cfd_0.5_from_normalized_raw"
            or baseline.get("end") != "CFD_sample_minus_round(20ns*sample_rate)"
            or estimator.get("kind") != "fixed_length_welch"
            or estimator.get("window") != "periodic_hann"
            or estimator.get("overlap_samples") != count // 2
            or estimator.get("detrend") != "segment_mean"
            or estimator.get("zero_padding") != 0 or estimator.get("dc") != "zero"
            or estimator.get("event_weighting") != "equal; normalize_each_event_psd_integral"
            or estimator.get("sidedness") != "one_sided_density; interior_x2_nyquist_x1"
            or estimator.get("density_units") != "1/MHz"
            or estimator.get("interpolation") != SPECTRUM_INTERPOLATION):
        raise ValueError("noise spectrum estimator/baseline contract is incompatible")
    contract = {"sample_rate_mhz": rate, "split_config": split,
                "baseline_contract": baseline, "estimator": estimator}
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if receipt.get("calibration_contract_sha256") != contract_hash:
        raise ValueError("noise spectrum calibration contract hash mismatch")
    frequency, power = _spectral_arrays(receipt.get("frequencies_mhz"),
                                       receipt.get("shape_psd_per_mhz"), rate)
    expected = np.fft.rfftfreq(count, d=1.0 / rate)
    if frequency.shape != expected.shape or not np.allclose(frequency, expected, atol=1e-10, rtol=0):
        raise ValueError("noise spectrum frequency bins do not match its fixed segment length")
    resolution = float(rate) / count
    if (not math.isclose(float(np.sum(power) * resolution), 1.0,
                         rel_tol=1e-10, abs_tol=1e-10)
            or receipt.get("frequency_resolution_mhz") != resolution
            or receipt.get("nyquist_mhz") != rate / 2
            or not str(receipt.get("limitation", "")).strip()):
        raise ValueError("noise spectrum shape normalization/resolution contract is invalid")
    return receipt
