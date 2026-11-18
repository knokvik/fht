"""
Clinical Cause Engine for Cough Prescreening
Combines:
  1. AI Audio Classification (CNN softmax probabilities)
  2. Acoustic Descriptors (Dry vs Wet, Bout Count, Harshness, Duration)
  3. Patient Reported Symptoms (Fever, Wheeze, Night worsening, Duration, Smoker, GERD, Allergy)

Outputs an explainable, ranked list of probable causes with explicit clinical evidence.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from server.config import config

logger = logging.getLogger("cough_screener.cause_engine")


@dataclass
class AudioDescriptors:
    duration_sec: float
    is_wet: bool
    wet_probability: float
    bout_count: int
    harshness_score: float
    is_harsh: bool
    spectral_centroid: float
    zero_crossing_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_sec": round(self.duration_sec, 2),
            "is_wet": self.is_wet,
            "wet_label": "Wet / Productive" if self.is_wet else "Dry / Non-productive",
            "wet_probability": round(self.wet_probability, 2),
            "bout_count": self.bout_count,
            "harshness_score": round(self.harshness_score, 2),
            "is_harsh": self.is_harsh,
            "spectral_centroid": round(self.spectral_centroid, 1),
            "zero_crossing_rate": round(self.zero_crossing_rate, 4),
        }


def compute_audio_descriptors(samples: np.ndarray, sample_rate: int = 16000) -> AudioDescriptors:
    """
    Computes bio-acoustic descriptors from raw 16-bit PCM cough audio:
      - Dry vs. Wet classification (spectral envelope + low-band bubble energy)
      - Bout count (number of cough explosive bursts)
      - Harshness / Stridor (spectral flatness & high-mid ratio)
    """
    if len(samples) == 0:
        return AudioDescriptors(0.0, False, 0.0, 0, 0.0, False, 0.0, 0.0)

    duration_sec = len(samples) / float(sample_rate)
    float_audio = samples.astype(np.float32) / 32768.0

    # 1. Zero Crossing Rate (ZCR)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(float_audio)))) / (2.0 * len(float_audio))

    # 2. Spectral Analysis (FFT)
    n_fft = 1024
    hop_length = 512
    window = np.hanning(n_fft)

    num_frames = max(1, (len(float_audio) - n_fft) // hop_length + 1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    spectra = []
    centroids = []
    low_band_energies = []   # 200 Hz - 1200 Hz (mucus/wet bubble resonances)
    high_band_energies = []  # 2000 Hz - 4500 Hz (dry friction & stridor)

    for i in range(num_frames):
        start = i * hop_length
        frame = float_audio[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        mag = np.abs(np.fft.rfft(frame * window))
        power = mag ** 2
        total_p = np.sum(power) + 1e-9

        # Spectral centroid
        centroid = np.sum(freqs * power) / total_p
        centroids.append(centroid)

        # Band energies
        low_mask = (freqs >= 200) & (freqs <= 1200)
        high_mask = (freqs >= 2000) & (freqs <= 4500)
        low_band_energies.append(np.sum(power[low_mask]) / total_p)
        high_band_energies.append(np.sum(power[high_mask]) / total_p)

        spectra.append(power)

    avg_centroid = float(np.mean(centroids)) if centroids else 1500.0
    avg_low_ratio = float(np.mean(low_band_energies)) if low_band_energies else 0.4
    avg_high_ratio = float(np.mean(high_band_energies)) if high_band_energies else 0.2

    # 3. Dry vs Wet Cough Classification
    # Wet coughs exhibit greater low-frequency power ratio and lower zero-crossing density
    wet_metric = (avg_low_ratio / (avg_high_ratio + 1e-6)) * (0.1 / (zero_crossings + 1e-4))
    # Normalize wet probability using sigmoid curve
    wet_probability = float(1.0 / (1.0 + np.exp(-(wet_metric - 1.8) * 1.5)))
    is_wet = wet_probability >= 0.50

    # 4. Harshness & Stridor (concentration in 1.5-4.0 kHz + spectral variance)
    harshness_score = float(np.clip((avg_high_ratio * 2.8) + (avg_centroid / 4000.0) * 0.4, 0.0, 1.0))
    is_harsh = harshness_score >= 0.45

    # 5. Bout Count (detect distinct energy peaks in smoothed amplitude envelope)
    frame_len_ms = 40
    frame_samples = int(sample_rate * (frame_len_ms / 1000.0))
    n_blocks = len(float_audio) // frame_samples
    if n_blocks > 2:
        envelope = np.array([np.sqrt(np.mean(float_audio[i*frame_samples:(i+1)*frame_samples]**2)) for i in range(n_blocks)])
        max_env = np.max(envelope) + 1e-6
        norm_env = envelope / max_env
        # Count peaks above 35% of max
        peaks = 0
        in_peak = False
        for val in norm_env:
            if val >= 0.35:
                if not in_peak:
                    peaks += 1
                    in_peak = True
            else:
                in_peak = False
        bout_count = max(1, peaks)
    else:
        bout_count = 1

    return AudioDescriptors(
        duration_sec=duration_sec,
        is_wet=is_wet,
        wet_probability=wet_probability,
        bout_count=bout_count,
        harshness_score=harshness_score,
        is_harsh=is_harsh,
        spectral_centroid=avg_centroid,
        zero_crossing_rate=float(zero_crossings),
    )


class ClinicalCauseEngine:
    """
    Weighted Multimodal Scorer implementing the MIT Paper's explainable diagnosis methodology.
    Combines acoustic neural logits, digital signal processing descriptors, and symptom quiz data.
    """

    def __init__(self, rules_file: Optional[Path] = None):
        self.rules_file = rules_file or (Path(__file__).parent / "cause_rules.yaml")
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict:
        if self.rules_file.exists():
            try:
                with open(self.rules_file, "r") as f:
                    data = yaml.safe_load(f)
                    return data.get("causes", {})
            except Exception as e:
                logger.error(f"Error reading cause rules from {self.rules_file}: {e}")
        return self._default_rules()

    def evaluate(
        self,
        ai_probs: Dict[str, float],
        descriptors: AudioDescriptors,
        symptoms: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Calculates ranked matching probabilities for all clinical causes.
        Returns sorted list of dictionaries with name, percentage, and matching reasons.
        """
        results = []

        # Extract parsed symptom flags
        fever = bool(symptoms.get("fever", False))
        wheezing = bool(symptoms.get("wheezing", False))
        worse_night = bool(symptoms.get("worse_at_night", False))
        duration_cat = symptoms.get("duration", "recent")  # recent, moderate, chronic
        smoker = bool(symptoms.get("smoker", False))
        heartburn = bool(symptoms.get("heartburn", False))
        allergy = bool(symptoms.get("allergy", False))
        phlegm = symptoms.get("phlegm", "none_clear")      # none_clear, yellow_green, white

        p_suspect = float(ai_probs.get("cough_suspect", 0.0))
        p_healthy = float(ai_probs.get("healthy", 0.0))
        p_nocough = float(ai_probs.get("no_cough", 0.0))

        # Rejection guard: if no cough was detected, causes are minimized
        if p_nocough > 0.70 and p_suspect < 0.20:
            return [{
                "cause_id": "no_active_cough",
                "name": "No Cough Detected / Ambient Noise",
                "subtext": "Normal Non-Cough Acoustic Profile",
                "probability": round(p_nocough * 100, 1),
                "severity": "low",
                "evidence": ["Acoustic signature matches ambient noise or non-cough speech"],
            }]

        for cause_id, rule in self.rules.items():
            score = float(rule.get("base_weight", 0.1))
            evidence_list = []
            ev_desc = rule.get("evidence_descriptions", {})

            # --- 1. Evaluate AI Audio Triggers ---
            audio_trig = rule.get("audio_triggers", {})
            if "cough_suspect_prob" in audio_trig:
                w = audio_trig["cough_suspect_prob"]
                score += w * p_suspect
                if p_suspect >= 0.40 and "cough_suspect" in ev_desc:
                    evidence_list.append(ev_desc["cough_suspect"])

            if "wet_cough" in audio_trig:
                if descriptors.is_wet:
                    score += audio_trig["wet_cough"] * descriptors.wet_probability
                    if "wet_cough" in ev_desc:
                        evidence_list.append(ev_desc["wet_cough"])

            if "dry_cough" in audio_trig:
                if not descriptors.is_wet:
                    score += audio_trig["dry_cough"] * (1.0 - descriptors.wet_probability)
                    if "dry_cough" in ev_desc:
                        evidence_list.append(ev_desc["dry_cough"])

            if "harshness_high" in audio_trig and descriptors.is_harsh:
                score += audio_trig["harshness_high"] * descriptors.harshness_score
                evidence_list.append("Harsh stridor acoustic frequency concentration")

            if "long_bout" in audio_trig and descriptors.bout_count >= 2:
                score += audio_trig["long_bout"]
                if "long_bout" in ev_desc:
                    evidence_list.append(ev_desc["long_bout"])

            if "brief_duration" in audio_trig and descriptors.duration_sec < 1.8:
                score += audio_trig["brief_duration"]
                if "brief_duration" in ev_desc:
                    evidence_list.append(ev_desc["brief_duration"])

            # --- 2. Evaluate Symptom Triggers ---
            symp_trig = rule.get("symptom_triggers", {})

            if fever:
                if "fever" in symp_trig:
                    score += symp_trig["fever"]
                    if "fever" in ev_desc:
                        evidence_list.append(ev_desc["fever"])
                elif "no_fever" in symp_trig:
                    score -= 0.25  # Incompatible with non-fever causes

            if not fever and "no_fever" in symp_trig:
                score += symp_trig["no_fever"]
                if "no_fever" in ev_desc:
                    evidence_list.append(ev_desc["no_fever"])

            if wheezing and "wheezing" in symp_trig:
                score += symp_trig["wheezing"]
                if "wheezing" in ev_desc:
                    evidence_list.append(ev_desc["wheezing"])

            if worse_night and "worse_at_night" in symp_trig:
                score += symp_trig["worse_at_night"]
                if "worse_at_night" in ev_desc:
                    evidence_list.append(ev_desc["worse_at_night"])

            if heartburn and "heartburn_acidity" in symp_trig:
                score += symp_trig["heartburn_acidity"]
                if "heartburn_acidity" in ev_desc:
                    evidence_list.append(ev_desc["heartburn_acidity"])

            if smoker and "smoker" in symp_trig:
                score += symp_trig["smoker"]
                if "smoker" in ev_desc:
                    evidence_list.append(ev_desc["smoker"])

            if allergy and "seasonal_allergy" in symp_trig:
                score += symp_trig["seasonal_allergy"]
                if "seasonal_allergy" in ev_desc:
                    evidence_list.append(ev_desc["seasonal_allergy"])

            # Duration matching
            if duration_cat == "recent" and "onset_recent" in symp_trig:
                score += symp_trig["onset_recent"]
                if "onset_recent" in ev_desc:
                    evidence_list.append(ev_desc["onset_recent"])
            elif duration_cat == "chronic" and "chronic_duration" in symp_trig:
                score += symp_trig["chronic_duration"]
                if "chronic_duration" in ev_desc:
                    evidence_list.append(ev_desc["chronic_duration"])

            # Phlegm matching
            if phlegm == "yellow_green" and "phlegm_yellow_green" in symp_trig:
                score += symp_trig["phlegm_yellow_green"]
                evidence_list.append("Purulent/colored phlegm reported")
            elif phlegm == "none_clear" and "phlegm_clear_or_none" in symp_trig:
                score += symp_trig["phlegm_clear_or_none"]

            # Clamp and convert to human percentage
            final_prob = float(np.clip(score, 0.05, 0.95))
            pct = round(final_prob * 100, 1)

            # Determine triage severity tag
            severity = "low"
            if pct >= 70 or (cause_id == "viral_infection" and fever and p_suspect > 0.5):
                severity = "high"
            elif pct >= 45:
                severity = "moderate"

            results.append({
                "cause_id": cause_id,
                "name": rule.get("name", cause_id),
                "subtext": rule.get("subtext", ""),
                "probability": pct,
                "severity": severity,
                "evidence": evidence_list or ["Consistent acoustic baseline parameters"],
            })

        # Sort descending by matched probability
        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    def _default_rules(self) -> Dict:
        """Fallback in case YAML file is missing."""
        return {
            "viral_infection": {
                "name": "Viral Respiratory Infection",
                "subtext": "COVID-19 / Influenza",
                "base_weight": 0.2,
            },
            "asthma_reactive": {
                "name": "Asthma / Reactive Airway",
                "subtext": "Bronchospasm",
                "base_weight": 0.15,
            },
            "dry_air_irritation": {
                "name": "Throat Irritation / Dry Air",
                "subtext": "Transient irritation",
                "base_weight": 0.15,
            },
        }


# Global instance
cause_engine = ClinicalCauseEngine()
