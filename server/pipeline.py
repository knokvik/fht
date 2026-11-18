"""
Real-time Audio Processing Pipeline, Ring Buffer, Sound Event Detector (VAD),
Feature Extraction, and Inference Dispatcher.
"""

import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile

from server.config import config
from server.model import CoughClassifierEngine
from server.cause_engine import compute_audio_descriptors, cause_engine

logger = logging.getLogger("cough_screener.pipeline")


class AudioRingBuffer:
    """
    Thread-safe circular ring buffer for 16-bit PCM mono audio.
    Maintains a rolling window of the last N seconds of audio.
    """

    def __init__(self, capacity_samples: int):
        self.capacity = capacity_samples
        self.buffer = np.zeros(self.capacity, dtype=np.int16)
        self.write_idx = 0
        self.total_samples_written = 0
        self.lock = threading.Lock()

    def append(self, pcm_bytes: bytes) -> int:
        """
        Appends raw byte buffer of int16 samples into the ring buffer.
        Returns number of samples written.
        """
        if not pcm_bytes:
            return 0

        # Little-endian int16
        new_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        n = len(new_samples)
        if n == 0:
            return 0

        with self.lock:
            if n >= self.capacity:
                # Overwrite entire buffer with the most recent slice
                self.buffer[:] = new_samples[-self.capacity:]
                self.write_idx = 0
            else:
                space_at_end = self.capacity - self.write_idx
                if n <= space_at_end:
                    self.buffer[self.write_idx:self.write_idx + n] = new_samples
                    self.write_idx = (self.write_idx + n) % self.capacity
                else:
                    self.buffer[self.write_idx:] = new_samples[:space_at_end]
                    overflow = n - space_at_end
                    self.buffer[:overflow] = new_samples[space_at_end:]
                    self.write_idx = overflow

            self.total_samples_written += n

        return n

    def get_latest(self, num_samples: int) -> np.ndarray:
        """
        Retrieves the latest `num_samples` in chronological order.
        If fewer samples have been written, pads with zeros at the beginning.
        """
        num_samples = min(num_samples, self.capacity)
        result = np.zeros(num_samples, dtype=np.int16)

        with self.lock:
            if self.total_samples_written == 0:
                return result

            available = min(self.total_samples_written, self.capacity)
            to_copy = min(num_samples, available)

            curr_idx = self.write_idx
            start_idx = (curr_idx - to_copy) % self.capacity

            if start_idx + to_copy <= self.capacity:
                result[-to_copy:] = self.buffer[start_idx:start_idx + to_copy]
            else:
                first_part = self.capacity - start_idx
                second_part = to_copy - first_part
                result[-to_copy:-to_copy + first_part] = self.buffer[start_idx:]
                result[-second_part:] = self.buffer[:second_part]

        return result


def compute_rms(samples: np.ndarray) -> float:
    """Computes Root Mean Square (RMS) energy of an integer or float sample array."""
    if len(samples) == 0:
        return 0.0
    float_samples = samples.astype(np.float64)
    mean_square = np.mean(float_samples ** 2)
    return float(np.sqrt(mean_square))


def extract_log_mel_spectrogram(
    audio_pcm: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 256,
    f_min: float = 50.0,
    f_max: float = 8000.0,
    target_frames: int = 128,
) -> np.ndarray:
    """
    Extracts 64-band Log-Mel Spectrogram and resamples/pads/crops to target_frames.
    Returns: (n_mels, target_frames) 2D float32 array.
    """
    # Convert to normalized float [-1.0, 1.0]
    audio_float = audio_pcm.astype(np.float32) / 32768.0

    try:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio_float,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
            power=2.0,
        )
        # Convert power to dB (log scale)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    except ImportError:
        # Fallback pure-numpy spectrogram if librosa is not yet installed
        window = np.hanning(n_fft)
        num_frames = max(1, (len(audio_float) - n_fft) // hop_length + 1)
        frames = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_length
            segment = audio_float[start:start + n_fft]
            if len(segment) < n_fft:
                segment = np.pad(segment, (0, n_fft - len(segment)))
            fft_mag = np.abs(np.fft.rfft(segment * window)) ** 2
            frames[:, i] = fft_mag

        # Rough binning into n_mels bands
        mel_indices = np.linspace(0, frames.shape[0] - 1, n_mels + 1, dtype=int)
        log_mel = np.zeros((n_mels, num_frames), dtype=np.float32)
        for m in range(n_mels):
            b_start, b_end = mel_indices[m], max(mel_indices[m] + 1, mel_indices[m + 1])
            log_mel[m, :] = np.mean(frames[b_start:b_end, :], axis=0)
        log_mel = 10.0 * np.log10(np.maximum(log_mel, 1e-9))
        log_mel = log_mel - np.max(log_mel)

    # Align time dimension to target_frames (128)
    curr_frames = log_mel.shape[1]
    if curr_frames < target_frames:
        pad_width = target_frames - curr_frames
        # Pad with minimum floor value (silence)
        min_val = np.min(log_mel)
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=min_val)
    elif curr_frames > target_frames:
        # Center crop the most energetic portion
        start_frame = (curr_frames - target_frames) // 2
        log_mel = log_mel[:, start_frame:start_frame + target_frames]

    return log_mel.astype(np.float32)


class AudioPipeline:
    """
    Main Audio Processing Pipeline:
    - Maintains rolling 3s ring buffer of incoming ESP32 audio chunks
    - Runs adaptive RMS + MAD Voice/Sound Activity Detection (VAD)
    - Captures discrete sound events with hangover padding
    - Runs Log-Mel feature extraction + CNN inference
    - Saves event WAV and JSON metadata
    - Broadcasts real-time level and event predictions via WebSockets
    """

    def __init__(self, classifier: Optional[CoughClassifierEngine] = None):
        capacity = int(config.SAMPLE_RATE * config.RING_BUFFER_SECONDS)
        self.ring_buffer = AudioRingBuffer(capacity)
        self.classifier = classifier or CoughClassifierEngine()

        # Noise floor history (running RMS measurements)
        self.rms_history: Deque[float] = deque(maxlen=config.NOISE_FLOOR_HISTORY_LEN)
        for _ in range(20):
            self.rms_history.append(config.MIN_RMS_THRESHOLD)

        # State machine
        self.in_event = False
        self.event_samples: List[np.ndarray] = []
        self.event_start_time: float = 0.0
        self.silence_timer_start: Optional[float] = None
        self.current_event_peak: float = 0.0

        # Broadcast callbacks
        self.level_subscribers: List[Callable[[Dict], None]] = []
        self.event_subscribers: List[Callable[[Dict], None]] = []

        # Manual Recording Session state
        self.manual_session_active = False
        self.manual_session_start_time = 0.0
        self.manual_session_end_time = 0.0
        self.manual_session_samples: List[np.ndarray] = []
        self.manual_session_duration = 0.0
        self._pending_symptoms: Dict = {}
        self._pending_finalize = False

        # Background worker control
        self.running = False
        self._worker_task: Optional[asyncio.Task] = None

        # Ensure events directory exists
        config.EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    def feed_pcm(self, pcm_bytes: bytes):
        """Feeds binary 16-bit PCM chunk from ESP32 into the ring buffer."""
        self.ring_buffer.append(pcm_bytes)

    def subscribe_level(self, callback: Callable[[Dict], None]):
        self.level_subscribers.append(callback)

    def subscribe_event(self, callback: Callable[[Dict], None]):
        self.event_subscribers.append(callback)

    def unsubscribe_level(self, callback: Callable[[Dict], None]):
        if callback in self.level_subscribers:
            self.level_subscribers.remove(callback)

    def unsubscribe_event(self, callback: Callable[[Dict], None]):
        if callback in self.event_subscribers:
            self.event_subscribers.remove(callback)

    def _compute_adaptive_threshold(self) -> Tuple[float, float]:
        """
        Computes running median and Median Absolute Deviation (MAD).
        Returns: (noise_floor_median, adaptive_threshold)
        """
        arr = np.array(self.rms_history, dtype=np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        threshold = med + config.MAD_MULTIPLIER * mad
        threshold = max(threshold, config.MIN_RMS_THRESHOLD)
        return med, threshold

    async def start(self):
        """Starts the asynchronous 100ms analysis loop."""
        if self.running:
            return
        self.running = True
        self._worker_task = asyncio.create_task(self._analysis_loop())
        logger.info("[PIPELINE] Audio analysis loop started (100 ms interval).")

    async def stop(self):
        """Stops the analysis loop."""
        self.running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[PIPELINE] Audio analysis loop stopped.")

    def start_manual_session(self):
        """Starts an open-ended manual recording session (user clicks Stop to end)."""
        self.manual_session_active = True
        self.manual_session_start_time = time.time()
        self.manual_session_end_time = time.time() + 30.0  # 30s safety guard
        # Seed with initial attack buffer from ring buffer
        pre_padding = int(config.SAMPLE_RATE * 0.1)
        self.manual_session_samples = [self.ring_buffer.get_latest(pre_padding)]
        logger.info("[SESSION] Manual recording session started (open-ended, max 30s).")

    def stop_manual_session(self, symptoms: dict = None):
        """Stops the manual recording session and triggers finalization.
        Returns True if a session was active and stopped."""
        if not self.manual_session_active:
            logger.warning("[SESSION] stop_manual_session called but no session is active.")
            return False
        self.manual_session_active = False
        self.manual_session_duration = time.time() - self.manual_session_start_time
        self._pending_symptoms = symptoms or {}
        logger.info(f"[SESSION] Manual recording stopped after {self.manual_session_duration:.2f}s.")
        return True

    async def _analysis_loop(self):
        """Processes 100ms audio windows periodically."""
        samples_per_frame = int(config.SAMPLE_RATE * config.ANALYSIS_INTERVAL_SEC)  # 1600 samples

        while self.running:
            loop_start = time.time()
            now = loop_start

            # Retrieve latest 100ms frame from ring buffer
            frame_samples = self.ring_buffer.get_latest(samples_per_frame)
            frame_rms = compute_rms(frame_samples)

            # Update noise floor history when not actively inside an event
            if not self.in_event:
                self.rms_history.append(frame_rms)

            noise_floor, threshold = self._compute_adaptive_threshold()

            # Handle Manual Session
            if self.manual_session_active:
                self.manual_session_samples.append(frame_samples)
                # Safety guard: auto-stop after 30s
                if now >= self.manual_session_end_time:
                    logger.warning("[SESSION] Safety guard: auto-stopping after 30s.")
                    self.manual_session_active = False
                    self.manual_session_duration = now - self.manual_session_start_time
                    self._pending_symptoms = {}
                    self._pending_finalize = True

            if not self.manual_session_active and not self._pending_finalize:
                # Background ambient VAD tracking
                is_above_threshold = frame_rms >= threshold
                await self._process_vad_step(frame_samples, frame_rms, is_above_threshold, noise_floor, threshold)

            # Check if finalization was requested (by stop_manual_session or safety guard)
            if self._pending_finalize or (not self.manual_session_active and len(self.manual_session_samples) > 1):
                # Only finalize if we actually have samples to process
                if len(self.manual_session_samples) > 1:
                    self._pending_finalize = False
                    await self._finalize_manual_session()

            # Broadcast lightweight level telemetry
            level_msg = {
                "type": "level",
                "rms": round(frame_rms, 1),
                "noise_floor": round(noise_floor, 1),
                "threshold": round(threshold, 1),
                "is_event": self.in_event or self.manual_session_active,
                "is_recording_session": self.manual_session_active,
                "timestamp": time.time(),
            }
            for cb in list(self.level_subscribers):
                try:
                    cb(level_msg)
                except Exception as e:
                    logger.debug(f"Error notifying level subscriber: {e}")

            # Sleep remaining time to maintain 100ms cadence
            elapsed = time.time() - loop_start
            sleep_time = max(0.01, config.ANALYSIS_INTERVAL_SEC - elapsed)
            await asyncio.sleep(sleep_time)

    async def _finalize_manual_session(self):
        """Processes audio recorded during the manual session, runs CNN + cause engine, and broadcasts."""
        all_samples = np.concatenate(self.manual_session_samples) if self.manual_session_samples else np.array([], dtype=np.int16)
        self.manual_session_samples = []
        duration = self.manual_session_duration
        symptoms = self._pending_symptoms
        self._pending_symptoms = {}

        logger.info(f"[SESSION] Finalizing manual recording session ({len(all_samples)} samples, {duration:.2f}s).")

        if len(all_samples) < config.SAMPLE_RATE * 0.3:
            logger.warning("[SESSION] Recording too short (<0.3s). Discarding.")
            error_payload = {
                "type": "session_error",
                "message": "Recording too short. Please record for at least 1 second.",
            }
            for cb in list(self.event_subscribers):
                try:
                    cb(error_payload)
                except Exception:
                    pass
            return

        try:
            mel_spec = extract_log_mel_spectrogram(
                all_samples,
                sample_rate=config.SAMPLE_RATE,
                n_mels=config.N_MELS,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                f_min=config.F_MIN,
                f_max=config.F_MAX,
                target_frames=config.TARGET_SPEC_FRAMES,
            )

            label, confidence, prob_dict = self.classifier.predict(mel_spec)
            waveform_downsampled = self._downsample_waveform(all_samples, target_points=200)
            spectrogram_ui = self._downsample_spectrogram(mel_spec, target_h=64, target_w=64)

            # Compute audio descriptors (dry/wet, bout count, harshness)
            descriptors = compute_audio_descriptors(all_samples, config.SAMPLE_RATE)
            descriptors_dict = descriptors.to_dict()

            # Run Cause Engine
            causes = cause_engine.evaluate(prob_dict, descriptors, symptoms)

            timestamp_iso = datetime.now(timezone.utc).isoformat()
            file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            wav_filename = f"session_{file_stamp}.wav"
            json_filename = f"session_{file_stamp}.json"

            wav_path = config.EVENTS_DIR / wav_filename
            json_path = config.EVENTS_DIR / json_filename

            wavfile.write(str(wav_path), config.SAMPLE_RATE, all_samples)

            peak_rms = compute_rms(all_samples)
            sidecar_data = {
                "timestamp": timestamp_iso,
                "label": label,
                "confidence": confidence,
                "probs": prob_dict,
                "duration_sec": round(duration, 3),
                "peak_rms": round(peak_rms, 1),
                "sample_rate": config.SAMPLE_RATE,
                "num_samples": len(all_samples),
                "wav_file": wav_filename,
                "is_session": True,
                "is_dummy_model": getattr(self.classifier, "is_dummy", False),
                "descriptors": descriptors_dict,
                "symptoms": symptoms,
                "causes": causes,
            }
            with open(json_path, "w") as f:
                json.dump(sidecar_data, f, indent=2)

            logger.info(
                f"[PREDICTION] Label: {label.upper()} | Conf: {confidence * 100:.1f}% | "
                f"Descriptors: {'Wet' if descriptors.is_wet else 'Dry'}, {descriptors.bout_count} bout(s) | "
                f"Top Cause: {causes[0]['name'] if causes else 'N/A'}"
            )

            event_payload = {
                "type": "prediction",
                "is_session": True,
                "label": label,
                "confidence": round(confidence, 4),
                "probs": prob_dict,
                "timestamp": timestamp_iso,
                "duration_sec": round(duration, 2),
                "peak_rms": round(peak_rms, 1),
                "wav_file": wav_filename,
                "is_dummy_model": getattr(self.classifier, "is_dummy", False),
                "waveform": waveform_downsampled,
                "spectrogram": spectrogram_ui,
                "descriptors": descriptors_dict,
                "causes": causes,
            }

            for cb in list(self.event_subscribers):
                try:
                    cb(event_payload)
                except Exception as e:
                    logger.error(f"Error notifying event subscriber: {e}")

        except Exception as e:
            logger.error(f"[SESSION ERROR] Error in manual session inference: {e}", exc_info=True)

    async def _process_vad_step(
        self,
        frame_samples: np.ndarray,
        frame_rms: float,
        is_above: bool,
        noise_floor: float,
        threshold: float,
    ):
        now = time.time()

        if not self.in_event:
            if is_above:
                # Trigger Event Onset!
                self.in_event = True
                self.event_start_time = now
                self.silence_timer_start = None
                self.current_event_peak = frame_rms

                # Pre-event buffer padding to preserve attack transient
                pre_padding_samples = int(config.SAMPLE_RATE * config.PRE_EVENT_PADDING_SEC)
                pre_buffer = self.ring_buffer.get_latest(pre_padding_samples)
                self.event_samples = [pre_buffer, frame_samples]
                logger.info(f"[VAD] Sound event onset triggered (RMS: {frame_rms:.1f} >= {threshold:.1f})")
        else:
            # Currently capturing an active event
            self.event_samples.append(frame_samples)
            if frame_rms > self.current_event_peak:
                self.current_event_peak = frame_rms

            if not is_above:
                # Frame dropped below threshold -> start or advance silence hangover
                if self.silence_timer_start is None:
                    self.silence_timer_start = now
                elif now - self.silence_timer_start >= config.SILENCE_HANGOVER_SEC:
                    # Silence hangover completed -> Finish event
                    await self._finalize_event(now)
            else:
                # Sound rose above threshold again -> reset silence hangover
                self.silence_timer_start = None

            # Hard cutoff check: Discard or truncate if event is absurdly long
            event_duration = now - self.event_start_time
            if event_duration > (config.MAX_EVENT_DURATION_SEC + 0.6):
                logger.warning(f"[VAD] Event exceeded maximum duration ({event_duration:.2f}s). Discarding.")
                self.in_event = False
                self.event_samples = []
                self.silence_timer_start = None

    async def _finalize_event(self, end_time: float):
        """Validates duration, extracts features, runs CNN, and broadcasts prediction."""
        self.in_event = False
        duration = end_time - self.event_start_time - config.SILENCE_HANGOVER_SEC
        all_samples = np.concatenate(self.event_samples) if self.event_samples else np.array([], dtype=np.int16)
        self.event_samples = []
        self.silence_timer_start = None

        if duration < config.MIN_EVENT_DURATION_SEC:
            logger.info(f"[VAD] Discarded transient spike: {duration:.2f}s < {config.MIN_EVENT_DURATION_SEC}s")
            return

        if duration > config.MAX_EVENT_DURATION_SEC:
            logger.info(f"[VAD] Discarded continuous sound: {duration:.2f}s > {config.MAX_EVENT_DURATION_SEC}s")
            return

        logger.info(f"[VAD] Valid sound event captured: {duration:.2f}s, Peak RMS: {self.current_event_peak:.1f}")

        # Run feature extraction and inference
        try:
            mel_spec = extract_log_mel_spectrogram(
                all_samples,
                sample_rate=config.SAMPLE_RATE,
                n_mels=config.N_MELS,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                f_min=config.F_MIN,
                f_max=config.F_MAX,
                target_frames=config.TARGET_SPEC_FRAMES,
            )

            # Predict
            label, confidence, prob_dict = self.classifier.predict(mel_spec)

            # Downsample waveform for UI rendering (200 points float [-1.0, 1.0])
            waveform_downsampled = self._downsample_waveform(all_samples, target_points=200)

            # Downsample spectrogram for UI visualization (64 x 64 matrix, scaled 0 to 1)
            spectrogram_ui = self._downsample_spectrogram(mel_spec, target_h=64, target_w=64)

            timestamp_iso = datetime.now(timezone.utc).isoformat()
            file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            wav_filename = f"event_{file_stamp}.wav"
            json_filename = f"event_{file_stamp}.json"

            wav_path = config.EVENTS_DIR / wav_filename
            json_path = config.EVENTS_DIR / json_filename

            # Save event WAV file (16 kHz, 16-bit mono)
            wavfile.write(str(wav_path), config.SAMPLE_RATE, all_samples)

            # Save JSON sidecar
            sidecar_data = {
                "timestamp": timestamp_iso,
                "label": label,
                "confidence": confidence,
                "probs": prob_dict,
                "duration_sec": round(duration, 3),
                "peak_rms": round(self.current_event_peak, 1),
                "sample_rate": config.SAMPLE_RATE,
                "num_samples": len(all_samples),
                "wav_file": wav_filename,
                "is_dummy_model": getattr(self.classifier, "is_dummy", False),
            }
            with open(json_path, "w") as f:
                json.dump(sidecar_data, f, indent=2)

            logger.info(
                f"[PREDICTION] Label: {label.upper()} | Conf: {confidence * 100:.1f}% | "
                f"Probs: {prob_dict} | Saved: {wav_filename}"
            )

            # Construct broadcast payload
            event_payload = {
                "type": "prediction",
                "label": label,
                "confidence": round(confidence, 4),
                "probs": prob_dict,
                "timestamp": timestamp_iso,
                "duration_sec": round(duration, 2),
                "peak_rms": round(self.current_event_peak, 1),
                "wav_file": wav_filename,
                "is_dummy_model": getattr(self.classifier, "is_dummy", False),
                "waveform": waveform_downsampled,
                "spectrogram": spectrogram_ui,
            }

            for cb in list(self.event_subscribers):
                try:
                    cb(event_payload)
                except Exception as e:
                    logger.error(f"Error notifying event subscriber: {e}")

        except Exception as e:
            logger.error(f"[PIPELINE ERROR] Failed processing sound event: {e}", exc_info=True)

    @staticmethod
    def _downsample_waveform(samples: np.ndarray, target_points: int = 200) -> List[float]:
        """Downsamples integer audio samples to target_points normalized float values."""
        if len(samples) == 0:
            return [0.0] * target_points

        float_samples = samples.astype(np.float32) / 32768.0
        if len(float_samples) <= target_points:
            return [round(float(v), 3) for v in float_samples]

        indices = np.linspace(0, len(float_samples) - 1, target_points, dtype=int)
        downsampled = float_samples[indices]
        return [round(float(v), 3) for v in downsampled]

    @staticmethod
    def _downsample_spectrogram(
        mel_spec: np.ndarray, target_h: int = 64, target_w: int = 64
    ) -> List[List[float]]:
        """
        Normalizes and downsamples 2D log-mel matrix into a 0.0 - 1.0 grid for Canvas rendering.
        """
        # Normalize spec to [0, 1] range for visual heatmap
        s_min = np.min(mel_spec)
        s_max = np.max(mel_spec)
        norm = (mel_spec - s_min) / (s_max - s_min + 1e-6)

        # Simple 2D grid resampling
        h_indices = np.linspace(0, norm.shape[0] - 1, target_h, dtype=int)
        w_indices = np.linspace(0, norm.shape[1] - 1, target_w, dtype=int)
        grid = norm[np.ix_(h_indices, w_indices)]

        # Convert to list of lists rounded to 2 decimals
        return [[round(float(val), 2) for val in row] for row in grid]
