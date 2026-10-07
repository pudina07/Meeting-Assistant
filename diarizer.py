"""
diarizer.py  --  OPTIONAL stage between STT and refinement: "who said it?"

Model : pyannote/speaker-diarization-3.1 (Hugging Face; gated - see README steps below)
Input : the original audio file + the TranscriptionResult from transcriber.py
Output: the same TranscriptionResult, but every segment now carries `speaker` ("Speaker 1", ...)

Why it matters
--------------
Action items are almost always phrased relative to the speaker: "I will send the report by Friday".
Without knowing WHO said "I", the extractor can only answer "Unspecified". With speaker labels in
the transcript the extractor can ground the owner of every first-person commitment.

Speed (why this version is much faster on CPU)
----------------------------------------------
pyannote's cost is dominated by the speaker-EMBEDDING step: by default it slides a 10 s window over the
audio with a 1 s step and embeds every window (25 min of audio -> ~1500 windows x up to 3 speakers).
This module therefore:
  1. widens the sliding-window step (DIARIZATION_SPEED = fast | balanced | accurate)   -> 2.5-5x fewer embeddings
  2. raises the segmentation / embedding batch sizes
  3. pins torch to the CPU cores that are really available (and lets you override it)
  4. enforces a TIME BUDGET (DIARIZATION_TIME_BUDGET_S): if diarization would take too long on a slow
     server it is aborted cleanly and the pipeline continues WITHOUT speaker labels instead of hanging
  5. reports real progress through pyannote's hook, so the UI progress bar moves
  6. (in app.py) runs concurrently with the Groq transcription, so their times overlap.
On a GPU the pipeline automatically uses the most accurate settings and no time limit.

Environment variables (all optional)
------------------------------------
  HF_TOKEN / HUGGINGFACE_TOKEN     Hugging Face READ token (required)
  DIARIZATION_MODEL                default pyannote/speaker-diarization-3.1
  DIARIZATION_SPEED                fast (default on CPU) | balanced | accurate (default on GPU)
  DIARIZATION_TIME_BUDGET_S        CPU default 900 (15 min); 0 = no limit (GPU default)
  DIARIZATION_THREADS              torch CPU threads (default: usable cores, max 8)

One-time setup
--------------
1. pip install pyannote.audio torch            (Python >= 3.10; GPU optional but much faster)
2. Create a Hugging Face READ token: https://huggingface.co/settings/tokens
3. Open both pages while logged in and click "Agree and access repository":
      https://huggingface.co/pyannote/speaker-diarization-3.1
      https://huggingface.co/pyannote/segmentation-3.0
4. Put  HF_TOKEN=hf_xxx  in your .env / Streamlit secrets / host environment variables.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from transcriber import Segment, TranscriptionResult

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
SAMPLE_RATE = 16000

# sliding-window step as a fraction of the 10 s window (pyannote default = 0.1  -> 1 s step)
_SPEED_STEP = {"fast": 0.5, "balanced": 0.25, "accurate": 0.1}
_SPEED_BATCH = {"fast": 32, "balanced": 32, "accurate": 16}

ProgressCB = Optional[Callable[[float, str], None]]


class DiarizationError(Exception):
    """Raised with a message that is safe to show directly to the end user."""


class _BudgetExceeded(Exception):
    pass


@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    segments: list[DiarSegment]
    model: str
    device: str = "cpu"
    warnings: list[str] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for s in sorted(self.segments, key=lambda x: x.start):
            if s.speaker not in seen:
                seen.append(s.speaker)
        return seen


# --------------------------------------------------------------------------------------
# Availability / audio loading / pipeline
# --------------------------------------------------------------------------------------
def diarization_available() -> tuple[bool, str]:
    """(ok, reason). Cheap check that does not load any model."""
    try:
        import pyannote.audio  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:  # ImportError, or a broken torch install
        return False, f"pyannote.audio / torch are not installed ({type(e).__name__}). Run: pip install pyannote.audio torch"
    return True, ""


def _load_audio(path: Path):
    """Decode ANY audio/video file to a mono 16 kHz float32 torch tensor (1, T) using ffmpeg,
    so we never depend on torchaudio/torchcodec being able to read the container."""
    import numpy as np
    import torch

    if not shutil.which("ffmpeg"):
        raise DiarizationError("ffmpeg not found - install it first (it is also needed for transcription).")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise DiarizationError("Could not decode the audio for speaker analysis.")
    arr = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return {"waveform": torch.from_numpy(arr).unsqueeze(0), "sample_rate": SAMPLE_RATE}


_PIPELINE_CACHE: dict[tuple[str, str], object] = {}


def _load_pipeline(model: str, token: str):
    """Load (and cache) the pyannote pipeline. pyannote 3.x calls the kwarg `use_auth_token`,
    4.x calls it `token`, so we try both."""
    key = (model, token[-6:])
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise DiarizationError(
            f"pyannote.audio is not installed or failed to import ({type(e).__name__}: {e}). "
            "Run: pip install pyannote.audio torch"
        )
    pipeline = None
    err: Optional[Exception] = None
    for kw in ("token", "use_auth_token"):
        try:
            pipeline = Pipeline.from_pretrained(model, **{kw: token})
            err = None
            break
        except TypeError as e:          # wrong kwarg name for this pyannote version
            err = e
            continue
        except Exception as e:
            err = e
            break
    if pipeline is None:
        text = str(err or "")
        low = text.lower()
        if any(k in low for k in ("401", "403", "gated", "access", "authorized", "token", "restricted")) or err is None:
            raise DiarizationError(
                "Hugging Face refused access to the diarization model. Check that (1) HF_TOKEN is a valid READ token and "
                "(2) you clicked 'Agree and access repository' on BOTH https://huggingface.co/pyannote/speaker-diarization-3.1 "
                "and https://huggingface.co/pyannote/segmentation-3.0 with the same account."
            )
        raise DiarizationError(f"Could not load the diarization model: {type(err).__name__}: {text[:300]}")
    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass
    _PIPELINE_CACHE[key] = pipeline
    return pipeline


def _device_name() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------------------
# Speed tuning
# --------------------------------------------------------------------------------------
def _usable_cores() -> int:
    """Cores this process may really use (respects CPU affinity; os.cpu_count() can report the whole host)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def _resolve_speed(device: str) -> str:
    speed = os.getenv("DIARIZATION_SPEED", "").strip().lower()
    if speed in _SPEED_STEP:
        return speed
    return "accurate" if device == "cuda" else "fast"


def _resolve_budget(device: str) -> float:
    raw = os.getenv("DIARIZATION_TIME_BUDGET_S", "").strip()
    try:
        if raw:
            return max(0.0, float(raw))
    except ValueError:
        pass
    return 0.0 if device == "cuda" else 900.0


def _tune(pipeline, device: str) -> str:
    """Apply threads / window step / batch sizes. Every step is guarded: if this pyannote version does not
    expose an attribute we simply keep its default. Returns the speed profile used."""
    speed = _resolve_speed(device)
    try:
        import torch
        if device == "cpu":
            try:
                n = int(os.getenv("DIARIZATION_THREADS", "") or min(_usable_cores(), 8))
            except ValueError:
                n = min(_usable_cores(), 8)
            torch.set_num_threads(max(1, n))
    except Exception:
        pass

    batch = _SPEED_BATCH[speed]
    try:
        seg = getattr(pipeline, "_segmentation", None)
        if seg is not None:
            if hasattr(seg, "batch_size"):
                seg.batch_size = batch
            dur = getattr(seg, "duration", None)
            if dur and hasattr(seg, "step"):
                seg.step = float(_SPEED_STEP[speed]) * float(dur)      # fewer windows -> far fewer embeddings
        if hasattr(pipeline, "segmentation_step"):
            pipeline.segmentation_step = _SPEED_STEP[speed]
        for attr in ("embedding_batch_size", "segmentation_batch_size"):
            if hasattr(pipeline, attr):
                setattr(pipeline, attr, batch)
        emb = getattr(pipeline, "_embedding", None)
        if emb is not None and hasattr(emb, "batch_size"):
            emb.batch_size = batch
    except Exception as e:      # never let tuning break diarization
        logger.warning("Could not apply diarization speed settings (%s); using pyannote defaults.", e)
    return speed


def _make_hook(report: Callable[[float, str], None], deadline: Optional[float]):
    """pyannote progress hook: reports progress and enforces the time budget by raising."""
    # (start, end) share of the 0.3 -> 1.0 progress range per pipeline step
    shares = {"segmentation": (0.00, 0.35, "Detecting speech"),
              "speaker_counting": (0.35, 0.37, "Counting speakers"),
              "embeddings": (0.37, 0.95, "Comparing voices"),
              "discrete_diarization": (0.95, 1.00, "Assigning speakers")}

    def hook(step_name, step_artifact=None, file=None, total=None, completed=None):
        if deadline is not None and time.monotonic() > deadline:
            raise _BudgetExceeded()
        share = shares.get(step_name)
        if share and total and completed is not None:
            lo, hi, label = share
            frac = lo + (hi - lo) * min(1.0, completed / total)
            report(0.3 + 0.7 * frac, f"Speaker analysis: {label} ({int(100 * min(1.0, completed / total))}%)...")

    return hook


# --------------------------------------------------------------------------------------
# Step 1: run pyannote
# --------------------------------------------------------------------------------------
def diarize(
    audio_path: str | Path,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    model: Optional[str] = None,
    progress_cb: ProgressCB = None,
) -> DiarizationResult:
    report = progress_cb or (lambda f, m: None)
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise DiarizationError(
            "HF_TOKEN is missing. Add your Hugging Face read token to the app secrets / environment "
            "to enable speaker labels."
        )
    model = model or DEFAULT_DIARIZATION_MODEL
    path = Path(audio_path)
    if not path.exists():
        raise DiarizationError("Audio file not found for speaker analysis.")

    report(0.05, "Loading speaker model...")
    pipeline = _load_pipeline(model, token)
    device = _device_name()
    speed = _tune(pipeline, device)
    budget = _resolve_budget(device)
    report(0.2, "Decoding audio...")
    audio = _load_audio(path)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            kwargs["max_speakers"] = int(max_speakers)

    report(0.3, f"Finding who speaks when ({device}, {speed} mode)...")
    deadline = time.monotonic() + budget if budget > 0 else None     # clock starts when the heavy work starts
    hook = _make_hook(report, deadline)
    try:
        try:
            out = pipeline(audio, hook=hook, **kwargs)
        except TypeError as e:
            if "hook" not in str(e):
                raise
            out = pipeline(audio, **kwargs)             # a pyannote build without hook support
    except _BudgetExceeded:
        raise DiarizationError(
            f"Speaker analysis was stopped after {budget / 60:.0f} min because this server is too slow for it "
            f"(running on {device}). Transcript and minutes were still produced, just without speaker labels. "
            "Use a smaller 'Exact speakers' number, shorter audio, or a GPU host."
        )
    except Exception as e:
        raise DiarizationError(f"Speaker analysis failed: {type(e).__name__}: {str(e)[:300]}")

    annotation = getattr(out, "speaker_diarization", out)   # 4.x returns an object, 3.x an Annotation
    segs: list[DiarSegment] = []
    try:
        for turn, _, label in annotation.itertracks(yield_label=True):
            if turn.end - turn.start >= 0.05:
                segs.append(DiarSegment(float(turn.start), float(turn.end), str(label)))
    except Exception as e:
        raise DiarizationError(f"Unexpected diarization output ({type(e).__name__}).")
    if not segs:
        raise DiarizationError("No speech turns were found by the speaker model.")
    report(1.0, "Speaker analysis complete.")
    return DiarizationResult(sorted(segs, key=lambda s: s.start), model=model, device=device)


# --------------------------------------------------------------------------------------
# Step 2: merge speakers into the transcript
# --------------------------------------------------------------------------------------
def _speaker_for_span(a: float, b: float, starts: list[float], segs: list[DiarSegment],
                      max_gap: float = 1.5) -> Optional[str]:
    """Speaker with the largest overlap with [a, b]; else the nearest turn within max_gap seconds."""
    overlap: dict[str, float] = {}
    i = max(0, bisect_left(starts, a) - 8)
    # diarization turns can overlap, so scan a small window rather than a single index
    while i < len(segs) and segs[i].start <= b:
        s = segs[i]
        ov = min(b, s.end) - max(a, s.start)
        if ov > 0:
            overlap[s.speaker] = overlap.get(s.speaker, 0.0) + ov
        i += 1
    if overlap:
        return max(overlap.items(), key=lambda kv: kv[1])[0]
    mid = (a + b) / 2
    best, best_d = None, max_gap
    j = max(0, bisect_left(starts, mid) - 8)
    while j < len(segs) and segs[j].start - mid < max_gap:
        s = segs[j]
        d = 0.0 if s.start <= mid <= s.end else min(abs(s.start - mid), abs(s.end - mid))
        if d < best_d:
            best, best_d = s.speaker, d
        j += 1
    return best


def _smooth(labels: list[Optional[str]], spans: list[tuple[float, float]], max_words: int = 2,
            max_dur: float = 0.9) -> list[Optional[str]]:
    """Remove 1-2 word speaker 'flickers' sandwiched between the same speaker."""
    out = list(labels)
    # fill unknowns from neighbours first
    for i, l in enumerate(out):
        if l is None:
            prev = next((out[k] for k in range(i - 1, -1, -1) if out[k]), None)
            nxt = next((out[k] for k in range(i + 1, len(out)) if out[k]), None)
            out[i] = prev or nxt
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1] == out[i]:
            j += 1
        n = j - i + 1
        dur = spans[j][1] - spans[i][0]
        if 0 < i and j < len(out) - 1 and n <= max_words and dur <= max_dur and out[i - 1] == out[j + 1] != out[i]:
            for k in range(i, j + 1):
                out[k] = out[i - 1]
        i = j + 1
    return out


def apply_diarization(result: TranscriptionResult, diar: DiarizationResult) -> TranscriptionResult:
    """Label every segment with a speaker (splitting segments at speaker changes when word timings
    are available). Returns the same TranscriptionResult object, modified in place."""
    segs = sorted(diar.segments, key=lambda s: s.start)
    starts = [s.start for s in segs]

    new_segments: list[Segment] = []
    for seg in result.segments:
        tokens = seg.text.split()
        words = seg.words
        if words and len(words) == len(tokens) and len(words) > 1:
            labels = [_speaker_for_span(w.start, w.end, starts, segs) for w in words]
            spans = [(w.start, w.end) for w in words]
            labels = _smooth(labels, spans)
            # split into runs of the same speaker
            run_start = 0
            for k in range(1, len(words) + 1):
                if k == len(words) or labels[k] != labels[run_start]:
                    new_segments.append(Segment(
                        start=words[run_start].start, end=words[k - 1].end,
                        text=" ".join(tokens[run_start:k]), speaker=labels[run_start],
                        words=words[run_start:k], avg_logprob=seg.avg_logprob,
                    ))
                    run_start = k
        else:
            seg.speaker = _speaker_for_span(seg.start, seg.end, starts, segs)
            new_segments.append(seg)

    # carry the previous speaker over segments that fell in a gap
    last = None
    for s in new_segments:
        if s.speaker is None:
            s.speaker = last
        last = s.speaker or last
    first = next((s.speaker for s in new_segments if s.speaker), None)
    for s in new_segments:
        if s.speaker is None:
            s.speaker = first

    # rename by order of first appearance
    names: dict[str, str] = {}
    for s in new_segments:
        if s.speaker and s.speaker not in names:
            names[s.speaker] = f"Speaker {len(names) + 1}"
    for s in new_segments:
        s.speaker = names.get(s.speaker) if s.speaker else None

    result.segments = new_segments
    if not result.has_word_timings:
        result.warnings.append(
            "Word-level timings were not available, so speakers were assigned per Whisper segment "
            "(a speaker change in the middle of a segment may be missed)."
        )
    return result


def speaker_stats(result: TranscriptionResult) -> dict[str, dict]:
    """words / seconds / share per speaker - for display in the UI."""
    stats: dict[str, dict] = {}
    for s in result.segments:
        if not s.speaker:
            continue
        d = stats.setdefault(s.speaker, {"words": 0, "seconds": 0.0})
        d["words"] += len(s.text.split())
        d["seconds"] += max(0.0, s.end - s.start)
    total = sum(d["words"] for d in stats.values()) or 1
    for d in stats.values():
        d["seconds"] = round(d["seconds"], 1)
        d["share"] = round(d["words"] / total, 3)
    return stats
