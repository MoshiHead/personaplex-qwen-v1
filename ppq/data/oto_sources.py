"""Readers for the four gated otoearth HF datasets used as real conversation data.

Datasets (all gated - accept the terms on each Hugging Face page first):

  video-21h  otoearth/otoSpeech-video-full-duplex-21h
             webdataset tars: <sid>.flac (stereo 48k) + <sid>.json +
             <sid>.transcript_speaker_{1,2}.json (word timestamps, ms) + MP4s
             (the MP4s are NEVER fetched - range reads skip them)
  task-20h   otoearth/otoSpeech-full-duplex-task-oriented-20h  (CC BY 4.0)
             webdataset tars: <sid>.flac + <sid>.json (no transcripts)
  task-v2    otoearth/otoSpeech-full-duplex-task-oriented-v2   (CC BY 4.0)
             same layout as task-20h
  turn-104h  otoearth/otoSpeech-full-duplex-turn-104h
             per-session dirs: speaker_{1,2}_audio.wav (mono 48k float32),
             speaker_{1,2}_annotation_a.srt (labeled segments), metadata.json
             (combined_audio.wav is NEVER downloaded)
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .hf_tar import make_fetch, resolve_hf_url, tar_index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceSpec:
    name: str                # short id used in file/manifest names
    repo_id: str             # HF dataset repo
    kind: str                # "video" | "task" | "turn"
    license: str
    asr_mode: str            # "none" | "full" | "long_segments"
    voice_prompt_mode: str   # "none" | "same_speaker"
    license_note: str


SOURCES: dict[str, SourceSpec] = {s.name: s for s in [
    SourceSpec(
        name="video-21h",
        repo_id="otoearth/otoSpeech-video-full-duplex-21h",
        kind="video",
        license="otoSpeech Video Dataset License 1.0",
        asr_mode="none",  # word-timestamp transcripts included
        voice_prompt_mode="none",  # license forbids voice cloning
        license_note=(
            "No redistribution of the dataset or anything derived from it; do "
            "not identify or contact participants; do not clone voices or "
            "faces; respect redaction segments; models released to others must "
            "carry these restrictions; removed sessions must be deleted within "
            "30 days of notice (see REMOVALS.md in the repo)."),
    ),
    SourceSpec(
        name="task-20h",
        repo_id="otoearth/otoSpeech-full-duplex-task-oriented-20h",
        kind="task",
        license="CC BY 4.0",
        asr_mode="full",
        voice_prompt_mode="same_speaker",
        license_note=(
            "CC BY 4.0: attribution required (otoearth otoSpeech task-oriented "
            "datasets). Do not attempt to identify speakers; respect redaction "
            "segments."),
    ),
    SourceSpec(
        name="task-v2",
        repo_id="otoearth/otoSpeech-full-duplex-task-oriented-v2",
        kind="task",
        license="CC BY 4.0",
        asr_mode="full",
        voice_prompt_mode="same_speaker",
        license_note=(
            "CC BY 4.0: attribution required (otoearth otoSpeech task-oriented "
            "datasets). Do not attempt to identify speakers; respect redaction "
            "segments."),
    ),
    SourceSpec(
        name="turn-104h",
        repo_id="otoearth/otoSpeech-full-duplex-turn-104h",
        kind="turn",
        license="otoSpeech turn-104h license (research use)",
        asr_mode="long_segments",  # SRT text for short segments, Whisper for long ones
        voice_prompt_mode="none",  # Voice Identity clause: no per-person voice prompts
        license_note=(
            "Research use only. Voice Identity clause: do not clone or imitate "
            "a specific person's voice - samples for this source are built "
            "WITHOUT a voice prompt. Do not redistribute the data or "
            "derivatives."),
    ),
]}


def get_hf_token(env_var: str = "HF_TOKEN") -> str | None:
    """Token from the environment, else the huggingface-cli login. Never printed."""
    tok = os.environ.get(env_var)
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def safe_name(s: str) -> str:
    """Filesystem-safe id (keeps alphanumerics, '-' and '_')."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

def merge_intervals(intervals, gap_s: float = 0.0) -> list[tuple[float, float]]:
    """Sort and merge overlapping (or gap-bridged, when gap_s > 0) intervals."""
    out: list[tuple[float, float]] = []
    for s, e in sorted((float(s), float(e)) for s, e in intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1] + gap_s:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def intervals_from_words(words, gap_s: float = 0.3) -> list[tuple[float, float]]:
    return merge_intervals([(s, e) for _, s, e in words], gap_s=gap_s)


def normalize_redactions(raw) -> list[tuple[float, float]]:
    """Clean up redaction intervals: accept dicts (start_sec/end_sec or
    start/end) or pairs, swap reversed bounds (end < start occurs in the
    task exports), drop degenerate ones, merge overlaps."""
    out: list[tuple[float, float]] = []
    for r in raw or []:
        if isinstance(r, dict):
            s = r.get("start_sec", r.get("start"))
            e = r.get("end_sec", r.get("end"))
        else:
            s, e = r
        if s is None or e is None:
            continue
        s, e = float(s), float(e)
        if e < s:
            s, e = e, s
        if e <= s:
            continue
        out.append((s, e))
    return merge_intervals(out)


# ---------------------------------------------------------------------------
# turn-104h: SRT annotations
# ---------------------------------------------------------------------------

SRT_TIME = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
SRT_LABEL = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$", re.S)

# Labels that mean "this channel's speaker is vocally active". Channel Bleed
# is the OTHER speaker bleeding in, Awkward Silence / Non-Speech Noise are not
# speech - none of those count as channel activity.
TURN_SPEECH_LABELS = {
    "Normal Turn", "Bounded Response", "Strong Floor Hold",
    "Acknowledgement Backchannel", "Continuer Backchannel", "Reaction Backchannel",
    "Filler", "Laughter", "Overlap", "Speech, Non-Linguistic",
    "Floor-taking Competitive Interruption", "Floor-taking Cooperative Interruption",
    "Non-floor Taking Competitive Interruption",
    "Non-floor Taking Cooperative Interruption",
}


@dataclass
class SrtSegment:
    start_s: float
    end_s: float
    label: str
    text: str


def parse_srt(text: str) -> list[SrtSegment]:
    """Parse a SubRip file with '[Label] transcript' payload lines."""
    out: list[SrtSegment] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        m, ti = None, -1
        for i, ln in enumerate(lines):
            m = SRT_TIME.search(ln)
            if m:
                ti = i
                break
        if m is None:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        payload = " ".join(ln.strip() for ln in lines[ti + 1:]).strip()
        lm = SRT_LABEL.match(payload)
        label, txt = (lm.group(1).strip(), lm.group(2).strip()) if lm else ("", payload)
        out.append(SrtSegment(start, end, label, txt))
    return out


def _even_spread(word_list: list[str], s: float, e: float) -> list[tuple[str, float, float]]:
    if not word_list or e <= s:
        return []
    step = (e - s) / len(word_list)
    return [(w, s + i * step, s + (i + 1) * step) for i, w in enumerate(word_list)]


def srt_words(segments: list[SrtSegment], max_segment_s: float = 4.0
              ) -> list[tuple[str, float, float]]:
    """Word times for SHORT speech segments (evenly spread inside the segment).
    Longer segments are left out - the ASR stage transcribes those."""
    words: list[tuple[str, float, float]] = []
    for seg in segments:
        if seg.label not in TURN_SPEECH_LABELS or not seg.text.strip():
            continue
        if seg.end_s - seg.start_s > max_segment_s:
            continue
        words.extend(_even_spread(seg.text.split(), seg.start_s, seg.end_s))
    words.sort(key=lambda w: (w[1], w[2]))
    return words


def long_srt_segments(segments: list[SrtSegment], min_segment_s: float = 4.0
                      ) -> list[tuple[float, float]]:
    """Speech segments longer than min_segment_s (transcribed by Whisper)."""
    return [(s.start_s, s.end_s) for s in segments
            if s.label in TURN_SPEECH_LABELS and s.text.strip()
            and s.end_s - s.start_s > min_segment_s]


# ---------------------------------------------------------------------------
# Session metadata (common structure for all sources)
# ---------------------------------------------------------------------------

@dataclass
class SessionMeta:
    source: str
    session_id: str
    duration_s: float
    speakers: list[dict]               # index = audio channel
    system_prompts: list[str]          # index = agent channel
    redactions: list[tuple[float, float]]
    words: list | None = None          # per channel [(word, s, e)] (video only)
    speech: list | None = None         # per channel [(s, e)] (video/turn)
    srt_segments: list | None = None   # per channel list[SrtSegment] (turn only)


def transcript_words(tr: dict) -> list[tuple[str, float, float]]:
    """video-21h transcript JSON -> word list. Token times are milliseconds."""
    words: list[tuple[str, float, float]] = []
    for seg in (tr or {}).get("segments") or []:
        toks = seg.get("tokens") or []
        for t in toks:
            w = (t.get("written") or "").strip()
            s, e = t.get("starttime"), t.get("endtime")
            if not w or s is None or e is None:
                continue
            words.append((w, float(s) / 1000.0, float(e) / 1000.0))
        if not toks:
            text = (seg.get("text") or "").strip()
            s, e = seg.get("starttime"), seg.get("endtime")
            if text and s is not None and e is not None and e > s:
                words.extend(_even_spread(text.split(), s / 1000.0, e / 1000.0))
    words.sort(key=lambda w: (w[1], w[2]))
    return words


def _channel_map(speakers_meta: list) -> dict[int, int]:
    """speaker-list index -> audio channel (defaults to the list order)."""
    ch_of = {}
    for i, sp in enumerate(speakers_meta):
        try:
            ch = int(sp.get("channel", i))
        except (TypeError, ValueError):
            ch = i
        ch_of[i] = ch if ch in (0, 1) else i
    return ch_of


def parse_video_session(spec: SourceSpec, sid: str, meta: dict,
                        tr1: dict, tr2: dict) -> SessionMeta:
    speakers_meta = meta.get("speakers") or []
    ch_of = _channel_map(speakers_meta)
    words: list = [[], []]
    for k, tr in enumerate((tr1, tr2)):
        ch = ch_of.get(k, k)
        if ch in (0, 1):
            words[ch] = transcript_words(tr)
    speech = [intervals_from_words(w) for w in words]
    speakers = [{"speaker_id": "", "role": "participant", "gender": ""} for _ in (0, 1)]
    for i, sp in enumerate(speakers_meta):
        ch = ch_of.get(i, i)
        if ch in (0, 1):
            speakers[ch] = {"speaker_id": sp.get("speaker_id", ""),
                            "role": "participant",
                            "gender": sp.get("gender", "") or ""}
    topics = []
    sim = meta.get("session_instruction_metadata") or {}
    for t in sim.get("topics") or []:
        topic = (t.get("topic") or "").strip()
        if topic:
            topics.append(topic)
    prompt = "You are having a casual video-call conversation"
    if topics:
        prompt += " about " + "; ".join(topics[:3])
    prompt += ". Speak naturally, as yourself."
    return SessionMeta(
        spec.name, sid, float(meta.get("duration", 0.0) or 0.0), speakers,
        [prompt, prompt], normalize_redactions(meta.get("redacted_segments")),
        words=words, speech=speech)


def parse_task_session(spec: SourceSpec, sid: str, meta: dict) -> SessionMeta:
    task = meta.get("task") or {}
    task_name = (task.get("task_name") or task.get("task_id")
                 or "collaborative task")
    task_name = str(task_name).replace("-", " ").replace("_", " ")
    ch_of = _channel_map(meta.get("speakers") or [])
    by_ch = {ch_of[i]: sp for i, sp in enumerate(meta.get("speakers") or [])}
    speakers, prompts = [], []
    for ch in (0, 1):
        sp = by_ch.get(ch, {})
        role = (sp.get("task_role_name") or sp.get("task_role") or "").strip()
        speakers.append({"speaker_id": sp.get("speaker_id", ""),
                         "role": role or "participant",
                         "gender": sp.get("gender", "") or ""})
        if role:
            prompts.append(
                f"You are the {role} in a collaborative task: {task_name}. "
                "Work together with your partner and speak naturally.")
        else:
            prompts.append(
                f"You are collaborating on a task: {task_name}. "
                "Work together with your partner and speak naturally.")
    red = normalize_redactions(meta.get("redacted_segments") or meta.get("redactions"))
    return SessionMeta(spec.name, sid, float(meta.get("duration", 0.0) or 0.0),
                       speakers, prompts, red)


def parse_turn_session(spec: SourceSpec, sid: str, meta: dict,
                       srt1: list[SrtSegment], srt2: list[SrtSegment]) -> SessionMeta:
    conv_type = (meta.get("conversation_type") or "casual").strip()
    speakers = [
        {"speaker_id": meta.get("speaker_1_actor_id", ""), "role": "participant",
         "gender": meta.get("speaker_1_actor_gender", "") or ""},
        {"speaker_id": meta.get("speaker_2_actor_id", ""), "role": "participant",
         "gender": meta.get("speaker_2_actor_gender", "") or ""},
    ]
    prompt = f"You are having a {conv_type.lower()} conversation. Speak naturally."
    speech = [merge_intervals([(s.start_s, s.end_s) for s in segs
                               if s.label in TURN_SPEECH_LABELS])
              for segs in (srt1, srt2)]
    dur = 0.0
    for sp in speech:
        if sp:
            dur = max(dur, max(e for _, e in sp))
    return SessionMeta(spec.name, sid, dur, speakers, [prompt, prompt], [],
                       speech=speech, srt_segments=[srt1, srt2])


# ---------------------------------------------------------------------------
# Loading sessions from HF (tar range reads / per-session downloads)
# ---------------------------------------------------------------------------

MEMBER_SUFFIXES = (
    ".transcript_speaker_1.json", ".transcript_speaker_2.json",
    ".video_speaker_1.mp4", ".video_speaker_2.mp4",
    ".flac", ".json",
)


def split_member_name(name: str) -> tuple[str | None, str | None]:
    base = name.rsplit("/", 1)[-1]
    for suf in MEMBER_SUFFIXES:
        if base.endswith(suf):
            return base[: -len(suf)], suf
    return None, None


def list_repo_files(repo_id: str, token: str | None) -> list[str]:
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(repo_id, repo_type="dataset", token=token)


class TarStore:
    """Range-read access to the webdataset shards of one tar-based source."""

    def __init__(self, spec: SourceSpec, data_root: str | Path, token: str | None):
        self.spec = spec
        self.cache_dir = Path(data_root) / "hf_cache" / spec.name
        self.token = token
        self._shards: list[str] | None = None
        self._clients: dict[str, tuple] = {}   # shard -> (fetch, size)
        self._indexes: dict[str, dict] = {}
        self._smap: dict | None = None

    def shards(self) -> list[str]:
        if self._shards is None:
            files = list_repo_files(self.spec.repo_id, self.token)
            self._shards = sorted(f for f in files
                                  if f.startswith("data/train/shard-")
                                  and f.endswith(".tar"))
            if not self._shards:
                raise RuntimeError(f"no data/train/shard-*.tar in {self.spec.repo_id}")
        return self._shards

    def _client(self, shard: str):
        if shard not in self._clients:
            url, size = resolve_hf_url(self.spec.repo_id, shard, self.token)
            # Signed CDN URLs expire after ~1 h; make_fetch re-resolves on
            # HTTP 401/403 so long-running workers don't start failing.
            def refresh() -> str:
                return resolve_hf_url(self.spec.repo_id, shard, self.token)[0]
            self._clients[shard] = (make_fetch(url, refresh=refresh), size)
        return self._clients[shard]

    def index(self, shard: str) -> dict[str, tuple[int, int]]:
        if shard not in self._indexes:
            fetch, size = self._client(shard)
            cache = self.cache_dir / (Path(shard).name + ".index.json")
            self._indexes[shard] = tar_index(size, fetch, cache)
        return self._indexes[shard]

    def session_map(self) -> dict[str, dict[str, tuple[str, int, int]]]:
        """session key -> {suffix: (shard, offset, size)}."""
        if self._smap is None:
            smap: dict[str, dict] = {}
            for shard in self.shards():
                for name, (off, size) in self.index(shard).items():
                    key, suffix = split_member_name(name)
                    if key and suffix:
                        smap.setdefault(key, {})[suffix] = (shard, off, size)
            self._smap = smap
        return self._smap

    def read(self, shard: str, off: int, size: int) -> bytes:
        fetch, _ = self._client(shard)
        return fetch(off, size)


def list_sessions(spec: SourceSpec, data_root: str | Path, token: str | None) -> list[str]:
    """Deterministic session id list for a source (cached as JSON)."""
    cache = Path(data_root) / "hf_cache" / spec.name / "sessions.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if spec.kind == "turn":
        files = list_repo_files(spec.repo_id, token)
        sids = sorted({f.split("/")[0] for f in files if f.endswith("/metadata.json")},
                      key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    else:
        sids = sorted(TarStore(spec, data_root, token).session_map().keys())
    if not sids:
        raise RuntimeError(f"no sessions found for {spec.name} ({spec.repo_id})")
    cache.parent.mkdir(parents=True, exist_ok=True)
    # pid-unique tmp: concurrent workers may build the session list together
    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(sids), encoding="utf-8")
    os.replace(tmp, cache)
    return sids


def load_tar_session(spec: SourceSpec, store: TarStore,
                     entry: dict[str, tuple[str, int, int]], sid: str):
    """Returns (SessionMeta, flac_bytes) for one tar-based session."""
    meta = json.loads(store.read(*entry[".json"]))
    flac = store.read(*entry[".flac"])
    if spec.kind == "video":
        tr1 = json.loads(store.read(*entry[".transcript_speaker_1.json"]))
        tr2 = json.loads(store.read(*entry[".transcript_speaker_2.json"]))
        return parse_video_session(spec, sid, meta, tr1, tr2), flac
    return parse_task_session(spec, sid, meta), flac


def load_turn_session(spec: SourceSpec, sid: str, data_root: str | Path,
                      token: str | None):
    """Returns (SessionMeta, (ch0_wav_path, ch1_wav_path)). Downloads the two
    per-speaker WAVs into <data_root>/raw/turn-104h/<sid>/ (the caller deletes
    them after processing unless --keep-raw). Annotations/metadata are small
    and stay in the HF cache."""
    from huggingface_hub import hf_hub_download
    meta = json.loads(Path(hf_hub_download(
        spec.repo_id, f"{sid}/metadata.json", repo_type="dataset",
        token=token)).read_text(encoding="utf-8"))
    srts = []
    for k in (1, 2):
        p = hf_hub_download(spec.repo_id, f"{sid}/speaker_{k}_annotation_a.srt",
                            repo_type="dataset", token=token)
        srts.append(parse_srt(Path(p).read_text(encoding="utf-8")))
    raw_dir = Path(data_root) / "raw" / spec.name
    wavs = []
    for k in (1, 2):
        p = hf_hub_download(spec.repo_id, f"{sid}/speaker_{k}_audio.wav",
                            repo_type="dataset", token=token,
                            local_dir=str(raw_dir))
        wavs.append(Path(p))
    return parse_turn_session(spec, sid, meta, srts[0], srts[1]), (wavs[0], wavs[1])



