"""
영상 자막 추출기 (Video -> Subtitle extractor)

FastAPI 백엔드:
  - 영상 업로드
  - ffmpeg 로 오디오 추출/압축
  - 필요 시 청크 분할 (OpenAI Whisper API 25MB 제한 대응)
  - OpenAI Whisper API 로 각 청크 전사(transcribe) 후 타임스탬프 병합
  - SRT / VTT / TXT 생성 및 다운로드
  - 정적 프론트엔드(../frontend) 서빙

로컬 실행:
    pip install -r requirements.txt
    cp .env.example .env   # 그리고 OPENAI_API_KEY 입력
    uvicorn main:app --reload

브라우저에서 http://127.0.0.1:8000 접속.
"""

import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
WORK_DIR = DATA_DIR / "work"
FRONTEND_DIR = APP_DIR.parent / "frontend"

for d in (DATA_DIR, UPLOAD_DIR, WORK_DIR):
    d.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-1")

# Whisper API 파일 크기 제한(25MB)보다 넉넉히 낮게 잡아 안전 마진 확보
CHUNK_SECONDS = 15 * 60  # 15분 단위로 오디오를 분할

app = FastAPI(title="영상 자막 추출기")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# job_id -> job 상태 dict
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def get_client():
    """OpenAI 클라이언트를 지연 생성한다 (API 키 누락 시 명확한 에러를 주기 위해)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-xxxx"):
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. backend/.env 파일에 발급받은 키를 입력해주세요."
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def update_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def run_ffmpeg(args: list[str]):
    result = subprocess.run(
        ["ffmpeg", "-y", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore")[-4000:]
        raise RuntimeError(f"ffmpeg 실행 실패: {stderr}")


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        return float(result.stdout.decode().strip())
    except ValueError:
        return 0.0


def extract_audio(video_path: Path, out_audio_path: Path):
    """영상에서 오디오만 추출하여 모노 16kHz mp3(저비트레이트)로 변환한다."""
    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "48k",
            str(out_audio_path),
        ]
    )


def split_audio(audio_path: Path, out_dir: Path) -> list[Path]:
    """오디오를 CHUNK_SECONDS 단위로 분할한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "chunk_%03d.mp3"
    run_ffmpeg(
        [
            "-i",
            str(audio_path),
            "-f",
            "segment",
            "-segment_time",
            str(CHUNK_SECONDS),
            "-c",
            "copy",
            "-reset_timestamps",
            "1",
            str(pattern),
        ]
    )
    chunks = sorted(out_dir.glob("chunk_*.mp3"))
    if not chunks:
        # 분할이 안 된 짧은 파일의 경우 원본을 그대로 사용
        chunks = [audio_path]
    return chunks


def fmt_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    ms = int(round((seconds - math.floor(seconds)) * 1000))
    if ms == 1000:
        ms = 0
        ss += 1
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def fmt_vtt_timestamp(seconds: float) -> str:
    return fmt_srt_timestamp(seconds).replace(",", ".")


def build_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_timestamp(seg['start'])} --> {fmt_srt_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def build_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{fmt_vtt_timestamp(seg['start'])} --> {fmt_vtt_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def build_txt(segments: list[dict]) -> str:
    return "\n".join(seg["text"].strip() for seg in segments)


def process_job(job_id: str, video_path: Path):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        update_job(job_id, status="extracting_audio", progress=5, message="오디오 추출 중...")
        audio_path = job_dir / "audio.mp3"
        extract_audio(video_path, audio_path)

        update_job(job_id, status="splitting", progress=15, message="오디오 분할 중...")
        chunks_dir = job_dir / "chunks"
        chunks = split_audio(audio_path, chunks_dir)

        client = get_client()

        all_segments: list[dict] = []
        offset = 0.0
        total = len(chunks)

        for idx, chunk_path in enumerate(chunks):
            progress = 15 + int(75 * (idx / max(total, 1)))
            update_job(
                job_id,
                status="transcribing",
                progress=progress,
                message=f"자막 추출 중... ({idx + 1}/{total})",
            )

            with open(chunk_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    response_format="verbose_json",
                )

            segments = getattr(resp, "segments", None)
            if segments is None and isinstance(resp, dict):
                segments = resp.get("segments")

            if segments:
                for seg in segments:
                    start = getattr(seg, "start", None)
                    end = getattr(seg, "end", None)
                    text = getattr(seg, "text", None)
                    if start is None and isinstance(seg, dict):
                        start = seg.get("start")
                        end = seg.get("end")
                        text = seg.get("text")
                    all_segments.append(
                        {
                            "start": float(start or 0) + offset,
                            "end": float(end or 0) + offset,
                            "text": (text or "").strip(),
                        }
                    )
            else:
                # segments 정보가 없으면 청크 전체를 하나의 구간으로 처리
                text = getattr(resp, "text", "") or ""
                duration = ffprobe_duration(chunk_path)
                all_segments.append(
                    {"start": offset, "end": offset + duration, "text": text.strip()}
                )

            offset += ffprobe_duration(chunk_path)

        full_text = " ".join(s["text"] for s in all_segments if s["text"])

        result = {
            "segments": all_segments,
            "text": full_text,
        }
        result_path = job_dir / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        update_job(
            job_id,
            status="done",
            progress=100,
            message="완료되었습니다.",
            segments=all_segments,
            text=full_text,
        )
    except Exception as e:  # noqa: BLE001
        update_job(job_id, status="error", message=str(e))
    finally:
        # 원본 업로드 영상은 처리 후 삭제하여 디스크 공간 절약 (오디오/결과는 보존)
        try:
            if video_path.exists():
                video_path.unlink()
        except OSError:
            pass


@app.get("/api/health")
def health():
    has_key = bool(os.environ.get("OPENAI_API_KEY")) and not os.environ.get(
        "OPENAI_API_KEY", ""
    ).startswith("sk-xxxx")
    return {"ok": True, "openai_key_configured": has_key}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "파일이 없습니다.")

    job_id = uuid.uuid4().hex
    suffix = Path(file.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"

    with open(video_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "status": "queued",
            "progress": 0,
            "message": "대기 중...",
            "segments": None,
            "text": None,
            "created_at": time.time(),
        }

    thread = threading.Thread(target=process_job, args=(job_id, video_path), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return job


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, fmt: str = "srt"):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "완료된 작업이 아닙니다.")

    segments = job["segments"]
    base_name = Path(job.get("filename", "subtitle")).stem

    if fmt == "srt":
        content = build_srt(segments)
        media_type = "application/x-subrip"
        filename = f"{base_name}.srt"
    elif fmt == "vtt":
        content = build_vtt(segments)
        media_type = "text/vtt"
        filename = f"{base_name}.vtt"
    elif fmt == "txt":
        content = build_txt(segments)
        media_type = "text/plain"
        filename = f"{base_name}.txt"
    else:
        raise HTTPException(400, "지원하지 않는 형식입니다. (srt, vtt, txt)")

    return Response(
        content=content.encode("utf-8"),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/jobs/{job_id}/save")
async def save_edits(job_id: str, payload: dict):
    """프론트엔드에서 사용자가 수정한 자막 내용을 서버 상태에도 반영(선택 사항)."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise HTTPException(400, "segments 형식이 올바르지 않습니다.")
        job["segments"] = segments
        job["text"] = " ".join(s.get("text", "").strip() for s in segments)
    return {"ok": True}


# 정적 프론트엔드 서빙 (반드시 API 라우트 등록 이후에 마운트)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
