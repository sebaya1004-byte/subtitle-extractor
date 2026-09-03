"""
영상/오디오 자막 추출기 (Video/Audio -> Transcript extractor)

FastAPI 백엔드:
  - 오디오(mp3/m4a/wav/aac) 또는 영상(mp4/mov/mkv) 업로드
  - 영상이면 ffmpeg 로 오디오 추출/압축, 오디오는 바로 정규화
  - 필요 시 청크 분할 (OpenAI Whisper API 25MB 제한 대응)
  - OpenAI Whisper API 로 각 청크 전사(transcribe, 파일당 1회) 후 타임스탬프 병합
  - 타임코드가 포함된 Transcript 생성 -> 사용자 검토/수정 -> 검토 완료 처리
  - 검토 완료 후, 텍스트 AI(1회)로 말자막/포인트자막(요약자막) 생성
  - TXT 다운로드, 정적 프론트엔드(../frontend) 서빙

1차: 업로드 -> Transcript 생성 -> 사용자 검토 완료까지.
2차(이번 추가분): 검토 완료된 Transcript를 기준으로
  - 말자막: 규칙 기반 줄바꿈만 적용, 타임코드는 원본 그대로 재사용 (AI 비용 없음)
  - 포인트자막(요약자막): 텍스트 AI가 source_segment_ids만 선택하고,
    실제 start/end는 backend가 원본 Transcript에서 조회해서 채운다
    (AI는 시간 값을 절대 만들지 않는다). 방송 자막 작가 스타일 프롬프트 적용.
스타일/빈도 선택, 자막 수정, SRT/VTT 다운로드 UI 완성은 3차에서 추가된다.

로컬 실행:
    pip install -r requirements.txt
    cp .env.example .env   # 그리고 OPENAI_API_KEY 입력
    uvicorn main:app --reload

브라우저에서 http://127.0.0.1:8000 접속.
"""

import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("subtitle_extractor")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
WORK_DIR = DATA_DIR / "work"
FRONTEND_DIR = APP_DIR.parent / "frontend"

for d in (DATA_DIR, UPLOAD_DIR, WORK_DIR):
    d.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-1")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gpt-4o")

# Whisper API 파일 크기 제한(25MB)보다 넉넉히 낮게 잡아 안전 마진 확보
CHUNK_SECONDS = 15 * 60  # 15분 단위로 오디오를 분할

# 권장: 오디오 / 지원: 영상 (영상은 서버에서 오디오만 추출해서 사용)
RECOMMENDED_AUDIO_EXT = {".mp3", ".m4a", ".wav", ".aac"}
SUPPORTED_VIDEO_EXT = {".mp4", ".mov", ".mkv"}
ALLOWED_EXT = RECOMMENDED_AUDIO_EXT | SUPPORTED_VIDEO_EXT

# 파일 길이 경고 기준
LONG_DURATION_SEC = 60 * 60  # 1시간
VERY_LONG_DURATION_SEC = 3 * 60 * 60  # 3시간

app = FastAPI(title="영상/오디오 자막 추출기")

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


class StageError(Exception):
    """process_job 내부 단계에서 발생한, 사용자에게 보여줄 한국어 오류."""

    def __init__(self, step: int, message: str):
        super().__init__(message)
        self.step = step
        self.message = message


def get_client():
    """OpenAI 클라이언트를 지연 생성한다 (API 키 누락 시 명확한 에러를 주기 위해)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-xxxx"):
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. backend/.env 파일에 발급받은 키를 입력해주세요."
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def friendly_whisper_error(e: Exception) -> str:
    msg = str(e).lower()
    if "insufficient_quota" in msg or "quota" in msg or "429" in msg:
        return "API 크레딧이 부족합니다."
    if "connection" in msg or "timeout" in msg or "network" in msg:
        return "서버 연결에 실패했습니다."
    return "음성 인식에 실패했습니다."


def friendly_text_ai_error(e: Exception) -> str:
    msg = str(e).lower()
    if "insufficient_quota" in msg or "quota" in msg or "429" in msg:
        return "API 크레딧이 부족합니다."
    if "connection" in msg or "timeout" in msg or "network" in msg:
        return "서버 연결에 실패했습니다."
    return "AI 자막 생성에 실패했습니다."


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


def extract_audio(source_path: Path, out_audio_path: Path):
    """오디오/영상에서 오디오만 모노 16kHz mp3(저비트레이트)로 추출/정규화한다.

    -vn(비디오 스트림 제거) 옵션은 오디오 전용 입력에도 안전하게 동작하므로
    영상/오디오 입력 모두 이 함수 하나로 처리한다.
    """
    run_ffmpeg(
        [
            "-i",
            str(source_path),
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


SPOKEN_ALLOWED_MAX_LINES = (1, 2)
SPOKEN_ALLOWED_MAX_CHARS = (20, 25, 30)


def _greedy_word_groups(tokens: list, max_chars: int, text_of) -> list[list]:
    """토큰(단어) 리스트를 앞에서부터 max_chars 상한 안에서 최대한 채워 그룹으로 묶는다.

    단어 중간을 자르지 않는다. 공란도 글자수에 포함해서 계산한다.
    """
    groups: list[list] = []
    current: list = []
    cur_len = 0
    for tok in tokens:
        s = text_of(tok)
        added_len = len(s) + (1 if current else 0)
        if current and cur_len + added_len > max_chars:
            groups.append(current)
            current = []
            cur_len = 0
            added_len = len(s)  # 새 그룹 시작이라 앞 공란은 더하지 않는다
        current.append(tok)
        cur_len += added_len
    if current:
        groups.append(current)
    return groups


def split_words_by_char_budget(words: list[dict], max_chars_per_line: int) -> list[list[dict]]:
    """단어별(word-level) 타임스탬프를 상한 안에서 그룹으로 묶는다.

    각 그룹의 시작/끝 시간은 그 그룹에 속한 단어들의 실제 Whisper word timestamp에서만
    가져온다 (임의로 시간을 추정하지 않는다).
    """
    return _greedy_word_groups(words, max_chars_per_line, lambda w: w["word"])


def wrap_reading_lines(text: str, max_chars_per_line: int = 25) -> list[str]:
    """2줄 모드 전용 줄바꿈: 첫 줄은 상한 안에서 최대한 채우고, 나머지는 전부 둘째 줄로 보낸다.
    (한 화면에 두 줄을 그대로 겹쳐서 보여주고 싶을 때 사용, 타임코드는 원본 그대로 재사용)

    max_chars_per_line은 "상한"일 뿐 목표 길이가 아니다 — 문장이 짧으면 그대로 한 줄로 둔다.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars_per_line:
        return [text]

    words = text.split(" ")
    if len(words) < 2:
        return [text]

    groups = _greedy_word_groups(words, max_chars_per_line, lambda w: w)
    if len(groups) <= 1:
        return [text]

    line1 = " ".join(groups[0])
    line2 = " ".join(w for g in groups[1:] for w in g).strip()
    return [line1, line2] if line2 else [line1]


def close_small_gaps(captions: list[dict], max_gap: float = 2.0) -> list[dict]:
    """말자막 사이의 빈 시간(공백)을 없앤다.

    다음 자막이 나오기 전까지 이전 자막을 계속 띄워두는 것이 기본이다 — 단,
    실제로 max_gap초 이상 말이 끊긴 경우(진짜 침묵)는 공백을 그대로 둔다.
    시작 시간은 절대 건드리지 않고, 끝나는 시간만 다음 자막의 시작 시간까지
    늘린다 (마지막 자막은 다음이 없으므로 그대로 둔다). 입력 captions는
    시간순으로 정렬돼 있다고 가정한다.
    """
    for i in range(len(captions) - 1):
        gap = captions[i + 1]["start"] - captions[i]["end"]
        if 0 < gap < max_gap:
            captions[i]["end"] = captions[i + 1]["start"]
    return captions


def build_spoken_captions_sequential(transcript: dict, max_chars_per_line: int = 25) -> list[dict]:
    """1줄 모드: 문장이 길면 여러 개의 순차적인 1줄짜리 자막으로 쪼갠다.

    각 조각의 start/end는 Whisper가 실제로 준 단어 단위 타임스탬프에서만 가져온다.
    단어 타임스탬프가 없으면(구버전 Transcript 등) 안전하게 나누지 않고 원본을 유지한다.
    """
    captions: list[dict] = []
    for seg in transcript["segments"]:
        text = seg["text"].strip()
        if not text:
            continue

        if len(text) <= max_chars_per_line:
            captions.append(
                {
                    "segment_id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "lines": [text],
                }
            )
            continue

        words = seg.get("words") or []
        if not words:
            captions.append(
                {
                    "segment_id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "lines": [text],
                }
            )
            continue

        for group in split_words_by_char_budget(words, max_chars_per_line):
            piece_text = " ".join(w["word"] for w in group).strip()
            if not piece_text:
                continue
            captions.append(
                {
                    "segment_id": seg["id"],
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "text": piece_text,
                    "lines": [piece_text],
                }
            )
    return captions


def build_spoken_captions(
    transcript: dict, max_lines: int = 1, max_chars_per_line: int = 25
) -> list[dict]:
    """말자막(B) 생성.

    1줄 모드: 문장이 길면 여러 개의 순차적인 1줄짜리 자막으로 쪼갠다 (실제 단어 시간 기준).
    2줄 모드: Transcript와 1:1 대응, 원본 start/end를 그대로 재사용 (한 화면에 최대 2줄 표시).
    """
    if max_lines >= 2:
        captions = []
        for seg in transcript["segments"]:
            lines = wrap_reading_lines(seg["text"], max_chars_per_line=max_chars_per_line)
            captions.append(
                {
                    "segment_id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": "\n".join(lines),
                    "lines": lines,
                }
            )
        return close_small_gaps(captions)

    return close_small_gaps(
        build_spoken_captions_sequential(transcript, max_chars_per_line=max_chars_per_line)
    )


POINT_PROMPT = """당신은 대한민국 TV 예능, 교양, 다큐멘터리, 인터뷰 프로그램에서 일하는
전문 방송 자막 작가입니다.

아래는 영상의 인터뷰·나레이션·대화 내용을 문장 단위로 자른 것입니다
(형식: "id: 문장"). 이 내용을 분석해서, 그대로 받아쓴 말자막이 아니라
"방송용 포인트 자막(요약 자막)"을 만들어 주세요.

포인트 자막은 단순한 내용 요약이 아닙니다. 시청자가 화면을 보는 순간
"지금 무슨 이야기를 하고 있는지" 바로 이해할 수 있도록, 핵심 의미를
방송 자막 문법으로 재구성해야 합니다.

## 1. 가장 중요한 원칙
인터뷰 내용을 그대로 줄여 쓰지 마세요. 화자가 말한 전체 맥락을 이해한 뒤
그 사람이 전달하려는 핵심을 방송 화면에 걸리는 제목·상황·강조 문장 형태로
다시 작성하세요.

나쁜 예: "진화론은 생명 다양성의 과정을 설명함" / "갈릴레오의 발견이 지구
중심 우주관에 도전함" / "종교와 과학의 관계에 대한 논의 시작"
좋은 예: "생명 다양성의 과정을 설명하는\\n진화론" / "지구 중심의 우주관을
뒤흔든\\n갈릴레오의 발견" / "과학과 종교를 둘러싼\\n오래된 논쟁"

## 2. 줄 수 규칙 (숫자 기준, 반드시 지킬 것)
자막 텍스트의 전체 글자 수(공백 포함)가 **12자 이상이면 무조건 2줄**로
나누세요(text 안에 \\n으로 표시). 12자 미만일 때만 1줄로 써도 됩니다.
이 기준은 예외 없이 지키세요 — 12자가 넘는데 한 줄로 욱여넣지 마세요.
(단, 12자 미만이더라도 의미상 두 덩어리로 자연스럽게 나뉘면 2줄로 써도
됩니다.)

2줄로 나눌 때 줄바꿈 위치를 정하는 기준: ① 앞부분이 상황/전제, 뒷부분이
핵심 대상/결론일 때 ② 수식어가 길고 마지막에 핵심 명사가 나올 때
③ 원인과 결과가 나뉠 때 ④ 두 개념을 대비할 때 ⑤ 상황과 감정을 함께
보여줄 때. 줄바꿈 위치는 글자 수 절반이 아니라 의미 단위로 정합니다
(조사와 핵심 명사를 억지로 분리하지 마세요).

## 3. 기본 표현 방식 (명사구로 끝내는 제목형이 기본, 나머지는 가끔만 섞기)
- 명사형·제목형 (기본, 전체의 대부분을 이 형태로): "지구 중심의 우주관을
  뒤흔든\\n갈릴레오의 발견" / "무너진 것은 신앙이 아닌\\n낡은 우주관"
- 상황형: "망원경을 통해\\n새로운 우주를 발견한 갈릴레오"
- 질문형 (가끔): "과학과 종교는\\n정말 충돌할 수밖에 없을까?"
- 대비형 (명사구로 대비): "세계의 작동을 묻는 과학,\\n삶의 방향을 묻는 종교"
모든 자막을 같은 형태("○○하는 △△")로만 쓰지 말고 위 형태를 섞되,
**명사형·제목형을 가장 많이 사용하세요.**

## 4. 절대 쓰면 안 되는 문장 형태
(A) AI 요약체: "~설명함", "~언급함", "~논의함", "~제기함", "~시작함",
"~이야기함", "~강조함", "~소개함", "~보여줌", "~밝힘", "~전달함"

(B) 완결된 서술문(에세이·내레이션 문장): "~이다", "~다", "~한다", "~묻는다",
"~비롯된다", "~아니다"로 끝나는 길고 완결된 문장. 이런 문장은 방송
자막이 아니라 책이나 다큐 내레이션 대본처럼 보입니다. **반드시 문장을
끝맺지 말고 명사(구)로 끝내세요.**
  나쁜 예 → 좋은 예:
  "지동설은 단순한 이론이 아니라 인간의 위치를 바꾼 사건이다."
    → "인간의 위치를 바꾼\\n사건, 지동설"
  "신앙을 무너뜨린 것은 하나님이 아닌 낡은 우주관이다."
    → "무너진 것은 신앙이 아닌\\n낡은 우주관"
  "과학은 세계의 작동을, 종교는 인간의 삶을 묻는다."
    → "세계의 작동을 묻는 과학,\\n삶의 방향을 묻는 종교"
  "충돌의 원인은 잘못된 이해에서 비롯된다."
    → "충돌의 원인은\\n잘못된 이해"
  "진화론은 생명의 다양성을 설명하지만, 삶의 의미를 부정하지 않는다."
    → "삶의 의미는 부정하지 않는\\n진화론"
질문형("~까?")과, 원래 짧고 강한 한 줄 자막은 이 규칙의 예외입니다.

## 5. 자막 개수 — 절대 적게 뽑지 마세요 (가장 중요한 규칙)
**기본값: 의미 있는 원본 문장 하나마다 포인트 자막을 하나씩 만듭니다
(1문장 : 1자막이 기본입니다).** 여러 문장을 묶어서 자막 하나로 만드는
것은, 그 문장들이 따로 떼어놓으면 너무 짧고 단편적이어서 의미가 안
통할 때만 예외적으로 하세요 (source_segment_ids에 문장 id를 2개 이상
넣는 경우는 예외이지 기본이 아닙니다).

절대 "여러 문장을 요약해서 자막 개수를 줄이는" 방향으로 가지 마세요.
이전에 10분 분량에서 포인트 자막이 10개밖에 안 나온 적이 있는데, 이는
문장 여러 개를 계속 하나로 묶었기 때문입니다. 원본 문장이 100개면
(인사말 등 의미 없는 문장 제외) 포인트 자막도 그와 비슷한 80~100개
수준으로 나와야 정상입니다. 자막 하나에는 메시지 하나만 담고, 의미
없는 인사말·감탄사·추임새만 제외하고 — 그 외에는 전부 만드세요.

## 6. 원문에 없는 사실을 만들지 마세요
표현은 방송적으로 재구성하되, 화자가 말하지 않은 사실(과학/역사/종교/
의학/정치/인물 관련 포함)을 새로 추가하거나 과장하거나 단정하지 마세요.
화자의 말에서 확인 가능한 범위 안에서 표현만 바꿉니다. 화자가 가능성을
말하면 가능성이 드러나게, 질문을 던지면 질문형으로, 확신하면 단정적인
명사구로 — 원문의 뉘앙스는 유지하되 4번 규칙(명사구로 끝내기)은 항상
지키세요. 유튜브 썸네일처럼 과도하게 자극적으로 만들지 마세요.

## 7. 자막 길이
(몇 줄로 나눌지는 2번 규칙의 12자 기준을 따르세요.) 한 줄의 길이는
대략 7~18자 정도가 읽기 좋습니다. 3줄 이상은 만들지 마세요
(1줄 또는 2줄만).

## 8. 최종 점검 기준
각 자막을 만든 뒤 스스로 확인하세요: 이 자막만 봐도 핵심을 알 수 있는가?
말자막 없이 흐름을 따라갈 수 있는가? AI 요약문처럼 보이지 않는가? 실제
TV 자막처럼 자연스러운가? 원문의 핵심 의미가 살아있는가? 너무 장황하지
않은가? 앞뒤 자막과 중복되지 않는가? 하나라도 걸리면 다시 쓰세요.

## 9. 타임코드 (이 프로젝트만의 규칙 — 반드시 지킬 것)
당신은 시작/끝 시각(start, end, timestamp 등)을 절대로 직접 만들지
않습니다. 대신 각 포인트 자막마다, 그 내용의 근거가 된 원본 문장들의
id를 source_segment_ids 배열로 표시하세요 (아래 [원본 대사] 목록에 실제
존재하는 id만 사용, 하나 이상, 새 id를 만들지 마세요). 실제 시작/끝
시각은 이후 시스템이 그 id들의 원본 시간을 이용해 직접 계산합니다.
하나의 발언이 길어서 여러 문장 id에 걸쳐 있으면, 그 문장 id들을 전부
배열에 포함하세요.

**id는 절대 겹치면 안 됩니다**: 한 문장 id는 오직 하나의 포인트 자막에서만
사용하세요. 이미 다른 포인트 자막에서 쓴 id를 또 쓰면 자막들의 시간이
서로 겹쳐버립니다. 한 포인트 자막 안의 source_segment_ids는 항상 서로
붙어 있는(연속된) 문장 id들이어야 합니다 (건너뛰어서 묶지 마세요).
전체 문장을 처음 id부터 끝 id까지 순서대로 죽 훑으면서, 각 문장이 어느
포인트 자막에 속하는지 하나씩 정한다고 생각하세요.

## 10. 출력 형식
아래 JSON 형식으로만 응답하세요 (다른 설명 텍스트 없이):
{"point_captions": [{"source_segment_ids": [12], "text": "..."}]}

[원본 대사]
{transcript_lines}
"""


def generate_point_captions(transcript: dict) -> list[dict]:
    """Transcript 텍스트만 보고 포인트자막(요약자막) 후보를 고른다 (오디오 재분석 없음).

    AI는 source_segment_ids만 선택하고, 실제 start/end는 이 함수가 원본 Transcript에서
    직접 조회해서 채운다. AI가 응답에 시간 값을 포함해도 절대 사용하지 않는다.
    """
    segments = [s for s in transcript["segments"] if s["text"]]
    if not segments:
        return []

    transcript_lines = "\n".join(f"{s['id']}: {s['text']}" for s in segments)
    prompt = POINT_PROMPT.replace("{transcript_lines}", transcript_lines)

    client = get_client()
    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(resp.choices[0].message.content)

    seg_by_id = {s["id"]: s for s in segments}
    raw_list = data.get("point_captions")

    resolved: list[dict] = []
    used_ids: set[int] = set()
    if isinstance(raw_list, list):
        for i, item in enumerate(raw_list, 1):
            if not isinstance(item, dict):
                continue
            ids = item.get("source_segment_ids")
            text = item.get("text")
            if not isinstance(ids, list) or not text:
                continue
            # 존재하지 않는 id는 버리고(환각 방지), 이미 다른 포인트 자막이 쓴 id도
            # 제외한다 — 그래야 포인트 자막끼리 시간이 절대 겹치지 않는다.
            valid_ids = [sid for sid in ids if sid in seg_by_id and sid not in used_ids]
            if not valid_ids:
                continue
            # AI가 중간 문장을 건너뛰고 참조하면(예: 6, 9만 골라 7·8을 빠뜨리면),
            # 이 자막이 어차피 그 시간대를 다 차지하므로 아직 아무도 안 쓴 사이 문장을
            # 자동으로 편입시킨다 (문장이 자막 없이 통째로 빠지는 것을 방지).
            lo, hi = min(valid_ids), max(valid_ids)
            for sid in seg_by_id:
                if lo < sid < hi and sid not in used_ids and sid not in valid_ids:
                    valid_ids.append(sid)
            valid_ids.sort()
            used_ids.update(valid_ids)
            start = min(seg_by_id[sid]["start"] for sid in valid_ids)
            end = max(seg_by_id[sid]["end"] for sid in valid_ids)
            resolved.append(
                {
                    "id": f"point_{i}",
                    "source_segment_ids": valid_ids,
                    "start": start,
                    "end": end,
                    "text": str(text).strip(),
                }
            )
    return resolved


def process_job(job_id: str, source_path: Path, source_type: str):
    """STEP 2(오디오 준비) ~ STEP 4(대본 검토 대기)까지 처리한다.

    Transcript가 만들어지면 여기서 멈춘다 (자동으로 다음 단계로 넘어가지 않음).
    """
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / "audio.mp3"
    chunks_dir = job_dir / "chunks"

    try:
        # ---- STEP 2: 오디오 준비 ----
        if source_type == "video":
            update_job(
                job_id,
                status="preparing_audio",
                progress=5,
                message="영상에서 오디오를 추출하고 있습니다...",
            )
        else:
            update_job(
                job_id,
                status="preparing_audio",
                progress=5,
                message="오디오를 준비하고 있습니다...",
            )

        try:
            extract_audio(source_path, audio_path)
        except Exception as e:  # noqa: BLE001
            logger.exception("job %s: extract_audio 실패", job_id)
            raise StageError(2, "영상에서 오디오를 추출하지 못했습니다.") from e

        update_job(
            job_id,
            progress=15,
            message="오디오 추출 완료" if source_type == "video" else "오디오 준비 완료",
        )

        update_job(job_id, status="splitting", progress=18, message="오디오를 나누고 있습니다...")
        try:
            chunks = split_audio(audio_path, chunks_dir)
        except Exception as e:  # noqa: BLE001
            logger.exception("job %s: split_audio 실패", job_id)
            raise StageError(2, "오디오 분할에 실패했습니다.") from e

        # ---- STEP 3: 음성을 텍스트로 변환 (Whisper API, 파일당 1회) ----
        try:
            client = get_client()
        except Exception as e:  # noqa: BLE001
            logger.exception("job %s: get_client 실패", job_id)
            raise StageError(3, str(e)) from e

        all_segments: list[dict] = []
        offset = 0.0
        seg_id = 1
        total = len(chunks)

        for idx, chunk_path in enumerate(chunks):
            progress = 20 + int(70 * (idx / max(total, 1)))
            update_job(
                job_id,
                status="transcribing",
                progress=progress,
                message=f"음성을 텍스트로 변환하는 중... ({idx + 1}/{total})",
            )

            try:
                with open(chunk_path, "rb") as f:
                    resp = client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=f,
                        response_format="verbose_json",
                        timestamp_granularities=["segment", "word"],
                    )
            except Exception as e:  # noqa: BLE001
                logger.exception("job %s: whisper transcription 실패 (chunk %s/%s)", job_id, idx + 1, total)
                raise StageError(3, friendly_whisper_error(e)) from e

            segments = getattr(resp, "segments", None)
            if segments is None and isinstance(resp, dict):
                segments = resp.get("segments")

            # 단어 단위(word-level) 타임스탬프. 같은 요청·같은 비용으로 함께 받는다.
            # 나중에 "1줄 모드"에서 문장을 나눌 때, 임의로 시간을 추정하지 않고
            # 여기서 받은 실제 시간만 사용하기 위함이다.
            words_resp = getattr(resp, "words", None)
            if words_resp is None and isinstance(resp, dict):
                words_resp = resp.get("words")
            words_resp = list(words_resp or [])

            def _word_fields(w):
                ws = getattr(w, "start", None)
                we = getattr(w, "end", None)
                wt = getattr(w, "word", None)
                if ws is None and isinstance(w, dict):
                    ws = w.get("start")
                    we = w.get("end")
                    wt = w.get("word")
                return wt, ws, we

            word_idx = 0

            if segments:
                segments = list(segments)
                for seg_idx, seg in enumerate(segments):
                    start = getattr(seg, "start", None)
                    end = getattr(seg, "end", None)
                    text = getattr(seg, "text", None)
                    if start is None and isinstance(seg, dict):
                        start = seg.get("start")
                        end = seg.get("end")
                        text = seg.get("text")
                    seg_end_local = float(end or 0)

                    # 다음 segment가 있으면 그 시작 시간을 경계로 쓴다. 문장 사이에 틈이
                    # 없이 붙어있어도(다음 segment가 이 segment의 end와 같은 시각에 시작해도)
                    # 다음 segment의 첫 단어가 이 segment로 잘못 끼어드는 것을 막기 위함이다.
                    next_start_local = None
                    if seg_idx + 1 < len(segments):
                        nxt = segments[seg_idx + 1]
                        next_start_local = getattr(nxt, "start", None)
                        if next_start_local is None and isinstance(nxt, dict):
                            next_start_local = nxt.get("start")
                    cutoff = float(next_start_local) if next_start_local is not None else seg_end_local + 0.05

                    # words_resp는 시간순으로 정렬돼 있으므로, 포인터를 이용해
                    # 이 segment 시간 범위 안에 들어오는 단어만 순서대로 소비한다.
                    seg_words = []
                    while word_idx < len(words_resp):
                        wt, ws, we = _word_fields(words_resp[word_idx])
                        if ws is None:
                            break
                        if ws >= cutoff:
                            break
                        seg_words.append(
                            {
                                "word": (wt or "").strip(),
                                "start": float(ws) + offset,
                                "end": float(we if we is not None else ws) + offset,
                            }
                        )
                        word_idx += 1

                    all_segments.append(
                        {
                            "id": seg_id,
                            "start": float(start or 0) + offset,
                            "end": seg_end_local + offset,
                            "text": (text or "").strip(),
                            "words": seg_words,
                        }
                    )
                    seg_id += 1
            else:
                # segments 정보가 없으면 청크 전체를 하나의 구간으로 처리
                text = getattr(resp, "text", "") or ""
                duration = ffprobe_duration(chunk_path)
                all_segments.append(
                    {
                        "id": seg_id,
                        "start": offset,
                        "end": offset + duration,
                        "text": text.strip(),
                        "words": [],
                    }
                )
                seg_id += 1

            offset += ffprobe_duration(chunk_path)

        full_text = " ".join(s["text"] for s in all_segments if s["text"])
        transcript = {
            "segments": all_segments,
            "text": full_text,
            "duration": offset,
        }

        result_path = job_dir / "result.json"
        result_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---- STEP 4: 대본 검토 대기 (여기서 자동으로 넘어가지 않음) ----
        update_job(
            job_id,
            status="transcript_ready",
            progress=100,
            message="대본을 검토해주세요.",
            transcript=transcript,
            transcript_reviewed=False,
        )

        # 임시 오디오 파일 정리 (Transcript와 Job 상태는 유지)
        try:
            if audio_path.exists():
                audio_path.unlink()
            if chunks_dir.exists():
                shutil.rmtree(chunks_dir)
        except OSError:
            pass

    except StageError as e:
        update_job(job_id, status="error", failed_step=e.step, message=e.message)
    except Exception:  # noqa: BLE001
        logger.exception("job %s: process_job에서 처리되지 않은 예외", job_id)
        update_job(job_id, status="error", failed_step=3, message="음성 인식에 실패했습니다.")
    finally:
        # 원본 업로드 파일은 처리 후 삭제하여 디스크 공간 절약
        try:
            if source_path.exists():
                source_path.unlink()
        except OSError:
            pass


def process_captions_job(job_id: str, spoken_max_lines: int = 1, spoken_max_chars_per_line: int = 25):
    """STEP 5: 검토 완료된 Transcript를 기준으로 말자막/포인트자막을 만든다.

    오디오는 다시 분석하지 않는다 (Whisper 재호출 없음). 텍스트 AI는 여기서 1회만 호출된다.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        transcript = job.get("transcript") if job else None

    if not job or not transcript:
        return

    try:
        update_job(
            job_id, status="generating_captions", progress=10, message="말자막을 만들고 있습니다..."
        )
        spoken_captions = build_spoken_captions(
            transcript, max_lines=spoken_max_lines, max_chars_per_line=spoken_max_chars_per_line
        )
        update_job(
            job_id,
            progress=40,
            message="포인트자막을 만들고 있습니다...",
            spoken_captions=spoken_captions,
        )

        try:
            point_captions = generate_point_captions(transcript)
        except Exception as e:  # noqa: BLE001
            logger.exception("job %s: generate_point_captions 실패", job_id)
            raise StageError(5, friendly_text_ai_error(e)) from e

        update_job(
            job_id,
            status="captions_ready",
            progress=100,
            message="AI 자막 생성이 완료되었습니다.",
            point_captions=point_captions,
        )
    except StageError as e:
        update_job(job_id, status="error", failed_step=e.step, message=e.message)
    except Exception:  # noqa: BLE001
        logger.exception("job %s: process_captions_job에서 처리되지 않은 예외", job_id)
        update_job(job_id, status="error", failed_step=5, message="AI 자막 생성에 실패했습니다.")


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

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "지원하지 않는 파일 형식입니다.")
    source_type = "audio" if ext in RECOMMENDED_AUDIO_EXT else "video"

    job_id = uuid.uuid4().hex
    source_path = UPLOAD_DIR / f"{job_id}{ext}"

    with open(source_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    duration = ffprobe_duration(source_path)
    duration_warning = None
    if duration >= VERY_LONG_DURATION_SEC:
        duration_warning = "very_long"
    elif duration >= LONG_DURATION_SEC:
        duration_warning = "long"

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "source_type": source_type,
            "duration": duration,
            "duration_warning": duration_warning,
            "status": "queued",
            "failed_step": None,
            "progress": 0,
            "message": "대기 중...",
            "transcript": None,
            "transcript_reviewed": False,
            "spoken_captions": None,
            "point_captions": None,
            "created_at": time.time(),
        }

    thread = threading.Thread(target=process_job, args=(job_id, source_path, source_type), daemon=True)
    thread.start()

    return {
        "job_id": job_id,
        "filename": file.filename,
        "source_type": source_type,
        "duration": duration,
        "duration_warning": duration_warning,
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return job


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, fmt: str = "srt", caption_type: str = "transcript"):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("transcript"):
        raise HTTPException(404, "아직 생성된 대본이 없습니다.")

    type_map = {
        "transcript": job["transcript"]["segments"],
        "spoken": job.get("spoken_captions") or [],
        "point": job.get("point_captions") or [],
    }
    if caption_type not in type_map:
        raise HTTPException(400, "지원하지 않는 자막 종류입니다. (transcript, spoken, point)")
    segments = type_map[caption_type]
    if not segments:
        raise HTTPException(404, "아직 생성되지 않은 자막입니다.")

    suffix_map = {"transcript": "", "spoken": "_spoken", "point": "_point"}
    base_name = Path(job.get("filename", "transcript")).stem + suffix_map[caption_type]

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

    # Content-Disposition 헤더는 ASCII(latin-1)만 담을 수 있어서, 한글 파일명은
    # RFC 5987 방식(filename*=UTF-8''...)으로 별도 인코딩해서 넣는다.
    # ASCII로만 된 대체 파일명도 같이 넣어 구버전 클라이언트도 대비한다.
    ascii_fallback = filename.encode("ascii", errors="ignore").decode("ascii") or "download"
    encoded_filename = quote(filename)
    return Response(
        content=content.encode("utf-8"),
        media_type=media_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded_filename}'
            )
        },
    )


@app.post("/api/jobs/{job_id}/save")
async def save_edits(job_id: str, payload: dict):
    """사용자가 검토 화면에서 수정한 대본 텍스트를 저장한다.

    id로 원본 segment를 찾아 text만 반영하고, start/end 타임코드는
    Whisper 원본 값을 그대로 유지한다 (절대 변경하지 않음).
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        if not job.get("transcript"):
            raise HTTPException(400, "아직 생성된 대본이 없습니다.")

        edited = payload.get("segments")
        if not isinstance(edited, list):
            raise HTTPException(400, "segments 형식이 올바르지 않습니다.")

        by_id = {s["id"]: s for s in job["transcript"]["segments"]}
        for seg in edited:
            if not isinstance(seg, dict):
                continue
            sid = seg.get("id")
            if sid in by_id and "text" in seg:
                by_id[sid]["text"] = str(seg["text"]).strip()

        job["transcript"]["text"] = " ".join(
            s["text"] for s in job["transcript"]["segments"] if s["text"]
        )

    return {"ok": True}


@app.post("/api/jobs/{job_id}/review")
def review_transcript(job_id: str):
    """[검토 완료 · AI 자막 만들기] 버튼: 대본 검토 완료 상태로 전환한다.

    이 시점까지는 텍스트 분석 AI를 호출하지 않는다 (1차 구현 범위 밖).
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        if not job.get("transcript"):
            raise HTTPException(400, "아직 생성된 대본이 없습니다.")

        job["transcript_reviewed"] = True
        job["status"] = "reviewed"
        job["message"] = "대본 검토가 완료되었습니다. 다음 단계에서 AI 자막을 생성할 수 있습니다."

    return {"ok": True, "status": "reviewed"}


@app.post("/api/jobs/{job_id}/generate-captions")
def generate_captions(job_id: str, payload: dict | None = None):
    """[AI 자막 생성 시작] 버튼: 검토 완료된 Transcript를 기준으로
    말자막/포인트자막(요약자막) 생성을 시작한다. 여기서 텍스트 AI 비용이 처음 발생하며,
    Whisper(음성 인식)는 다시 호출되지 않는다.

    payload로 말자막 설정(spoken_max_lines: 1=순차분할/2=한 화면에 2줄,
    spoken_max_chars_per_line)을 선택적으로 받는다.
    """
    payload = payload or {}
    max_lines = payload.get("spoken_max_lines", 1)
    max_chars = payload.get("spoken_max_chars_per_line", 25)
    if max_lines not in SPOKEN_ALLOWED_MAX_LINES:
        max_lines = 1
    if max_chars not in SPOKEN_ALLOWED_MAX_CHARS:
        max_chars = 25

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        if not job.get("transcript"):
            raise HTTPException(400, "아직 생성된 대본이 없습니다.")
        if not job.get("transcript_reviewed"):
            raise HTTPException(400, "먼저 대본 검토를 완료해주세요.")

        job["status"] = "generating_captions"
        job["failed_step"] = None
        job["progress"] = 0
        job["message"] = "말자막을 만들고 있습니다..."

    thread = threading.Thread(
        target=process_captions_job, args=(job_id, max_lines, max_chars), daemon=True
    )
    thread.start()

    return {"ok": True, "status": "generating_captions"}


@app.post("/api/jobs/{job_id}/rewrap-spoken")
def rewrap_spoken(job_id: str, payload: dict | None = None):
    """말자막의 줄 수 / 줄당 최대 글자수만 바꿔서 즉시 다시 만든다.

    AI를 호출하지 않는 규칙 기반 작업이라 비용이 들지 않고, 포인트자막에는
    영향을 주지 않는다.
    """
    payload = payload or {}
    max_lines = payload.get("max_lines", 1)
    max_chars = payload.get("max_chars_per_line", 25)
    if max_lines not in SPOKEN_ALLOWED_MAX_LINES:
        max_lines = 1
    if max_chars not in SPOKEN_ALLOWED_MAX_CHARS:
        max_chars = 25

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        if not job.get("transcript"):
            raise HTTPException(400, "아직 생성된 대본이 없습니다.")
        spoken_captions = build_spoken_captions(
            job["transcript"], max_lines=max_lines, max_chars_per_line=max_chars
        )
        job["spoken_captions"] = spoken_captions

    return {"ok": True, "spoken_captions": spoken_captions}


# 정적 프론트엔드 서빙 (반드시 API 라우트 등록 이후에 마운트)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
