# 영상 자막 추출기 (Video Subtitle Extractor)

영상을 업로드하면 OpenAI Whisper API로 음성을 인식해 타임스탬프가 포함된 자막을 자동으로 추출하는 로컬 웹앱입니다.
화면에서 바로 자막을 확인/수정하고, SRT / VTT / TXT 파일로 다운로드할 수 있습니다.

## 준비물

1. **Python 3.9 이상**
2. **ffmpeg** (영상에서 오디오를 추출하는 데 사용)
   - macOS: `brew install ffmpeg`
   - Windows: [ffmpeg.org](https://ffmpeg.org/download.html)에서 다운로드 후 PATH에 추가, 또는 `choco install ffmpeg`
   - Linux(Ubuntu/Debian): `sudo apt install ffmpeg`
3. **OpenAI API 키** — https://platform.openai.com/api-keys 에서 발급 (결제 수단 등록 필요, 사용한 만큼만 과금)

## 설치 및 실행

```bash
cd subtitle-extractor/backend

# (권장) 가상환경 생성
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# API 키 설정
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY=발급받은키 로 수정

# 서버 실행
uvicorn main:app --reload
```

터미널에 아래와 같이 뜨면 정상 실행된 것입니다.

```
Uvicorn running on http://127.0.0.1:8000
```

브라우저에서 **http://127.0.0.1:8000** 접속 → 영상 업로드 → 자막 자동 추출.

## 사용 방법

1. 화면의 업로드 영역을 클릭하거나 영상 파일을 드래그해서 놓습니다.
2. 진행률 바가 채워지며 오디오 추출 → 자막 추출 순서로 처리됩니다. (영상 길이에 비례해 시간이 걸립니다)
3. 완료되면 타임스탬프별 자막 목록이 나타나며, 각 줄을 클릭해서 바로 수정할 수 있습니다.
4. `SRT 다운로드` / `VTT 다운로드` / `TXT 다운로드` 버튼으로 원하는 형식의 자막 파일을 저장합니다.
   (수정한 내용이 다운로드 파일에 그대로 반영됩니다)

## 동작 원리 / 구조

```
subtitle-extractor/
├── Dockerfile            # 배포용 컨테이너 정의 (ffmpeg 포함)
├── render.yaml           # Render 배포 설정 (무료 플랜, 환경변수)
├── backend/
│   ├── main.py          # FastAPI 서버 (업로드, ffmpeg 처리, Whisper API 호출, 파일 생성)
│   ├── requirements.txt
│   ├── .env.example
│   └── data/             # 업로드/작업 파일 저장 (자동 생성, git에 커밋하지 마세요)
└── frontend/
    └── index.html        # 업로드 UI + 자막 편집/다운로드 (백엔드가 정적 파일로 서빙)
```

- 업로드된 영상은 ffmpeg로 **모노 16kHz mp3(저비트레이트)** 오디오로 변환되어 용량을 크게 줄입니다.
- OpenAI Whisper API는 요청 1건당 25MB 파일 크기 제한이 있어, 긴 영상은 자동으로 **15분 단위 청크**로 분할한 뒤 각각 전사하고 타임스탬프를 이어붙입니다.
- 처리는 백그라운드 스레드에서 진행되며, 프론트엔드는 1.5초마다 진행 상태를 polling합니다.
- 원본 업로드 영상 파일은 처리가 끝나면 자동 삭제되어 디스크 공간을 절약합니다 (추출된 오디오/자막 결과는 `backend/data/work/<job_id>/`에 보존됩니다).

## 비용 관련

OpenAI Whisper API(`whisper-1`)는 2025년 기준 분당 과금 방식입니다. 정확한 최신 요금은
https://openai.com/api/pricing 에서 확인하세요. 오디오를 저비트레이트로 압축해서 보내지만,
요금은 원본 오디오 **길이(분)** 기준이므로 압축과 무관하게 영상 길이에 비례해 비용이 발생합니다.

## 실제 인터넷에 배포하기 (Render, 무료, 터미널 불필요)

이 프로젝트에는 이미 `Dockerfile`과 `render.yaml`이 포함되어 있어서, 아래 순서대로 웹 화면 클릭만으로
실제 URL(`https://xxxx.onrender.com`)을 가진 사이트를 만들 수 있습니다. 터미널/코딩 지식이 필요 없습니다.

### 1. GitHub에 코드 올리기

1. [github.com](https://github.com) 에서 무료 계정을 만듭니다 (이미 있다면 로그인).
2. 오른쪽 위 `+` → `New repository` 클릭 → 이름 입력(예: `subtitle-extractor`) → `Create repository`.
3. 생성된 저장소 페이지에서 `uploading an existing file` 링크를 클릭합니다.
4. 이 프로젝트 폴더(`subtitle-extractor/`) 안의 **모든 파일과 폴더**를 통째로 끌어다 놓습니다
   (`.gitignore`, `Dockerfile`, `README.md`, `render.yaml`, `backend/`, `frontend/` 전부).
5. 아래 `Commit changes` 버튼을 눌러 업로드를 완료합니다.

### 2. Render에서 배포하기

1. [render.com](https://render.com) 에서 무료 계정을 만듭니다 (GitHub 계정으로 바로 가입 가능).
2. 대시보드에서 `New +` → `Web Service` 클릭.
3. 방금 만든 GitHub 저장소를 연결합니다 (처음이면 GitHub 권한 허용 필요).
4. Render가 저장소 안의 `render.yaml`을 자동으로 인식합니다. `Plan`은 `Free`를 선택합니다.
5. `OPENAI_API_KEY` 환경변수 입력란이 나타나면, 발급받은 OpenAI API 키를 붙여넣습니다.
6. `Create Web Service` (또는 `Deploy`) 클릭 → 몇 분 정도 빌드가 진행됩니다.
7. 빌드가 끝나면 화면 상단에 `https://subtitle-extractor-xxxx.onrender.com` 같은 실제 URL이 생깁니다.
   이 주소로 접속하면 어디서든(휴대폰 포함) 영상 업로드 → 자막 추출을 바로 사용할 수 있습니다.

### 참고 (무료 요금제 특성)

- 무료 요금제는 **15분 동안 요청이 없으면 서버가 잠들고**, 다시 접속하면 약 1분 정도 깨어나는 시간이 걸립니다.
- 매달 750시간의 무료 사용 시간이 제공됩니다 (개인 사용에는 충분합니다).
- 무료 요금제는 파일 저장 공간이 "휘발성"이라 서버가 재시작되면 이전에 처리한 결과가 사라질 수 있습니다.
  (이 앱은 처리 후 자막을 화면에서 바로 다운로드하는 구조라 문제없습니다.)
- 코드를 수정한 뒤 다시 GitHub에 업로드하면 Render가 자동으로 재배포합니다.

### 이후 여러 사람이 쓰는 서비스로 키우고 싶다면

- **작업 상태 저장소**: 현재는 메모리(`dict`)에 작업 상태를 저장하므로 서버 재시작 시 사라지고, 여러 서버 인스턴스로 확장할 수 없습니다. 여러 사용자가 쓰게 되면 Redis나 데이터베이스로 교체하는 것을 권장합니다.
- **업로드 용량 제한**: 여러 사용자가 쓰는 서비스라면 `main.py`의 업로드 처리에 파일 크기 제한과 사용자별 사용량 제한을 추가하세요.
- **동시 처리량**: 현재는 요청마다 새 스레드를 띄우는 단순한 방식입니다. 트래픽이 늘면 Celery/RQ 같은 작업 큐로 교체하는 것을 권장합니다.
- **인증**: 여러 사람이 쓰게 되면 간단한 로그인/토큰 체계를 추가해 업로드를 제한하는 것이 좋습니다.
- **유료 플랜**: 무료 요금제의 잠들기/시간 제한이 불편하다면 Render의 유료 플랜(Starter 등)으로 올리면 상시 구동됩니다.

## 문제 해결

- **"ffmpeg 실행 실패"**: ffmpeg가 설치되어 있는지, 터미널에서 `ffmpeg -version`이 실행되는지 확인하세요.
- **"OPENAI_API_KEY가 설정되지 않았습니다"**: `backend/.env` 파일이 있는지, 안에 실제 키 값이 들어있는지 확인하세요. 서버를 재시작해야 반영됩니다.
- **업로드가 너무 오래 걸림**: 영상이 길수록 오디오 추출/청크 분할/API 호출에 시간이 걸립니다. 진행률 바의 상태 메시지를 참고하세요.
