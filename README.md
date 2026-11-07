# EKF 기반 보행자 궤적 예측

VIRAT 고정 카메라 영상에서 YOLO26s로 사람을 검출하고 원본 DeepSORT로
추적합니다. 박스 하단 중앙을 Homography로 지면 좌표로 변환하여 ID별 EKF에
입력하고, 미래 12프레임을 예측한 뒤 원본 GT와 ADE/FDE를 비교합니다.
라이브 카메라와 GRU 실행 코드는 포함하지 않습니다.

## 구성

```text
src/
  tracking system.py   # 터미널에서 실행할 파일
  track.py             # 실행기가 사용하는 데이터셋 처리·평가 구현
  tracking_core.py     # YOLO, MARS, 원본 DeepSORT 연결
  ekf.py               # 프레임 단위 상태 추정과 미래 예측
  homography.py        # 픽셀 ↔ 지면 좌표 변환
  forecast.py          # 예측/GT 궤적 표시와 ID 대응
  estimate_r.py        # 개발용 GT에서 R 사전 보정
  evaluate.py          # ADE/FDE, 직선 기준, 데이터 분할, 배치 평가
  latency.py           # EKF 처리 지연 측정
  test_*.py            # 수식, 초기화, GT 매칭, 지표 계산 검증
deep_sort/             # 원본 DeepSORT와 MARS 인코더
Resources/homographies/ # Scene별 변환 행렬과 기준 영상
```

## 설치 및 확인

Python 3.10 환경에서 프로젝트 폴더(`README.md`, `requirements.txt`, `src/`가
있는 폴더)를 터미널로 열고 순서대로 실행합니다. 설치는 처음 한 번 진행합니다.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -c "import torchvision; import tensorflow; import cv2; import ultralytics; print('imports OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -B -m unittest discover -s src -t . -p 'test_*.py'
```

`requirements.txt`는 로컬 검증 환경의 버전입니다. 설치는 루트
`requirements.txt`를 사용합니다. 다음으로 GPU를 확인합니다.

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

아래 실행 예시는 GPU를 사용하는 `--device 0` 기준입니다. `nvidia-smi`가
정상이고 CUDA가 `True`여야 합니다. GPU를 사용할 수 없는 환경에서는
`--device cpu`로 바꾸면 되며, 전체 처리 FPS가 낮아질 수 있습니다.
EKF와 MARS는 CPU에서 실행합니다.

## 데이터 및 가중치

VIRAT 원본 영상과 동일한 이름의 objects GT, YOLO26s 및 MARS 가중치를 준비합니다.

```text
Resources/videos_original/VIRAT_S_010208_07_000768_000791.mp4
Resources/annotations/VIRAT_S_010208_07_000768_000791.viratdata.objects.txt
weights/yolo26s.pt
weights/mars-small128.pb
runs/calibration/ekf_r_0102_yolo26s_imgsz960_maxage60_nms0p65.json
```

영상·GT·가중치·R 보정 결과·실행 결과·가상환경은 Git 업로드에서 제외하므로
새 clone에는 별도로 준비해야 합니다. Homography는 영상의 Scene에 맞게
자동 선택하며, 영상은 원본 해상도와 VIRAT 파일 이름을 유지해야 합니다.

R 파일이 없으면 **평가 영상과 다른 촬영본**의 개발 영상/GT를 준비한 뒤 보정합니다.
최소 100개 유효 대응 표본과 서로 다른 GT 보행자 3명이 필요합니다.

```bash
python src/estimate_r.py \
  --source Resources/videos_original/VIRAT_S_010201_00_000000_000053.mp4 \
  --device 0 --imgsz 960
```

기존 R 파일은 덮어쓰지 않습니다. 모델·Homography·입력 크기·NMS·추적 설정이
변경되면 해당 조건으로 새 R을 보정해야 합니다. `--ekf-r VAR_X VAR_Y`는
미보정 분산으로 동작을 확인할 때만 사용하며 평가 결과에도 provisional로 기록합니다.

## 실행

설치와 데이터 준비를 마쳤으면, 새 터미널을 열 때마다 프로젝트 폴더에서
가상환경을 활성화합니다.

```bash
source .venv/bin/activate
```

기존 터미널 사용 방식과 동일하게 **실행할 Python 파일 뒤에 `--source`로
적용할 데이터셋 영상 경로를 지정**합니다. 파일 이름에 공백이 있으므로
`"src/tracking system.py"`를 따옴표로 감쌉니다.

보행자 추적·EKF 궤적 예측 화면을 표시하고 결과 영상을 저장하려면:

```bash
python "src/tracking system.py" \
  --source Resources/videos_original/VIRAT_S_010208_07_000768_000791.mp4 \
  --device 0
```

**ADE/FDE·처리 FPS·EKF 지연시간까지 파일로 저장하려면** 다음 명령을 사용합니다.
이 명령도 화면을 표시하고 결과 영상을 저장합니다.

```bash
python "src/tracking system.py" \
  --source Resources/videos_original/VIRAT_S_010208_07_000768_000791.mp4 \
  --device 0 \
  --evaluate \
  --measure-predictor-latency
```

다른 데이터셋 영상을 적용할 때는 `--source` 뒤의 경로를 바꿉니다.
해당 영상과 이름이 같은 GT를 `Resources/annotations/`에 준비하고,
그 Scene 및 실행 설정에 맞는 R 보정 파일도 준비해야 합니다.
적용 영상을 바꾸기 위해 코드를 수정하거나 모델을 다시 학습할 필요는 없습니다.

화면 표시와 MP4 저장을 생략하고 성능만 평가하려면:

```bash
python "src/tracking system.py" \
  --source Resources/videos_original/VIRAT_S_010208_07_000768_000791.mp4 \
  --device 0 \
  --evaluate \
  --measure-predictor-latency \
  --no-display \
  --no-save
```

| 옵션 | 역할 |
|---|---|
| `--source 영상_경로` | 적용할 VIRAT 데이터셋 영상 |
| `--device 0` | YOLO를 첫 번째 GPU에서 실행. CPU는 `--device cpu` |
| `--evaluate` | ADE/FDE와 전체 처리 FPS 등의 평가 JSON 저장 |
| `--measure-predictor-latency` | EKF 갱신·미래 예측 지연 측정. `--evaluate`와 함께 사용 |
| `--no-display` | 실행 화면 표시 생략 |
| `--no-save` | 결과 MP4 저장 생략 |

`--max-frames 60`은 짧은 동작 확인에만 사용합니다. 이때 `complete_video=false`이며,
12프레임의 미래가 실제로 처리된 구간만 평가합니다. `q`, `Esc`, 창 닫기로 종료합니다.
전체 영상의 성능을 평가하려면 영상이 끝날 때까지 실행합니다.
`tracking system.py`와 `track.py`는 동일한 실행 구현을 사용합니다.

결과 영상은 `runs/tracking/`, 평가 JSON은 `runs/evaluation/`에 저장합니다.
`--output`, `--evaluation-output`으로 경로를 지정할 수 있습니다.

| JSON 항목 | 의미 |
|---|---|
| `metrics.ekf.ade` / `fde` | 미래 12개 위치 오차 평균 / 마지막 위치 오차 |
| `metrics.linear` | 같은 예측 구간에서 최근 9개 실제 관측으로 회귀한 직선 예측 기준 |
| `processing_speed.processing_fps` | 초기 10프레임을 제외한 전체 처리 FPS |
| `predictor_latency.methods.ekf` | EKF 갱신+미래 예측 지연의 평균 및 P95 등 |
| `evaluated_windows` / `status_counts` | 유효 평가 구간 수 및 제외 사유 |
| `ellipse_95_gt_coverage` | 예측 공분산의 명목 95% 영역에 포함된 GT 비율 |

전체 처리 FPS에는 활성화된 화면·영상 저장·로그·평가가 포함됩니다.
예측기 지연은 검출·추적·그리기를 제외하며, 기본 초기 50프레임을 제외합니다.
같은 실행 옵션으로 속도를 비교해야 합니다.

## 평가 규칙 및 배치 실행

- GT는 매칭·평가·표시에만 사용하며 EKF 관측에 넣지 않습니다.
- IoU 0.5 이상, 후보 간 차이 0.1 이상, 상호 최우선, 3프레임 연속 조건으로
  대응을 확정합니다. DeepSORT ID와 GT ID가 같다고 가정하지 않습니다.
- 시작 시 GT ID를 고정한 12프레임 예측을 평가하고 다음 구간은 t+13 이후 시작합니다.
  미래 GT가 누락되면 보간하지 않고 구간을 제외합니다. 조기 종료도 미처리 미래를 평가하지 않습니다.
- 실제 관측 9개로 초기화하고 추가 9프레임 준비 조건을 사용합니다.
  초기화 중 영상 경계에 닿는 박스는 제외하며 유효 관측 간격이 3프레임을 넘으면 다시 모읍니다.
- 거리 단위는 Scene 좌표 단위입니다. 미터로 간주하거나 서로 다른 Scene의 거리를 합산하지 않습니다.
- 자동 대응이 가능한 부분집합의 성능입니다. ADE/FDE가 전체 보행자 검출률이나 ID 유지 성능을 뜻하지 않습니다.
  명목 95% 타원이 실제로 95%를 포함한다는 보장도 없습니다.

여러 촬영본과 Scene별 R이 준비된 경우 다음 명령으로 분할·배치 평가합니다.
영상 한 개로는 독립 validation/test 분할을 만들 수 없습니다.

```bash
python src/evaluate.py --prepare-split --all-scenes
python src/evaluate.py --split validation --device 0
python src/evaluate.py --split test --device 0
```

촬영본 단위로 분리하고 R 보정 촬영본은 평가에서 제외합니다. Q는 validation으로
결정한 뒤 고정하여 test를 평가합니다. 기본 Q는 기존 값
`(0.001, 0.001, 0.0001, 0.0001, 0.00001)`이며 자동 최적화를 수행하지 않습니다.
