# """실습 07~08: 정상 데이터 20건. train 12건 학습, valid 4건 확인, test 4건 보관."""
# from pathlib import Path
# import csv
# import json
# import torch
# from torch import nn

# # 파일 위치를 기준으로 읽으므로 다른 작업 폴더에서 실행해도 됩니다.
# base = Path(__file__).resolve().parent
# with (base / "points.csv").open(encoding="utf-8", newline="") as f:
#     rows = list(csv.DictReader(f))
# train_rows = [r for r in rows if r["split"] == "train"]
# valid_rows = [r for r in rows if r["split"] == "valid"]
# X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
# y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
# X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
# y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# # 기본필수는 CPU 고정, full-batch 80회입니다. test는 Tensor로 만들지 않습니다.
# torch.manual_seed(42)
# model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# history = [[0, loss_fn(model(X_train), y_train).item()]]
# model.train()
# for epoch in range(1, 81):
#     optimizer.zero_grad()
#     loss = loss_fn(model(X_train), y_train)
#     # 06번에서 배운 역전파와 파라미터 갱신 두 문장을 연결하세요.
#     # 이미 위에서 이전 gradient를 비우고 loss를 계산했습니다.
#     # >>> B08
#     # TODO B08: 위 요구사항을 읽고 이 블록을 완성하세요.
     
#     # <<< B08
#     # 갱신 후 같은 train 자료의 loss를 기록합니다. 마지막 batch loss와 구분합니다.
#     with torch.no_grad():
#         history.append([epoch, loss_fn(model(X_train), y_train).item()])
# model.eval()
# with torch.no_grad():
#     logits = model(X_valid)
#     # 각 행에서 가장 큰 logit의 열 번호를 prediction으로 구하세요.
#     # argmax의 dim=1은 클래스 방향입니다. 결과 shape는 (4,)입니다.
#     # >>> B09
#     # TODO B09: 위 요구사항을 읽고 이 블록을 완성하세요.
#     prediction = torch.zeros(len(y_valid), dtype=torch.long)
#     # <<< B09
#     accuracy = (prediction == y_valid).float().mean().item()
# with (base / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["epoch", "train_loss_eval"])
#     writer.writerows(history)
# metrics = {"device": "cpu", "epochs": 80, "train_rows": 12, "valid_rows": 4,
#            "test_used": False, "valid_accuracy": accuracy,
#            "initial_train_loss": history[0][1], "final_train_loss": history[-1][1],
#            "valid_ids": [r["id"] for r in valid_rows], "actual": y_valid.tolist(),
#            "predicted": prediction.tolist()}
# (base / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
# (base / "gpu_check.txt").write_text(
#     f"torch={torch.__version__}\ncuda_build={torch.version.cuda}\n"
#     f"cuda_available={torch.cuda.is_available()}\nselected_device=cpu\n", encoding="utf-8")
# print("분할: train 12 / valid 4 / test 4(보관)")
# print("train loss:", round(history[0][1], 6), "->", round(history[-1][1], 6))
# print("valid 정확도:", round(accuracy, 4))
# print("valid 예측:", prediction.tolist())
# print("생성: gpu_check.txt, training_history.csv, final_metrics.json")





"""실습 07~08: 정상 데이터 20건. train 12건 학습, valid 4건 확인, test 4건 보관."""
from pathlib import Path  # 경로 처리를 객체 지향적으로 수행하는 모듈입니다.
import csv                 # CSV 로그 기록 및 파일 읽기를 위한 모듈입니다.
import json                # 최종 평가지표를 구조화된 파일로 내보내기 위한 모듈입니다.
import torch               # PyTorch 프레임워크 핵심 라이브러리입니다.
from torch import nn       # 인공신경망 레이어 컨테이너 모듈입니다.

# 파일 위치를 기준으로 읽으므로 다른 작업 폴더에서 실행해도 작동합니다.
base = Path(__file__).resolve().parent

# base 경로 폴더 안의 points.csv 파일을 파이썬 딕셔너리 리스트 구조로 읽어옵니다.
with (base / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

# 데이터의 split 속성을 확인하여 훈련용(train) 행과 검증용(valid) 행을 분할 추출합니다.
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]

# 훈련 및 검증 데이터를 신경망이 읽을 수 있는 PyTorch 텐서(Tensor) 객체로 변환합니다.
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)

# 기본필수는 CPU 고정, full-batch 80회입니다. test는 오염 방지를 위해 텐서로 만들지 않고 보관합니다.
torch.manual_seed(42) # 가중치 난수 초기화 시드를 42로 고정하여 재현성을 보장합니다.

# 입력 특성 2개 -> 은닉층 뉴런 8개(ReLU) -> 출력 클래스 2개(0번, 1번 점수) 구조의 신경망 모델을 만듭니다.
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

# 교차 엔트로피 손실함수 채점관과 확률적 경사하강법(SGD, 학습률 0.1) 최적화 도구를 선언합니다.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 학습을 시작하기 전(0 에폭) 무작위 상태인 가중치 기준으로 초기 훈련 손실값을 리스트에 채워둡니다.
history = [[0, loss_fn(model(X_train), y_train).item()]]

# 모델의 내부 하드웨어 상태를 '학습 모드'로 전환합니다.
model.train()

# 1 에폭부터 81 에폭까지 학습 반복 루프를 돕니다. (총 81회 업데이트)
for epoch in range(1, 81):
    # 1단계. 가중치 파라미터 메모리에 기록된 이전 에폭의 잔여 미분 계수를 청소합니다.
    optimizer.zero_grad()
    
    # 2단계 & 3단계. 12건의 전체 훈련 세트를 한꺼번에 통과(Full-batch)시켜 예측하고 손실 오차를 도출합니다.
    loss = loss_fn(model(X_train), y_train)
    
    # 06번에서 배운 역전파와 파라미터 갱신 두 문장을 연결하세요.
    # 이미 위에서 이전 gradient를 비우고 loss를 계산했습니다.
    # >>> B08
    # TODO B08: 위 요구사항을 읽고 이 블록을 완성하세요.
    loss.backward()   # 4단계. 오차 기반 역전파를 작동시켜 모델 파라미터별 실제 미분값(.grad)을 계산해 냅니다.
    optimizer.step()  # 5단계. 연산 완료된 미분 가중치 방향으로 모델 내부 파라미터 수치를 갱신합니다.
    # <<< B08
    
    # 갱신 후 같은 train 자료의 loss를 기록합니다. 마지막 batch loss와 구분합니다.
    # 기록을 채점하는 과정은 미분이 필요 없으므로 no_grad 컨텍스트 안에서 불필요한 메모리 소모를 아낍니다.
    with torch.no_grad():
        history.append([epoch, loss_fn(model(X_train), y_train).item()])

# 80에폭의 학습이 완수되었으므로 모델 상태를 '평가/추론 모드'로 고정합니다.
model.eval()

# 미분 계수 기록을 정지하고 4건의 검증 데이터를 사용해 최종 검증 점수를 도출합니다.
with torch.no_grad():
    logits = model(X_valid) # 4건의 검증 데이터를 통과시켜 각 원시 클래스 점수(Logits) 배열을 뽑습니다.
    
    # 각 행에서 가장 큰 logit의 열 번호를 prediction으로 구하세요.
    # argmax의 dim=1은 클래스 방향입니다. 결과 shape는 (4,)입니다.
    # >>> B09
    # TODO B09: 위 요구사항을 읽고 이 블록을 완성하세요.
    # 가로축 방향(dim=1)으로 가장 수치가 높은(예측 확률이 높은) 방 번호 인덱스(0 또는 1)를 모아 클래스를 확정합니다.
    prediction = logits.argmax(dim=1)
    # <<< B09
    
    # 예측해낸 방 번호 배열과 실제 정답(y_valid) 텐서의 정수가 일치하는지 비교하고 참(1.0)/거짓(0.0) 평균으로 정확도를 집계합니다.
    accuracy = (prediction == y_valid).float().mean().item()

# 기록된 훈련 손실 추이 데이터 배열을 csv 텍스트 구조로 파일에 기록합니다.
with (base / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss_eval"])
    writer.writerows(history)

# 프로젝트 결과물 검수 검증 계약 조건에 맞춘 지표를 파이썬 딕셔너리로 작성합니다.
metrics = {"device": "cpu", "epochs": 80, "train_rows": 12, "valid_rows": 4,
           "test_used": False, "valid_accuracy": accuracy,
           "initial_train_loss": history[0][1], "final_train_loss": history[-1][1],
           "valid_ids": [r["id"] for r in valid_rows], "actual": y_valid.tolist(),
           "predicted": prediction.tolist()}

# 생성 완료된 딕셔너리 지표 사전을 깔끔한 들여쓰기가 포함된 JSON 형식 파일 문자열로 직렬화하여 내보냅니다.
(base / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

# 강의 검수 규격 자동 채점 엔진용 가상환경 GPU 탑재 정보 체크용 텍스트 문서를 강제로 떨굽니다.
(base / "gpu_check.txt").write_text(
    f"torch={torch.__version__}\ncuda_build={torch.version.cuda}\n"
    f"cuda_available={torch.cuda.is_available()}\nselected_device=cpu\n", encoding="utf-8")

# 최종 완성 연산 지표 스코어를 인간 개발자가 읽을 수 있도록 표준 CLI 콘솔에 프린트합니다.
print("분할: train 12 / valid 4 / test 4(보관)")
print("train loss:", round(history[0][1], 6), "->", round(history[-1][1], 6))
print("valid 정확도:", round(accuracy, 4))
print("valid 예측:", prediction.tolist())
print("생성: gpu_check.txt, training_history.csv, final_metrics.json")
