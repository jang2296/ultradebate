# 작업 배분 모델 재현

프로젝트 루트에서 가상환경을 활성화하고 선택 의존성을 설치합니다. 공개본에는 실제 사용 로그나 학습된 모델이 포함되지 않습니다.

```bash
python -m pip install -e '.[mlp]'
python3 scripts/generate_synthetic_dataset.py \
  --out artifacts/features_synth.csv --num-samples 1500 --seed 42
python3 scripts/train_allocator_mlp.py \
  --data artifacts/features_synth.csv \
  --out-model artifacts/allocator_mlp.joblib \
  --out-meta artifacts/allocator_meta.json \
  --out-report artifacts/training_report.md \
  --out-plots artifacts/plots/ --seed 42
```

모델과 `model_sha256`이 포함된 메타데이터를 생성합니다. 결과는 합성 데이터에서의 검증이며 실제 작업 배분 성능을 뜻하지 않습니다. MCP의 `ultradebate_v2_allocator_info`로 모델 로드 상태를 확인하세요.

직접 수집한 로그를 사용하려면 다음 입력 인자를 확인합니다. 대화 내용·프로젝트 경로·사용 이력이 포함될 수 있으므로 원본과 추출 데이터를 공개 저장소에 추가하지 마세요.

```bash
python3 scripts/extract_features.py --help
python3 scripts/label_real_features.py --help
```
