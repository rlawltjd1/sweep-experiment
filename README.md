# Steering Vectors for Implicit Hate Detection

Minimal-pair(Cell A/B/C)에서 뽑은 스티어링 벡터를 Llama-3.2-3B의 특정 레이어에 주입해, 외부 implicit hate 평가셋에서 baseline 대비 탐지 성능(macro F1·FN recovery)을 올릴 수 있는지를 실험한다.

**한 줄 가설.** 모델 내부에 harm 방향이 존재하지만 implicit hate에서는 잘 활성화되지 않는다(dehatebert P(hate) 0.80→0.04). 추론 시점에 minimal-pair 벡터를 그 방향으로 주입하면 implicit hate 탐지가, 특히 baseline이 놓치던 케이스에서 회복된다.

- **H1 (메인)** `v_AB` = A−B (cue 고정·target 변화) → target 축
- **H2 (보조)** `v_AC` = A−C (target 고정·cue 변화) → cue 축

---

## 1. 전체 흐름

```
[Week 1]  week1_pipeline.ipynb
     평가셋 잠금 · probe 학습 · B0 측정
        │  
        ▼
[Week 2]  submit_sweep.sh → week2_sweep.py
     벡터 4종 생성 · coarse+fine sweep
        │  (sweep csv 다운로드)
        ▼
[Week 3]  week3_analysis.ipynb
     ablation · 유의성 검정 · axis 연결 · 최종 표/그래프
```

핵심 불변식(전 단계 공통): **모든 hidden·주입 = left-padding 마지막 토큰**, **probe는 no-steering HateXplain으로 한 번만 학습**, **4벡터 모두 레이어별 단위정규화**, **평가셋·split·라벨매핑은 1주차 이후 불변**.

---

## 2. 디렉터리 구조

```
프로젝트/
├─ README.md
├─ week1_pipeline.ipynb        # Week 1: 평가셋·probe·B0 (로컬 실행)
├─ week2_sweep.py                # Week 2: sweep (서버 실행, 체크포인트/resume)
├─ submit_sweep.sh             # Week 2: SLURM 제출 스크립트
├─ week3_analysis.ipynb        # Week 3: ablation·유의성·최종 산출물
│
├─ cell_c_test_final.csv               # 입력: Cell A 원문 + Cell C
├─ cell_bbb_domain_v10_256_revised.csv # 입력: Cell B
│
├─ data/
│  ├─ hatexplain_train.csv         # (HF 생성 시) HateXplain 전체 풀 — probe는 이 중 90% 학습
│  └─ eval/
│     ├─ eval_latent_v2.csv        # 입력(직접 제작): Latent Hatred 2,000건
│     ├─ eval_toxigen_v1.csv       # 산출: ToxiGen-HumanVal 평가셋
│     ├─ sanity_hatexplain.csv     # 산출: HateXplain held-out 10% (in-domain sanity)
│     └─ leakage_check.md          # 산출: 겹침 0건 로그
├─ src/
│  └─ eval/
│     └─ metrics.py                # 산출: 단일 evaluate() 함수
└─ results/
   ├─ probe.pkl                    # 산출: scaler + logistic regression
   ├─ b0_baseline.json             # 산출: No-steering 기준 숫자
   ├─ fn_subset_eval_latent_v2.npy
   ├─ fn_subset_eval_toxigen_v1.npy
   ├─ v_AB.npy v_AC.npy v_random.npy v_harm.npy
   ├─ harm_gap_z_lasttoken.npy
   ├─ sweep_coarse_*.csv  sweep_fine_*.csv
   ├─ main_table_*.csv  final_table_*.csv
   ├─ ablation_*.csv  significance.json
   └─ graphA_axis_*.png
```

---

## 3. 실행 방법

### 환경
```
python>=3.10
torch (CUDA), transformers, datasets, scikit-learn,
pandas, numpy, scipy, matplotlib, joblib, tqdm
```
Llama-3.2-3B · HateXplain · ToxiGen은 gated이므로 HF 토큰 필요(`huggingface-cli login` 또는 `HF_TOKEN`).

### Week 1 
`week1_pipeline.ipynb`를 위에서부터 실행. **입력으로 `data/eval/eval_latent_v2.csv`가 미리 있어야 함**(Latent Hatred 2,000건, `subtype` 포함). 끝나면 `results/`·`data/eval/`·`src/eval/`에 산출물이 생긴다.

### Week 2
Week 1 산출물이 존재한다면:
```bash
sbatch submit_sweep.sh        # precheck 통과하면 week2_sweep.py 실행
# 또는 직접:
python week2_sweep.py --batch 64 --all-layers --eval both
# latent 평가셋만 돌릴 경우:
python week2_sweep.py --batch 64 --all-layers --eval latent
```
walltime에 잘려도 **같은 명령 재제출 시 끝난 셋업은 건너뛰고 이어서 돈다.**

### Week 3 
sweep csv가 `results/`에 있는 상태에서 `week3_analysis.ipynb`를 실행.

---

## 4. 파일별 상세

### 4.1 입력 데이터

**`cell_c_test_final.csv`** — 행 ~256
| 컬럼 | 내용 |
| --- | --- |
| `idx` | 쌍 식별자 (S1, S2, …) |
| `cell_a` | **Cell A**: 노골적 혐오 원문 (slur+cue+target) |
| `cell_c_generated` | Cell C 생성 원본 |
| `cell_c_modified` | **Cell C**: cue 제거·target 유지 (implicit hate) ← 실제 사용 |

**`cell_bbb_domain_v10_256_revised.csv`** — 행 ~256
| 컬럼 | 내용 |
| --- | --- |
| `text_clean` | Cell A 원문 (위 `cell_a`와 매칭 키) |
| `generated_text` | **Cell B**: target 중립화·cue 유지 ← 실제 사용(최종) |
| (그 외) `rewrite, domain, check, turn_1, turn_2` | 중간 생성물 |

> A/B/C는 같은 원문에서 파생된 minimal-pair. `cell_a`(=`text_clean`) 텍스트로 join해 **A/B/C 모두 존재하는 triple**만 벡터 추출에 사용.

**`data/eval/eval_latent_v2.csv`** — 2,000행 (직접 제작, 잠금)
| 컬럼 | 내용 |
| --- | --- |
| `text` | Latent Hatred 글 |
| `label` | `"hate"` / `"non-hate"` (코드에서 1/0 매핑) |
| `subtype` | `implicit_hate` / `explicit_hate` / `not_hate` (implicit-only 분석용) |

---

### 4.2 `week1_pipeline.ipynb` 

평가 파이프라인을 잠그는 단계. 셀 그룹별 동작:

1. **설정·모델 로드** — Llama-3.2-3B fp16, **left-padding**(마지막 토큰=위치 −1 고정).
2. **공용 함수** — `extract_hidden`(left-pad 마지막 토큰), `SteeringHook`(레이어 출력 마지막 토큰에 +α·v; tuple/tensor 출력 모두 처리), leakage용 `norm`.
3. **`metrics.py` 작성** — `evaluate(preds,labels)`가 `macro_f1`,`hate_recall`만 반환.
4. **eval_latent_v2 로드·정렬** — 라벨 1/0 매핑, Cell A/B/C join → triple.
5. **leakage check** — eval_latent_v2 ∩ Cell, eval_latent_v2 ∩ HateXplain 교집합 크기를 잰다.
6. **eval_toxigen_v1 구축** — ToxiGen `annotated` split, `toxicity_ai≥4`→hate·`≤2`→non-hate·`==3`제외, 그룹×label 셀당 60건 stratified.
7. **probe 학습** — HateXplain(explicit 중심)을 **90/10 stratified split**, **90%** 마지막 토큰 hidden → StandardScaler + LogisticRegression. **no-steering으로 한 번만 학습.** 나머지 10%는 sanity로.
8. **sanity_hatexplain 구축** — held-out 10%에서 균형 ~500건 → in-domain sanity 평가셋.
9. **B0 측정** — 세 평가셋(eval_latent_v2·eval_toxigen_v1·sanity_hatexplain) 무개입 분류 → macro F1·hate recall.
10. **FN subset 저장** — 메인 두 평가셋에서 B0가 놓친 hate 인덱스(이후 sweep 내내 고정).
11. **v_AB sanity** — 한 레이어 주입 시 F1이 B0와 달라지는지(배선 점검) + `v_AB.npy` 저장.

**산출물**
| 파일 | 형식 | 내용 |
| --- | --- | --- |
| `src/eval/metrics.py` | py | 단일 `evaluate()` |
| `data/eval/eval_toxigen_v1.csv` | csv | `id,text,label,target_group,toxicity_ai,source` |
| `data/eval/sanity_hatexplain.csv` | csv | `text,label,source` (HateXplain held-out 10%, 균형 ~500) |
| `data/eval/leakage_check.md` | md | eval_latent_v2 ∩ Cell / ∩ HateXplain 건수 |
| `results/probe.pkl` | pickle | `{"scaler":…, "clf":…}` (joblib) — HateXplain **90%** 학습 |
| `results/b0_baseline.json` | json | 세 평가셋 B0 (아래 4.6 참조) |
| `results/fn_subset_eval_latent_v2.npy` | npy int[] | B0 false-negative 인덱스 |
| `results/fn_subset_eval_toxigen_v1.npy` | npy int[] | 〃 |
| `results/v_AB.npy` | npy (29,3072) | target 축 벡터 (sweep에서 재생성됨) |
| `data/hatexplain_train.csv` | csv | (HF 생성 시) HateXplain 전체 풀; probe는 이 중 90% 사용 |

이미 로컬에서 돌린 산출물도 함께 push 되어 있지만 긴 시간 소요 없으므로 서버에서 처음부터 Week 1 ipynb 노트북부터 다시 돌리는 것 권장

---

### 4.3 `week2_sweep.py` 

1주차 자산을 받아 **스티어링을 주입하며 sweep**. 체크포인트/resume 지원.

**동작**
- 1주차 자산(probe·b0·fn·평가셋) 로드.
- **벡터 4종 생성**(없으면): `v_AB`,`v_AC`(Cell A/B/C에서), `v_random`, **`v_harm`(HateXplain harm/safe에서 last-token 재추출)** — 전부 레이어별 단위정규화.
- 평가셋을 한 번만 토큰화(`prebatch`).
- **coarse sweep**: 4벡터 × 레이어 × α{0.5,1,2,4}, 각 셋업을 csv에 **한 줄씩 append**(중간에 죽어도 보존).
- **fine sweep**: `v_AB`,`v_AC` best 주변 레이어±2·α 0.25 간격 + 부호 −α.
- **메인 표** 저장.

**주요 옵션**
| 옵션 | 기본 | 의미 |
| --- | --- | --- |
| `--batch` | 32 | GPU 메모리에 맞게(서버 64~128) |
| `--all-layers` | off | 켜면 0~28 전체, 아니면 우선순위 17개 |
| `--eval` | both | `latent` / `tg` / `both` |
| `--alphas` | 0.5,1,2,4 | coarse α 격자 |
| `--no-fine` | off | fine sweep 생략 |

**산출물**
| 파일 | 형식 | 내용 |
| --- | --- | --- |
| `results/v_AB.npy v_AC.npy v_random.npy v_harm.npy` | npy (29,3072) | 단위정규화 스티어링 벡터 4종 |
| `results/sweep_coarse_eval_latent_v2.csv` | csv | 컬럼: `vector,layer,alpha,macro_f1,hate_recall,fn_recovery,d_f1` |
| `results/sweep_coarse_eval_toxigen_v1.csv` | csv | 〃 |
| `results/sweep_fine_eval_latent_v2.csv` | csv | 〃 (best 주변) |
| `results/sweep_fine_eval_toxigen_v1.csv` | csv | 〃 |
| `results/main_table_eval_latent_v2.csv` | csv | B0/B1/B2/E1/E2 best 행 |
| `results/main_table_eval_toxigen_v1.csv` | csv | 〃 |

> `d_f1` = 해당 셋업 macro_f1 − B0 macro_f1. `fn_recovery` = B0가 놓친 hate 중 이 셋업이 hate로 잡은 비율.

---

### 4.4 `submit_sweep.sh` 

GPU 1장 작업으로 `week2_sweep.py`를 제출. 실행 전 **1주차 산출물 9개 존재 여부를 precheck**하고 하나라도 없으면 어떤 파일인지 찍고 종료. 죽어도 재제출하면 resume.
손볼 곳: `--partition`/`--account`, 환경 활성화 줄(conda/venv/module), `HF_TOKEN`/`HF_HOME`.

---

### 4.5 `week3_analysis.ipynb`

sweep 결과로 **해석을 굳히는** 단계. 새 분석은 추가하지 않는다.

**동작**
1. sweep csv·벡터·probe·b0·fn 로드, best 셋업 확정(coarse+fine).
2. **Ablation** — 메인 셋업에서 `α=0`(=B0 확인), `부호반전 −α`, `first-token`(left-pad 보정해 첫 실제 토큰), `all-token` 주입.
3. **유의성 검정** — B0 vs E1(v_AB): **McNemar** p값 + **ΔmacroF1 bootstrap 95% CI**.
4. **implicit-only 분석** — `eval_latent_v2.csv`의 `subtype`별로 hate recall B0→E1 회복. implicit_hate Δ가 explicit_hate Δ보다 큰지가 핵심 메시지.
5. **Axis 연결 (last-token 재산출)** — Cell A/C를 last-token으로 뽑아 레이어별 harm 축 사영 갭(`harm_gap_z`)을 계산하고, 그래프 A에 **이중축 곡선**으로 겹친다. 성능 봉우리와 harm 축 봉우리가 같은 레이어에서 겹치는지를 sweep과 **같은 풀링(last-token)** 으로 보여 풀링 차이 각주를 없앤다. 0528 mean-pool 리스트(`CFG.HARM_LAYERS`)는 비교 출력용 motivation으로만 유지.
6. **그래프 B (α dose-response)** — v_AB·v_AC best layer에서 α를 sweep한 macro F1 곡선(양수 α). 효과의 sweet spot(너무 작으면 무효·너무 크면 과왜곡)을 보인다. 평가셋별 1장.
7. **최종 표 / 한계 단락 템플릿** — 표 컬럼은 문서 §5와 동일하게 `Setup, layer, alpha, macro_f1, hate_recall, **fn_recovery**, d_f1`.

**산출물**
| 파일 | 형식 | 내용 |
| --- | --- | --- |
| `results/ablation_eval_latent_v2.csv` | csv | `vector,L,setting,macro_f1,d_f1,fn_recovery` |
| `results/ablation_eval_toxigen_v1.csv` | csv | 〃 |
| `results/significance.json` | json | McNemar(b,c,p) + bootstrap CI + best 셋업 |
| `results/harm_gap_z_lasttoken.npy` | npy (29,) | 레이어별 harm 축 A−C 갭 z (last-token) |
| `results/final_table_eval_latent_v2.csv` | csv | 본문용 최종 표 (`Setup,layer,alpha,macro_f1,hate_recall,fn_recovery,d_f1`) |
| `results/final_table_eval_toxigen_v1.csv` | csv | 〃 |
| `results/graphA_axis_eval_latent_v2.png` | png | 레이어별 F1 + harm 축 곡선(last-token, 이중축) |
| `results/graphA_axis_eval_toxigen_v1.png` | png | 〃 |
| `results/graphB_eval_latent_v2.png` | png | best layer에서 α dose-response (v_AB·v_AC) |
| `results/graphB_eval_toxigen_v1.png` | png | 〃 |
---

### 4.6 산출물 형식 상세

**`results/probe.pkl`** (joblib) — `{"scaler": StandardScaler, "clf": LogisticRegression}`. 입력 3072-d(Llama 마지막 레이어 last-token), 출력 hate(1)/non-hate(0).

**`results/b0_baseline.json`**
```json
{
  "model": "meta-llama/Llama-3.2-3B",
  "probe": "logistic_regression",
  "layer": -1,
  "results": {
    "eval_latent_v2":    {"macro_f1": 0.5987, "hate_recall": ..., "n": 2000},
    "eval_toxigen_v1":   {"macro_f1": 0.6313, "hate_recall": ..., "n": 2397},
    "sanity_hatexplain": {"macro_f1": ...,    "hate_recall": ..., "n": ~500}
  }
}
```

**`results/v_*.npy`** — `np.float` 배열 `(29, 3072)` = (임베딩 포함 29개 레이어, hidden). 각 행은 단위벡터.

**`results/fn_subset_*.npy`** — `np.int` 1차원. 해당 평가셋에서 B0가 hate를 non-hate로 놓친 행 인덱스.

**`results/significance.json`** (평가셋별 키)
```json
{"eval_latent_v2": {
   "B0_macro_f1":0.5987,"E1_macro_f1":...,"delta":...,
   "mcnemar":{"b_only_B0":...,"c_only_E1":...,"p_value":...},
   "bootstrap_delta_ci":{"lo":...,"med":...,"hi":...},
   "best":{"layer":...,"alpha":...}}}
```

---

## 5. 핵심 설계 결정 및 변경 사항

- **probe 학습 = HateXplain(explicit 중심) 90%.** in-domain(Latent Hatred)으로 학습하면 B0가 implicit을 이미 맞혀 회복 여지가 줄고 OOD 서사가 깨진다. HateXplain 90% 학습 → Latent Hatred 2,000건 전체가 test. 남은 **10%는 sanity_hatexplain**(in-domain sanity, probe·v_harm 양쪽에서 held-out).
- **B2 = v_harm control.** "implicit을 안 본 probe니 당연히 좋아지는 것 아니냐"는 반론은, 같은 HateXplain harm 방향 v_harm을 v_AB/v_AC가 이겨야 "minimal-pair의 추가 가치"가 증명되는 구조로 막는다.
- **left-padding + 마지막 토큰.** 배치 추론에서 위치 −1을 항상 진짜 마지막 토큰으로 정렬 → 주입·probe 위치 일치.
- **벡터 last-token·단위정규화.** 0528의 v_harm은 mean-pool로 뽑혀 공간이 달랐으므로 sweep 직전 last-token으로 재추출(B2 공정성). 단위정규화로 α가 벡터·레이어 무관하게 같은 의미.
- **leakage guard.** 벡터를 뽑는 Cell/HateXplain 문장이 평가셋과 겹치면 점수가 부풀려지므로 모두 제거·기록. v_harm 추출 시 `sanity_hatexplain` 텍스트도 가드에 포함해 held-out을 보장.

**벡터 풀링**
- 기존 문서: mean-pool => 실제 한 것: last-token(B0/probe는 마지막 토큰, 벡터는 mean-pool, 주입은 마지막 토큰으로 셋을 마지막 토큰으로 통일. 따라서 0528 mean-pool 기반 harm-dominant 레이어와 best layer가 다를 수 있음)
    -> 향후 mean-pool 혹은 last token 방식 결정 필요

**v_harm**
- 기존 문서: mean-pool 파일 재사용 => 실제 한 것: last-token으로 재추출(위와 같은 이유. B2 비교 공정성)


**논의 사항**
기존 문서에서는 v_AB, v_AC의 vector extraction과 0528 harm-dominant layer 분석이 mean-pool hidden state 기준으로 설명되어 있었다. 반면 B0/probe는 causal LM의 최종 판단 위치와 직접 연결되는 last-token hidden state를 사용하는 것으로 계획되어 있었고, steering intervention 역시 해당 layer의 마지막 토큰 위치에 +α·v를 주입하는 방식으로 정리되어 있었다. 즉, 기존 계획에는 vector extraction 및 layer prior는 mean-pool 기준, B0/probe 및 steering 주입 위치는 last-token 기준이라는 불일치가 있었다. 

이에 따라 현재 구현에서는 B0/probe가 last-token representation을 사용하고 steering 주입도 마지막 토큰 위치에 적용되는 점을 고려하여, B2 비교 공정성을 위해 v_harm 역시 기존 mean-pool 파일을 재사용하지 않고 last-token 기준으로 재추출하도록 하였다.

다만 이 변경으로 인해 0528 mean-pool 기반 harm-dominant layer 및 best layer prior와 현재 last-token 기반 sweep 결과는 동일 조건의 layer 분석으로 직접 비교하기 어렵다. mean-pool은 문장 전체 token representation을 평균낸 sentence-level signal에 가깝고, last-token은 causal LM의 다음 토큰 예측 및 최종 판단 위치와 더 직접적으로 연결된 signal이므로, 두 방식에서 강하게 나타나는 layer가 달라질 수 있다.

따라서 향후 main pooling 기준을 명시적으로 결정해야 한다. 선택지는 크게 세 가지이다. 
- 첫째, 현재 구현처럼 last-token을 main setting으로 고정하여 B0/probe, vector extraction, steering 주입 위치를 모두 causal LM의 최종 판단 위치 기준으로 통일하는 방식이다. 이 경우 0528 mean-pool 기반 harm-dominant layer 및 best layer prior를 그대로 사용하지 않고, 동일한 분석을 last-token 기준으로 재수행하여 layer prior를 재산출해야 한다.
- 둘째, 기존 0528 분석과의 연속성을 유지하기 위해 mean-pool을 main setting으로 두되, 그에 맞춰 B0/probe와 steering 방식까지 mean-pool 기준으로 재정의 및 재구현하는 방식이다. 
- 셋째, 기존 계획대로 mean-pooling으로 추출한 벡터를 last token에 주입하는 방식으로 진행한다. 하지만 이는 성능 변화가 벡터가 포착한 정보 때문인지 아니면 풀링 방식의 불일치로 인한 노이즈 때문인지 분리하기 어려울 것으로 우려된다.

가장 안전한 방법은 last-token과 mean-pool을 모두 ablation으로 비교하여 pooling 방식 자체가 성능 및 best layer 선택에 미치는 영향을 확인하는 것이다.


## 6. 알려진 한계 / 주의

- harm-dominant 레이어·`|z|` 수치는 mean-pool 공간 산출 → last-token best layer가 다를 수 있음(harm-dominant 레이어를 last-token 기준으로 재산출해야할 수 있음).
- probe·B0는 로컬, sweep은 서버 GPU → fp16 미세차 가능. 엄밀히 하려면 Week 3에서 B0 재계산값으로 교차확인(significance.json).
- ToxiGen `target_group`이 원본 free-text라 그룹 수가 13개보다 많을 수 있음(현재 27). 필요 시 표준 카테고리로 병합.
- 인과 주장은 단일 모델(Llama-3.2-3B)·추론 시점 개입 수준에 한정.
- Week 1 파이프라인은 로컬에서 돌려 문제 없이 동작하는 것을 확인했지만 Week 2, 3은 실제로 실행 후 동작 확인 필요
---
