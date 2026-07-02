# GR00T + PLADIS / GR00T + ACG + PLADIS 통합 — 태스크 브레이크다운

> 발표자료 `260629 GROOT-N1 기반 단일 패스 제어 최적화: ACG 한계 극복을 위한 PLADIS 절제 연구`의
> 구현 계획. 작성 2026-07-01. 모든 코드 변경은 이 저장소(`~/parkkwanjoon/workspace/ACG`) 안에서 이루어짐.
> PLADIS 알고리즘 원본 참조는 `~/parkkwanjoon/workspace/PLADIS/pipeline/`.

---

## 0. 배경 / 설계 요지 (읽고 시작)

- **ACG** (`libs/robomimic/robomimic/algo/guidance/acg.py`): DiT의 **self-attention** 을 Identity로
  교체한 "incoherent" 예측을 따로 만들어 `v = (1+λ)v_θ − λ·v_incoherent` 로 steering. **2-pass**
  (기준 1회 + skip_blocks 재실행 1회) → 연산 ~1.5배(13/8).
- **PLADIS** (`~/parkkwanjoon/workspace/PLADIS`): **cross-attention** 에서 dense(softmax)와
  sparse(α-entmax)를 한 번에 계산해 `Attn_out = dense + λ(sparse − dense)` 로 외삽. **1-pass**
  (추가 NFE 없음). 목표: 추론속도 바닐라 수준(~64ms) 유지.
- **핵심 사실 (검증됨):** GR00T DiT는 `interleave_self_attention=True`, 실제 체크포인트 **16 블록**.
  `DiT.forward` (`libs/Isaac-GR00T-N1/gr00t/model/action_head/cross_attention_dit.py:274-290`)에서
  - **짝수 idx 블록(0,2,…,14) = cross-attention** (`encoder_hidden_states=vl_embs`)
  - **홀수 idx 블록(1,3,…,15) = self-attention** (`encoder_hidden_states=None`) = self-attn 레이어 1–8
  - ACG 타깃 `skip_blocks=[7,9,11]` = self-attn 레이어 4·5·6 (발표 "Layer 4–6"과 일치).
  - **⇒ PLADIS(cross=짝수)와 ACG(self=홀수 일부)는 서로 다른 블록** → 프로세서 교체 레벨에서 충돌 없음.
- 모든 블록은 `attn1` 단일 Attention 모듈을 사용. 가이던스는 `transformer_blocks[i].attn1.processor`
  를 교체하는 방식(ACG가 이미 이렇게 함). PLADIS도 동일 패턴으로 구현.

### 디스패치/패치 관례 (반드시 준수)
`algo/guidance/__init__.py`의 `modify_gr00t_policy(name, policy)`가 name→클래스로 4개 메서드를
monkey-patch: `Gr00tPolicy.get_action`, `Gr00tPolicy._get_action_from_normalized_input`,
`GR00T_N1.get_action`, `FlowmatchingActionHead.get_action`. 새 방식은 이 4개를 그대로 미러링.

---

## Phase 0 — 준비 / 기준선 확정  ✅ 완료 (2026-07-01)
- [x] `pip install entmax` (conda env `acg`). → **entmax 1.3 설치·검증 완료.**
      attention 형태(1,8,17,99) 텐서에서 entmax15=89% 희소 / sparsemax=97% 희소 / dense=0%,
      각 행 합=1, **bf16→float32 캐스팅 시 NaN 없음** 확인.
- [x] PLADIS 원본 프로세서 정독: `PLADIS/pipeline/pipeline_sdxl_seg.py`의
      `sparse_scaled_dot_product_attention` (L68-104) + `SparseAttnProcessor` (L107-).
      → 공식 `attn = pladis_scale*sparse + (1-pladis_scale)*dense` = `dense + λ(sparse-dense)`, float 캐스팅.
- [x] ACG 프로세서/흐름 정독: `acg.py` `ACGAttnProcessor2_0`, `FlowmatchingActionHead_ACG.get_action`
      (denoising 루프 L176-267). **반환부 확인:** `pred = pred + (scale-1)(pred-pred_perturb)` (L259) →
      euler `actions += dt*pred_velocity`, `pred_velocity=pred[:,-action_horizon:]`,
      `return BatchFeature({"action_pred": actions})`. PLADIS는 L259만 `pred=model_output`으로 바꾸고 2-pass 제거.
- [x] baseline offset 인지: 재현에서 baseline이 논문보다 ~8.4%p 높음(sdpa 유력). PLADIS 효과도
      head-room 축소로 작게 보일 수 있음 — ablation 해석 시 감안. (근거: `EXPERIMENT_NOTES.md §8`)

**완료 기준:** entmax 동작 ✅, 두 알고리즘의 attention 처리 지점 코드로 지목 가능 ✅.

### ⚠️ Phase 1 배선을 위한 필수 확인 사실 (Phase 0에서 발견)
- **config→get_action 인자 전달 경로:** `gr00t_guidance.py:45` `policy_get_action_kwargs =
  self.algo_config.guidance.to_dict()` → `:66` `policy.get_action(obs, **kwargs)`. 즉 **`algo.guidance`
  하위 전체 키가 통째로 kwargs로 전달됨** (`name, scale, num_inference_timesteps, skip_blocks,
  threshold, n_ensemble, sigma, noise_std`, + 새로 추가할 `pladis_scale, method`).
  → **`Gr00tPolicy_PLADIS.get_action`는 `pladis_scale/method/num_inference_timesteps`만 명시 인자로 받고
  나머지(특히 `name`, ACG용 `scale/skip_blocks`)는 반드시 `**kwargs`로 흡수**해야 죽지 않음. (ACG도 동일 관례)
- **config 위치:** `gr00t_config.py:41-53`. 모든 guidance 키가 평평하게 존재. 여기에
  `self.algo.guidance.pladis_scale = 1.5`, `self.algo.guidance.method = "ent15max"` 추가.
- **cross/self 블록:** 체크포인트 config `action_head_cfg.diffusion_model_cfg` = `num_layers:16`,
  `interleave_self_attention:true` 확인됨 → 짝수=cross, 홀수=self 가정 유효.

---

## Phase 1 — GR00T + PLADIS (단일 패스) 구현
- [ ] **파일 생성** `libs/robomimic/robomimic/algo/guidance/pladis.py` — `acg.py` 구조 복제하여
      `PLADISAttnProcessor`, `Gr00tPolicy_PLADIS`, `GR00T_N1_PLADIS`, `FlowmatchingActionHead_PLADIS`.
- [ ] **`PLADISAttnProcessor.__call__`**: cross-attention 계산을 수동으로 수행
      (q=`attn.to_q(hidden_states)`, k/v=`attn.to_k/v(encoder_hidden_states)`), sdpa 대신 명시적으로:
      - `logits = q@kᵀ / sqrt(d)`
      - `dense = softmax(logits)`, `sparse = entmax15(logits)` (또는 sparsemax; `method` 파라미터)
      - `attn = dense + λ·(sparse − dense)` = `λ·sparse + (1−λ)·dense`
      - 수치 안정 위해 entmax/softmax는 **float32**로 계산 후 원래 dtype(bf16)으로 캐스팅.
      - `out = attn @ v` → `to_out` 투영. (원본 `SparseAttnProcessor`와 동일 뼈대)
- [ ] **`FlowmatchingActionHead_PLADIS.get_action`**: `acg.py` denoising 루프 복제하되 **2-pass 제거**.
      루프 진입 전 1회, **짝수(cross) 블록**의 `attn1.processor`를 `PLADISAttnProcessor(λ, method)`로 교체
      (또는 `pladis_applied_layers` 지정 서브셋). denoising은 단일 forward만. `scale/skip_blocks` 대신
      `pladis_scale`, `method`, `pladis_layers` 인자.
      - 짝수 블록 선별: `[i for i in range(len(transformer_blocks)) if i % 2 == 0]`
        (또는 `config.interleave_self_attention` 확인 후). **런타임에 실제 cross인지 검증 로그 1회 출력.**
- [ ] **디스패치 등록** `algo/guidance/__init__.py`에 `elif name == "pladis":` 추가(4개 메서드 patch).
- [ ] **설정** `libs/robomimic/robomimic/config/gr00t_config.py`에
      `self.algo.guidance.pladis_scale`, `.method`("ent15max"/"sparsemax"), `.pladis_layers` 기본값 추가.
      `algo.guidance.name`이 인자를 get_action까지 전달하는 경로 확인(acg의 scale/skip_blocks 전달 경로와 동일).

**완료 기준:** `algo.guidance.name=pladis algo.guidance.pladis_scale=1.5`로 rollout이 에러 없이 돌고,
단일 forward만 실행됨(2-pass 아님).

---

## Phase 2 — GR00T + PLADIS 검증 & 스모크
- [ ] **1-task 스모크**: `robocasa_smoke_1task.json` 또는 `base_rollout.sh ... n=2 nbe=2`로 크래시/차원 확인.
      영상(`videos/*.mp4`) 육안 확인 — 손떨림/드리프트 개선 방향인지.
- [ ] **단일 패스 검증(중요)**: forward 호출 횟수 = `num_inference_timesteps`(16)인지 로그/카운터로 확인
      (ACG는 2배). 추론 시간이 baseline 수준(~64ms/step급)인지 대략 측정.
- [ ] **λ 민감도 사전 점검**: pladis_scale ∈ {1.0(=off), 1.5, 2.0}에서 발산/NaN 없나 확인.
- [ ] **3-seed 평가**: 기존 인프라 재사용. `run_resilient.sh`에 pladis phase 추가하거나
      별도 래퍼로 `algo.guidance.name=pladis` 실행 (seeds 123/456/789, n=24, nbe=2, `setsid` 분리).
      결과 요약은 `summarize_multiseed.py`로 baseline과 비교.

**완료 기준:** 3-seed baseline vs PLADIS 성공률 비교표 확보, 단일 패스(속도 이점) 실측.

---

## Phase 3 — GR00T + ACG + PLADIS 결합 구현
- [ ] **파일 생성** `algo/guidance/pladis_acg.py` (또는 `acg.py`에 combined 모드). 두 프로세서를 서로 다른
      블록에 동시 설치: **PLADIS→짝수(cross) 블록**, **ACG Identity→홀수 {7,9,11}(self) 블록**.
- [ ] **get_action 흐름 (설계 결정 필요 — 문서화):**
      - Base pass: PLADIS cross-attn 프로세서 설치 상태로 1회 (PLADIS 이점 포함된 `v_θ`).
      - Incoherent pass: 홀수 {7,9,11}에 ACG Identity 설치 후 1회 → `v_incoherent`.
        - **결정1:** incoherent pass에서 PLADIS(cross)를 유지할지/dense로 되돌릴지. 발표 p18은 둘 다 활성.
          기본값 = 유지(모델 cross-attn의 일부로 간주), 되돌림도 ablation 대상으로 남김.
      - 합성: `v = (1+scale−1)…` 즉 기존 ACG 식 `pred + (scale−1)(pred − pred_perturb)` 그대로.
- [ ] **디스패치 등록** `name == "acg-pladis"` (또는 `pladis+acg`) 추가.
- [ ] **설정**: acg(scale, skip_blocks) + pladis(pladis_scale, method) 파라미터 모두 노출.

**완료 기준:** `algo.guidance.name=acg-pladis`로 rollout 동작, 짝수=PLADIS/홀수 일부=ACG가 실제로
설치됐는지 1회 검증 로그.

---

## Phase 4 — 결합 검증 & 평가
- [ ] 1-task 스모크 + 차원/NaN 확인.
- [ ] 연산량 확인: ACG 2-pass 유지되므로 ~1.5배 예상(단일 패스 아님) — 실측해 발표 수치와 대조.
- [ ] 3-seed 평가(seeds 123/456/789, n=24) → `summarize_multiseed.py` 비교표.

---

## Phase 5 — Ablation Study (발표 계획 반영)
비교군: **baseline / +ACG / +PLADIS / +ACG+PLADIS** (동일 seed·task).
- [ ] `pladis_scale` 스윕: {1.0, 1.5, 2.0, 2.5} — 최적 λ.
- [ ] `method` 비교: `ent15max`(entmax1.5) vs `sparsemax`.
- [ ] `pladis_layers` 서브셋: 전체 cross vs 일부(예: 발표의 "초반 레이어 비적용" 캐싱 아이디어, p18).
- [ ] (검토) `GR00T + PLADIS + α` — 발표의 미정 항목. 스코프 정의 후 태스크화.
- [ ] 결과 표/그래프 정리 → `EXPERIMENT_NOTES.md`에 섹션 추가, 발표자료 업데이트.

---

## 리스크 / 주의
- **dtype**: bf16 autocast 하에서 entmax/softmax는 float32 계산 권장(원본 PLADIS도 `.float()` 캐스팅). NaN 주의.
- **cross 블록 식별**: 짝수=cross는 `interleave_self_attention=True` 가정. 로딩 시 config에서 값 확인 후 분기
  (False면 전 블록이 cross이므로 로직 달라짐).
- **attention_mask/GQA**: cross-attn에서 vl_embeds(길이 ~99) 대상. 마스크 유무 확인(현재 rollout은 mask=None으로 보임).
- **baseline offset**: sdpa로 baseline이 높아 효과가 작게 보일 수 있음 → 상대 비교(동일 환경 내)로 결론.
- **인프라 재사용**: 새 실행은 반드시 `setsid` 분리 + `run_resilient.sh` 스타일 resume/재시도로 무인 완주.
  RAM 상 `num_batch_envs=2` 유지.

## 산출물 체크
- [ ] `pladis.py`, (선택)`pladis_acg.py`, `__init__.py`·`gr00t_config.py` diff
- [ ] 3-seed 결과 (baseline/ACG/PLADIS/ACG+PLADIS) 비교표
- [ ] 단일 패스 속도 실측치 (PLADIS vs ACG)
- [ ] ablation 표 + 발표자료 반영

---
*참고 파일*: `acg.py`(참조 구현) · `cross_attention_dit.py:274-290`(cross/self 분기) ·
`gr00t_config.py:47`(guidance 기본값) · `PLADIS/pipeline/pipeline_sdxl_seg.py`(sparse 외삽 원본) ·
`run_resilient.sh`/`summarize_multiseed.py`(실행·집계) · `EXPERIMENT_NOTES.md §8`(재현 현황).
