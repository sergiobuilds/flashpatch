---
doc_kind: project-material
status: canonical
version: 2026-07-26_v1
canonical_path: self
---

# WCAG 2.2 점멸 기준, Day 2 원문 추출

## 범위와 판정 경계

이 문서는 WCAG 2.2 SC 2.3.1, 설명 문서, G15, G19, G176 원문에서 FlashPatch 구현에 필요한 수식과 경계값을 추출한 정본이다. W3C는 WCAG 본문을 권고안으로, Understanding과 Techniques를 정보성 설명 및 충분기법으로 구분한다. 따라서 이 기록은 원문을 정확히 구현 계약으로 옮기기 위한 것이며, 현재 bootstrap detector는 not a complete WCAG 2.2 conformance implementation 이다.

- 확인일: 2026-07-26
- 입력 가정: sRGB, 8-bit RGB, standard zoom, 1024 × 768 표시와 22부터 26 inch 시청 거리의 WCAG 근사
- 적용 금지: 이 수치만으로 의료 안전성이나 다른 표준 적합성을 주장하지 않는다.
- 원문 보관: HTML은 저장소에 넣지 않는다. URL, retrieval date, 응답 SHA-256, locator, W3C Document License 2023 정보를 `sources.json`에 남긴다.

## SC 2.3.1과 일반 점멸

SC 2.3.1은 어느 1초 구간에서도 3회를 초과해 점멸하지 않거나, general flash 및 red flash threshold 아래여야 한다고 규정한다.

일반 점멸의 원문 정의는 다음과 같다.

- 반대 방향의 relative luminance 변화 쌍이다.
- 각 변화의 크기는 최대 relative luminance 1.0의 10%, 즉 `|ΔL| ≥ 0.10`이다.
- 더 어두운 상태는 `L < 0.80`이다.
- 반대 방향 쌍은 증가 뒤 감소 또는 감소 뒤 증가다.

sRGB 상대 휘도는 선형화한 `R`, `G`, `B`로 `L = 0.2126R + 0.7152G + 0.0722B`다. 이 정의는 bootstrap profile의 sRGB 선형화와 일치하지만, profile이 CSS display-area 및 red flash 계약을 아직 구현했다는 뜻은 아니다.

## red flash와 색도 계산

원문 용어 정의는 red flash를 saturated red를 포함하는 반대 방향 전이 쌍으로 둔다. Understanding SC 2.3.1은 field working definition도 제공한다.

- 한 상태에서 `R / (R + G + B) ≥ 0.8`
- 두 상태의 CIE 1976 UCS 색도 거리가 `Δu′v′ > 0.2`
- `Δu′v′ = sqrt((u′1-u′2)^2 + (v′1-v′2)^2)`
- `u′ = 4X / (X + 15Y + 3Z)`, `v′ = 9Y / (X + 15Y + 3Z)`

이 field working definition은 WCAG 본문 용어보다 구현에 구체적이지만, 해당 문서가 이를 working definition이라고 부른다. FlashPatch는 이후 red detector에서 source provenance와 profile 상태를 분리해 표시한다.

## 면적과 시간 벡터

원문은 아래 둘 중 하나면 threshold 아래라고 정의한다.

1. 임의의 1초 구간에 general flash 및 red flash가 각각 3회 이하
2. 동시에 발생하는 flash의 합산 면적이 전형적 시청 거리의 어느 10도 시야에서도 `.006 steradians`, 즉 10도 시야의 25% 이하

W3C의 일반 소프트웨어 근사는 1024 × 768 표시에서 `341 × 256` 픽셀 직사각형이다. 그 25%는 21,824 pixels다. G176은 한 번에 하나의 점멸 영역만 있을 때 그 영역을 담는 contiguous container가 21,824보다 작으면 general 및 red threshold를 통과하는 간이 기준으로 설명한다.

초기 threshold vectors는 다음으로 고정한다.

| Vector | 입력 | 기대 판정 | 근거 |
|---|---|---|---|
| general-delta-below | 어두운 상태 `< 0.80`, `|ΔL| = 0.099` | general transition 아님 | SC 2.3.1 용어 정의의 0.10 경계 |
| general-delta-at | 어두운 상태 `< 0.80`, `|ΔL| = 0.10` | general transition | SC 2.3.1 용어 정의의 포함 경계 |
| general-dark-at | `|ΔL| ≥ 0.10`, 더 어두운 상태 `L = 0.80` | general transition 아님 | `L < 0.80`의 엄격 경계 |
| frequency-three-closed-same-state | closed 1-second window에 정확히 3 flash, 시작과 끝이 같은 light/dark state | 빈도 경로 통과 | G19 Tests Procedure의 3회 및 end-state 조건 |
| frequency-three-closed-different-state | closed 1-second window에 정확히 3 flash, 시작과 끝이 다른 light/dark state | 빈도 경로 실패 | G19 Tests Procedure의 3회 end-state 조건 |
| frequency-four | 임의의 1초에 4 flash | 빈도 경로 실패, 면적 경로 별도 평가 | SC 2.3.1 |
| area-21823 | 단일 contiguous area 21,823 pixels | G176 간이 경로 통과 | 21,824보다 작음 |
| area-21824 | 단일 contiguous area 21,824 pixels | G176 간이 경로에 넣지 않음 | G176의 엄격한 smaller than |
| red-ratio-at | 한 상태 `R/(R+G+B)=0.8`, `Δu′v′>0.2` | red 후보 | Understanding field working definition |
| red-distance-at | saturated red 상태, `Δu′v′=0.2` | red 후보 아님 | `Δu′v′ > 0.2`의 엄격 경계 |

## G15, G19, G176의 구현 귀결

- G15는 복합 점멸을 time-continuous하게 추적하는 도구가 필요하다고 설명한다. FlashPatch는 PTS rolling window, event evidence, 입력 color-space metadata를 계약으로 둔다.
- G19는 임의의 1초 동안 3회를 넘지 않아야 하며 정확히 3회면 1초 시작과 끝의 light/dark 상태가 같아야 한다고 둔다. frequency test는 end-state vector를 포함해야 한다.
- G176은 동시 점멸이 하나뿐이고 contiguous container가 small safe area보다 작은 경우에만 간이 경로다. disconnected regions와 여러 동시 영역은 G176 통과로 단정하지 않는다.

## 후속 구현 항목

1. Day 13 threshold corpus에 위 vector의 synthetic manifest와 expected label을 추가한다.
2. Day 18에 CSS display assumption 및 21,823, 21,824, 21,825 경계를 구현한다.
3. Day 19에 CIE 1976 UCS와 `0.8`, `0.2` 경계를 독립 verifier와 별도로 구현한다.
4. Day 17 frequency profile에는 closed 1-second window 및 G19 end-state vector를 명시한다.
