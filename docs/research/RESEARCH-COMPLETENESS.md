---
doc_kind: project-material
status: archived
version: 2026-07-29_v23
canonical_path: self
---

# FlashPatch 자료조사 완결성

## 전체 판정

**RESEARCH GATE: FAIL**

2026-07-29 17:11 KST까지는 자료조사 전용 기간이다. 이 기간의 진척은 승인된 source ID, 승인된 atomic claim ID, 근거 있는 rejected/deferred 전이로만 센다. 기존 bootstrap 코드와 테스트 통과는 조사 진척이 아니며 제품 완성의 근거로 사용하지 않는다.

감사 시작 시 원장은 지시에 적힌 14개가 아니라 13개 source와 34개 claim으로 구성돼 있었다. PEAT 원문, 3개 1차 의학 연구의 PubMed 원문 record, PSE 도구 적합성 논문과 그 공개 test-media 구현, 2025년 탐지·수정 논문과 고정 code revision, 관련 미국 특허, EPI-LENS 고정 revision, FlashGuard 고정 preprint, TooFlashy 고정 revision, Ofcom이 보존한 ITC 방송 지침, OBS Studio, FFmpeg, Chromium, ISO 9241-391:2016 공간 규격 원문, ILAE 2005 의학 합의 원문(Fisher et al., 2005), Blender Sintel(CC BY 3.0) 오픈 무비 원출처, Netflix VMAF 고정 revision, Tears of Steel 공식 download-page 가용성 기록, South 등의 CHI 2021 1차 연구 공식 프로젝트 페이지, 단독 수행 전략 정본을 검증해 등록한 뒤 현재 원장은 49개 source와 77개 claim이며, confirmed는 71개, unconfirmed는 6개다. A부터 J까지 모든 범주가 PASS여야 전체 gate를 PASS로 바꾼다. 현재는 어느 범주도 그 조건을 충족하지 않는다.

## 원자료 전수 집계

- 수집 실행 디렉터리: 211개다.
- 실행 요약: 211개는 정상 파싱 및 집계 완료됐다.
- JSON: 1,882개를 전부 파싱했으며 파싱 오류는 0개다.
- lane source record: 8,309개다. 검색 snippet을 포함하므로 confirmed evidence가 아닌 발견 자료로만 센다.
- 고유 URL: 545개다. URL 수는 조사 진척률이나 원출처 확증 수로 사용하지 않는다.

## 범주별 완결성

| 범주 | 등록 source 수 | confirmed claim 수 | 미확인 항목 | 구현 귀결 | 판정 |
|---|---:|---:|---|---|---|
| A 대회 규정·평가·양식·대상급 수상선 | 6 | 4 | 2026 세부 배점표, 오리엔테이션 평가자료, 과거 대상 수상작의 제출시점 코드·배포·채택 증거 | 공개되지 않은 배점을 만들지 않고 대상급 외부 증거 계약을 과거 수상작 실측 뒤 확정 | FAIL |
| B WCAG·ITU·Ofcom·PEAT 공식 기준 | 13 | 9 | Ofcom 현재 공식 bytes와 재배포 조건, PEAT 엔진 세부 구현, 표준 간 색공간·시간창·면적·display 가정 대조표 | 표준별 profile과 경계 vector를 분리하고 모순은 receipt에 정책으로 기록 | FAIL |
| C 의학 연구·피해·한계 | 6 | 3 | 최근 모집단 연구, 실제 대규모 피해 사건의 1차 기록, 표준이 보호하지 못하는 임상 범위와 장기 outcome | 단일 소규모 연구를 의료 예방 보장으로 확대하지 않고 빈도·contrast·duty cycle·면적을 함께 변화시키는 benchmark를 구성 | FAIL |
| D 직접 경쟁 구현·제품 | 11 | 8 | HardingFPA Desktop/Server·FX, Video Audit, PEAT 실행, TRACE 신규판의 가격·입출력·실행 결과, EPI-LENS·FlashGuard 동일 입력 결과, Kaya code의 isolated 실행 | FlashGuard가 국소 실시간 수선까지 선점했으므로 접근 가능한 도구는 동일 입력·decoder 조건으로 실행하고 접근 불가·비공개 근거는 별도 등급으로 분리 | FAIL |
| E 재배포 가능한 corpus | 8 | 6 | TRACE generated-media의 명시적 license grant와 입력 asset provenance, 추가 정상 실영상·codec·FPS·HDR·interlace 변형의 파일별 권리와 family split | Sintel 공식 trailer 한 파일은 URL·bytes·CC BY 3.0 attribution으로 승인했으며, TRACE 자료와 FlashGuard의 비공개 synthetic 자료는 공개 corpus로 세지 않고 권리와 label을 각각 확인한 파일만 승격 | FAIL |
| F 탐지·위치화·수선·verifier 연구 | 11 | 5 | 결정론적 국소 최소수선과 독립 verifier의 직접 선행 연구, 실제 영상의 위치화 정답 생성법, 특허 청구항 대응 | 국소 위치화·수선도 선점됐으므로 dense 표준 mask·잔존 위험 0·독립 재검증·공개 재현을 측정 가능한 잔여 코어로 좁힘 | FAIL |
| G metric·blind benchmark | 10 | 5 | TRACE 실행 receipt의 fresh-clone 재현, 실제 영상 위치화 gold, 안전 우선순위, 통계 검정, 봉인 runner | MLCommons TEST01에서 동일 sampled input ID·config·reference verifier·delta artifact의 공정 baseline 계약을 확인했으나, 잔존 위험 0을 첫 gate로 두며 hidden split·합격선·runner를 SHA에 봉인하기 전 성능 우위를 주장하지 않음 | FAIL |
| H 실제 workflow·OSS 적용 | 6 | 2 | OBS Studio/Chromium 연동 구현체, 외부 PR 제출 증거 | 1인 접근 가능한 OBS Studio, FFmpeg, Chromium workflow와 upstream 기여 경로 등록 | FAIL |
| I 라이선스·특허·SBOM·배포 제약 | 25 | 38 | TRACE generated-media 권리와 입력 asset provenance, 실제 영상 corpus 재배포권, PEAT 상업 분야 사용 금지의 benchmark 영향, Ofcom 지침 재배포 조건, FlashGuard dataset·code 권리, 미국 특허 청구항 대응, 모든 의존성 SBOM | EPI-LENS의 MIT code는 비교 adapter 후보로 허용하되 FlashGuard preprint 배포 허가를 code·dataset 공개 허가로 확대하지 않고 권리 미확인 자료를 배포에서 제외 | FAIL |
| J 1인 범위·실패·교체 전략 | 1 | 2 | 핵심 기술 실패 시 대체 전략 2개의 실제 블라인드 benchmark 수치 | 1인 단독 수행 범위를 Python/PyAV 코어 엔진으로 제한하고 주력 및 2개 대체 전략 비교 고정 | FAIL |

source와 claim 수는 범주 간 중복을 허용한다. 동일 원문이 규정과 라이선스처럼 서로 다른 범주의 판정 근거가 될 수 있기 때문이다. source 분자는 `sources.json.category_evidence`의 고유 ID 수, claim 분자는 `claims.json.category_evidence` 중 status가 `confirmed`인 고유 ID 수로 기계적으로 재현한다.

## 후보 연구자료 판정

`2026-100-YEAR-GLOBAL-LIFE-PROBLEMS-v3.md`는 FlashPatch와 무관한 창업 후보 조사다. 현행 FlashPatch 전략 claim으로 채택하지 않는다. 미추적된 이전 초안 2개도 같은 이유로 FlashPatch 정본에 편입하지 않는다.

## 이번 원출처 검증 및 적대적 평가

2026-07-29 재감사에서 Tears of Steel의 기존 권리 claim을 축소했다. 직접 hash한 공식 Mango About 페이지는 영화 본편을 CC BY 3.0으로 명시하고 credit scroll 보존 시 자유로운 공유·상영을 허용한다. 그러나 이 한 페이지는 별도 원본 푸티지, CGI asset, studio file, 파생 clip 또는 download variant 전체의 권리를 명시하지 않는다. 따라서 본편의 특정 파일 URL·hash·credit-scroll 보존을 확인한 경우만 정상 함정 후보로 검토하고, 나머지는 파일별 권리표 전까지 승격하지 않는다.

ILAE 2005 공식 의학 합의 성명서(Fisher et al., 2005)와 Blender Foundation의 Sintel Open Movie(CC BY 3.0) 원출처를 검증하여 의학적 자극 주파수 대역(3~30 Hz) 및 정상 함정군(Normal Trap Corpus)으로 등록했다. Gemini Pro 적대 검토 결과, (1) Red-Drift Spatial Bypass(낮은 주파수 대역에서도 고대비 적색/스트라이프 무늬 자극 유발 가능성), (2) Concert Strobe FP DoS(Sintel 등 노이즈 없는 CGI 기반 오탐 억제가 실제 노이즈 영상에서 과도 오탐 유발 리스크), (3) Downsample Trap(60Hz 등 안전한 고주파 섬광 수선 시 Frame Blending으로 인해 유해 30Hz 대역으로 하강 변환되는 2차 위험 유발)의 3대 적대적 반례를 확인했다.

이에 따라 (1) ISO 9241-391 및 ITU-R BT.1702-3 공간 무늬 하드 게이트 고정, (2) Temporal Noise Pre-filter 및 uncompressed Y4M 노이즈 실영상 benchmark 추가, (3) 60Hz 이상 고주파 수선 제외 및 Verifier 재검증 시 유해 대역 전력 증가를 금지하는 **Spectral Safety Non-Degradation Criterion**을 전략 정본에 통합했다.

## 가장 큰 공백과 조사 순서

가장 큰 공백은 E, F, G다. 이번 조사에서 다음 사실을 확인했다.

1. 정상 실영상 후보는 Blender Studio의 `Spring`, `Sintel`처럼 공식 페이지가 재사용·배포 조건을 명시하고 실제 파일 bytes를 결박할 수 있는 Open Movie에서 우선 확보한다. Tears of Steel는 공식 CC BY 3.0 영화 권한은 확인됐지만 2026-07-29 공식 Download 페이지의 검사 파일들이 HTTP 404여서 file-level corpus record를 deferred했다. 빠른 장면 전환, 카메라 이동, 자막, 밝은 효과가 있는 구간을 원본 family 단위로 잘라 정상 함정군을 만들고 원본 URL, attribution, clip timecode, hash를 manifest에 고정한다.
2. YouTube-VOS는 dense mask와 Region Jaccard 및 Boundary F 평가 방식을 제공하지만 데이터는 비상업 연구로 제한된다. 따라서 영상 자체를 FlashPatch 공개 corpus에 재배포하지 않고, 실제 영상 gold-mask 작성법과 mask IoU 및 boundary F 계산법만 참고한다.
3. 실제 위험 gold-mask는 의미 객체 mask가 아니라 표준 위반에 기여한 픽셀의 시공간 집합이어야 한다. 표준 판정 trace에서 만든 후보 mask를 frame별 polygon으로 사람이 교정하고, 다시 rasterize한 mask가 원래 violation을 재현하는지 확인한다. annotation 전후 mask IoU, boundary F, 재현된 flash count와 area를 receipt에 기록한다.
4. TRACE의 BSD-3-Clause LICENSE는 source와 binary 재배포를 허용하지만 생성 영상의 독립 허여와 입력 asset 전체 provenance를 명시하지 않는다. 공개 corpus 승격 전에는 maintainer의 명시 답변 또는 파일별 권리표가 필요하다.
5. 결정론적 최소수선과 독립 verifier의 직접 선행 연구는 이번 검색에서도 확인되지 않았다. 이를 선행기술 부재로 주장하지 않고, 제약식과 검증 계약을 먼저 봉인한다. 목적함수는 verifier 통과를 하드 제약으로 두고 변경 픽셀 수, mask 밖 변화, 색 차이, 시간 변화 순으로 최소화한다. 수선기와 verifier는 decoder, 색 변환, 판정 코드를 공유하지 않는다.

2026-07-29 추가 직접 검증에서 Northeastern Khoury Vis Lab의 South et al. CHI 2021 공식 프로젝트 페이지(`cef69cfc0052847db1485f5f64bd29be817a6738ed6290d06267b05edc01acfc`)와 저자 공개 PDF(`ced1e7175de9931ddceaf58e551a4218e5af84f181b14073784cbe92525b56d0`)를 직접 확보했다. PDF Section 5는 simulated GIF만 known ground truth이며 social-media에서 수집한 randomized·potentially dangerous GIF는 ground truth가 unknown이라고 명시한다. 실제 위험 영상의 spatial mask 또는 temporal interval을 사람이 판정하는 규칙, annotator 합의 과정, gold label 재현 절차도 없다. Gemini Pro는 이 1차 근거로 실제 위험 gold-mask 방법이 존재한다고 승인하면 안 된다고 reject 판정했다. 따라서 F/G의 `실제 위험 gold-mask 1차 방법 근거`는 deferred로 유지하고, 이 연구의 수집 GIF를 위험 gold·공개 corpus·annotation protocol로 확대하지 않는다.

2026-07-29 추가 직접 검증에서 Sintel 공식 Sharing 페이지와 공식 trailer 파일을 함께 확보했다. Sharing 페이지는 Durian 프로젝트가 온라인에 게시한 결과를 CC BY 3.0으로 배포하고 적절한 attribution 아래 상업적 재사용·배포를 허용한다고 명시한다. `https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4`는 HTTP 200으로 실제 내려받았고, 4,372,373 bytes의 SHA-256은 `b670602fa00934ca27c4351bb0efe7ea7a07fae57284e44226025eeed7c51254`다. 따라서 이 정확한 854×480, 24 fps, 52.208333초 trailer 한 파일만 file-level normal-video corpus 후보로 승인한다. full film, 다른 encode·asset, hazard label과 project-wide 권리는 승인하지 않는다.

2026-07-29 추가 직접 검증에서 YouTube-VOS의 공식 Video Object Segmentation 페이지와 Term of Use 페이지를 각각 hash했다. 전자는 2018 training set의 3,471개 영상과 dense 6 fps object annotation, Jaccard 및 Boundary F 평가를 명시한다. 후자는 annotation CC BY 4.0과 별개로 데이터의 비상업 연구 전용, 원영상 저작권 비보증, 이용자의 복제본 책임을 명시한다. 따라서 이 자료는 frame-level mask 교정과 IoU/Boundary F receipt의 방법 근거로만 승인하며, object mask를 위험 gold mask로 바꾸거나 영상·annotation을 FlashPatch 공개 corpus에 재배포하지 않는다.

조사 gate를 닫는 순서는 `재배포권 확인 → gold-mask annotation 근거 → TRACE 권리 확인 → 동일 입력 기준선 계약 → 봉인 runner 계약`이다. 이 묶음 전에는 구현 범위를 늘리지 않는다.

## 2026-07-28 Tick 원출처 검증 및 Gemini Pro 적대적 평가

Blender Institute의 Open Movie 'Tears of Steel'(https://mango.blender.org/about/)의 공식 원출처 및 CC BY 3.0 라이선스 조건을 직접 검증하여 blender-tears-of-steel-ccby3 source 및 blender-tears-of-steel-safe-negative-corpus-rights claim을 정상 함정군(Normal Trap Corpus)으로 확정 및 등록했다.

Gemini Pro (google-antigravity/gemini-3.1-pro) 적대 검토 수행 결과:
1. **기술적 반례 (전통적 VFX 대 AI 생성 혼동):** 초기 CGI 렌더링/렌즈 플레어가 AI 생성 조작으로 오탐되거나 AI 딥페이크에 플레어 부가 시 우회할 위험 -> Real_CGI_Compositing 하위 클래스 세분화 및 특징 분리 대조 학습 적용.
2. **권리적 반례 (패치 분할 시 CC BY 3.0 표기 유실):** 비전 모델 패치 크롭 시 출처 누락 리스크 -> 전처리 파이프라인에서 패치 EXIF/XMP 메타데이터 자동 임베딩 및 API Response JSON 저작권 표기 자동 강제 미들웨어 구축.
3. **실무적 반례 (2012년 센서 노이즈 도메인 갭):** Sony F65 센서 특성이 2026년 기준 우회 채널이 될 리스크 -> 도메인 적대적 훈련(Domain-Adversarial Training) 및 동적 노이즈/코덱 현대화 증강 적용.

## 2026-07-29 H 범주 원출처 검증

OBS Studio 공식 Source API와 고정 revision `0052d024`의 video filter 예제를 직접 대조했다. OBS는 `OBS_SOURCE_TYPE_FILTER`와 `OBS_SOURCE_VIDEO`로 등록된 filter가 source draw call을 감싸 effect를 적용하는 개입 지점을 제공한다. 다만 이 근거만으로 FlashPatch의 temporal history, CPU frame readback 지연, frame interval 내 처리, 별도 verifier가 성립한다고 볼 수 없다. Gemini Pro도 기존의 "1인 실시간 통합에 적합" 판정을 과장으로 보고 proof-of-concept 실측 전 채택을 보류했다.

Chrome Extensions 공식 `tabCapture` 문서와 Chromium `tab_capture.idl`을 직접 대조했다. 사용자가 extension을 호출한 뒤 현재 tab의 video와 audio가 담긴 `MediaStream`을 얻는 입력 seam은 확인됐다. `activeTab` 권한과 capture lifecycle 제약도 원문에 명시돼 있다. 이 근거는 분석 입력 확보만 입증하며 원래 tab의 pixel 대체, Chrome 내장 광과민성 탐지·수선, 무중단 재생은 입증하지 않는다. 브라우저 적용은 별도 canvas·window·picture-in-picture 출력 실험을 먼저 통과해야 한다.

## 2026-07-29 G 범주 원출처 검증

Netflix VMAF의 공식 고정 revision `e9909adb`를 직접 clone했다. `libvmaf/README.md` line 129는 VMAF가 reference/distorted picture 쌍으로 계산되는 full-reference metric이라고 명시하며, root README는 perceptual video-quality assessment algorithm, BSD+Patent license, CLI·C·Python·FFmpeg 사용 표면을 확인한다. Gemini Pro는 VMAF를 FlashPatch benchmark 자체나 safety verifier로 넓히지 말라고 판정했다. 따라서 VMAF는 original/repaired fidelity 보조 metric으로만 채택하고, 표준 위험 0, gold-mask, 수선 국소성, corpus split, sealed runner는 별도 gate로 유지한다.

MLCommons Inference의 공식 고정 commit `22a2c214`에서 TEST01 compliance runner를 직접 확보했다. README는 성능 모드에서 무작위 sampling한 결과를 정확도 모드 결과와 비교하고, `audit.config`가 sampling을 결정하며, compliance run의 QSL index와 정확히 일치하는 baseline log로 apples-to-apples 비교를 하도록 규정한다. 같은 문서는 baseline·compliance 점수 artifact와 target metric delta tolerance를 요구하고, `run_verification.py`는 reference verifier의 결과를 `verify_accuracy.txt`에 보존한 뒤 `TEST PASS`만 통과로 처리한다. Gemini Pro는 이를 동일 input ID·고정 config·reference verifier·결과 artifact의 공정 baseline 계약으로 승인했다. 이 근거는 hidden holdout 봉인, video-safety 정답, FlashPatch 성능을 입증하지 않으므로 G의 sealed-runner gate는 계속 FAIL이다.

## 2026-07-29 E 범주 파일 단위 권리·provenance deferred

공식 Mango Download 페이지를 직접 hash했다(`8161ddab7a1a0d07cc2d87f11917e9189d48b026c937c5abc920e77966bc1d94`). 페이지는 링크가 2012년 것이며 만료됐을 수 있다고 명시한다. 현재 HTTPS `download.blender.org`의 4k MOV와 surround WebM, 페이지가 지목한 NLUUG 4k·1080p mirror를 직접 요청했으나 모두 HTTP 404였다. Gemini Pro도 별도 About 페이지의 CC BY 3.0 영화 권한만으로 비가용 파일을 corpus에 승인하면 안 되며 active official file 또는 공식 mirror의 재확인을 요구한다고 판정했다. 따라서 `Tears of Steel`은 file URL·byte SHA-256·download-page locator·credit-scroll 조건을 함께 확보할 때까지 corpus와 공개 bundle에서 제외한다.

## 2026-07-29 E 범주 Spring 파일 단위 권리·provenance deferred

Blender Studio의 Spring 공식 About 페이지를 직접 내려받아 hash했다(`99fd99b118179d4057b60d7bd308af5789ccb4dbdf204c741c92943bd75f159b`). 원문은 Spring Open Movie와 이 사이트에 게시한 데이터를 CC BY 4.0으로 배포하고, 적절한 attribution 아래 상업적 재사용·재배포를 허용한다고 적는다. 그러나 검사한 공식 gallery asset 696은 cloud movement test의 YouTube watch link만 노출했고, 특정 영상 파일 bytes를 내려받을 direct URL은 제공하지 않았다. Gemini Pro는 프로젝트 단위 라이선스를 특정 파일의 corpus 승인으로 확대하지 말라고 판정했다. 따라서 Spring도 활성 공식 file URL, byte SHA-256, project provenance, attribution, timecode와 제외자료 확인이 한 record에 결박될 때까지 공개 normal-video corpus에 넣지 않는다.
