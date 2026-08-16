---
doc_kind: project-material
status: canonical
version: 2026-07-26_v1
canonical_path: self
---

# 2026 오픈소스 개발자대회 공식 요구사항

## 1. 목적과 증거 경계

이 문서는 FlashPatch Day 1에 확인한 대회 공식 요구사항의 정본이다. 기계 판독 가능한 출처는 `docs/research/sources.json`, 주장 단위의 상태와 구현 귀결은 `docs/research/claims.json`에서 관리한다. RED→GREEN과 품질 검사 실행 증거는 `docs/research/day1-validation.json`에 기록한다.

- 기준 확인일: 2026-07-26
- 확인 상태: `confirmed`는 공식 원문에서 locator를 확인한 사실이고, `unconfirmed`는 추가 공식 자료가 필요한 항목이다.
- 원문 보관: PDF, ZIP, PNG 파일은 저작권과 저장소 크기를 고려해 commit하지 않는다. 공식 URL, SHA-256이 확인된 파일의 hash, 원문 locator만 보관한다.
- 주장 경계: 공식 원문에 없는 세부 배점, 녹화 공개 여부와 평가자료 내용은 추정하지 않는다.

## 2. 공식 출처 고정

| Source ID | 원문 | 고정 정보 |
|---|---|---|
| `contest-overview-2026` | 대회 개요 | 2026 공식 overview URL |
| `contest-operating-rules-2026` | 운영규정 PDF | 2026-06, 15쪽, SHA-256 `5c129ed9f389ecc04b6f7ba8b97f719a313efaf32aea9178e635500023ae1da1` |
| `result-report-template-2026` | 결과보고서 ZIP | SHA-256 `9a5d2968d48ff8a8fd85ce991dc72dc2b0818d7e8c06ebb871cc97ce5cc62d95` |
| `orientation-notice-2026` | 오리엔테이션 공지 | 게시일 2026-07-22 |
| `orientation-program-image-2026` | 오리엔테이션 프로그램 이미지 | 공지 첨부 원본 URL, SHA-256 `808e6b3c6830a6b05431795dfe1fa4ea325cbee81ac5feaf450985b0153a77d7` |
| `official-youtube-videos` | 공식 YouTube 영상 목록 | 2026-07-26 확인 상태 |

전체 URL과 retrieval metadata는 source registry를 따른다.

## 3. 운영규정 확정 요구사항

### 3.1 라이선스와 출처 공개

운영규정 제8조에 따라 직접 작성한 코드는 OSI 인증 오픈소스 라이선스를 적용한다. FlashPatch의 직접 작성 코드는 Apache-2.0 정책을 유지한다. 타 저작물과 모든 library, framework, model의 출처와 라이선스를 공개한다. 이에 따라 dependency, model, dataset, 인용 자료를 provenance 및 license 원장에 연결해야 한다.

### 3.2 전체 소스와 공개 저장소

운영규정 제10조에 따라 전체 소스코드를 제출하고 공개 저장소에 게시해야 한다. 5년 Public 유지 대상은 우수팀 및 수상팀이며, 수상일부터 5년 동안 저장소를 Public 상태로 유지해야 한다.

**현재 확정 gap:** FlashPatch GitHub 저장소는 현재 Private이다. 제출 전에 Public으로 전환하고 비로그인 상태의 clone과 실행 경로를 검증해야 한다.

### 3.3 평가 시점과 개발

운영규정 제11조에 따라 각 평가는 해당 평가 시점의 저장소 상태를 기준으로 진행한다. 평가 기간 외에는 자유롭게 개발할 수 있다. 각 평가 시작 시점의 commit SHA와 재현 결과를 기록하고, 심사 대상 상태를 깨뜨리지 않는 release 및 branch 운영이 필요하다.

### 3.4 AI 사용

운영규정 제9조에 따라 제품에 AI 모델을 탑재할 때는 최소 open-weight 모델이어야 한다. API로만 이용하는 closed model을 출품작의 탑재 AI로 사용하는 것은 제한된다. 다만 MCP·AI 연동 생태계 software와 테스트 목적 API 호출은 공식 예외로 허용된다. 코딩과 디버깅을 위한 AI assistant 사용은 허용되지만, 작성 코드의 원리를 충분히 이해하지 못하면 감점될 수 있다.

공식 예외의 사용은 허용하되 FlashPatch 제품 핵심 실행 경로는 closed model API에 의존하지 않도록 한다. assistant 활용 범위, 사람의 검토, 핵심 알고리즘의 설명 가능성과 test evidence를 함께 보존한다.

### 3.5 정부지원 동일 및 유사 프로젝트

운영규정 제12조부터 제15조와 결과보고서 양식에 따라 정부지원사업에서 동일하거나 유사한 프로젝트로 현재 참여 중, 결과 대기 중이거나 이미 수혜한 이력을 기재해야 하며 주최 측 검증 대상이다. 이 세 상태를 별도 원장에 기록하고 증빙 포인터를 보존한다. 단순 신청 전체를 공식 의무로 확대하지 않는다.

### 3.6 세부 배점

운영규정 제16조는 세부 배점이 모집 공고 또는 심사계획을 따른다고 규정한다. 운영규정 자체에는 배점표가 없다. 따라서 세부 평가 배점은 `unconfirmed` 상태로 유지하며 임의 점수를 만들지 않는다.

## 4. 결과보고서 양식 확정 요구사항

| 항목 | 확정 요구사항 | 구현 귀결 |
|---|---|---|
| 마감 | 2026-08-27 18:00 | 제출 패키지와 공개 상태를 마감 전에 검증한다. |
| 제출 파일 | HWP, HWPX, DOC, DOCX 중 원본 1부와 PDF 1부 | 동일 내용의 편집 원본과 PDF를 생성하고 대조한다. |
| 본문 형식 | 본문 5페이지 이내, 맑은고딕 10pt, 여백 변경 금지 | 페이지 수, 글꼴, 글자 크기와 여백을 검사한다. |
| 저장소 | 공개 저장소 URL 기재 | Public 전환과 비로그인 접근을 확인한다. |
| 시연 | YouTube 시연영상 URL 기재 | URL과 재생 가능 상태를 확인한다. |
| 붙임 1 | SBOM 필수, 분량 제한 없음 | 본문과 분리해 전체 component inventory를 제출한다. |
| SBOM 필드 | library, version, license, official URL, use | 모든 dependency에 필수 필드를 채운다. |
| 붙임 2 | AI 모델 활용 시 작성 | 탑재 모델이 있으면 유형과 활용 범위를 기록한다. |
| 상용 AI 보조도구 | AI 모델 활용 명세서를 작성하는 경우 4번 항목에 상용 AI 보조도구 활용 여부와 범위를 기재 | 사용한 상용 AI 보조도구와 활용 범위를 4번 항목에 기록한다. |

## 5. 오리엔테이션 확인 결과

2026-07-22 공지에 따르면 오리엔테이션은 2026-07-23 14:30 Zoom으로 진행되었고 접속정보는 참가자 이메일로 개별 발송되었다. 프로그램 이미지에는 15:20부터 15:40까지 대회 소개 및 평가기준 안내가 배정되어 있다.

2026-07-26 현재 공지와 공식 YouTube 영상 목록에서는 2026 오리엔테이션 녹화나 평가자료의 공식 공개 링크를 확인하지 못했다. 공식 YouTube의 최신 공개 영상은 2025 수상작이다. 다음 중 하나를 확보하면 해당 claim을 다시 판정한다.

- 주최 측의 2026 오리엔테이션 녹화 URL
- 공식 발표자료 URL
- 참가자에게 배포된 공식 평가자료

## 6. Day 1 gap과 후속 게이트

| 상태 | Gap | 닫는 조건 |
|---|---|---|
| confirmed gap | 저장소가 현재 Private | 제출 전 Public 전환, 비로그인 clone 및 실행 성공 |
| unconfirmed | 2026 세부 평가 배점 | 모집 공고의 배점표 또는 공식 심사계획 원문 확보 |
| unconfirmed | 오리엔테이션 녹화 및 평가자료 공개 링크 | 주최 측 공식 URL 또는 참가자 배포 원문 확보 |

Day 1 이후 공식 요구사항을 변경할 때는 source registry에 원문을 먼저 등록하고, claim ledger의 status, locator와 implementation consequence를 함께 갱신한다.
