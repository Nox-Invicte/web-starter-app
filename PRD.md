ScreenShield — Product Requirements Document (PRD) (Expanded: models, memory, benchmarking, updates)

Short pitch: ScreenShield is a privacy-first, on-device screenshot sanitizer. It inspects screenshots locally, highlights sensitive regions, and offers safe redaction — all offline, instant, and private.

1 — Product Overview (concise)

Product Name: ScreenShield
Platform: Mobile (React Native). Admin / dashboard (optional): React + Vite.

ScreenShield analyzes screenshots on-device using RunAnywhere on-device models to detect and redact sensitive information (passwords, API keys, tokens, CC numbers, OTPs, PII) before sharing. All inference runs locally — no screenshot or extracted text leaves the device unless the user explicitly opts into encrypted sync.

RunAnywhere

2 — New additions (what this doc adds)

Model size & quantization plan (MBs)

Memory / runtime budget per device class

Fallback & degraded-mode behaviour for low-end devices

Short benchmarking plan + test-cases (with acceptance criteria)

Model update, secure rollout, and rollback plan

(Everything below is fully actionable — no vague promises. Ready for engineering + ops.)

3 — Model sizing & quantization plan (concrete MB targets)

Assumptions: models are small, optimized for edge: ONNX / TFLite / RunAnywhere format. Quantization targets are INT8 where possible; FP16 on iOS Metal where useful.

Models (baseline configuration — “Balanced” bundle)

OCR model (light) — task: text detection + recognition (single-stage small OCR):
FP32 size (approx): 20 MB → INT8 quantized size: ~6–8 MB
Target runtime latency (single 1080×1920 screenshot): 60–200 ms on mid-range device.

Pattern detector (regex+ML classifier) — task: structured pattern & visual tokens (credit cards, API keys):
FP32: 6 MB → INT8: ~2–3 MB
Latency: <50 ms.

Contextual classifier (tiny LLM / intent verifier) — task: determine whether text is sensitive in context, reduce false positives:
Tiny LLM FP16/FP32: 120 MB (not preferred) → preferred tiny model family:

Tiny (strong default): quantized INT8 ~18–28 MB (recommended)

Micro fallback: INT8 distilled classifier ~1.5–3 MB for low-end devices
Latency: 80–400 ms depending on model and device.

Total (Balanced bundle, all INT8): ~30–40 MB on-device model store.

High-end / Optional (quality-focused) bundle

Improved OCR + larger LLM for fewer false positives: add +30–60 MB, total ~70–100 MB (opt-in for high-end devices).

Low-end / Minimal bundle

OCR (very small) + Micro contextual classifier: ~10 MB total. Sacrifices context sensitivity but keeps privacy and responsiveness.

Quantization strategy

Primary: Post-training INT8 quantization (symmetric/asymmetric as appropriate) — maximum model size reduction with minimal accuracy loss.

Secondary: FP16 (half-precision) builds for iOS Metal where INT8 support is limited or FP16 is faster.

Fallback: Distilled micro-models (binary classifiers) for devices unable to host full models.

Calibration: Per-model calibration with representative dataset (see benchmarking section) to keep accuracy drop <5% vs FP32.

4 — Memory & runtime budget (targets by device class)

Memory numbers include model memory + runtime working memory (activations) + app overhead for AI pipeline, and assume no other heavy app usage.

Device classes

Low-end (LE) — 1.5–2 GB RAM, older ARM cores (Cortex-A53), Android 8–10.
Target memory budget: ≤ 200 MB (model + runtime).
Model bundle: Minimal (~10 MB), micro classifier fallback.
Processing time target: < 2.5 s for full pipeline.

Mid-range (MR) — 3–6 GB RAM, Snapdragon 6xx/7xx, Android 10–13 / recent iPhone SE.
Target memory budget: 200–400 MB.
Model bundle: Balanced (~30–40 MB).
Processing time target: < 1.5 s.

High-end (HE) — 8+ GB RAM, Snapdragon 8xx / A-series iPhones.
Target memory budget: 400–800 MB (if using heavier LLM).
Model bundle: Full (70–100 MB opt-in).
Processing time target: < 0.6–1.0 s.

Additional runtime considerations

Use NNAPI / GPU delegate / Metal delegate where available to reduce CPU overhead and energy.

Ensure peak allocated memory is limited via streaming / chunked OCR where possible (process text regions rather than entire high-res images simultaneously).

Graceful degradation: if memory pressure is detected, switch to micro models and notify user.

5 — Fallback & degraded-mode behaviour (low-end devices)

Goal: preserve privacy and core value (prevent accidental leaks) even on constrained devices.

Degraded modes (ranked)

Full mode (default on MR/HE): OCR + Pattern detection + Contextual LLM → highest accuracy, low FP.

Intermediate mode (LE preferred): Lightweight OCR + Pattern detection + micro-context classifier → good balance.

Conservative mode (ultra-low): Only high-confidence regex/pattern detection (credit cards, API keys, phone numbers) — no LLM → highest speed, higher FP risk for contextual items.

Behavioural flow

On app install / first-run, detect device profile (RAM, CPU, NNAPI/Metal availability). Automatically select recommended model bundle. Show short notice: “Optimized for your device: Balanced / Lightweight / Minimal.” User can opt for “High-recall” (bigger model) or “Low-memory” (smaller).

Dynamic switching: If inference fails due to OOM or slow perf, switch to next lower mode and log event locally. Show unobtrusive notification: “Switched to lightweight detection for smoother performance.”

User-facing UX in degraded mode:

Always highlight definite matches (regex-based).

For uncertain detections that were handled by LLM in full mode, show a “Possible sensitive content” badge with “Review manually” CTA.

Offer one-tap “Auto-redact all definite matches” (safe default).

Optional server-assisted mode (opt-in, enterprise):

If user consents and enables encrypted sync (explicit consent), offload complex checks to cloud for a short session. This is opt-in only and never the default. All communication must be end-to-end encrypted and signed.

6 — Benchmarking plan (how we measure success)
Objectives (what to measure)

Latency (total end-to-end from image load → redaction UI ready)

Peak memory usage (MB) during inference

CPU usage & energy (approx battery impact per minute)

Detection accuracy (Precision / Recall / F1) per sensitive class (API key, password, CC, email, phone, SSN)

False positive rate (FPR) and false negative rate (FNR)

Throughput (images processed per minute in batch)

Perceptual quality of redaction (no bleed-through, visual artifacts)

Device matrix (minimum test devices)

Low-end Android: MediaTek/Qualcomm with 1.5–2 GB RAM, e.g., ARM Cortex-A53-based phone (Android 9).

Mid-range Android: Snapdragon 720G, 4–6 GB RAM (Android 11/12).

High-end Android: Snapdragon 8xx (latest), 8+ GB.

iOS mid: iPhone 11 / 12 (iOS 15/16).

iOS low: iPhone SE 2 (if needed).

Emulator/CI: Android x86 + instrumented CPU + optional NNAPI delegate simulation.

Test dataset composition (representative heatmap)

Banks & finance: 350 images — bank statements, mobile bank screenshots, card images, masked/unmasked.

Developer consoles: 300 images — code snippets, terminal outputs, .env files, API key lines.

Messaging & emails: 300 images — messages with OTP, phone numbers, emails.

Receipts & invoices: 200 images — multi-column receipts, small fonts.

Multi-lingual UIs: 200 images — Tamil, Hindi, English mixed text, RTL samples if relevant.

Noisy/real-world: 150 images — low-light photos of phone screen, compressed images, screenshot with annotations/stickers, UI overlays.

Edge cases: 100 images — rotated, cropped, partial text, stylized fonts.

Total dataset target: ~1,600 sample images for initial benchmarking. Augment with synthetic variants for robustness (scaling, compression, rotation, occlusion).

Benchmark runs & metrics collection

For each device, run the entire dataset (1,600 images) and collect: latency histogram, memory peak, accuracy per class, energy estimate (if possible), and failure cases.

Compute precision/recall per sensitive type and macro-averaged F1.

Acceptance thresholds:

Latency: MR < 1.5 s, LE < 2.5 s

Detection Accuracy: >90% overall (target), per-class minimums: CC 95%, API keys 92%, passwords 90%

Redaction success: >95% (visual integrity)

FNR (critical items): <5% for CC & API keys.

CI & regression

Integrate a small, curated regression suite (200 images) into CI. Use model diff tests: if new model lowers F1 by >1.5% on regression set, reject automatic rollout.

7 — Test cases (concrete examples & acceptance criteria)
Test Case A — Bank app with visible account and partial card

Input: mobile banking screenshot with account number and masked card with last 4 visible.

Expected output: detect account number (highlight), detect card number (highlight masked parts), suggest redaction.

Acceptance: both detections present, redaction exports with redacted area covering numbers. Pass if precision and recall for this case = 100%.

Test Case B — Terminal / .env file with API key

Input: developer console screenshot with AKIA... style key and surrounding logs.

Expected: detect API key and token patterns, high confidence.

Acceptance: auto-highlight + suggested redact; detection confidence >0.9.

Test Case C — Messaging app containing OTP and phone number

Input: chat screenshot with OTP (4–6 digits) and phone number.

Expected: both detected; OTP low TTL flagged (suggest quick redact).

Acceptance: OTP and phone detected; redaction export hides both.

Test Case D — Mixed-language UI (Tamil + English)

Input: screenshot with Tamil labels and English email address.

Expected: OCR extracts both languages; email detected.

Acceptance: email detected correctly; non-sensitive Tamil text ignored (no false positive).

Test Case E — Low-quality photo of a screen (glare, angle)

Input: phone camera photo of another phone showing an app with a visible password.

Expected: OCR may degrade; still detect visible digits/keywords if legible.

Acceptance: if readable, detection; if not readable, app should indicate low-confidence and allow manual redaction.

Test Case F — False-positive reduction

Input: snippet of code with long hex strings that are not secrets.

Expected: LLM/context classifier should avoid flagging non-secrets or explain risk as low.

Acceptance: major false positives reduced; show contextual “likely not a secret” flag.

8 — Model update, secure rollout & rollback (engineering-grade)
Goals

Deliver model improvements safely and auditable.

Ensure on-device model authenticity (signed) and atomic updates.

Provide secure rollback if a model causes regressions.

Components

Model Registry (backend) — HTTPS endpoint (GET /models) returns model metadata: version, model hash (SHA-256), signature, size, minDeviceProfile, release notes, canary flag.

Model Package — ZIP contains model(s), manifest.json, signature.sig. Manifest fields: version, sha256 per file, compatibility profile, digital signature metadata.

Signing & Verification — Models signed by private key held in CI/CD. Devices verify using embedded public key pinned in app (rotate via secure update process). Use elliptic curve signatures (Ed25519 or ECDSA P-256).

Delta/Chunked downloads — support partial downloads (resume) and optional delta patching (bsdiff/rpmb) for bandwidth efficiency.

Atomic Swap — Download to temp location, verify hashes & signature, validate model on-device (sanity checks), then atomically rename to active folder. If verification fails, delete temp and keep current model.

Rollout strategy

Canary rollout: rollout % by device class (e.g., 1% MR, 0.5% LE) for 48–72 hours. Collect local metrics (no sensitive data) and model health telemetry (inference time, crash rates, precision/recall on a tiny anonymized test set built into app). Telemetry is opt-in.

Automated validation gates: On-device quick self-check runs regression suite (small curated set) after model install. If metrics degrade beyond threshold, the device auto-rolls back and reports the failure to backend (only telemetry, no screenshot/text).

Manual rollback: Admin can mark a model as revoked in registry; devices will detect revocation and revert to last-good model.

Secure rollback mechanics

Maintain last two successful model versions on device (current + fallback). On failure, swap back to fallback. If fallback missing, revert to minimal micro-model bundled inside app (never deleted) and notify user.

Key rotation & secure bootstrapping

Public key pinning stored in app with a key-rotation schedule. Rotations require signed key bundles and cross-signed tokens from old & new keys to allow smooth transition.

Recovery: If key compromised, emergency manifest rotation via new app update (signed by store-signed update).

Privacy & audit

Telemetry limited to non-sensitive metrics: model version, inference latency, memory peak, detection counts. Never include extracted text or images. Telemetry must be explicit opt-in.

9 — CI/CD & governance for model lifecycle

Model training pipeline → versioned artifacts, unit tests, regression tests.

Automated evaluation → run full benchmark suite, produce report with precision/recall and delta vs prior version.

Sign & publish → only after passing thresholds and review.

Canary rollout → 48–72h, automated health checks.

Full rollout → after canary success.

Rollback → automatic if thresholds breached or manual via admin.

Artifacts to store in registry: model binary, manifest.json, signed hash, changelog, expected performance numbers.

10 — UX / User communication considerations

First-run: explain privacy and the model bundle size chosen for the device (e.g., “Lightweight models (12 MB) chosen for your device”).

Settings: “Detection sensitivity” slider (High Recall / Balanced / Low False-Positives). Higher sensitivity may use larger model (ask for permission to download).

Transparent consent for model telemetry and optional cloud assistance.

Show model version & last update in Settings → About → Model info (signed hash).

11 — Acceptance criteria (summary)

Model bundle size for default Balanced <= 40 MB (INT8).

Low-end fallback bundle <= 12 MB.

MR inference time: < 1.5 s end-to-end. LE: < 2.5 s.

Detection accuracy overall >= 90%, per-class minima as specified.

Redaction success rate >= 95%.

Canary rollout and rollback implemented with signed models and atomic swap.

CI regression gate prevents regressions >1.5% F1 on curated suite.

12 — Appendix — quick reference (tables)
Model sizes (recommended builds)
Model	INT8 (MB)	FP16/FP32 (MB)	Role
OCR (small)	6–8	20	Text detection+recognition
Pattern detector	2–3	6	Structured token detection
Context LLM (tiny)	18–28	60–120	Contextual filtering
Micro classifier	1.5–3	5–8	Low-end fallback
Balanced total	~30–40	—	Combined
Memory budgets (peak)

Low-end: ≤200 MB

Mid-range: 200–400 MB

High-end: 400–800 MB

13 — Next steps for engineering (milestones)

Create representative benchmarking dataset (1,600 images) — label ground truth.

Train/quantize OCR, detector, micro-classifier. Produce INT8 builds and run calibration.

Integrate model registry + signing in CI.

Implement atomic model swap & canary rollout client logic.

Run device matrix benchmarking, iterate models until acceptance thresholds met.

Prepare demo scenarios and a 3-minute video showing full flow on MR and LE devices (include “fallback” story).