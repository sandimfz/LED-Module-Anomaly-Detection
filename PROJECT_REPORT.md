# LED Module Anomaly Detection — Project Report

**Versi:** v3.0
**Tanggal:** 2026-07-17
**Status:** Aktif dalam pengembangan

---

## 1. Ringkasan Proyek

**Tujuan:** Deteksi otomatis kerusakan modul LED pada videotron dari foto/screenshot CCTV.

**Target defect:**
- **Blocking** — area gelap/putih yang memblokir konten LED
- **Module error** — modul rusak, glitch, stuck, dead

**Lokasi:** Lengkong (Bandung), Paskal (Bandung), Sigma Cirebon

**Stack:** Python, OpenCV, NumPy, scikit-learn, Anomalib (opsional)

---

## 2. Arsitektur Sistem

### Alur Deteksi
```
Input image (CCTV screenshot)
  → Perspective transform (screen_points calibration)
  → LEDAnalyzer: jalankan 11 sub-detectors
  → EnsemblePipeline: gabungkan hasil (weighted average)
  → Output: annotated image + report JSON
```

### Detector yang Aktif

| Detector | File | Fungsi | Threshold |
|----------|------|--------|-----------|
| blocking | `blocking.py` | Area gelap/putih memblokir konten | base_severity=0.3 + area/contrast bonus |
| flat_content | `flat_content.py` | Blank/stuck regions (white/black) | var < 15, mean > 200 / < 15 |
| module_error | `module_error.py` | Module glitch/stuck | chaos > 0.75 OR flat_stuck = 0.8 |
| color_errors | `color_errors.py` | Color shift per block | hue_shift > 70° + ≥3 saturated neighbors |
| region_contrast | `region_contrast.py` | Region vs neighbors | MAD z-score > 4.5 |
| dead_blocks | `dead_blocks.py` | Dead pixel blocks | mean < active_mean × 0.2 |
| dark_regions | `dark_regions.py` | Dark regions | mean < global_mean - 2.5×std |
| line_defects | `line_defects.py` | H/V line defects | local minimum + >40% dark in row |
| uniformity | `uniformity.py` | Frozen/stuck detection | std < 10 + edge check |
| horizontal_line | `horizontal_line.py` | Horizontal patterns | row variance > mean + 5×std |
| grid | `grid.py` | Neighbor contrast | adaptive dark/flat/color thresholds |
| dark_spot | `dark_spot.py` | Dark spots in LED | mean < max(ratio, mean-k×std, 30) |
| temporal | `temporal.py` | Frozen/stuck (multi-frame) | activity_score + robust_z |

### Ensemble Scoring
```python
weights = {
    "led_analyzer": 0.50,    # 11 sub-detectors digabung
    "temporal": 0.15,         # frozen/stuck detection
    "dark_spot": 0.08,
    "grid": 0.07,
    "patchcore": 0.03,        # ML-based (opsional)
    "anomaly_detector": 0.02  # IsolationForest (opsional)
}
# Level: NORMAL <0.30, WARNING 0.30-0.55, CRITICAL >=0.55
# Force CRITICAL: if ≥2 detectors agree AND max_score > 0.7
```

---

## 3. Screen Points Calibration

Perspective transform untuk crop gambar ke area LED screen.

| Lokasi | Resolusi | TL | TR | BR | BL | Status |
|--------|----------|-----|-----|-----|-----|--------|
| Lengkong | 1280×720 | [349,258] | [1046,190] | [988,434] | [441,666] | ✅ Direcalibrate 2026-07-17 |
| Paskal | 1280×720 | [52,44] | [943,2] | [904,705] | [148,386] | ✅ Direcalibrate 2026-07-17 |
| Sigma | 1920×1080 | [648,122] | [1811,42] | [1748,546] | [847,1065] | ✅ Direcalibrate 2026-07-17 |

**Lookup order:** `screen_points_map[res]` → `screen_points` (exact match) → linear scaling (fallback)

---

## 4. Dataset

```
dataset/
├── lengkong/
│   ├── good/ (142 images, Jul 7-15 2026, 4 time slots: 08/12/16/20)
│   └── bad/ (23 images, berbagai defect types)
├── paskal/
│   ├── good/ (17 images, Jul 13-15 2026)
│   └── bad/ (12 images)
└── sigma_cirebon/
    ├── good/ (6 images, Jun 24 2026)
    └── bad/ (1 image)
```

### Jenis Defect di Dataset Bad

| Jenis | Jumlah | Detektor | Status |
|-------|--------|----------|--------|
| Flat white stuck | ~15 | flat_content | ⚠️ Partial (64% tertangkap) |
| Black blocking | ~4 | blocking | ⚠️ Partial (depend contrast) |
| Subtle bar | ~5 | line_defects | ❌ Sulit (beda tipis dari konten) |
| Dark content | ~3 | dark_regions | ⚠️ Bisa jadi TN (konten normal redup) |
| Color cast | ~2 | color_errors (insidental) | ❌ Tidak dedicated |
| Mixed/small | ~7 | campuran | ⚠️ Bergantung detector |

---

## 5. Hasil Test (Full Batch — 2026-07-17)

### Per Lokasi

| Lokasi | Bad | TP | Recall | Good | TN | FP | FP Rate |
|--------|-----|-----|--------|------|-----|-----|---------|
| Lengkong | 23 | 14 | **61%** | 142 | 76 | 66 | **46.5%** |
| Paskal | 12 | 8 | **67%** | 17 | 6 | 11 | **64.7%** |
| Sigma | 1 | 0 | **0%** | 6 | 4 | 2 | **33.3%** |
| **TOTAL** | **36** | **22** | **61%** | **165** | **86** | **79** | **47.9%** |

### FN Analysis (14 gambar masih miss)

| Defect Type | Jumlah FN | Kenapa Miss |
|-------------|-----------|-------------|
| Flat white stuck | 9 | `flat_content` butuh `var<15 AND mean>200`, miss mean 170-190 |
| Subtle bar | 3 | Beda brightness hanya 11-20 dari panel, tidak terdeteksi |
| Dark content | 2 | Bisa jadi TN (konten normal redup), bukan defect |

### FP Analysis (79 gambar salah ter-flag)

| Detector | FP Rate | Penyebab |
|----------|---------|----------|
| color_errors | 23.0% | Hue shift normal (teks putih vs bg berwarna) |
| module_error | 25.5% | Flat stuck trigger di konten flat alami |
| grid | ~15% | Neighbor contrast false positive |
| flat_content | 10.3% | Panel mask fallback termasuk area normal |
| lainnya | <10% | |

### Perubahan dari v2.2 ke v3.0

| Metrik | v2.2 (awal) | v3.0 (sekarang) |
|--------|-------------|-----------------|
| Recall (bad) | 39% | **61%** |
| FP rate (good) | 0%* | **47.9%** |
| Screen points | Belum dikalibrasi | ✅ 3 lokasi |
| Flat white detection | ❌ Tidak ada | ⚠️ Partial |

*FP 0% di v2.2 karena threshold sangat tinggi (0.55 CRITICAL), banyak defect asli juga miss.

---

## 6. Perubahan di v3.0

### Quick Wins (Telah Di-commit)

1. **`flat_content.py`** — panel_mask fallback (base_ratio 0.5), block 64px
2. **`region_contrast.py`** — MAD-based z-score (z>4.5), min_region_cells=6
3. **`blocking.py`** — base_severity=0.3 untuk module-scale defect
4. **`module_error.py`** — flat_stuck_score=0.8 + min_region_cells=1 + per-lokasi config
5. **`color_errors.py`** — hue_shift 55°→70° (FP turun 15.8pp)
6. **`config.py`** — recalibrate screen_points 3 lokasi
7. **`types.py`** — per-lokasi flat_stuck_std_threshold + flat_stuck_neighbor_threshold
8. **`calibrate_screen.py`** — fix info bar overlap (50px offset)

### File Baru (Dibuat, Belum Terpasang Penuh)

| File | Fungsi | Status |
|------|--------|--------|
| `module_grid.py` | Grid fisik cabinet LED | Dibuat, belum terpasang ke pipeline |
| `feature_extractor.py` | Fitur per-modul (median/MAD relatif) | Dibuat, belum terpasang |
| `spatial_analyzer.py` | Spatial decorrelation z-score | Dibuat, belum terpasang |
| `defect_scorer.py` | Unified scoring per modul | Dibuat, belum terpasang |
| `temporal_correlation.py` | Corr(signal_M, global_signal) | Dibuat, belum teruji |
| `baseline_store.py` | Rolling statistics per modul | Belum dibuat |
| `content_classifier.py` | Klasifikasi konten (soccer/text/video) | Dibuat, belum ter-wire |
| `anomaly_detector.py` | IsolationForest ML | Dibuat, belum ter-wire |
| `simple_defect_detector.py` | Baseline statistik JSON | Dibuat, belum terpakai |
| `temporal_analyzer.py` | Frozen detection + persistence | Dibuat, belum ter-prioritas |

---

## 7. Masalah yang Masih Ada

### Prioritas Tinggi

1. **FP 47.9%** — Detektor terlalu sensitif pada konten normal. Grid + color_errors + module_error flat_stuck semua trigger di iklan solid color.

2. **FN 39% (14 gambar)** — Flat white stuck tidak tertangkap `flat_content` karena `mean > 200` terlalu tinggi. Subtle bar tidak terdeteksi tanpa temporal.

3. **Tidak ada module-grid fisik** — Semua grid arbitrary (32/48/64px). Module fisik LED cabinet tidak diketahui.

4. **Tidak ada temporal correlation** — Single image tidak bisa bedakan "panel fade" dari "modul stuck". `temporal_correlation.py` dibuat tapi belum teruji (butuh ≥8 frame burst).

### Prioritas Sedang

5. **Tidak ada baseline historis** — Rolling statistics per modul belum ada. Adaptive threshold masih manual per lokasi.

6. **Scoring belum per-modul** — Masih per-gambar. Tidak bisa jawab "modul mana yang defect".

7. **Consensus voting belum ada** — Ensemble pakai weighted average, 1 detector sensitif sudah cukup push WARNING.

8. **Label ground truth tidak ada** — Tidak ada dataset "modul ini defect" untuk validasi precision/recall per-modul.

### Prioritas Rendah

9. **Color cast global** — Tidak ada dedicated detector. Heuristik `hue_std>30 AND sat_mean>60` dicatat sebagai backlog.

10. **Partial dimming** — Module redup tapi tidak mati total, tidak ada detector.

---

## 8. Cara Pakai

### Deteksi Satu Gambar
```bash
python detect.py --location lengkong --image path/to/image.jpg
```

### Deteksi Multiple Frames
```bash
python detect.py --location lengkong --frames path/to/folder/
```

### Training PatchCore
```bash
python train.py --location lengkong
```

### Training Anomaly Detector (IsolationForest)
```bash
python train_anomaly_detector.py --location lengkong
```

### Kalibrasi Screen Points
```bash
python calibrate_screen.py --location lengkong --resolution 1280x720
```

---

## 9. Struktur File

```
led/
├── detect.py                    # Entry point deteksi
├── train.py                     # Training PatchCore
├── train_anomaly_detector.py    # Training IsolationForest
├── calibrate_screen.py          # Kalibrasi screen points
├── README.md                    # Dokumentasi user
├── PROJECT_REPORT.md            # Dokumentasi teknis (file ini)
│
├── src/
│   ├── core/
│   │   ├── config.py            # Konfigurasi lokasi + screen_points
│   │   ├── types.py             # Dataclass (LocationConfig, DetectionResult, dll)
│   │   └── utils.py             # Utility functions
│   ├── detectors/
│   │   ├── base.py              # BaseDetector ABC
│   │   ├── grid.py              # Grid neighbor-contrast detector
│   │   ├── dark_spot.py         # Dark spot detector
│   │   ├── temporal.py          # Temporal frozen/stuck detector
│   │   ├── patchcore.py         # PatchCore anomaly detector
│   │   └── led/
│   │       ├── analyzer.py      # LEDAnalyzer orchestrator
│   │       ├── scoring.py       # Anomaly score calculation
│   │       ├── content_mask.py  # LED content masking
│   │       ├── panel_finder.py  # Auto-detect LED panel
│   │       ├── annotation.py    # Image annotation
│   │       ├── types.py         # LEDAnomaly, LEDPanelInfo
│   │       ├── helpers.py       # Safe brightness calculations
│   │       ├── module_grid.py   # [BARU] Module grid calibration
│   │       ├── feature_extractor.py  # [BARU] Per-module features
│   │       ├── spatial_analyzer.py   # [BARU] Spatial decorrelation
│   │       ├── defect_scorer.py      # [BARU] Unified scoring
│   │       ├── temporal_correlation.py # [BARU] Signal correlation
│   │       ├── content_classifier.py  # [BARU] Content type classification
│   │       ├── anomaly_detector.py    # [BARU] IsolationForest
│   │       ├── simple_defect_detector.py # [BARU] Baseline stats
│   │       └── detectors/
│   │           ├── blocking.py         # Blocking detection
│   │           ├── flat_content.py     # Flat/stuck detection
│   │           ├── module_error.py     # Module error + flat_stuck
│   │           ├── color_errors.py     # Color shift detection
│   │           ├── region_contrast.py  # Region contrast (MAD-based)
│   │           ├── dead_blocks.py      # Dead pixel blocks
│   │           ├── dark_regions.py     # Dark regions
│   │           ├── line_defects.py     # Line defects
│   │           ├── uniformity.py       # Uniform content detection
│   │           ├── horizontal_line.py  # Horizontal patterns
│   │           └── pixel_chaos.py      # Pixel chaos (disabled)
│   └── pipeline/
│       └── ensemble.py          # Ensemble pipeline + scoring
│
├── dataset/                     # Input data per lokasi
├── models/                      # Trained models
├── output/                      # Detection results
└── sample/                      # Sample images
```

---

## 10. Commit History

```
bebff35 docs: add color cast global to known limitations
8702eb4 fix: raise color_errors hue_shift threshold 55°→70°
5c9ddb4 fix: paskal per-lokasi flat_stuck tuning
2468b72 fix: recalibrate paskal + sigma, per-lokasi flat_stuck config, fix calibrate UI
27fffe9 fix: recalibrate screen_points for lengkong location
f5ca268 fix: blocking severity formula + module_error flat_stuck_score
31b538c feat: LED detection accuracy improvements v3.0
2e995d6 feat: false positive fixes v2.2
```

---

## 11. Roadmap

### Phase 1: Foundation (Selesai)
- [x] Screen points calibration 3 lokasi
- [x] Flat white stuck detection (panel_mask fallback)
- [x] Region contrast MAD-based z-score
- [x] Module error flat_stuck_score
- [x] Blocking severity fix
- [x] Color errors threshold tuning
- [x] Per-lokasi config

### Phase 2: Accuracy (Dikerjakan)
- [ ] Module grid fisik calibration
- [ ] Feature extractor (relative normalization)
- [ ] Spatial analyzer integration
- [ ] Flat content `diff > 25 dari neighbor median`
- [ ] Consensus voting antar detector

### Phase 3: Temporal (Belum Mulai)
- [ ] Temporal correlation testing (≥8 frame burst)
- [ ] Persistence check (N frame berturut)
- [ ] Baseline historis per modul

### Phase 4: Production (Belum Mulai)
- [ ] Label ground truth per modul
- [ ] Precision/recall metrics
- [ ] Content-aware masking
- [ ] Color cast dedicated detector
- [ ] Partial dimming detection

---

## 12. Catatan Teknis

### Kenapa FP Tinggi (47.9%)

Root cause bukan 1 detector, tapi kombinasi:
1. **color_errors (23%)** — hue shift normal di konten iklan (teks putih vs bg)
2. **module_error flat_stuck (25%)** — trigger di background solid alami
3. **grid (~15%)** — neighbor contrast false positive
4. **flat_content (10%)** — panel mask termasuk area normal

### Kenapa FN 39%

Root cause:
1. **flat_content** `mean > 200` terlalu tinggi → miss white stuck mean 170-190
2. **Tidak ada temporal** → subtle bar tidak terdeteksi tanpa multi-frame
3. **Single image analysis** tidak bisa bedakan defect dari konten normal

### Kenapa Perubahan Satu Detector Mempengaruhi Lain

Karena `LEDAnalyzer` gabungan 11 sub-detector → `calculate_score` pakai `area_contribution + avg_severity + type_boost`. Ketika 1 detector (misal color_errors) fire, ia menyumbang ke `avg_severity` yang menaikkan `base_score` yang menaikkan `led_analyzer` score yang menaikkan `ensemble` score.

### Kenapa Paskal FP Lebih Tinggi dari Lengkong

Paskal konten iklan banyak menggunakan background solid warna (merah, biru) dengan teks putih → flat_stuck trigger lebih sering. Lengkong konten lebih bervariasi (video, animasi).
