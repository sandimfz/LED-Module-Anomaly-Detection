# LED Module Anomaly Detection v2.2

Sistem deteksi kerusakan modul LED pada videotron dengan pendekatan **hybrid**: LED Analyzer + Grid + Temporal + PatchCore.

## Fitur

- **LED Analyzer** — Analisis seperti AI: auto-detect panel LED, deteksi blocking, dead blocks, line defects, color errors
- **Dark Spot Detector** — Deteksi area gelap kecil di dalam LED
- **Grid Detector** — Deteksi cepat dengan membandingkan cell ke tetangganya
- **Temporal Detector** — Deteksi modul yang tidak berubah seiring waktu (cocok untuk videotron)
- **PatchCore** — Deteksi berdasarkan memory bank gambar normal (perlu training)
- **Ensemble Pipeline** — Menggabungkan hasil semua detector untuk keputusan lebih akurat
- **Perspective Transform** — Crop otomatis ke area LED screen dengan kalibrasi per resolusi
- **Calibration Tool** — Interactive tool untuk kalibrasi screen points
- **Auto-organize Output** — Output otomatis terpisah berdasarkan status (good/warning/critical)

## Struktur Folder

```
led/
├── dataset/               # Input data
│   ├── sigma_cirebon/
│   │   ├── good/          # Gambar normal (WAJIB)
│   │   └── bad/           # Gambar rusak (opsional)
│   ├── lengkong/
│   │   ├── good/
│   │   └── bad/
│   └── paskal/
│       ├── good/
│       └── bad/
│
├── models/                # Model trained (otomatis)
│   ├── sigma_cirebon/
│   ├── lengkong/
│   └── paskal/
│
├── output/                # Hasil inference (otomatis)
│   └── <DD-MM-YYYY>/<HH>/
│       ├── good/          # Output gambar NORMAL
│       ├── warning/       # Output gambar WARNING
│       ├── critical/      # Output gambar CRITICAL
│       └── report_*.json  # JSON reports
│
├── src/                   # Source code
│   ├── core/              # Config, types, utils
│   ├── detectors/         # Detection methods
│   │   ├── led/           # LED Analyzer sub-detectors
│   │   │   ├── blocking.py
│   │   │   ├── dark_regions.py
│   │   │   ├── module_error.py
│   │   │   ├── line_defects.py
│   │   │   ├── color_errors.py
│   │   │   ├── dead_blocks.py
│   │   │   ├── uniformity.py
│   │   │   ├── pixel_chaos.py
│   │   │   ├── flat_content.py
│   │   │   ├── horizontal_line.py
│   │   │   └── region_contrast.py
│   │   ├── grid.py
│   │   ├── dark_spot.py
│   │   ├── temporal.py
│   │   └── patchcore.py
│   └── pipeline/
│       └── ensemble.py
│
├── calibrate_screen.py    # Calibration tool
├── train.py               # Training script
├── detect.py              # Detection script
└── README.md
```

## Instalasi

```bash
python -m venv venv
source venv/bin/activate
pip install opencv-python numpy torch anomalib
```

## Cara Pakai

### 1. Deteksi Satu Gambar (Default)

```bash
python detect.py --location lengkong --image <path>
```

### 2. Deteksi dengan Semua Detector (Termasuk PatchCore)

```bash
python detect.py --location lengkong --image <path> --all
```

### 3. Deteksi Basic (Grid + DarkSpot saja)

```bash
python detect.py --location lengkong --image <path> --basic
```

### 4. Deteksi Temporal (Multiple Frames)

```bash
python detect.py --location lengkong --frames <folder>
```

### 5. Training PatchCore

```bash
python train.py --location lengkong
```

### 6. Kalibrasi Screen Points

```bash
python calibrate_screen.py --location lengkong --resolution 1920x1080
```

## Perspective Transform & Screen Points

Sistem menggunakan perspective transform untuk crop gambar ke area LED screen. Setiap lokasi dan resolusi memiliki screen points (4 corner) yang dikalibrasi.

### Screen Points Structure

```python
LocationConfig(
    screen_points=[[313,240],[1079,169],[1008,448],[422,700]],  # TL, TR, BR, BL
    screen_resolution="1280x720",
    screen_points_map={
        "1920x1080": [[490,356],[1598,256],[1499,670],[634,1047]],
    },
)
```

### Lookup Order

1. **screen_points_map[current_resolution]** — Kalibrasi manual untuk resolusi ini
2. **screen_points** — Jika resolusi cocok dengan screen_resolution
3. **Linear scaling** — Fallback: scale points berdasarkan rasio resolusi

### Kalibrasi Lokasi Baru

```bash
# 1. Jalankan calibration tool
python calibrate_screen.py --location <lokasi> --resolution <WxH>

# 2. Klik 4 corner LED: TL → TR → BR → BL
# 3. Tekan 'y' untuk terima, 'r' untuk reset, ESC untuk selesai
# 4. Paste hasil ke config.py di screen_points_map
```

## Output Structure

Output otomatis terorganisir berdasarkan tanggal dan waktu:

```
output/16-07-2026/12/
├── good/
│   └── <image>_darkspot.jpg
├── warning/
│   └── <image>_darkspot.jpg
├── critical/
│   └── <image>_darkspot.jpg
└── report_20260716_120153.json
```

## LED Analyzer Sub-Detectors

| Detector | Fungsi | Threshold |
|----------|--------|-----------|
| blocking | Area gelap yang memblokir konten | contrast > 0.4, area > 5% panel |
| dark_regions | Area gelap dari local contrast | cell < mean - 2.5×std |
| module_error | Modul dengan chaos tinggi | chaos score > 0.75 |
| line_defects | Garis horizontal/vertical abnormal | local minimum detection |
| color_errors | Warna menyimpang dari neighbors | hue shift > 55°, min 3 neighbors |
| dead_blocks | Blok mati di dalam content mask | brightness < 25 |
| uniformity | Konten terlalu uniform (frozen?) | std < 10 AND ratio < 0.10 + edge check |
| pixel_chaos | Area dengan pixel-level chaos | entropy analysis |
| flat_content | Area terlalu rata | std dev rendah |
| horizontal_line | Pola garis horizontal | variance > 3500, smoothing +5×std |
| region_contrast | Kontras antar region | diff > 2.5×neighbor_std, abs min 35 |

## Scoring

### LED Analyzer Score

```
base_score = (area_contribution × 0.5) + (avg_severity × 0.5)
type_boost = min(unique_priority_types × 0.10, 0.15) + max_severity × 0.1
final_score = min(base_score + type_boost, 1.0)
```

### Ensemble Score

```
led_analyzer:  0.60 weight
dark_spot:     0.15 weight
grid:          0.10 weight
temporal:      0.10 weight
patchcore:     0.05 weight
```

### Level Thresholds

| Score | Level |
|-------|-------|
| < 0.30 | NORMAL |
| 0.30 - 0.55 | WARNING |
| ≥ 0.55 | CRITICAL |

## Interpretasi Hasil

| Score | Level | Keterangan |
|---|---|---|
| 0.0 - 0.30 | NORMAL | LED berfungsi baik |
| 0.30 - 0.55 | WARNING | Perlu dicek lebih lanjut |
| 0.55 - 1.0 | CRITICAL | Kerusakan terdeteksi |

## Menambah Lokasi Baru

```bash
# 1. Buat folder dataset
mkdir -p dataset/<lokasi>/good
mkdir -p dataset/<lokasi>/bad

# 2. Masukkan gambar normal ke good/

# 3. Jalankan deteksi
python detect.py --location <lokasi> --image <path>

# 4. Kalibrasi screen points
python calibrate_screen.py --location <lokasi> --resolution <WxH>

# 5. Tambahkan ke config.py
```

## Troubleshooting

### Output tidak muncul di folder yang benar
- Output terorganisir per tanggal: `output/<DD-MM-YYYY>/<HH>/`

### LED Analyzer tidak menemukan panel LED
- Pastikan gambar memiliki area LED yang terang dan berwarna
- Kalibrasi screen points dengan `calibrate_screen.py`

### Green border melebihi area LED
- Screen points perlu dikalibrasi ulang
- Jalankan `calibrate_screen.py --location <lokasi> --resolution <resolusi>`

### False positive pada gambar normal
- Detector terlalu sensitif terhadap variasi warna normal
- Perlu adjust threshold di `src/detectors/led/detectors/`

## Perubahan v2.2 (False Positive Fixes)

Perbaikan untuk mengurangi false positive pada gambar normal:

### 1. Horizontal Line Detector
- Tambah `_smooth_row_variances()` untuk smooth noise dari content transitions
- Naikkan `min_variance_threshold` dari 2000 → 3500
- Naikkan std multiplier dari 4× → 5×

### 2. Color Errors Detector
- Naikkan hue_shift threshold dari 40° → 55°
- Tambah requirement minimal 3 saturated neighbors

### 3. Region Contrast Detector
- Naikkan `contrast_threshold` dari 1.5 → 2.5
- Tambah absolute threshold minimum 35

### 4. Blocking Detector
- Naikkan std multiplier dari 2× → 2.5×
- Naikkan absolute threshold dari 60 → 80
- Perketat `_classify_blocking()` - require higher aspect ratios
- Tambah requirement contrast > 0.4 untuk large dark areas

### 5. Uniformity Detector
- Naikkan thresholds: std < 10 (dari 15), ratio < 0.10 (dari 0.15)
- Tambah edge detection check - skip jika ada edges (> 5%)

### 6. Ensemble Pipeline
- Hapus aggressive `has_critical` override
- Require ≥2 detectors agree sebelum force CRITICAL
- Naikkan thresholds untuk score boosts

### Hasil Test

| Status | Sebelum v2.2 | Sesudah v2.2 |
|--------|--------------|--------------|
| NORMAL | 0% | 59% |
| WARNING | 0% | 39% |
| CRITICAL | 100% | 1.4% |

### Known Limitations

1. **Single image analysis** tidak bisa bedakan frozen/stuck dari normal uniform content
2. **Moiré patterns** dari kamera masih bisa trigger false positives
3. **Edge cases** dengan content sangat uniform (soccer field) masih bisa terdeteksi sebagai WARNING

### Planned Improvements (v2.3)

1. **Temporal Analysis** - Bandingkan multiple frames untuk detect frozen/stuck
2. **Content Classification** - Classify content type (soccer, text, ad) untuk adaptive thresholds
3. **Location Profiling** - Baseline normal per lokasi

## Development
