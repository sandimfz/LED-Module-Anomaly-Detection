# LED Module Anomaly Detection v2.1

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
| blocking | Area gelap yang memblokir konten | area > 0.2% panel |
| dark_regions | Area gelap dari local contrast | cell < mean - 1.5×std |
| module_error | Modul dengan chaos tinggi | chaos score > 0.3 |
| line_defects | Garis horizontal/vertical abnormal | local minimum detection |
| color_errors | Warna menyimpang dari neighbors | hue delta > 35 |
| dead_blocks | Blok mati di dalam content mask | brightness < 25 |
| uniformity | Konten terlalu uniform (frozen?) | std < 25 AND ratio < 0.20 |
| pixel_chaos | Area dengan pixel-level chaos | entropy analysis |
| flat_content | Area terlalu rata | std dev rendah |
| horizontal_line | Pola garis horizontal | edge detection |
| region_contrast | Kontras antar region | diff > 1.5×neighbor_std |

## Scoring

### LED Analyzer Score

```
base_score = (area_contribution × 0.5) + (avg_severity × 0.5)
type_boost = min(unique_priority_types × 0.15, 0.3)
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
