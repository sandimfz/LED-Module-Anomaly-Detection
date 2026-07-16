# Graph Report - led  (2026-07-15)

## Corpus Check
- 35 files · ~16,502 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 400 nodes · 867 edges · 20 communities (19 shown, 1 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 39 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7fdf34a2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DetectionResult
- LEDPanelInfo
- EnsemblePipeline
- analyzer.py
- __init__.py
- Path
- utils.py
- line_defects.py
- detect_pixel_chaos
- .detect
- __init__.py
- Path
- Config
- AnomalyLevel
- annotation.py
- .detect
- .detect
- TemporalCellResult
- .detect
- .detect

## God Nodes (most connected - your core abstractions)
1. `LEDPanelInfo` - 48 edges
2. `LEDAnomaly` - 45 edges
3. `DetectionResult` - 33 edges
4. `EnsemblePipeline` - 32 edges
5. `Config` - 26 edges
6. `LocationConfig` - 26 edges
7. `BaseDetector` - 23 edges
8. `AnomalyLevel` - 21 edges
9. `GridDetector` - 21 edges
10. `LEDAnalyzer` - 21 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `LEDAnalyzer`  [INFERRED]
  detect.py → src/detectors/led/analyzer.py
- `detect_single()` --references--> `EnsemblePipeline`  [EXTRACTED]
  detect.py → src/pipeline/ensemble.py
- `detect_frames()` --calls--> `load_frames()`  [EXTRACTED]
  detect.py → src/core/utils.py
- `detect_frames()` --references--> `EnsemblePipeline`  [EXTRACTED]
  detect.py → src/pipeline/ensemble.py
- `main()` --calls--> `EnsemblePipeline`  [EXTRACTED]
  detect.py → src/pipeline/ensemble.py

## Import Cycles
- None detected.

## Communities (20 total, 1 thin omitted)

### Community 0 - "DetectionResult"
Cohesion: 0.15
Nodes (16): ABC, DetectionResult, Hasil deteksi anomali untuk satu gambar.      Attributes:         location: Nama, BaseDetector, Base class untuk semua detectors., Tentukan level anomali berdasarkan skor.          Args:             score: Skor, Base class untuk semua anomaly detector.      Semua detector harus mengimplement, Hitung skor anomali berdasarkan jumlah cell terdeteksi.          Args: (+8 more)

### Community 1 - "LEDPanelInfo"
Cohesion: 0.11
Nodes (14): LEDAnalyzer, ndarray, LED Panel Analyzer (Legacy Compatibility).  This module is kept for backward com, Find LED area in image without cropping.          Detects LED panel location and, Convert screen points to LEDPanelInfo.          Args:             screen_points:, Create result when no LED panel is found.          Args:             image_path:, Detect all anomalies in LED area.          Always runs sub-detectors regardless, Create anomaly for completely off LED.          Args:             panel: LED pan (+6 more)

### Community 2 - "EnsemblePipeline"
Cohesion: 0.06
Nodes (29): detect_frames(), detect_single(), main(), Deteksi anomali pada satu gambar.      Args:         pipeline: Ensemble pipeline, Deteksi anomali dari multiple frames.      Args:         pipeline: Ensemble pipe, Ambil konfigurasi untuk lokasi tertentu.          Args:             location: Na, load_frames(), Load semua frame dari folder.      Args:         folder: Path ke folder yang ber (+21 more)

### Community 3 - "analyzer.py"
Cohesion: 0.23
Nodes (12): _apply_morphological_cleanup(), create_full_image_mask(), create_led_content_mask(), _extract_largest_component(), ndarray, Content mask module.  Functions for creating LED content masks., Create mask for LED content in full image.      Creates a binary mask covering t, Refine panel mask to detect actual LED content.      Applies brightness and satu (+4 more)

### Community 4 - "__init__.py"
Cohesion: 0.05
Nodes (68): LED Analyzer module.  Orchestrates all detection algorithms for LED panel analys, Analyze LED content for anomalies on full image.          Runs all detectors on, _classify_blocking(), detect_blocking(), ndarray, Blocking detection module.  Detects large dark areas that block LED content., Detect blocking anomalies in LED panel.      Args:         gray: Grayscale LED r, Classify type of blocking anomaly.      Args:         cw: Contour width. (+60 more)

### Community 5 - "Path"
Cohesion: 0.10
Nodes (15): crop_image(), main(), ndarray, Apply perspective crop to image.      Args:         image: Input BGR image., Preprocess dataset for a location.      Args:         location: Location name., Path, Ambil path output untuk lokasi.          Struktur folder:             output/<DD, Ambil path untuk reports.          Report ada di folder jam, sama dengan gambar. (+7 more)

### Community 6 - "utils.py"
Cohesion: 0.18
Nodes (13): create_heatmap_overlay(), extract_grid_stats(), get_neighbor_stats(), hue_delta(), load_image(), ndarray, Utility functions untuk LED Anomaly Detection., Ekstrak statistik brightness, std, dan hue untuk tiap cell grid.      Args: (+5 more)

### Community 7 - "line_defects.py"
Cohesion: 0.17
Nodes (18): _detect_horizontal_lines(), detect_line_defects(), _detect_vertical_lines(), _is_local_minimum_x(), _is_local_minimum_y(), ndarray, Line defect detection module.  Detects horizontal and vertical line defects in L, Detect vertical line defects.      Args:         gray: Grayscale LED region. (+10 more)

### Community 8 - "detect_pixel_chaos"
Cohesion: 0.15
Nodes (18): _compute_chaos_stats(), _compute_chaos_threshold(), _compute_panel_brightness(), _confirm_chaos_clusters(), _create_chaos_anomalies(), detect_pixel_chaos(), _find_suspicious_blocks(), merge_chaos_anomalies() (+10 more)

### Community 9 - ".detect"
Cohesion: 0.13
Nodes (14): CellResult, Hasil analisis untuk satu cell/grid.      Attributes:         row: Index baris., GridDetector, ndarray, Deteksi anomali menggunakan neighbor contrast.          Args:             image:, Analisis grid untuk deteksi anomali.          Args:             grid: Grid stati, Cek apakah cell gelap dari tetangga.          Menggunakan adaptive threshold yan, Cek apakah cell terlalu rata/flatt.          Args:             stat: Statistik c (+6 more)

### Community 11 - "Path"
Cohesion: 0.16
Nodes (20): _cleanup_mask(), _compute_candidate_score(), _compute_rectangularity(), _compute_region_stats(), _create_initial_mask(), _evaluate_candidates(), find_led_panel(), perspective_crop() (+12 more)

### Community 12 - "Config"
Cohesion: 0.15
Nodes (10): Config, Konfigurasi untuk LED Anomaly Detection System., Konfigurasi global sistem.      Attributes:         BASE_DIR: Directory root pro, Buat semua direktori yang diperlukan., Core modules untuk LED Anomaly Detection., LocationConfig, Konfigurasi untuk satu lokasi LED.      Attributes:         name: Nama lokasi (m, Initialize detector.          Args:             config: Konfigurasi lokasi. (+2 more)

### Community 13 - "AnomalyLevel"
Cohesion: 0.19
Nodes (11): Enum, AnomalyLevel, FrameStats, Type definitions untuk LED Anomaly Detection., Level keparahan anomali., Statistik untuk satu frame.      Attributes:         frame_index: Index frame da, PatchCoreDetector, PatchCore Anomaly Detector.  Wrapper untuk Anomalib PatchCore. Memerlukan traini (+3 more)

### Community 14 - "annotation.py"
Cohesion: 0.22
Nodes (14): annotate_image(), _draw_anomalies(), _draw_label(), _draw_panel_border(), _draw_summary(), _get_severity_color(), ndarray, Annotation module.  Functions for annotating images with detected anomalies. (+6 more)

### Community 15 - ".detect"
Cohesion: 0.22
Nodes (6): ndarray, Buat heatmap dari cell results.          Args:             cells: List hasil ana, Buat overlay gambar dengan heatmap.          Args:             image: Gambar asl, Generate pesan deskriptif.          Args:             score: Skor anomali., Initialize temporal detector.          Args:             config: Konfigurasi lok, Deteksi anomali menggunakan multiple frames.          Args:             image: F

### Community 16 - ".detect"
Cohesion: 0.22
Nodes (6): ndarray, Path, Deteksi anomali menggunakan PatchCore.          Args:             image: Gambar, Generate pesan deskriptif.          Args:             score: Skor anomali., Load model dari checkpoint.          Raises:             FileNotFoundError: Jika, Cari checkpoint terbaru.          Args:             search_dir: Directory pencar

### Community 17 - "TemporalCellResult"
Cohesion: 0.25
Nodes (7): Hasil analisis temporal untuk satu cell.      Attributes:         row: Index bar, TemporalCellResult, calculate_activity_score(), calculate_robust_z(), Hitung activity score dari sequence nilai.      Activity score = rata-rata selis, Hitung z-score robust menggunakan MAD.      Args:         value: Nilai yang akan, Analisis temporal semua cell.          Returns:             List TemporalCellRes

### Community 18 - ".detect"
Cohesion: 0.40
Nodes (3): ndarray, Deteksi anomali pada gambar.          Args:             image: Gambar BGR., Simpan gambar annotated/overlay.          Args:             image: Gambar yang s

### Community 19 - ".detect"
Cohesion: 0.50
Nodes (3): ndarray, Buat visualisasi hasil deteksi.          Args:             image: Gambar asli., Deteksi dark spots di area LED.          Args:             image: Gambar BGR.

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DetectionResult` connect `DetectionResult` to `LEDPanelInfo`, `EnsemblePipeline`, `__init__.py`, `Path`, `utils.py`, `.detect`, `Config`, `AnomalyLevel`, `.detect`, `.detect`, `.detect`, `.detect`?**
  _High betweenness centrality (0.187) - this node is a cross-community bridge._
- **Why does `LocationConfig` connect `Config` to `DetectionResult`, `LEDPanelInfo`, `EnsemblePipeline`, `__init__.py`, `utils.py`, `.detect`, `AnomalyLevel`, `.detect`?**
  _High betweenness centrality (0.133) - this node is a cross-community bridge._
- **Why does `LEDPanelInfo` connect `__init__.py` to `LEDPanelInfo`, `analyzer.py`, `line_defects.py`, `detect_pixel_chaos`, `Path`, `annotation.py`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `DetectionResult` (e.g. with `BaseDetector` and `DarkSpotDetector`) actually correct?**
  _`DetectionResult` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `EnsemblePipeline` (e.g. with `Config` and `AnomalyLevel`) actually correct?**
  _`EnsemblePipeline` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `Config` (e.g. with `LocationConfig` and `BaseDetector`) actually correct?**
  _`Config` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Should `LEDPanelInfo` be split into smaller, more focused modules?**
  _Cohesion score 0.11333333333333333 - nodes in this community are weakly interconnected._