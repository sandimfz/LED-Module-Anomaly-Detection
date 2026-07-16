# Graph Report - led  (2026-07-16)

## Corpus Check
- 38 files · ~20,284 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 441 nodes · 946 edges · 18 communities (17 shown, 1 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `59a223c2`
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

## God Nodes (most connected - your core abstractions)
1. `LEDPanelInfo` - 54 edges
2. `LEDAnomaly` - 51 edges
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

## Communities (18 total, 1 thin omitted)

### Community 0 - "DetectionResult"
Cohesion: 0.05
Nodes (58): ABC, Enum, Config, Konfigurasi untuk LED Anomaly Detection System., Konfigurasi global sistem.      Attributes:         BASE_DIR: Directory root pro, Buat semua direktori yang diperlukan., Core modules untuk LED Anomaly Detection., AnomalyLevel (+50 more)

### Community 1 - "LEDPanelInfo"
Cohesion: 0.12
Nodes (14): LEDAnalyzer, ndarray, LED Panel Analyzer (Legacy Compatibility).  This module is kept for backward com, Find LED area in image without cropping.          Detects LED panel location and, Convert screen points to LEDPanelInfo.          Args:             screen_points:, Create result when no LED panel is found.          Args:             image_path:, Detect all anomalies in LED area.          Always runs sub-detectors regardless, Create anomaly for completely off LED.          Args:             panel: LED pan (+6 more)

### Community 2 - "EnsemblePipeline"
Cohesion: 0.06
Nodes (34): detect_frames(), detect_single(), main(), Deteksi anomali pada satu gambar.      Args:         pipeline: Ensemble pipeline, Deteksi anomali dari multiple frames.      Args:         pipeline: Ensemble pipe, create_heatmap_overlay(), extract_grid_stats(), load_frames() (+26 more)

### Community 3 - "analyzer.py"
Cohesion: 0.22
Nodes (13): LED Analyzer module.  Orchestrates all detection algorithms for LED panel analys, _apply_morphological_cleanup(), create_full_image_mask(), create_led_content_mask(), _extract_largest_component(), ndarray, Content mask module.  Functions for creating LED content masks., Create mask for LED content in full image.      Creates a binary mask covering t (+5 more)

### Community 4 - "__init__.py"
Cohesion: 0.11
Nodes (23): _calculate_neighborhood_contrast(), _classify_blocking(), detect_blocking(), ndarray, Blocking detection module.  Detects large dark areas that block LED content., Classify type of blocking anomaly.      Args:         cw: Contour width., Calculate contrast between dark region and its neighborhood.      Args:, Detect blocking anomalies in LED panel.      Args:         gray: Grayscale LED r (+15 more)

### Community 5 - "Path"
Cohesion: 0.09
Nodes (16): Path, Ambil path model untuk lokasi.          Args:             location: Nama lokasi., Ambil/buat session folder berdasarkan tanggal & waktu.          Satu runtime = s, Ambil path output untuk lokasi.          Struktur folder:             output/<DD, Ambil path untuk reports.          Report ada di folder jam, sama dengan gambar., Ambil path dataset untuk lokasi.          Args:             location: Nama lokas, Generate output path berdasarkan status.          Args:             image_path:, ndarray (+8 more)

### Community 6 - "utils.py"
Cohesion: 0.15
Nodes (15): detect_dead_blocks(), detect_dead_blocks_in_mask(), ndarray, Dead block detection module.  Detects dead pixel blocks in LED content., Detect dead pixel blocks without mask.      Args:         gray: Grayscale LED re, Detect dead pixel blocks with mask.      Args:         gray: Grayscale LED regio, LED Analyzer package.  Provides LED panel analysis and anomaly detection., calculate_score() (+7 more)

### Community 7 - "line_defects.py"
Cohesion: 0.17
Nodes (18): _detect_horizontal_lines(), detect_line_defects(), _detect_vertical_lines(), _is_local_minimum_x(), _is_local_minimum_y(), ndarray, Line defect detection module.  Detects horizontal and vertical line defects in L, Detect vertical line defects.      Args:         gray: Grayscale LED region. (+10 more)

### Community 8 - "detect_pixel_chaos"
Cohesion: 0.15
Nodes (18): _compute_chaos_stats(), _compute_chaos_threshold(), _compute_panel_brightness(), _confirm_chaos_clusters(), _create_chaos_anomalies(), detect_pixel_chaos(), _find_suspicious_blocks(), merge_chaos_anomalies() (+10 more)

### Community 9 - ".detect"
Cohesion: 0.10
Nodes (19): CellResult, Hasil analisis untuk satu cell/grid.      Attributes:         row: Index baris., get_neighbor_stats(), hue_delta(), Ambil rata-rata statistik dari 8 tetangga.      Args:         grid: Grid statist, Selisih hue yang benar secara sirkular.      Di OpenCV HSV, hue range 0-180. Nil, GridDetector, ndarray (+11 more)

### Community 11 - "Path"
Cohesion: 0.16
Nodes (20): _cleanup_mask(), _compute_candidate_score(), _compute_rectangularity(), _compute_region_stats(), _create_initial_mask(), _evaluate_candidates(), find_led_panel(), perspective_crop() (+12 more)

### Community 12 - "Config"
Cohesion: 0.16
Nodes (16): _compute_block_stats(), _compute_block_stats_full(), detect_color_errors(), detect_color_errors_in_mask(), _get_neighbor_sats(), _get_neighbor_stats(), _merge_color_anomalies(), ndarray (+8 more)

### Community 13 - "AnomalyLevel"
Cohesion: 0.19
Nodes (14): _compute_cell_stats(), _create_region_anomaly(), detect_region_contrast_anomalies(), _find_anomalous_cells(), _get_neighbors(), _merge_cells_to_regions(), ndarray, Region-based contrast detection module.  Detects anomalies by comparing each reg (+6 more)

### Community 14 - "annotation.py"
Cohesion: 0.08
Nodes (38): annotate_image(), _draw_anomalies(), _draw_label(), _draw_panel_border(), _draw_summary(), _get_severity_color(), ndarray, Annotation module.  Functions for annotating images with detected anomalies. (+30 more)

### Community 15 - ".detect"
Cohesion: 0.22
Nodes (7): crop_image(), main(), ndarray, Apply perspective crop to image.      Args:         image: Input BGR image., Preprocess dataset for a location.      Args:         location: Location name., Ambil konfigurasi untuk lokasi tertentu.          Args:             location: Na, Initialize ensemble pipeline.          Args:             location: Nama lokasi.

### Community 16 - ".detect"
Cohesion: 0.33
Nodes (7): calibrate_image(), ClickState, main(), mouse_callback(), preview(), Calibration tool untuk screen points LED.  Cara pakai:     python calibrate_scre, Klik 4 corner. Return points atau None (ESC) atau [] (skip).

### Community 17 - "TemporalCellResult"
Cohesion: 0.40
Nodes (4): detect_uniform_content(), ndarray, Uniformity detection module.  Detects LED panels with abnormally uniform content, Detect abnormally uniform LED content.      A normal LED ad has natural brightne

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DetectionResult` connect `DetectionResult` to `LEDPanelInfo`, `EnsemblePipeline`, `analyzer.py`, `Path`, `.detect`?**
  _High betweenness centrality (0.172) - this node is a cross-community bridge._
- **Why does `LEDPanelInfo` connect `utils.py` to `LEDPanelInfo`, `analyzer.py`, `__init__.py`, `line_defects.py`, `detect_pixel_chaos`, `Path`, `Config`, `AnomalyLevel`, `annotation.py`, `TemporalCellResult`?**
  _High betweenness centrality (0.133) - this node is a cross-community bridge._
- **Why does `LocationConfig` connect `DetectionResult` to `LEDPanelInfo`, `EnsemblePipeline`, `analyzer.py`, `.detect`, `.detect`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `DetectionResult` (e.g. with `BaseDetector` and `DarkSpotDetector`) actually correct?**
  _`DetectionResult` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `EnsemblePipeline` (e.g. with `Config` and `AnomalyLevel`) actually correct?**
  _`EnsemblePipeline` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `Config` (e.g. with `LocationConfig` and `BaseDetector`) actually correct?**
  _`Config` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Should `DetectionResult` be split into smaller, more focused modules?**
  _Cohesion score 0.05172413793103448 - nodes in this community are weakly interconnected._