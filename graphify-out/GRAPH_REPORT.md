# Graph Report - led  (2026-07-17)

## Corpus Check
- 48 files · ~28,545 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 601 nodes · 1256 edges · 41 communities (34 shown, 7 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 69 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `31b538c1`
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
- LEDAnomaly
- detect_flat_content
- TemporalAnalyzer
- ModuleFeatures
- AnomalyDetector
- ._annotate_results_on_original
- TemporalCorrelationAnalyzer
- SimpleDefectDetector
- content_mask.py
- load_image
- module_grid.py
- FeatureExtractor
- SpatialAnalyzer
- ModuleGrid
- DetectionResult
- .detect
- .detect
- .__init__
- .load_from_folder
- ._rect_to_original
- .temporal_correlation
- .temporal_detector
- __init__.py

## God Nodes (most connected - your core abstractions)
1. `LEDPanelInfo` - 59 edges
2. `LEDAnomaly` - 56 edges
3. `DetectionResult` - 45 edges
4. `EnsemblePipeline` - 40 edges
5. `AnomalyLevel` - 29 edges
6. `Config` - 27 edges
7. `LocationConfig` - 26 edges
8. `LEDAnalyzer` - 25 edges
9. `BaseDetector` - 23 edges
10. `GridDetector` - 21 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `LEDAnalyzer`  [INFERRED]
  detect.py → src/detectors/led/analyzer.py
- `train_location()` --calls--> `AnomalyDetector`  [EXTRACTED]
  train_anomaly_detector.py → src/detectors/led/anomaly_detector.py
- `load_normal_images()` --calls--> `EnsemblePipeline`  [EXTRACTED]
  train_anomaly_detector.py → src/pipeline/ensemble.py
- `detect_single()` --references--> `EnsemblePipeline`  [EXTRACTED]
  detect.py → src/pipeline/ensemble.py
- `detect_frames()` --calls--> `load_frames()`  [EXTRACTED]
  detect.py → src/core/utils.py

## Import Cycles
- None detected.

## Communities (41 total, 7 thin omitted)

### Community 0 - "DetectionResult"
Cohesion: 0.06
Nodes (56): ABC, Config, Konfigurasi untuk LED Anomaly Detection System., Konfigurasi global sistem.      Attributes:         BASE_DIR: Directory root pro, Buat semua direktori yang diperlukan., Ambil konfigurasi untuk lokasi tertentu.          Args:             location: Na, Core modules untuk LED Anomaly Detection., AnomalyLevel (+48 more)

### Community 1 - "LEDPanelInfo"
Cohesion: 0.07
Nodes (23): LEDAnalyzer, ndarray, LED Panel Analyzer (Legacy Compatibility).  This module is kept for backward com, Find LED area in image without cropping.          Detects LED panel location and, Convert screen points to LEDPanelInfo.          Args:             screen_points:, Create result when no LED panel is found.          Args:             image_path:, Detect all anomalies in LED area.          Always runs sub-detectors regardless, Create anomaly for completely off LED.          Args:             panel: LED pan (+15 more)

### Community 2 - "EnsemblePipeline"
Cohesion: 0.12
Nodes (15): detect_frames(), detect_single(), main(), Deteksi anomali pada satu gambar.      Args:         pipeline: Ensemble pipeline, Deteksi anomali dari multiple frames.      Args:         pipeline: Ensemble pipe, load_frames(), Load semua frame dari folder.      Args:         folder: Path ke folder yang ber, EnsemblePipeline (+7 more)

### Community 3 - "analyzer.py"
Cohesion: 0.20
Nodes (14): _calculate_cell_chaos(), _compute_chaos_scores(), _create_module_error_anomaly(), detect_module_errors(), _find_chaotic_cells(), _merge_chaotic_cells(), ndarray, Module error detection module.  Detects corrupted or glitchy sections in LED con (+6 more)

### Community 4 - "__init__.py"
Cohesion: 0.25
Nodes (10): _compute_row_variances(), detect_horizontal_line_pattern(), _is_local_maximum_variance(), ndarray, Horizontal line pattern detection module.  Detects abnormal horizontal line patt, Compute variance for each row.      Args:         gray: Grayscale image., Smooth row variances using moving average.      Reduces noise from single-row co, Check if row has locally maximum variance.      Args:         y: Row index. (+2 more)

### Community 5 - "Path"
Cohesion: 0.07
Nodes (21): crop_image(), main(), ndarray, Apply perspective crop to image.      Args:         image: Input BGR image., Preprocess dataset for a location.      Args:         location: Location name., Path, Ambil path dataset untuk lokasi.          Args:             location: Nama lokas, Ambil path model untuk lokasi.          Args:             location: Nama lokasi. (+13 more)

### Community 6 - "utils.py"
Cohesion: 0.18
Nodes (13): LED Analyzer module.  Orchestrates all detection algorithms for LED panel analys, detect_uniform_content(), ndarray, Uniformity detection module.  Detects LED panels with abnormally uniform content, Detect abnormally uniform LED content.      A normal LED ad has natural brightne, calculate_score(), filter_anomalies_in_panel(), Scoring module.  Functions for calculating anomaly scores and filtering anomalie (+5 more)

### Community 7 - "line_defects.py"
Cohesion: 0.17
Nodes (18): _detect_horizontal_lines(), detect_line_defects(), _detect_vertical_lines(), _is_local_minimum_x(), _is_local_minimum_y(), ndarray, Line defect detection module.  Detects horizontal and vertical line defects in L, Detect vertical line defects.      Args:         gray: Grayscale LED region. (+10 more)

### Community 8 - "detect_pixel_chaos"
Cohesion: 0.15
Nodes (18): _compute_chaos_stats(), _compute_chaos_threshold(), _compute_panel_brightness(), _confirm_chaos_clusters(), _create_chaos_anomalies(), detect_pixel_chaos(), _find_suspicious_blocks(), merge_chaos_anomalies() (+10 more)

### Community 9 - ".detect"
Cohesion: 0.12
Nodes (15): CellResult, Hasil analisis untuk satu cell/grid.      Attributes:         row: Index baris., GridDetector, ndarray, Deteksi anomali menggunakan neighbor contrast.          Args:             image:, Analisis grid untuk deteksi anomali.          Args:             grid: Grid stati, Cek apakah cell gelap dari tetangga.          Menggunakan adaptive threshold yan, Cek apakah cell terlalu rata/flatt.          Args:             stat: Statistik c (+7 more)

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
Cohesion: 0.22
Nodes (14): annotate_image(), _draw_anomalies(), _draw_label(), _draw_panel_border(), _draw_summary(), _get_severity_color(), ndarray, Annotation module.  Functions for annotating images with detected anomalies. (+6 more)

### Community 15 - ".detect"
Cohesion: 0.27
Nodes (10): detect_dark_regions_by_local_contrast(), _find_dark_cells(), _merge_dark_cells(), ndarray, Dark region detection module.  Detects dark regions using local contrast analysi, Merge adjacent dark cells into regions.      Args:         dark_cells: List of d, Detect dark regions using local contrast analysis.      Divides panel into 8x8 g, Find cells that are darker than threshold.      Args:         gray: Grayscale im (+2 more)

### Community 16 - ".detect"
Cohesion: 0.33
Nodes (7): calibrate_image(), ClickState, main(), mouse_callback(), preview(), Calibration tool untuk screen points LED.  Cara pakai:     python calibrate_scre, Klik 4 corner. Return points atau None (ESC) atau [] (skip).

### Community 17 - "TemporalCellResult"
Cohesion: 0.31
Nodes (8): _calculate_neighborhood_contrast(), _classify_blocking(), detect_blocking(), ndarray, Blocking detection module.  Detects large dark areas that block LED content., Classify type of blocking anomaly.      Args:         cw: Contour width., Calculate contrast between dark region and its neighborhood.      Args:, Detect blocking anomalies in LED panel.      Args:         gray: Grayscale LED r

### Community 18 - "LEDAnomaly"
Cohesion: 0.31
Nodes (7): detect_dead_blocks(), detect_dead_blocks_in_mask(), ndarray, Dead block detection module.  Detects dead pixel blocks in LED content., Detect dead pixel blocks without mask.      Args:         gray: Grayscale LED re, Detect dead pixel blocks with mask.      Args:         gray: Grayscale LED regio, LED anomaly detectors.  Individual detection algorithms for different types of L

### Community 19 - "detect_flat_content"
Cohesion: 0.22
Nodes (8): Analyze LED content for anomalies on full image.          Runs all detectors on, detect_flat_content(), merge_flat_anomalies(), ndarray, Flat content detection module.  Detects blank or abnormally flat regions in LED, Merge adjacent flat anomalies into larger regions.      Groups anomalies that ar, Detect flat or blank content regions.      Original detection logic: block 64px,, Merge adjacent flat anomalies into larger regions.      Args:         anomalies:

### Community 20 - "TemporalAnalyzer"
Cohesion: 0.13
Nodes (12): ndarray, Temporal analyzer for LED anomaly detection.  Compares multiple frames over time, Detect if content is frozen/stuck.          Args:             frames: List of fr, Detect persistent defects that appear across multiple frames.          Args:, Check for persistent dark areas across frames.          Args:             gray_f, Check for persistent bright areas across frames.          Args:             gray, Temporal analyzer for LED anomaly detection.      Compares multiple frames over, Get temporal analysis statistics for a location.          Args:             loca (+4 more)

### Community 21 - "ModuleFeatures"
Cohesion: 0.19
Nodes (14): DefectScorer, ModuleScore, Defect Scorer for LED anomaly detection.  Combines signals from multiple detecto, Compute global panel score from module scores.          Args:             module, Score for a single module., Combine detector signals into unified scores.      Per-module scoring:     - Spa, Score each module based on multiple signals.          Args:             spatial_, ModuleFeatures (+6 more)

### Community 22 - "AnomalyDetector"
Cohesion: 0.15
Nodes (10): AnomalyDetector, ndarray, Predict if image is normal or anomalous.          Args:             image: Cropp, ML-based anomaly detector using Isolation Forest.      Learns "normal" pattern p, Extract feature vector from image.          Args:             image: Cropped LED, Initialize anomaly detector.          Args:             model_dir: Directory to, Check if anomaly detector is available., Load trained model for location.          Args:             location: Location n (+2 more)

### Community 23 - "._annotate_results_on_original"
Cohesion: 0.18
Nodes (9): ndarray, Crop gambar ke area screen LED menggunakan perspective transform.          Menyi, Gambar border hijau di sekeliling area screen LED.          Args:             im, Analisis satu gambar dengan semua detector.          Deteksi dilakukan di area c, Re-annotate LED Analyzer results on full original image.          Args:, Re-annotate Grid results on full original image.          Args:             orig, Re-annotate DarkSpot results on full original image., Re-annotate semua detector results ke gambar original (full).          Args: (+1 more)

### Community 24 - "TemporalCorrelationAnalyzer"
Cohesion: 0.14
Nodes (10): calibrate_grid(), Calibrate module grid by dividing panel evenly.      Default grid is 12x16 (matc, ndarray, Temporal Correlation Analyzer for LED defect detection.  Compares per-module sig, Get per-module correlation values for external use.          Args:             l, Get temporal analysis statistics.          Args:             location: Location, Detect defects via temporal correlation analysis.      Buffers N frames per loca, Initialize temporal correlation analyzer.          Args:             buffer_size (+2 more)

### Community 25 - "SimpleDefectDetector"
Cohesion: 0.18
Nodes (9): ndarray, Detect local anomalies (module errors).          Focus on specific defect patter, Get variance of neighboring blocks.          Args:             gray: Grayscale i, Simple defect detector using normal baseline., Get mean brightness of neighboring blocks.          Args:             gray: Gray, Initialize detector.          Args:             model_dir: Directory to load bas, Load normal baseline for location.          Args:             location: Location, Detect defects in image.          Args:             image: BGR image. (+1 more)

### Community 26 - "content_mask.py"
Cohesion: 0.23
Nodes (12): _apply_morphological_cleanup(), create_full_image_mask(), create_led_content_mask(), _extract_largest_component(), ndarray, Content mask module.  Functions for creating LED content masks., Create mask for LED content in full image.      Creates a binary mask covering t, Refine panel mask to detect actual LED content.      Applies brightness and satu (+4 more)

### Community 27 - "load_image"
Cohesion: 0.24
Nodes (10): create_heatmap_overlay(), load_image(), ndarray, Load gambar dari path.      Args:         path: Path ke file gambar.      Return, Buat heatmap overlay di atas gambar asli.      Args:         original: Gambar as, load_normal_images(), main(), Load normal images from dataset folder.      Args:         location: Location na (+2 more)

### Community 28 - "module_grid.py"
Cohesion: 0.22
Nodes (8): auto_refine_grid(), _find_projection_minima(), ndarray, Module Grid calibration for LED panels.  Maps the physical LED module grid (cabi, Refine grid using projection profile to detect bezel lines.      LED cabinets ha, Find local minima in projection profile.      Minima correspond to bezel lines (, Get pixel bounds for a module.          Args:             row: Module row index., Get binary mask for a module.          Args:             row: Module row index.

### Community 29 - "FeatureExtractor"
Cohesion: 0.22
Nodes (6): FeatureExtractor, Extract features for all modules.          Returns:             Dict mapping (ro, Extract features for a single module.          Args:             row: Module row, Get global panel brightness (for temporal correlation).          Returns:, Get global panel statistics.          Returns:             Dict with global_medi, Extract per-module features with global normalization.      Usage:         extra

### Community 30 - "SpatialAnalyzer"
Cohesion: 0.24
Nodes (6): Analyze a single module against its neighbors.          Args:             row: M, Merge adjacent anomalous modules into regions.          Args:             anomal, Detect defects via spatial decorrelation analysis.      For each module, compute, Initialize spatial analyzer.          Args:             z_threshold: Z-score thr, Analyze spatial decorrelation across all modules.          Args:             fea, SpatialAnalyzer

### Community 31 - "ModuleGrid"
Cohesion: 0.22
Nodes (6): ndarray, Initialize feature extractor.          Args:             gray: Grayscale LED pan, ModuleGrid, Get all module coordinates.          Returns:             List of (row, col) tup, Grid mapping for LED module structure.      Attributes:         rows: Number of, Get neighboring module coordinates (8-connected).          Args:             row

### Community 32 - "DetectionResult"
Cohesion: 0.25
Nodes (5): DetectionResult, Hasil deteksi anomali untuk satu gambar.      Attributes:         location: Nama, ML-based anomaly detector using Isolation Forest.  Learns "normal" pattern per l, Simple defect detector using normal baseline.  Detects: 1. Blocking (area gelap, Analyze temporal patterns for a location.          Args:             location: L

### Community 33 - ".detect"
Cohesion: 0.33
Nodes (4): ndarray, Buat visualisasi hasil deteksi.          Args:             image: Gambar asli., Buat DetectionResult.          Args:             image_path: Path gambar., Deteksi dark spots di area LED.          Args:             image: Gambar BGR.

### Community 34 - ".detect"
Cohesion: 0.40
Nodes (3): ndarray, Deteksi anomali pada gambar.          Args:             image: Gambar BGR., Simpan gambar annotated/overlay.          Args:             image: Gambar yang s

## Knowledge Gaps
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DetectionResult` connect `DetectionResult` to `DetectionResult`, `.detect`, `.detect`, `LEDPanelInfo`, `EnsemblePipeline`, `Path`, `utils.py`, `.detect`, `TemporalAnalyzer`, `AnomalyDetector`, `._annotate_results_on_original`, `TemporalCorrelationAnalyzer`, `SimpleDefectDetector`?**
  _High betweenness centrality (0.221) - this node is a cross-community bridge._
- **Why does `AnomalyLevel` connect `DetectionResult` to `DetectionResult`, `.detect`, `LEDPanelInfo`, `EnsemblePipeline`, `Path`, `utils.py`, `.detect`, `TemporalAnalyzer`, `AnomalyDetector`, `TemporalCorrelationAnalyzer`, `SimpleDefectDetector`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `LEDPanelInfo` connect `utils.py` to `LEDPanelInfo`, `analyzer.py`, `__init__.py`, `line_defects.py`, `detect_pixel_chaos`, `Path`, `Config`, `AnomalyLevel`, `annotation.py`, `.detect`, `TemporalCellResult`, `LEDAnomaly`, `detect_flat_content`, `ModuleFeatures`, `content_mask.py`, `SpatialAnalyzer`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `DetectionResult` (e.g. with `BaseDetector` and `DarkSpotDetector`) actually correct?**
  _`DetectionResult` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `EnsemblePipeline` (e.g. with `Config` and `AnomalyLevel`) actually correct?**
  _`EnsemblePipeline` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `AnomalyLevel` (e.g. with `DarkSpotDetector` and `GridDetector`) actually correct?**
  _`AnomalyLevel` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Should `DetectionResult` be split into smaller, more focused modules?**
  _Cohesion score 0.05510388437217705 - nodes in this community are weakly interconnected._