"""
LED anomaly detectors.

Individual detection algorithms for different types of LED anomalies.
"""

from src.detectors.led.detectors.blocking import detect_blocking
from src.detectors.led.detectors.flat_content import (
    detect_flat_content,
    merge_flat_anomalies,
)
from src.detectors.led.detectors.line_defects import detect_line_defects
from src.detectors.led.detectors.dead_blocks import (
    detect_dead_blocks,
    detect_dead_blocks_in_mask,
)
from src.detectors.led.detectors.color_errors import (
    detect_color_errors,
    detect_color_errors_in_mask,
)
from src.detectors.led.detectors.dark_regions import (
    detect_dark_regions_by_local_contrast,
)
from src.detectors.led.detectors.pixel_chaos import (
    detect_pixel_chaos,
    merge_chaos_anomalies,
)
from src.detectors.led.detectors.horizontal_line import (
    detect_horizontal_line_pattern,
)
from src.detectors.led.detectors.uniformity import (
    detect_uniform_content,
)
from src.detectors.led.detectors.region_contrast import (
    detect_region_contrast_anomalies,
)
from src.detectors.led.detectors.module_error import (
    detect_module_errors,
)
