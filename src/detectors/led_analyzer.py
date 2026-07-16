"""
LED Panel Analyzer (Legacy Compatibility).

This module is kept for backward compatibility.
The actual implementation has been moved to src/detectors/led/

Import LEDAnalyzer from src.detectors.led instead.
"""

from src.detectors.led import LEDAnalyzer, LEDAnomaly, LEDPanelInfo

__all__ = ["LEDAnalyzer", "LEDAnomaly", "LEDPanelInfo"]
