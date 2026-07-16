"""
Type definitions for LED Analyzer.

Provides dataclasses for LED panel information and detected anomalies.
"""

from dataclasses import dataclass


@dataclass
class LEDPanelInfo:
    """Information about detected LED panel.

    Attributes:
        x: X coordinate of panel top-left corner.
        y: Y coordinate of panel top-left corner.
        width: Width of panel in pixels.
        height: Height of panel in pixels.
        confidence: Detection confidence score (0-1).
        mean_brightness: Mean brightness of panel region.
        mean_saturation: Mean saturation of panel region.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float
    mean_brightness: float
    mean_saturation: float


@dataclass
class LEDAnomaly:
    """Detected anomaly in LED panel.

    Attributes:
        x: X coordinate of anomaly.
        y: Y coordinate of anomaly.
        width: Width of anomaly region.
        height: Height of anomaly region.
        anomaly_type: Type of anomaly detected.
        severity: Severity score (0-1).
        description: Human-readable description.
    """

    x: int
    y: int
    width: int
    height: int
    anomaly_type: str
    severity: float
    description: str
