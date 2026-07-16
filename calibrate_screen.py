"""
Calibration tool untuk screen points LED.

Cara pakai:
    python calibrate_screen.py --location lengkong --resolution 1920x1080

Langkah:
    1. Klik 4 corner LED: TL → TR → BR → BL
    2. Tekan 'y' untuk terima, 'r' untuk reset, 'q' untuk skip
    3. Tekan ESC untuk selesai semua gambar
    4. Preview muncul → 'y' untuk simpan
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


class ClickState:
    def __init__(self):
        self.points = []
        self.scale = 1.0


def mouse_callback(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN and len(state.points) < 4:
        orig_x = int(x / state.scale)
        orig_y = int(y / state.scale)
        state.points.append([orig_x, orig_y])
        print(f"  [{len(state.points)}/4] ({orig_x}, {orig_y})")


def calibrate_image(image_path: str) -> list:
    """Klik 4 corner. Return points atau None (ESC) atau [] (skip)."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Error: {image_path}")
        return []

    h_orig, w_orig = img.shape[:2]
    win = "Calibrate LED Screen"
    scale = min(1600 / w_orig, 900 / h_orig, 1.0)
    display = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
    disp_h, disp_w = display.shape[:2]

    state = ClickState()
    state.scale = scale

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, mouse_callback, state)

    labels = ["TL", "TR", "BR", "BL"]

    while True:
        show = display.copy()

        # Titik
        for i, p in enumerate(state.points):
            sx, sy = int(p[0] * scale), int(p[1] * scale)
            cv2.circle(show, (sx, sy), 8, (0, 0, 255), -1)
            cv2.circle(show, (sx, sy), 10, (255, 255, 255), 2)
            cv2.putText(show, labels[i], (sx + 15, sy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Garis
        for i in range(len(state.points) - 1):
            p1 = (int(state.points[i][0] * scale), int(state.points[i][1] * scale))
            p2 = (int(state.points[i+1][0] * scale), int(state.points[i+1][1] * scale))
            cv2.line(show, p1, p2, (0, 255, 0), 2)

        # Tutup polygon
        if len(state.points) == 4:
            pts_s = (np.array(state.points) * scale).astype(np.int32)
            cv2.polylines(show, [pts_s], True, (0, 255, 0), 2)

        # Info bar
        cv2.rectangle(show, (0, 0), (disp_w, 50), (0, 0, 0), -1)
        if len(state.points) < 4:
            txt = f"Klik {labels[len(state.points)]} ({len(state.points)+1}/4)"
        else:
            txt = "y=TERIMA  r=RESET  q=SKIP"
        cv2.putText(show, txt, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(win, show)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(win)
            return None
        elif key == ord('q'):
            cv2.destroyWindow(win)
            return []
        elif key == ord('r'):
            state.points.clear()
            print("  Reset.")
        elif key == ord('y') and len(state.points) == 4:
            result = [[int(p[0]), int(p[1])] for p in state.points]
            cv2.destroyWindow(win)
            return result

    cv2.destroyAllWindows()
    return []


def preview(image_path: str, points: list) -> bool:
    img = cv2.imread(image_path)
    if img is None:
        return True

    h, w = img.shape[:2]
    scale = min(1600 / w, 900 / h, 1.0)

    annotated = img.copy()
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)
    for i, p in enumerate(pts):
        cv2.circle(annotated, tuple(p), 6, (0, 0, 255), -1)
        cv2.putText(annotated, f"{i+1}: {p}", (p[0] + 10, p[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if scale != 1.0:
        annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    win = "Preview - y=ACCEPT, n=REJECT"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, annotated)

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('y'):
            cv2.destroyAllWindows()
            return True
        elif key == ord('n') or key == 27:
            cv2.destroyAllWindows()
            return False


def main():
    parser = argparse.ArgumentParser(description="Calibrate LED screen points")
    parser.add_argument("--location", required=True)
    parser.add_argument("--resolution", required=True, help="e.g. 1920x1080")
    parser.add_argument("--images", nargs="*")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--max-images", type=int, default=5)
    args = parser.parse_args()

    print("=" * 50)
    print("  LED Screen Calibration Tool")
    print("=" * 50)
    print(f"  Location:  {args.location}")
    print(f"  Resolution: {args.resolution}")
    print()

    # Cari gambar
    if args.images:
        images = args.images
    else:
        base = Path(args.dataset_dir) / args.location
        tw, th = map(int, args.resolution.split("x"))
        images = []
        for p in sorted(base.glob("**/*.jpg")):
            img = cv2.imread(str(p))
            if img is not None and img.shape[1] == tw and img.shape[0] == th:
                images.append(str(p))
            if len(images) >= args.max_images:
                break

    if not images:
        print(f"  Tidak ada gambar {args.resolution} di dataset/{args.location}/")
        sys.exit(1)

    print(f"  Ditemukan {len(images)} gambar:")
    for p in images:
        print(f"    - {Path(p).name}")
    print()
    print("  Instruksi:")
    print("    1. Klik 4 corner LED: TL -> TR -> BR -> BL")
    print("    2. y = terima, r = reset, q = skip, ESC = selesai")
    print()

    all_points = []
    for i, path in enumerate(images):
        print(f"[{i+1}/{len(images)}] {Path(path).name}")
        pts = calibrate_image(path)

        if pts is None:  # ESC
            print("  Selesai.")
            break
        if pts:
            print(f"  Diterima: {pts}")
            all_points.append(pts)
            # Jika cuma 1 gambar, langsung selesai
            if len(images) == 1:
                break

    if not all_points:
        print("\n  Tidak ada points yang di-calibrate.")
        sys.exit(0)

    # Average
    avg = np.mean(all_points, axis=0)
    result = [[int(round(x)), int(round(y))] for x, y in avg]

    print()
    print("=" * 50)
    print("  HASIL CALIBRATION")
    print("=" * 50)
    for i, label in enumerate(["TL", "TR", "BR", "BL"]):
        print(f"  {label}: {result[i]}")
    print()

    # Preview
    print("  Menampilkan preview...")
    if preview(images[0], result):
        print()
        print("  DITERIMA! Tambahkan ke src/core/config.py:\n")
        print(f'    "{args.resolution}": {result},')
        print()
    else:
        print("  Ditolak. Coba lagi.")


if __name__ == "__main__":
    main()
