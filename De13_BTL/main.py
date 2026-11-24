import sys
import os
import csv
import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QAction, QImage, QPixmap, QPainter, QColor, QPen, QBrush,
    QPainterPath, QIcon, QFont, QCursor
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QComboBox, QMessageBox,
    QStatusBar, QSizePolicy, QButtonGroup, QStackedWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QDialog,
    QScrollArea, QAbstractItemView, QGroupBox, QCheckBox, QPlainTextEdit,
    QSlider, QSpinBox, QDoubleSpinBox, QProgressBar, QStyledItemDelegate, QStyle, QMenu
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
from matplotlib.figure import Figure
try:
    from PIL import Image as PILImage, ImageOps as PILImageOps
except Exception:
    PILImage = None
    PILImageOps = None
try:
    _pillow_heif = importlib.import_module("pillow_heif")
    _pillow_heif.register_heif_opener()
except Exception:
    pass

# Cấu hình đường dẫn & định dạng ảnh hỗ trợ toàn hệ thống
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"
DATASET_100_DIR = BASE_DIR / "dataset_100"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
ARROW_PATH = (ASSETS_DIR / "arrow_down.png").as_posix()

SUPPORTED_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".jpe", ".jfif",
    ".bmp", ".dib",
    ".tif", ".tiff",
    ".webp", ".avif", ".heic", ".heif",
    ".jp2", ".j2k", ".jpf", ".jpx",
    ".gif", ".ico", ".icns",
    ".tga", ".pcx", ".ppm", ".pgm", ".pbm", ".pnm", ".pxm",
    ".sgi", ".rgb", ".rgba", ".dds", ".qoi", ".psd", ".hdr", ".pic", ".sr", ".ras", ".xbm", ".xpm",
}

IMAGE_FILE_FILTER = (
    "Tất cả tệp ảnh ("
    "*.png *.jpg *.jpeg *.jpe *.jfif *.webp *.bmp *.dib *.tif *.tiff "
    "*.gif *.ico *.jp2 *.j2k *.avif *.heic *.heif *.tga *.ppm *.pgm *.pbm *.pnm *.pcx *.dds *.qoi *.psd"
    ");;Tất cả tệp (*.*)"
)


def read_image_universal(path, grayscale: bool = False):
    """
    Đọc đa định dạng ảnh (hỗ trợ đường dẫn tiếng Việt Unicode):
    1. Ưu tiên OpenCV (cv2.imdecode) cho tốc độ cao với PNG/JPG/JFIF/WEBP/BMP/TIFF/JP2/PNM...
    2. Fallback tự động sang Pillow (GIF/ICO/TGA/PCX/DDS/PSD/QOI/AVIF/HEIC/CMYK/16-bit...).
    3. Fallback cuối sang Qt QImage nếu còn định dạng đặc biệt khác.
    """
    fpath = str(path)
    # 1. OpenCV nhanh (hỗ trợ tốt đa số định dạng phổ thông)
    try:
        raw = np.fromfile(fpath, dtype=np.uint8)
        if raw.size > 0:
            flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            img = cv2.imdecode(raw, flag)
            if img is not None and img.size > 0:
                return img
    except Exception:
        pass

    # 2. Pillow fallback (xử lý GIF, ICO, TGA, PCX, DDS, QOI, PSD, AVIF, HEIC, CMYK...)
    if PILImage is not None:
        try:
            with PILImage.open(fpath) as pil_im:
                if getattr(pil_im, "is_animated", False):
                    pil_im.seek(0)
                pil_im = PILImageOps.exif_transpose(pil_im)
                if grayscale:
                    arr = np.asarray(pil_im.convert("L"), dtype=np.uint8)
                    return arr.copy()
                # Xử lý ảnh có nền trong suốt RGBA trên nền trắng/RGB sạch
                if pil_im.mode in ("RGBA", "LA") or (pil_im.mode == "P" and "transparency" in pil_im.info):
                    rgba = pil_im.convert("RGBA")
                    bg = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
                    rgb = PILImage.alpha_composite(bg, rgba).convert("RGB")
                else:
                    rgb = pil_im.convert("RGB")
                arr = np.asarray(rgb, dtype=np.uint8)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    # 3. Qt QImage fallback
    try:
        qim = QImage(fpath)
        if not qim.isNull():
            qim = qim.convertToFormat(
                QImage.Format.Format_Grayscale8 if grayscale else QImage.Format.Format_RGB888
            )
            w, h = qim.width(), qim.height()
            bpl = qim.bytesPerLine()
            ptr = qim.constBits()
            if grayscale:
                arr = np.frombuffer(ptr, dtype=np.uint8, count=h * bpl).reshape((h, bpl))[:, :w]
                return arr.copy()
            arr = np.frombuffer(ptr, dtype=np.uint8, count=h * bpl).reshape((h, bpl))[:, :w * 3].reshape((h, w, 3))
            return cv2.cvtColor(arr.copy(), cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    return None


# ============================================================
# CÁC THUẬT TOÁN CỐT LÕI - CÀI ĐẶT BẰNG NUMPY
# (Bám sát yêu cầu Đề tài 13 - Không dùng cv2.calcHist,
#  cv2.equalizeHist, cv2.threshold(..., THRESH_OTSU))
# ============================================================

def calculate_histogram(gray_image: np.ndarray) -> np.ndarray:
    """
    Tính histogram 256 mức xám bằng NumPy (tối ưu với np.bincount):
    h(r_k) = n_k với k = 0..255
    """
    return np.bincount(gray_image.ravel(), minlength=256)[:256].astype(np.int64, copy=False)


def contrast_stretch(gray_image: np.ndarray,
                     r_min: float = None,
                     r_max: float = None,
                     s_min: float = 0.0,
                     s_max: float = 255.0) -> np.ndarray:
    """
    Kéo giãn tương phản tuyến tính (Contrast Stretching) dùng bảng tra 256 mức xám (LUT):
    s = clip((r - r_min) / (r_max - r_min) * (s_max - s_min) + s_min, 0, 255)
    - Nếu không truyền r_min / r_max: tự động lấy [min, max] thực tế của ảnh.
    - Nếu truyền r_min / r_max (và s_min / s_max): kéo giãn chính xác theo ngưỡng người dùng chọn.
    """
    low = float(gray_image.min()) if r_min is None else float(r_min)
    high = float(gray_image.max()) if r_max is None else float(r_max)
    if high <= low:
        return gray_image.copy()
    out_low = float(np.clip(s_min, 0.0, 255.0))
    out_high = float(np.clip(s_max, 0.0, 255.0))
    levels = np.arange(256, dtype=np.float32)
    lut = np.clip((levels - low) * (out_high - out_low) / (high - low) + out_low, 0.0, 255.0).astype(np.uint8)
    return lut[gray_image]


def histogram_equalization(gray_image: np.ndarray, hist: np.ndarray = None) -> np.ndarray:
    """
    Cân bằng histogram (Histogram Equalization) bằng hàm phân bố tích lũy CDF:
    s_k = round((CDF(k) - CDF_min) * 255 / (N - CDF_min))
    """
    if hist is None:
        hist = calculate_histogram(gray_image)
    cdf = hist.cumsum()
    nonzero = np.flatnonzero(hist)
    if len(nonzero) <= 1:
        return gray_image.copy()
    cdf_min = cdf[nonzero[0]]
    denominator = gray_image.size - cdf_min
    if denominator <= 0:
        return gray_image.copy()
    lut = np.round((cdf - cdf_min) * 255.0 / denominator)
    return np.clip(lut, 0, 255).astype(np.uint8)[gray_image]


def gamma_correction(gray_image: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """
    Biến đổi hàm mũ Gamma (Power-Law Transformation) dùng bảng tra 256 mức xám (LUT):
    s = round(255 * (r / 255)^gamma)
    - gamma < 1.0: tăng sáng vùng tối (khắc phục ảnh thiếu sáng, ngược sáng).
    - gamma > 1.0: làm đậm / giảm chói vùng sáng (khắc phục ảnh lóa sáng).
    """
    g = float(max(0.05, gamma))
    levels = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.round((levels ** g) * 255.0), 0.0, 255.0).astype(np.uint8)
    return lut[gray_image]


def clahe_enhance(gray_image: np.ndarray,
                  clip_limit: float = 2.0,
                  tile_grid: int = 8) -> np.ndarray:
    """
    Cân bằng histogram thích nghi có giới hạn tương phản (CLAHE) tự cài đặt bằng NumPy:
    1. Chia ảnh thành lưới tile_grid x tile_grid ô cục bộ.
    2. Tính histogram từng ô, cắt đỉnh vượt ngưỡng clip_limit và phân phối lại phần dư
       đều cho 256 mức xám (chặn khuếch đại nhiễu trên vùng nền phẳng).
    3. Xây dựng bảng ánh xạ CDF cục bộ cho mỗi ô và nội suy song tuyến tính (Bilinear)
       giữa 4 ô lân cận để xóa hoàn toàn vết ghép biên giữa các ô.
    """
    h, w = gray_image.shape[:2]
    grid = int(max(2, min(16, tile_grid)))
    if h < grid * 2 or w < grid * 2:
        grid = max(2, min(h // 2, w // 2, grid))
        if grid < 2:
            return histogram_equalization(gray_image)

    y_edges = np.linspace(0, h, grid + 1, dtype=np.int32)
    x_edges = np.linspace(0, w, grid + 1, dtype=np.int32)
    luts = np.zeros((grid, grid, 256), dtype=np.uint8)

    clip_val = float(max(1.0, clip_limit))
    for iy in range(grid):
        y0, y1 = int(y_edges[iy]), int(y_edges[iy + 1])
        for ix in range(grid):
            x0, x1 = int(x_edges[ix]), int(x_edges[ix + 1])
            tile = gray_image[y0:y1, x0:x1]
            n_pix = tile.size
            if n_pix == 0:
                luts[iy, ix] = np.arange(256, dtype=np.uint8)
                continue
            hist = np.bincount(tile.ravel(), minlength=256)[:256].astype(np.int64)
            limit = max(1, int(round(clip_val * (n_pix / 256.0))))
            over = hist > limit
            if np.any(over):
                excess = int(np.sum(hist[over] - limit))
                hist[over] = limit
                add_each, rem = divmod(excess, 256)
                hist += add_each
                if rem > 0:
                    step = max(1, 256 // rem)
                    idx = np.arange(0, 256, step)[:rem]
                    hist[idx] += 1
            cdf = hist.cumsum()
            cdf_total = float(cdf[-1])
            if cdf_total > 0:
                luts[iy, ix] = np.clip(np.round(cdf * 255.0 / cdf_total), 0, 255).astype(np.uint8)
            else:
                luts[iy, ix] = np.arange(256, dtype=np.uint8)

    # Tọa độ tâm các ô và nội suy song tuyến tính theo từng khối (1D LUT cực nhanh, 100% NumPy)
    cy = 0.5 * (y_edges[:-1] + y_edges[1:]) - 0.5
    cx = 0.5 * (x_edges[:-1] + x_edges[1:]) - 0.5

    yy = np.arange(h, dtype=np.float32)
    xx = np.arange(w, dtype=np.float32)

    iy1 = np.clip(np.searchsorted(cy, yy), 0, grid - 1)
    iy0 = np.clip(iy1 - 1, 0, grid - 1)
    denom_y = np.maximum(cy[iy1] - cy[iy0], 1e-6)
    wy = np.where(iy1 == iy0, 0.0, np.clip((yy - cy[iy0]) / denom_y, 0.0, 1.0)).astype(np.float32)

    ix1 = np.clip(np.searchsorted(cx, xx), 0, grid - 1)
    ix0 = np.clip(ix1 - 1, 0, grid - 1)
    denom_x = np.maximum(cx[ix1] - cx[ix0], 1e-6)
    wx = np.where(ix1 == ix0, 0.0, np.clip((xx - cx[ix0]) / denom_x, 0.0, 1.0)).astype(np.float32)

    # Tìm các đoạn hàng/cột có cùng cặp (iy0, iy1) và (ix0, ix1) -> tra bảng 1D LUT cực nhanh
    y_breaks = np.flatnonzero(np.diff(iy0) | np.diff(iy1)) + 1
    y_spans = np.split(np.arange(h), y_breaks)
    x_breaks = np.flatnonzero(np.diff(ix0) | np.diff(ix1)) + 1
    x_spans = np.split(np.arange(w), x_breaks)

    out = np.empty((h, w), dtype=np.uint8)
    for ys in y_spans:
        r0, r1 = int(ys[0]), int(ys[-1]) + 1
        ty0, ty1 = int(iy0[r0]), int(iy1[r0])
        wy_b = wy[r0:r1, None]
        one_my = 1.0 - wy_b
        for xs in x_spans:
            c0, c1 = int(xs[0]), int(xs[-1]) + 1
            tx0, tx1 = int(ix0[c0]), int(ix1[c0])
            patch = gray_image[r0:r1, c0:c1]
            if ty0 == ty1 and tx0 == tx1:
                out[r0:r1, c0:c1] = luts[ty0, tx0][patch]
            elif ty0 == ty1:
                wx_b = wx[None, c0:c1]
                v0 = luts[ty0, tx0][patch]
                v1 = luts[ty0, tx1][patch]
                out[r0:r1, c0:c1] = np.clip(np.round(v0 * (1.0 - wx_b) + v1 * wx_b), 0, 255).astype(np.uint8)
            elif tx0 == tx1:
                v0 = luts[ty0, tx0][patch]
                v1 = luts[ty1, tx0][patch]
                out[r0:r1, c0:c1] = np.clip(np.round(v0 * one_my + v1 * wy_b), 0, 255).astype(np.uint8)
            else:
                wx_b = wx[None, c0:c1]
                one_mx = 1.0 - wx_b
                top = luts[ty0, tx0][patch] * one_mx + luts[ty0, tx1][patch] * wx_b
                bot = luts[ty1, tx0][patch] * one_mx + luts[ty1, tx1][patch] * wx_b
                out[r0:r1, c0:c1] = np.clip(np.round(top * one_my + bot * wy_b), 0, 255).astype(np.uint8)
    return out


def otsu_threshold_full(gray_image: np.ndarray, hist: np.ndarray = None):
    """
    Phân đoạn Otsu tự cài đặt:
    Tối đa hóa phương sai giữa hai lớp (Between-Class Variance):
    sigma_B^2(t) = omega_0(t) * omega_1(t) * [mu_0(t) - mu_1(t)]^2
    Trả về: (ngưỡng tối ưu t, mặt nạ nhị phân 0-255, mảng phương sai 256 phần tử).
    """
    h_arr = calculate_histogram(gray_image) if hist is None else hist
    hist_f = h_arr.astype(np.float64)
    total = float(gray_image.size)
    levels = np.arange(256, dtype=np.float64)
    probability = hist_f / total
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * levels)
    mu_total = mu[-1]
    denominator = omega * (1.0 - omega)
    numerator = (mu_total * omega - mu) ** 2
    variance = np.zeros(256, dtype=np.float64)
    valid = denominator > 1e-7
    variance[valid] = numerator[valid] / denominator[valid]
    threshold = int(np.argmax(variance))
    mask = (gray_image > threshold).astype(np.uint8) * 255
    return threshold, mask, variance


def to_gray(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh sang mức xám nếu là ảnh màu BGR."""
    if image is None:
        return None
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_lightness_channel(image: np.ndarray) -> np.ndarray:
    """Lấy kênh độ sáng L (nếu là ảnh màu BGR) hoặc chính ảnh xám (nếu là ảnh 1 kênh)."""
    if image is None:
        return None
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]


def enhance_color_image(image: np.ndarray,
                        method: str,
                        r_min: float = None,
                        r_max: float = None,
                        s_min: float = 0.0,
                        s_max: float = 255.0,
                        clip_limit: float = 2.0,
                        tile_grid: int = 8,
                        gamma: float = 0.7,
                        lab_cache: tuple = None) -> np.ndarray:
    """
    Nâng cao chất lượng ảnh:
    - Nếu là ảnh xám: áp dụng trực tiếp thuật toán.
    - Nếu là ảnh màu BGR: chuyển sang không gian màu CIE LAB,
      chỉ áp dụng nâng cao trên kênh độ sáng L (Lightness), sau đó
      ghép lại kênh A và B để bảo toàn màu sắc tự nhiên không bị sai màu.
    """
    def _apply_1ch(ch: np.ndarray) -> np.ndarray:
        if "Kéo giãn" in method:
            return contrast_stretch(ch, r_min=r_min, r_max=r_max, s_min=s_min, s_max=s_max)
        if "CLAHE" in method:
            return clahe_enhance(ch, clip_limit=clip_limit, tile_grid=tile_grid)
        if "Gamma" in method:
            return gamma_correction(ch, gamma=gamma)
        return histogram_equalization(ch)

    if image.ndim == 2:
        return _apply_1ch(image)

    if lab_cache is not None:
        lightness, a, b = lab_cache
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)

    new_lightness = _apply_1ch(lightness)
    merged = cv2.merge((new_lightness, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def calculate_image_stats(gray_image: np.ndarray, hist: np.ndarray = None) -> dict:
    """Tính nhanh các chỉ số thống kê phân bố mức xám, độ tương phản, sắc nét và bão hòa."""
    if gray_image is None or gray_image.size == 0:
        return {}
    if hist is None:
        hist = calculate_histogram(gray_image)
    total = float(gray_image.size)
    prob = hist / total
    levels = np.arange(256, dtype=np.float64)
    mean_val = float(np.sum(prob * levels))
    variance_val = float(np.sum(prob * ((levels - mean_val) ** 2)))
    std_val = float(np.sqrt(variance_val))

    nz = np.flatnonzero(hist)
    min_val = int(nz[0]) if len(nz) > 0 else 0
    max_val = int(nz[-1]) if len(nz) > 0 else 0

    # Tính trung vị chính xác từ CDF (O(256) thay vì sắp xếp hàng triệu pixel)
    cdf = hist.cumsum()
    n_int = int(gray_image.size)
    if n_int % 2 == 1:
        median_val = float(np.searchsorted(cdf, (n_int + 1) // 2))
    else:
        m1 = np.searchsorted(cdf, n_int // 2)
        m2 = np.searchsorted(cdf, n_int // 2 + 1)
        median_val = 0.5 * (float(m1) + float(m2))

    nz_prob = prob[nz]
    entropy_val = float(-np.sum(nz_prob * np.log2(nz_prob))) if len(nz_prob) > 0 else 0.0

    # Độ sắc nét biên (Gradient trung bình tối ưu bằng float32)
    h_img, w_img = gray_image.shape[:2]
    if h_img > 1 and w_img > 1:
        f_img = gray_image.astype(np.float32)
        gx = f_img[:-1, 1:] - f_img[:-1, :-1]
        gy = f_img[1:, :-1] - f_img[:-1, :-1]
        sharpness_val = float(np.mean(np.hypot(gx, gy)))
    else:
        sharpness_val = 0.0

    # Tỷ lệ pixel bão hòa hai đầu tối/sáng (<= 2 hoặc >= 253)
    clip_count = int(hist[:3].sum() + hist[253:].sum())
    clip_rate = float(clip_count / total * 100.0)

    return {
        "min": min_val,
        "max": max_val,
        "range": max_val - min_val,
        "mean": mean_val,
        "variance": variance_val,
        "std": std_val,
        "median": median_val,
        "entropy": entropy_val,
        "sharpness": sharpness_val,
        "clip_rate": clip_rate,
    }


def compute_pair_quality_metrics(gray_orig: np.ndarray,
                                 gray_proc: np.ndarray,
                                 s1: dict = None,
                                 s2: dict = None) -> dict:
    """
    Tính nhanh các chỉ số chất lượng so sánh cặp ảnh trước & sau nâng cao:
    - MSE & PSNR (dB)
    - SSIM (Structural Similarity Index, cửa sổ Gaussian chuẩn trên float32)
    - AMBE (Absolute Mean Brightness Error): |mean_after - mean_before|
    - CII (Contrast Improvement Index): std_after / std_before
    """
    diff = gray_orig.astype(np.int32) - gray_proc.astype(np.int32)
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 1e-10 else float(10.0 * np.log10((255.0 ** 2) / mse))

    # SSIM chuẩn (float32 + thu gọn nếu ảnh quá lớn > 512px để phản hồi tức thì)
    h, w = gray_orig.shape[:2]
    max_dim = max(h, w)
    if max_dim > 512:
        scale = 512.0 / max_dim
        nw, nh = max(16, int(w * scale)), max(16, int(h * scale))
        x = cv2.resize(gray_orig, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
        y = cv2.resize(gray_proc, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        x = gray_orig.astype(np.float32)
        y = gray_proc.astype(np.float32)

    c1 = np.float32((0.01 * 255.0) ** 2)
    c2 = np.float32((0.03 * 255.0) ** 2)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = np.maximum(cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2, 0.0)
    sigma_y2 = np.maximum(cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2, 0.0)
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    ssim = float(np.clip(np.mean(ssim_map), -1.0, 1.0))

    mean1 = s1["mean"] if s1 else float(np.mean(gray_orig))
    mean2 = s2["mean"] if s2 else float(np.mean(gray_proc))
    std1 = s1["std"] if s1 else float(np.std(gray_orig))
    std2 = s2["std"] if s2 else float(np.std(gray_proc))
    ambe = abs(mean2 - mean1)
    cii = std2 / max(std1, 1e-5)

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "ambe": ambe,
        "cii": cii,
    }


def evaluate_enhancement_outcome(s1: dict, s2: dict, pair_m: dict):
    """
    Đánh giá định lượng xem ảnh sau nâng cao 'Tốt hơn', 'Tương đương' hay 'Xấu đi'.
    Trả về: (nhãn_phân_loại, điểm_tổng_hợp_delta_Q, lý_do_ngắn_gọn).
    Tiêu chí kết hợp chuẩn xử lý ảnh:
    1. Cải thiện độ tương phản (std hướng về vùng tối ưu [48..76]) & mở rộng dải động.
    2. Cải thiện độ rõ biên (sharpness) & bảo toàn/tăng Entropy (H).
    3. Cân bằng độ sáng trung bình (hướng về vùng 105..155 nếu ảnh gốc quá tối/quá sáng).
    4. Phạt nặng khi mất chi tiết bão hòa (clip_rate tăng mạnh), biến dạng cấu trúc (SSIM thấp),
       lệch sáng quá mức (AMBE lớn trên ảnh đã cân bằng), hoặc khuếch đại nhiễu/quá gắt.
    """
    std1, std2 = s1["std"], s2["std"]
    ent1, ent2 = s1["entropy"], s2["entropy"]
    sh1, sh2 = s1["sharpness"], s2["sharpness"]
    clip1, clip2 = s1["clip_rate"], s2["clip_rate"]
    mean1, mean2 = s1["mean"], s2["mean"]
    rng1, rng2 = s1["range"], s2["range"]
    ssim = pair_m["ssim"]
    ambe = pair_m["ambe"]

    d_std_pct = (std2 - std1) / max(std1, 1.0) * 100.0
    d_sh_pct = (sh2 - sh1) / max(sh1, 0.5) * 100.0
    d_ent = ent2 - ent1
    d_clip = clip2 - clip1
    d_rng = rng2 - rng1

    # Nếu ảnh gần như không thay đổi
    if pair_m["mse"] < 1.5:
        return "Tương đương", 0.0, "Ảnh gần như không thay đổi sau xử lý"

    score = 0.0
    positives = []
    negatives = []

    # 1. Đánh giá tương phản (std): tối ưu tự nhiên nằm quanh 50..74
    if std1 < 50.0 and std2 > std1 + 2.5:
        gain = min(d_std_pct, 90.0) * 0.38
        score += gain
        positives.append(f"tăng tương phản ({d_std_pct:+.1f}%)")
    elif std1 >= 60.0 and std2 > 76.0 and d_std_pct > 5.0:
        # Ảnh gốc vốn đã tương phản cao mà đẩy gắt thêm -> quá gắt
        score -= 16.0
        negatives.append(f"tương phản quá gắt (σ={std2:.1f})")
    elif d_std_pct < -5.0:
        score -= min(abs(d_std_pct) * 0.6, 25.0)
        negatives.append(f"giảm tương phản ({d_std_pct:+.1f}%)")
    elif d_std_pct > 3.0 and std2 <= 76.0:
        score += min(d_std_pct * 0.25, 12.0)
        positives.append(f"tăng nhẹ tương phản ({d_std_pct:+.1f}%)")

    # 2. Mở rộng dải động
    if d_rng >= 25:
        score += min(d_rng * 0.14, 18.0)
        positives.append(f"mở rộng dải động (+{d_rng})")

    # 3. Cân bằng độ sáng (ảnh quá tối <85 hoặc quá sáng >185 tiến về vùng hài hòa 128)
    dist1 = abs(mean1 - 128.0)
    dist2 = abs(mean2 - 128.0)
    if (mean1 < 85.0 or mean1 > 185.0) and dist2 < dist1 - 10.0:
        score += min((dist1 - dist2) * 0.25, 16.0)
        positives.append("cân bằng lại độ sáng")
    elif 95.0 <= mean1 <= 165.0 and ambe > 28.0:
        score -= min((ambe - 25.0) * 0.55, 22.0)
        negatives.append(f"lệch sáng mạnh (AMBE={ambe:.1f})")

    # 4. Độ sắc nét biên (chỉ cộng điểm khi không vỡ cấu trúc/nhiễu nặng)
    if d_sh_pct > 8.0 and ssim >= 0.62:
        score += min(d_sh_pct * 0.15, 14.0)
        positives.append(f"rõ biên ({d_sh_pct:+.1f}%)")
    elif d_sh_pct > 75.0 and ssim < 0.62:
        # Khuếch đại nhiễu mạnh trên ảnh phẳng/nhiễu
        score -= 22.0
        negatives.append(f"khuếch đại nhiễu biên ({d_sh_pct:+.0f}%)")
    elif d_sh_pct < -8.0:
        score -= 10.0
        negatives.append(f"mờ biên ({d_sh_pct:+.1f}%)")

    # 5. Entropy thông tin
    if d_ent > 0.08:
        score += min(d_ent * 14.0, 10.0)
        positives.append(f"tăng Entropy ({d_ent:+.2f})")
    elif d_ent < -0.20:
        score -= min(abs(d_ent) * 32.0, 26.0)
        negatives.append(f"giảm Entropy ({d_ent:+.2f})")

    # 6. Tỷ lệ bão hòa sáng/tối (Clipping)
    if d_clip > 3.5:
        score -= min(d_clip * 2.4, 34.0)
        negatives.append(f"bão hòa sáng/tối (+{d_clip:.1f}%)")
    elif d_clip < -2.0:
        score += min(abs(d_clip) * 1.5, 10.0)
        positives.append("giảm vùng bão hòa")

    # 7. Bảo toàn cấu trúc SSIM
    if ssim < 0.58:
        score -= (0.58 - ssim) * 85.0
        negatives.append(f"biến dạng cấu trúc (SSIM={ssim:.2f})")
    elif ssim >= 0.85 and score > 0:
        score += 4.0

    # Phân loại cuối cùng
    if score >= 6.0:
        reason = "; ".join(positives[:2]) if positives else "Cải thiện tổng thể chất lượng hiển thị"
        return "Tốt hơn", round(score, 1), reason.capitalize()
    elif score <= -5.0:
        reason = "; ".join(negatives[:2]) if negatives else "Suy giảm chất lượng/bảo toàn cấu trúc"
        return "Xấu đi", round(score, 1), reason.capitalize()
    else:
        detail = (positives[0] if positives else (negatives[0] if negatives else "biến đổi nhỏ trên ảnh đã ổn định"))
        return "Tương đương", round(score, 1), f"Thay đổi nhẹ ({detail})"


def generate_diverse_100_dataset(output_dir: Path) -> list:
    """
    Sinh bộ 100 ảnh thực nghiệm đa dạng (480x320) gồm 10 nhóm kịch bản quang học
    (thiếu sáng, mờ sương, tối cục bộ, nhạt màu, cân bằng tốt, cháy sáng, nhiễu nền phẳng,
    tương phản gắt sẵn, ít mức xám, ngược sáng) để đánh giá thống kê khách quan
    cả trường hợp cải thiện lẫn suy giảm.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)

    # Nạp tối đa các ảnh thật sẵn có trong assets/ làm nguồn kết cấu phong phú
    real_sources = []
    for name in ("thiennhien.jpg", "hehehe.jpg", "sample_test.png"):
        p = ASSETS_DIR / name
        if p.exists():
            im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                real_sources.append(cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA))

    # Nếu không có ảnh mẫu, tự tạo ảnh nền phong cảnh màu chuẩn
    if not real_sources:
        yy, xx = np.mgrid[0:320, 0:480].astype(np.float32)
        base = np.zeros((320, 480, 3), dtype=np.uint8)
        base[:, :, 0] = np.clip(90 + 80 * np.sin(xx / 60.0) + 40 * (yy / 320.0), 0, 255)
        base[:, :, 1] = np.clip(110 + 70 * np.cos(yy / 50.0), 0, 255)
        base[:, :, 2] = np.clip(120 + 60 * np.sin((xx + yy) / 80.0), 0, 255)
        real_sources.append(base)

    def pick_crop(idx: int) -> np.ndarray:
        src = real_sources[idx % len(real_sources)].copy()
        if idx % 2 == 1:
            src = cv2.flip(src, 1)
        # Biến đổi góc nhìn / crop nhẹ khác nhau cho từng ảnh
        h, w = src.shape[:2]
        scale = rng.uniform(0.76, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0 = int(rng.integers(0, max(1, h - ch + 1)))
        x0 = int(rng.integers(0, max(1, w - cw + 1)))
        crop = cv2.resize(src[y0:y0 + ch, x0:x0 + cw], (480, 320), interpolation=cv2.INTER_LINEAR)
        # Dịch chuyển sắc độ nhẹ để mỗi ảnh có đặc trưng riêng
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.integers(-18, 19))) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def make_flat_scene(base_level: int, noise_sigma: float) -> np.ndarray:
        """Tạo ảnh nền phẳng có vài hình khối nhẹ + nhiễu hạt (ca điển hình HE làm xấu đi)."""
        img = np.full((320, 480, 3), base_level, dtype=np.float32)
        yy, xx = np.mgrid[0:320, 0:480].astype(np.float32)
        img += 10.0 * np.sin(xx / 120.0)[:, :, None]
        cx, cy = int(rng.integers(140, 340)), int(rng.integers(90, 230))
        rad = int(rng.integers(38, 72))
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rad ** 2
        img[mask] += rng.uniform(14.0, 26.0)
        noise = rng.normal(0.0, noise_sigma, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def make_poster_scene(levels_count: int) -> np.ndarray:
        """Tạo ảnh đồ họa ít mức xám (sơ đồ/hình khối phẳng)."""
        img = np.full((320, 480, 3), int(rng.integers(180, 235)), dtype=np.uint8)
        for _ in range(levels_count):
            x1, y1 = int(rng.integers(20, 340)), int(rng.integers(20, 220))
            x2, y2 = min(460, x1 + int(rng.integers(50, 140))), min(300, y1 + int(rng.integers(40, 110)))
            col = [int(rng.integers(30, 220)) for _ in range(3)]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)
        return img

    # 10 nhóm kịch bản = đúng 100 ảnh
    plan = (
        [("thieu_sang", 14)] +
        [("mo_suong", 14)] +
        [("toi_cuc_bo", 12)] +
        [("nhat_mau", 10)] +
        [("can_bang_tot", 10)] +
        [("chay_sang", 10)] +
        [("nhieu_nen_phang", 10)] +
        [("tuong_phan_gat", 10)] +
        [("it_muc_xam", 5)] +
        [("nguoc_sang", 5)]
    )

    saved_paths = []
    idx = 0
    for cat, count in plan:
        for j in range(count):
            idx += 1
            if cat == "thieu_sang":
                im = pick_crop(idx).astype(np.float32)
                alpha = rng.uniform(0.22, 0.42)
                beta = rng.uniform(4.0, 16.0)
                out = np.clip(im * alpha + beta, 0, 255).astype(np.uint8)
            elif cat == "mo_suong":
                im = pick_crop(idx).astype(np.float32)
                alpha = rng.uniform(0.22, 0.40)
                mid = rng.uniform(100.0, 145.0)
                out = np.clip((im - 128.0) * alpha + mid, 0, 255).astype(np.uint8)
            elif cat == "toi_cuc_bo":
                im = pick_crop(idx).astype(np.float32) / 255.0
                gamma = rng.uniform(2.2, 3.4)
                out = np.clip((im ** gamma) * 255.0, 0, 255).astype(np.uint8)
            elif cat == "nhat_mau":
                im = pick_crop(idx).astype(np.float32)
                alpha = rng.uniform(0.32, 0.48)
                beta = rng.uniform(105.0, 130.0)
                out = np.clip(im * alpha + beta, 0, 255).astype(np.uint8)
            elif cat == "can_bang_tot":
                im = pick_crop(idx).astype(np.float32)
                alpha = rng.uniform(0.85, 1.02)
                beta = rng.uniform(-6.0, 6.0)
                out = np.clip((im - 128.0) * alpha + 128.0 + beta, 0, 255).astype(np.uint8)
            elif cat == "chay_sang":
                im = pick_crop(idx).astype(np.float32)
                # Vùng sáng lớn chiếm ưu thế + lóa mạnh
                out = np.clip(im * rng.uniform(0.55, 0.80) + rng.uniform(115.0, 145.0), 0, 255).astype(np.uint8)
            elif cat == "nhieu_nen_phang":
                out = make_flat_scene(int(rng.integers(95, 155)), rng.uniform(6.5, 13.0))
            elif cat == "tuong_phan_gat":
                im = pick_crop(idx).astype(np.float32)
                gain = rng.uniform(1.55, 2.05)
                out = np.clip((im - 128.0) * gain + 128.0, 0, 255).astype(np.uint8)
            elif cat == "it_muc_xam":
                out = make_poster_scene(int(rng.integers(4, 9)))
            else:  # nguoc_sang
                im = pick_crop(idx).astype(np.float32)
                h, w = im.shape[:2]
                grad = np.linspace(0.25, 1.55, w, dtype=np.float32)[None, :, None]
                out = np.clip(im * grad + rng.uniform(-15.0, 15.0), 0, 255).astype(np.uint8)

            fname = output_dir / f"{idx:03d}_{cat}_{j + 1:02d}.png"
            ok, buf = cv2.imencode(".png", out)
            if ok:
                buf.tofile(str(fname))
                saved_paths.append(fname)

    return saved_paths


# ============================================================
# CÁC HÀM SO SÁNH PHÂN ĐOẠN ĐỊNH LƯỢNG & TRỰC QUAN HÓA
# (Thỏa mãn yêu cầu: So sánh kết quả phân đoạn trước & sau nâng cao)
# ============================================================

def compute_mask_metrics(mask1: np.ndarray, mask2: np.ndarray) -> dict:
    """
    Tính các chỉ số định lượng so sánh giữa 2 mặt nạ phân đoạn nhị phân:
    - IoU (Jaccard Index): |M1 ∩ M2| / |M1 ∪ M2|
    - Dice Similarity Coefficient: 2 * |M1 ∩ M2| / (|M1| + |M2|)
    - Tỷ lệ pixel sai khác: count(M1 != M2) / total * 100%
    - Số lượng pixel mới bổ sung (added) và loại bỏ (removed).
    """
    b1 = (mask1 > 127)
    b2 = (mask2 > 127)
    total = float(mask1.size)

    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()

    iou = float(inter / union) if union > 0 else 1.0
    sum_fg = b1.sum() + b2.sum()
    dice = float((2.0 * inter) / sum_fg) if sum_fg > 0 else 1.0

    diff_pixels = np.logical_xor(b1, b2).sum()
    diff_rate = float(diff_pixels / total * 100.0)

    added_pixels = np.logical_and(~b1, b2).sum()    # chỉ có ở mặt nạ 2
    removed_pixels = np.logical_and(b1, ~b2).sum()  # chỉ có ở mặt nạ 1

    return {
        "iou": iou * 100.0,
        "dice": dice * 100.0,
        "diff_pixels": int(diff_pixels),
        "diff_rate": diff_rate,
        "common_pixels": int(inter),
        "added_pixels": int(added_pixels),
        "removed_pixels": int(removed_pixels),
        "fg1_count": int(b1.sum()),
        "fg2_count": int(b2.sum()),
        "fg1_percent": float(b1.sum()) / total * 100.0,
        "fg2_percent": float(b2.sum()) / total * 100.0,
    }


def create_diff_map(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Tạo bản đồ khác biệt phân đoạn (RGB Difference Map):
    - Trắng [245, 245, 245]: Đối tượng chung (cả 2 đều nhận diện là đối tượng).
    - Xanh lục [40, 200, 60]: Đối tượng mới bổ sung sau nâng cao chất lượng ảnh.
    - Đỏ cam [50, 60, 230]: Đối tượng bị loại bỏ sau nâng cao chất lượng ảnh.
    - Xám than [28, 33, 39]: Nền đồng thuận (cả 2 đều nhận diện là nền).
    """
    b1 = (mask1 > 127)
    b2 = (mask2 > 127)
    h, w = mask1.shape[:2]
    diff = np.zeros((h, w, 3), dtype=np.uint8)

    diff[~b1 & ~b2] = [28, 33, 39]       # Nền chung
    diff[b1 & b2] = [245, 245, 245]       # Đối tượng chung
    diff[~b1 & b2] = [40, 200, 60]       # Thêm mới (Xanh lục BGR)
    diff[b1 & ~b2] = [50, 60, 230]       # Loại bỏ (Đỏ tươi BGR)

    return diff


def create_segmented_overlay(img: np.ndarray, mask: np.ndarray,
                             color=(220, 160, 40), alpha=0.42) -> np.ndarray:
    """
    Phủ màu đối tượng (Color Overlay): Phủ lớp màu bán trong suốt lên vùng đối tượng
    kèm đường viền mỏng làm nổi bật ranh giới phân đoạn trên nền ảnh thực tế.
    """
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    overlay = base.copy()
    fg = (mask > 127)
    overlay[fg] = np.clip(
        (1.0 - alpha) * base[fg].astype(np.float32) + alpha * np.array(color, dtype=np.float32),
        0, 255
    ).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay


def create_cutout(img: np.ndarray, mask: np.ndarray, bg_color=(24, 28, 33)) -> np.ndarray:
    """
    Tách biệt đối tượng (Foreground Cutout):
    Giữ nguyên màu sắc đối tượng tại mask == 255, xóa nền về màu tối studio.
    """
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cutout = np.full_like(base, bg_color)
    fg = (mask > 127)
    cutout[fg] = base[fg]
    return cutout


# ============================================================
# CÁC WIDGET GIAO DIỆN CHUYÊN DỤNG
# ============================================================

class ImageCanvas(QWidget):
    """
    Widget hiển thị ảnh chuyên dụng:
    - Tự động căn giữa, bảo toàn đúng tỷ lệ gốc (KeepAspectRatio).
    - Khắc phục triệt để lỗi phình layout của QLabel.setPixmap.
    - Hỗ trợ kéo thả tệp ảnh trực tiếp (với hiệu ứng nhận diện) và nhấp đúp để phóng to.
    """
    file_dropped = Signal(str)
    double_clicked = Signal()

    def __init__(self, placeholder="Chưa có ảnh", subtext="", accept_drop=False, parent=None):
        super().__init__(parent)
        self.image = None
        self.qimage = None
        self.gray_view = False
        self.placeholder = placeholder
        self.subtext = subtext
        self._accept_drop = accept_drop
        self._is_dragging = False
        self._scaled_pixmap = None
        self._scaled_key = None
        self._img_version = 0
        self.setAcceptDrops(accept_drop)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(120, 90)

    def minimumSizeHint(self):
        return QSize(120, 90)

    def sizeHint(self):
        return QSize(360, 260)

    def set_image(self, image, gray_view=None):
        self.image = image
        if gray_view is not None:
            self.gray_view = gray_view
        self._img_version += 1
        self._scaled_pixmap = None
        self._scaled_key = None
        if image is None:
            self.qimage = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            show = to_gray(image) if (self.gray_view and image.ndim == 3) else image
            if show.ndim == 2:
                data = np.ascontiguousarray(show)
                h, w = data.shape
                self.qimage = QImage(data.data, w, h, data.strides[0],
                                     QImage.Format.Format_Grayscale8).copy()
            else:
                data = np.ascontiguousarray(cv2.cvtColor(show, cv2.COLOR_BGR2RGB))
                h, w, _ = data.shape
                self.qimage = QImage(data.data, w, h, data.strides[0],
                                     QImage.Format.Format_RGB888).copy()
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()

    def _draw_empty_icon(self, p: QPainter, cx: float, cy: float, active: bool):
        """Vẽ biểu tượng khung ảnh / tải lên tối giản bằng vector sắc nét."""
        stroke = QColor("#1B56B5" if active else "#64748B")
        fill_bg = QColor("#DBEAFE" if active else "#EEF2F7")

        # Nền tròn nhạt bao quanh biểu tượng
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(fill_bg))
        p.drawEllipse(QPointF(cx, cy), 22, 22)

        pen = QPen(stroke, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        if self._accept_drop:
            # Biểu tượng mũi tên tải lên + khay nhận ảnh
            p.drawLine(QPointF(cx, cy + 5), QPointF(cx, cy - 8))
            p.drawLine(QPointF(cx, cy - 8), QPointF(cx - 5.5, cy - 2.5))
            p.drawLine(QPointF(cx, cy - 8), QPointF(cx + 5.5, cy - 2.5))
            tray = QPainterPath()
            tray.moveTo(cx - 10, cy + 4)
            tray.lineTo(cx - 10, cy + 9)
            tray.lineTo(cx + 10, cy + 9)
            tray.lineTo(cx + 10, cy + 4)
            p.drawPath(tray)
        else:
            # Biểu tượng khung ảnh tối giản
            frame_rect = QRectF(cx - 11, cy - 8.5, 22, 17)
            p.drawRoundedRect(frame_rect, 3, 3)
            p.drawEllipse(QPointF(cx - 4.5, cy - 2.5), 2.0, 2.0)
            mtn = QPainterPath()
            mtn.moveTo(cx - 8.5, cy + 6)
            mtn.lineTo(cx - 2.5, cy + 0.5)
            mtn.lineTo(cx + 2.0, cy + 4.5)
            mtn.lineTo(cx + 5.5, cy + 1.5)
            mtn.lineTo(cx + 9.0, cy + 6)
            p.drawPath(mtn)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        clip = QPainterPath()
        clip.addRoundedRect(rect, 8, 8)
        p.setClipPath(clip)

        if self.qimage is None:
            bg_color = "#EFF6FF" if self._is_dragging else "#F8FAFC"
            p.fillRect(self.rect(), QColor(bg_color))
        else:
            p.fillRect(self.rect(), QColor("#1C2128"))

        p.setClipping(False)

        if self.qimage is None:
            # Viền nét đứt chuyên nghiệp khi đang ở trạng thái trống
            border_color = QColor("#1B56B5" if self._is_dragging else "#CBD5E1")
            pen_style = Qt.PenStyle.DashLine if self._accept_drop or self._is_dragging else Qt.PenStyle.SolidLine
            p.setPen(QPen(border_color, 1.5 if self._is_dragging else 1.2, pen_style))
            p.setBrush(Qt.BrushStyle.NoBrush)
            inner_rect = rect.adjusted(1.0, 1.0, -1.0, -1.0)
            p.drawRoundedRect(inner_rect, 8, 8)

            w, h = self.width(), self.height()
            show_rich = (w >= 210 and h >= 135)

            if show_rich:
                cx = w / 2.0
                cy = h / 2.0 - (18 if self.subtext else 10)
                self._draw_empty_icon(p, cx, cy - 10, self._is_dragging)

                title_font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
                p.setFont(title_font)
                p.setPen(QColor("#1B56B5" if self._is_dragging else "#1E293B"))
                title_rect = QRectF(16, cy + 18, w - 32, 24)
                p.drawText(title_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                           "Thả tệp ảnh để mở ngay" if self._is_dragging else self.placeholder)

                if self.subtext:
                    sub_font = QFont("Segoe UI", 9, QFont.Weight.Normal)
                    p.setFont(sub_font)
                    p.setPen(QColor("#64748B"))
                    sub_rect = QRectF(16, cy + 42, w - 32, 34)
                    p.drawText(sub_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                               self.subtext)
            else:
                small_font = QFont("Segoe UI", 9, QFont.Weight.Medium)
                p.setFont(small_font)
                p.setPen(QColor("#64748B"))
                p.drawText(self.rect().adjusted(10, 0, -10, 0),
                           Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                           self.placeholder)
            return

        # Viền mảnh quanh khung ảnh đã tải
        p.setPen(QPen(QColor("#CBD5E1"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect, 8, 8)

        aw, ah = self.width() - 14, self.height() - 14
        iw, ih = self.qimage.width(), self.qimage.height()
        if aw <= 0 or ah <= 0:
            return

        s = min(aw / iw, ah / ih)
        dw, dh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
        target = QRectF(self.width() / 2 - dw / 2, self.height() / 2 - dh / 2, dw, dh)

        dpr = self.devicePixelRatioF()
        pw, ph = max(1, int(round(dw * dpr))), max(1, int(round(dh * dpr)))
        cache_key = (self._img_version, pw, ph)
        if self._scaled_key != cache_key or self._scaled_pixmap is None:
            scaled = self.qimage.scaled(
                pw, ph, Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            pm = QPixmap.fromImage(scaled)
            pm.setDevicePixelRatio(dpr)
            self._scaled_pixmap = pm
            self._scaled_key = cache_key

        img_clip = QPainterPath()
        img_clip.addRoundedRect(target, 4, 4)
        p.setClipPath(img_clip)
        p.drawPixmap(target.topLeft(), self._scaled_pixmap)
        p.setClipping(False)

        # Gợi ý phóng to tinh tế ở góc dưới phải khi khung đủ lớn
        if self.width() >= 270 and self.height() >= 150:
            hint_text = "Nhấp đúp để phóng to"
            hint_font = QFont("Segoe UI", 8, QFont.Weight.Medium)
            p.setFont(hint_font)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(hint_text)
            th = fm.height()
            pill = QRectF(self.width() - tw - 24, self.height() - th - 14, tw + 16, th + 6)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(15, 23, 42, 165)))
            p.drawRoundedRect(pill, 5, 5)
            p.setPen(QColor("#F1F5F9"))
            p.drawText(pill, Qt.AlignmentFlag.AlignCenter, hint_text)

    def mouseDoubleClickEvent(self, event):
        if self.image is not None:
            self.double_clicked.emit()

    def dragEnterEvent(self, event):
        if self._accept_drop and event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._is_dragging = True
            self.update()

    def dragLeaveEvent(self, event):
        if self._is_dragging:
            self._is_dragging = False
            self.update()

    def dropEvent(self, event):
        self._is_dragging = False
        self.update()
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).is_file():
                self.file_dropped.emit(path)


class ZoomDialog(QDialog):
    """Cửa sổ xem chi tiết ảnh phóng to."""
    def __init__(self, title, image, parent=None):
        super().__init__(parent)
        h, w = image.shape[:2]
        self.setWindowTitle(f"{title} — {w} × {h} px")
        self.resize(960, 700)
        self.setStyleSheet("QDialog { background: #F4F6F9; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        canvas = ImageCanvas()
        canvas.set_image(image)
        lay.addWidget(canvas)


class BatchAnalyticsDialog(QDialog):
    """
    Cửa sổ biểu đồ trực quan hóa kết quả thống kê nhiều ảnh & so sánh đối đầu
    các phương pháp (HE vs Kéo giãn vs CLAHE vs Gamma) trên cùng thư mục ảnh.
    Hỗ trợ xuất ảnh biểu đồ PNG chất lượng cao (200 DPI) cho báo cáo Word / Slide.
    """
    def __init__(self, results: list, folder: Path, compare_callback=None, parent=None):
        super().__init__(parent)
        self.results = results
        self.folder = folder
        self.compare_callback = compare_callback
        self.comparison_data = None

        n = len(results)
        method_name = results[0]["method"] if results else ""
        self.setWindowTitle(f"Biểu đồ thống kê thực nghiệm ({n} ảnh) — {folder.name if folder else ''}")
        self.resize(1140, 720)
        self.setStyleSheet("""
            QDialog { background: #F4F6F9; font-family: 'Segoe UI', 'Tahoma', sans-serif; }
            QPushButton { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 7px;
                          padding: 6px 14px; color: #1E293B; font-size: 9.5pt; font-weight: 600; }
            QPushButton:hover { background: #F8FAFC; border-color: #94A3B8; }
            QPushButton#primary { background: #1B56B5; color: #FFFFFF; border: 1px solid #164899; font-weight: 700; }
            QPushButton#primary:hover { background: #144493; }
            QPushButton#accentTeal { background: #0F766E; color: #FFFFFF; border: 1px solid #0C5F59; font-weight: 700; }
            QPushButton#accentTeal:hover { background: #0D625C; }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        # Thanh nút phía trên
        top = QHBoxLayout()
        top.setSpacing(8)
        self.lbl_info = QLabel(
            f"Tập ảnh: {folder.name if folder else '—'} ({n} ảnh)  •  Phương pháp đang xem: {method_name}"
        )
        self.lbl_info.setStyleSheet("font-size: 10pt; font-weight: 700; color: #0F172A;")
        top.addWidget(self.lbl_info)
        top.addStretch()

        self.btn_compare_all = QPushButton("So sánh cả 4 phương pháp (HE / Kéo giãn / CLAHE / Gamma)")
        self.btn_compare_all.setObjectName("accentTeal")
        self.btn_compare_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare_all.clicked.connect(self.run_compare_all)
        if compare_callback is None:
            self.btn_compare_all.setVisible(False)
        top.addWidget(self.btn_compare_all)

        self.btn_save_png = QPushButton("Lưu biểu đồ PNG...")
        self.btn_save_png.setObjectName("primary")
        self.btn_save_png.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_png.clicked.connect(self.save_chart_png)
        top.addWidget(self.btn_save_png)
        lay.addLayout(top)

        # Khung Figure Matplotlib
        card = QFrame()
        card.setStyleSheet("QFrame { background: #FFFFFF; border: 1px solid #D8E0EA; border-radius: 10px; }")
        c_lay = QVBoxLayout(card)
        c_lay.setContentsMargins(8, 8, 8, 8)

        self.figure = Figure(figsize=(11, 6.2), dpi=100, facecolor="#FFFFFF")
        self.canvas = FigureCanvas(self.figure)
        c_lay.addWidget(self.canvas, 1)
        lay.addWidget(card, 1)

        self._render_charts()

    def _style_ax(self, ax):
        ax.set_facecolor("#FFFFFF")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color("#CBD5E1")
        ax.tick_params(labelsize=8.5, colors="#475569")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color="#E2E8F0")
        ax.set_axisbelow(True)

    def _render_charts(self):
        self.figure.clear()
        res = self.results
        if not res:
            self.canvas.draw_idle()
            return

        n = len(res)
        n_better = sum(1 for r in res if r["verdict"] == "Tốt hơn")
        n_same = sum(1 for r in res if r["verdict"] == "Tương đương")
        n_worse = sum(1 for r in res if r["verdict"] == "Xấu đi")

        gs = self.figure.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.42, wspace=0.25)

        # 1. Biểu đồ cột phân loại Tốt hơn / Tương đương / Xấu đi
        ax1 = self.figure.add_subplot(gs[0, 0])
        self._style_ax(ax1)
        cats = ["Tốt hơn", "Tương đương", "Xấu đi"]
        vals = [n_better, n_same, n_worse]
        colors = ["#0F766E", "#64748B", "#DC2626"]
        bars = ax1.bar(cats, vals, color=colors, width=0.52, edgecolor="none")
        ax1.set_title(f"Phân loại chất lượng ({n} ảnh)", fontsize=10.5, fontweight="bold", color="#0F172A", pad=8)
        ax1.set_ylabel("Số lượng ảnh", fontsize=8.5, fontweight="semibold", color="#334155")
        ax1.set_ylim(0, max(vals) * 1.24 + 1)
        for b, v in zip(bars, vals):
            pct = v / n * 100.0
            ax1.text(
                b.get_x() + b.get_width() / 2.0, v + max(vals) * 0.03 + 0.2,
                f"{v} ({pct:.1f}%)", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#0F172A"
            )

        # 2. Biểu đồ phân tán Độ tương phản σ (Trước vs Sau)
        ax2 = self.figure.add_subplot(gs[0, 1])
        self._style_ax(ax2)
        ax2.grid(True, linestyle="--", linewidth=0.6, color="#E2E8F0")
        s1 = np.array([r["std1"] for r in res], dtype=np.float32)
        s2 = np.array([r["std2"] for r in res], dtype=np.float32)
        verdicts = [r["verdict"] for r in res]
        c_map = {"Tốt hơn": "#0F766E", "Tương đương": "#64748B", "Xấu đi": "#DC2626"}
        for lbl in ("Tốt hơn", "Tương đương", "Xấu đi"):
            idx = [i for i, v in enumerate(verdicts) if v == lbl]
            if idx:
                ax2.scatter(s1[idx], s2[idx], s=26, alpha=0.82, color=c_map[lbl], label=lbl, edgecolors="white", linewidths=0.4)
        lim_max = max(float(s1.max()), float(s2.max()), 90.0) * 1.06
        ax2.plot([0, lim_max], [0, lim_max], linestyle=":", color="#94A3B8", linewidth=1.2, label="Không đổi (σ_sau = σ_gốc)")
        ax2.set_xlim(0, lim_max)
        ax2.set_ylim(0, lim_max)
        ax2.set_title("Độ tương phản σ: Ảnh gốc vs Sau nâng cao", fontsize=10.5, fontweight="bold", color="#0F172A", pad=8)
        ax2.set_xlabel("σ ảnh gốc", fontsize=8.5, fontweight="semibold", color="#334155")
        ax2.set_ylabel("σ sau nâng cao", fontsize=8.5, fontweight="semibold", color="#334155")
        ax2.legend(fontsize=7.8, loc="lower right", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")

        # 3. Biểu đồ dưới: So sánh 4 phương pháp (nếu đã bấm so sánh) hoặc Phân bố thay đổi chỉ số TB
        ax3 = self.figure.add_subplot(gs[1, :])
        self._style_ax(ax3)
        if self.comparison_data:
            names = [d["short"] for d in self.comparison_data]
            pct_better = [d["better_pct"] for d in self.comparison_data]
            pct_worse = [d["worse_pct"] for d in self.comparison_data]
            avg_ssim_100 = [d["avg_ssim"] * 100.0 for d in self.comparison_data]
            x = np.arange(len(names))
            w = 0.25

            b1 = ax3.bar(x - w, pct_better, width=w, color="#0F766E", label="Tỷ lệ Tốt hơn (%)")
            b2 = ax3.bar(x, pct_worse, width=w, color="#DC2626", label="Tỷ lệ Xấu đi (%)")
            b3 = ax3.bar(x + w, avg_ssim_100, width=w, color="#1B56B5", label="SSIM trung bình (×100%)")

            ax3.set_xticks(x)
            ax3.set_xticklabels(names, fontsize=9.2, fontweight="bold", color="#0F172A")
            ax3.set_ylim(0, 116)
            ax3.set_ylabel("Tỷ lệ (%)", fontsize=8.5, fontweight="semibold", color="#334155")
            ax3.set_title(
                f"So sánh đối đầu 4 phương pháp trên cùng thư mục ({n} ảnh)",
                fontsize=10.5, fontweight="bold", color="#0F172A", pad=8
            )
            ax3.legend(fontsize=8.2, loc="upper right", ncol=3, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")

            for group in (b1, b2, b3):
                for bar in group:
                    h_val = bar.get_height()
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2.0, h_val + 1.5,
                        f"{h_val:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#1E293B"
                    )
        else:
            # Trước khi bấm so sánh 4 phương pháp: hiển thị so sánh TB 4 chỉ số Trước vs Sau
            metrics_lbl = ["TB Mức xám (μ)", "TB Tương phản (σ)", "TB Entropy (H×10)", "TB Sắc nét (×5)"]
            before_v = [
                float(np.mean([r["mean1"] for r in res])),
                float(np.mean([r["std1"] for r in res])),
                float(np.mean([r["ent1"] for r in res])) * 10.0,
                float(np.mean([r["sh1"] for r in res])) * 5.0,
            ]
            after_v = [
                float(np.mean([r["mean2"] for r in res])),
                float(np.mean([r["std2"] for r in res])),
                float(np.mean([r["ent2"] for r in res])) * 10.0,
                float(np.mean([r["sh2"] for r in res])) * 5.0,
            ]
            x = np.arange(len(metrics_lbl))
            w = 0.30
            b1 = ax3.bar(x - w / 2, before_v, width=w, color="#94A3B8", label="Ảnh gốc (TB)")
            b2 = ax3.bar(x + w / 2, after_v, width=w, color="#1B56B5", label="Sau nâng cao (TB)")
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics_lbl, fontsize=9, fontweight="semibold", color="#0F172A")
            ax3.set_ylim(0, max(max(before_v), max(after_v)) * 1.24 + 5)
            ax3.set_title(
                "So sánh giá trị trung bình toàn tập (Nhấn 'So sánh cả 4 phương pháp' ở góc trên để đối chiếu HE / Kéo giãn / CLAHE / Gamma)",
                fontsize=9.5, fontweight="bold", color="#0F172A", pad=8
            )
            ax3.legend(fontsize=8.2, loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
            for bar in b1:
                v = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2.0, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#475569", fontweight="bold")
            for bar in b2:
                v = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2.0, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#1B56B5", fontweight="bold")

        self.figure.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.09)
        self.canvas.draw_idle()

    def run_compare_all(self):
        if self.compare_callback is None:
            return
        self.btn_compare_all.setEnabled(False)
        self.btn_compare_all.setText("Đang chạy so sánh 4 phương pháp...")
        QApplication.processEvents()
        try:
            self.comparison_data = self.compare_callback()
            self._render_charts()
        finally:
            self.btn_compare_all.setEnabled(True)
            self.btn_compare_all.setText("So sánh cả 4 phương pháp (HE / Kéo giãn / CLAHE / Gamma)")

    def save_chart_png(self):
        default_name = f"BieuDo_ThongKe_{self.folder.name if self.folder else 'De13'}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh biểu đồ thống kê", default_name, "PNG Image (*.png)"
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, facecolor="#FFFFFF")
            QMessageBox.information(self, "Lưu thành công", f"Đã lưu ảnh biểu đồ chất lượng cao (200 DPI) tại:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi khi lưu biểu đồ", str(exc))


class PlotCanvas(FigureCanvas):
    """Widget vẽ đồ thị matplotlib tối ưu hiệu năng cao, đồng bộ phông chữ hệ thống."""
    _X256 = np.arange(256, dtype=np.int32)
    _RC_APPLIED = False

    def __init__(self, parent=None):
        if not PlotCanvas._RC_APPLIED:
            matplotlib.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Segoe UI", "Tahoma", "DejaVu Sans"],
                "axes.edgecolor": "#CBD5E1",
                "axes.labelcolor": "#334155",
                "xtick.color": "#475569",
                "ytick.color": "#475569",
            })
            PlotCanvas._RC_APPLIED = True

        self.figure = Figure(figsize=(4, 2.2), dpi=100, facecolor="#FFFFFF")
        self.ax = self.figure.add_subplot(111)
        self._last_key = None
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(110)

    def _reset_ax(self, title: str, compact: bool = False):
        ax = self.ax
        ax.cla()
        ax.set_facecolor("#FFFFFF")
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold", color="#0F172A", pad=7)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Ngưỡng t" if compact else "Mức xám (0 - 255)",
                      fontsize=8 if compact else 8.5, fontweight="semibold", color="#334155")
        ax.set_ylabel("σ²_B(t)" if compact else "Số pixel",
                      fontsize=8 if compact else 8.5, fontweight="semibold", color="#334155")
        ax.tick_params(labelsize=7.5 if compact else 8, colors="#475569")
        ax.grid(True, linewidth=0.6, linestyle="--", color="#E2E8F0")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color("#CBD5E1")
            ax.spines[sp].set_linewidth(1.0)
        if compact:
            self.figure.subplots_adjust(left=0.21, right=0.96, top=0.94, bottom=0.22)
        else:
            self.figure.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.23)
        return ax

    def draw_histogram(self, gray, title, threshold=None, hist=None, cache_key=None):
        if cache_key is not None and cache_key == self._last_key:
            return
        self._last_key = cache_key

        ax = self._reset_ax(title, compact=False)
        if hist is None and gray is not None:
            hist = calculate_histogram(gray)
        if hist is not None:
            ax.fill_between(self._X256, 0, hist, step="mid", color="#2563EB", alpha=0.84, linewidth=0.0)
            y_max = float(hist.max()) * 1.08 if hist.max() > 0 else 1.0
            ax.set_ylim(0, y_max)
            if threshold is not None:
                ax.axvline(threshold, color="#DC2626", linewidth=1.6, linestyle="--")
                x_pos = threshold + 5 if threshold < 195 else threshold - 48
                ax.text(
                    x_pos, y_max * 0.84,
                    f"T = {threshold}",
                    color="#B91C1C", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#FCA5A5", linewidth=0.9)
                )
        self.draw_idle()

    def draw_variance(self, var_before, var_after, t_before, t_after):
        key = ("var_empty",) if var_before is None else ("var", t_before, t_after, float(var_before[t_before]), float(var_after[t_after]))
        if key == self._last_key:
            return
        self._last_key = key

        ax = self._reset_ax("", compact=True)
        if var_before is not None:
            ax.plot(self._X256, var_before, color="#64748B", linewidth=1.3, linestyle="--",
                    label=f"Gốc (T={t_before})")
            ax.plot(self._X256, var_after, color="#1B56B5", linewidth=1.8,
                    label=f"Nâng cao (T={t_after})")
            ax.axvline(t_before, color="#94A3B8", linestyle=":", linewidth=1.0)
            ax.axvline(t_after, color="#1B56B5", linestyle=":", linewidth=1.1)
            ax.scatter([t_before], [var_before[t_before]], color="#475569", s=22, zorder=5)
            ax.scatter([t_after], [var_after[t_after]], color="#1B56B5", s=28, zorder=6)
            ax.legend(fontsize=7.2, loc="lower center", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
        self.draw_idle()


class MetricSubtitleDelegate(QStyledItemDelegate):
    """Vẽ ô 2 dòng (tên chỉ số đậm ở trên + chú thích ngắn chữ bé ở dưới) mượt, không đè chữ."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._font_title_bold = QFont("Segoe UI", 9, QFont.Weight.Bold)
        self._font_title_demi = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        self._font_sub = QFont("Segoe UI", 7.6, QFont.Weight.Normal)

    def paint(self, painter, option, index):
        painter.save()
        # Vẽ nền chuẩn (bao gồm màu xen kẽ dòng, nền nổi bật hoặc trạng thái chọn)
        opt = option
        if opt.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(opt.rect, QColor("#EFF6FF"))
        else:
            bg = index.data(Qt.ItemDataRole.BackgroundRole)
            if bg is not None:
                painter.fillRect(opt.rect, bg)
            elif index.row() % 2 == 1:
                painter.fillRect(opt.rect, QColor("#F8FAFC"))
            else:
                painter.fillRect(opt.rect, QColor("#FFFFFF"))

        title = index.data(Qt.ItemDataRole.DisplayRole) or ""
        subtitle = index.data(Qt.ItemDataRole.UserRole) or ""
        is_bold = bool(index.data(Qt.ItemDataRole.UserRole + 1))

        rect = opt.rect.adjusted(8, 3, -6, -3)
        if subtitle:
            top_rect = rect.adjusted(0, 0, 0, -rect.height() // 2 + 2)
            bot_rect = rect.adjusted(0, rect.height() // 2 - 1, 0, 0)
            painter.setFont(self._font_title_bold if is_bold else self._font_title_demi)
            painter.setPen(QColor("#0F172A"))
            painter.drawText(top_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)

            painter.setFont(self._font_sub)
            painter.setPen(QColor("#64748B"))
            elided_sub = painter.fontMetrics().elidedText(subtitle, Qt.TextElideMode.ElideRight, bot_rect.width())
            painter.drawText(bot_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided_sub)
        else:
            painter.setFont(self._font_title_bold if is_bold else self._font_title_demi)
            painter.setPen(QColor("#0F172A"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        painter.restore()


# ============================================================
# CỬA SỔ CHÍNH CỦA CHƯƠNG TRÌNH
# ============================================================

class ImageProcessingStudio(QMainWindow):
    METHODS = [
        "Cân bằng histogram",
        "Kéo giãn tương phản",
        "CLAHE (Cân bằng cục bộ)",
        "Biến đổi Gamma (γ)",
    ]
    VIEW_MODES = [
        "Mặt nạ nhị phân (Binary Mask)",
        "Phủ màu đối tượng (Color Overlay)",
        "Tách riêng đối tượng (Cutout)"
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tính histogram và nâng cao chất lượng ảnh - Đề tài 13")
        self.resize(1320, 840)
        self.setMinimumSize(960, 660)

        icon = ASSETS_DIR / "logo.ico"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "btl.xlavtg.histogram.de13")
            except Exception:
                pass

        # Trạng thái dữ liệu & bộ nhớ đệm tăng tốc phản hồi
        self.path = None
        self.original = None
        self.processed = None
        self.method_used = None
        self.otsu = None          # Kết quả phân đoạn Otsu
        self.diff_map = None      # Bản đồ khác biệt
        self.gt_mask = None       # Mặt nạ chuẩn Ground Truth (tùy chọn)
        self.view_mode = self.VIEW_MODES[0]
        self.batch_folder = None
        self.batch_results = []   # Danh sách kết quả thực nghiệm hàng loạt

        self._img_id = 0
        self._proc_id = 0
        self._orig_gray = None
        self._orig_hist = None
        self._orig_stats = None
        self._orig_lab = None
        self._proc_gray = None
        self._proc_hist = None
        self._proc_stats = None
        self._pair_metrics = None

        self.build_ui()
        self.apply_style()
        self.init_history_log()

    # ---------------- Xây dựng giao diện ----------------
    def _make_vsep(self):
        sep = QFrame()
        sep.setObjectName("vSep")
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setFixedHeight(26)
        return sep

    def _make_slider_spin(self, label_text: str, low: int, high: int, init_val: int):
        """Tạo cụm nhãn + QSlider + QSpinBox đồng bộ hai chiều."""
        lbl = QLabel(label_text)
        lbl.setObjectName("paramLabel")

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(init_val)
        slider.setFixedWidth(115)
        slider.setCursor(Qt.CursorShape.PointingHandCursor)

        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(init_val)
        spin.setFixedWidth(62)

        return lbl, slider, spin

    def _make_param_group(self, title: str, widgets: list):
        """Đóng khung nhóm tham số kèm tiêu đề nằm trên cạnh khung (giống khung chú thích kỹ thuật)."""
        box = QGroupBox(title)
        box.setObjectName("paramGroup")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(9, 4, 9, 4)
        lay.setSpacing(6)
        for i, w in enumerate(widgets):
            if i == 3:
                lay.addSpacing(6)
            lay.addWidget(w)
        return box

    def build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(8)
        self.setCentralWidget(central)

        # 1. Thanh công cụ ngang cao cấp (Card Toolbar gọn, rõ ràng)
        toolbar_card = QFrame()
        toolbar_card.setObjectName("toolbarCard")
        bar = QHBoxLayout(toolbar_card)
        bar.setContentsMargins(10, 6, 10, 6)
        bar.setSpacing(5)

        # Nhóm 1: Chọn ảnh / tệp (menu chọn riêng ảnh đơn hoặc thư mục nhiều ảnh)
        self.btn_open = QPushButton("Chọn ảnh/ tệp ")
        self.btn_open.setObjectName("primary")
        self.btn_open.setCursor(Qt.CursorShape.PointingHandCursor)
        open_menu = QMenu(self)
        act_open_img = open_menu.addAction("Chọn ảnh...")
        act_open_img.triggered.connect(self.open_dialog)
        act_open_dir = open_menu.addAction("Chọn thư mục ảnh (nhiều ảnh)...")
        act_open_dir.triggered.connect(self.choose_batch_folder)
        self.btn_open.setMenu(open_menu)

        self.btn_reset = QPushButton("Đặt lại")
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset)

        # Nhóm 2: Xử lý cốt lõi
        lbl_method = QLabel("Phương pháp:")
        lbl_method.setObjectName("toolbarLabel")

        self.combo_method = QComboBox()
        self.combo_method.addItems(self.METHODS)
        self.combo_method.setFixedWidth(182)
        self.combo_method.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_method.currentTextChanged.connect(self.on_method_changed)

        self.btn_apply = QPushButton("Nâng cao ảnh")
        self.btn_apply.setObjectName("primary")
        self.btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply.clicked.connect(self.apply_enhancement)

        self.btn_otsu = QPushButton("Phân đoạn Otsu")
        self.btn_otsu.setObjectName("accentTeal")
        self.btn_otsu.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_otsu.clicked.connect(self.apply_otsu)

        # Nhóm 3: Thống kê nhiều ảnh (chỉ sáng lên khi đã chọn thư mục tệp)
        self.btn_run_batch = QPushButton("Chạy thống kê")
        self.btn_run_batch.setObjectName("primary")
        self.btn_run_batch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_batch.clicked.connect(self.run_batch_benchmark)

        self.lbl_batch_filter = QLabel("Lọc:")
        self.lbl_batch_filter.setObjectName("toolbarLabel")
        self.combo_batch_filter = QComboBox()
        self.combo_batch_filter.addItems(["Tất cả ảnh", "Chỉ ảnh tốt hơn", "Chỉ ảnh xấu đi", "Chỉ ảnh tương đương"])
        self.combo_batch_filter.setFixedWidth(134)
        self.combo_batch_filter.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_batch_filter.currentTextChanged.connect(self.refresh_batch_table)

        self.lbl_batch_folder = QLabel("Nguồn: Chưa chọn")
        self.lbl_batch_folder.setObjectName("logPathBadge")

        # Nhóm 4: Xuất kết quả
        self.btn_save = QPushButton("Lưu ảnh...")
        self.btn_save.setObjectName("btnExport")
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.clicked.connect(self.save_results)

        self.btn_report = QPushButton("Xuất báo cáo...")
        self.btn_report.setObjectName("btnExport")
        self.btn_report.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_report.clicked.connect(self.export_report)

        bar.addWidget(self.btn_open)
        bar.addWidget(self.btn_reset)
        bar.addWidget(self._make_vsep())
        bar.addWidget(lbl_method)
        bar.addWidget(self.combo_method)
        bar.addWidget(self.btn_apply)
        bar.addWidget(self.btn_otsu)
        bar.addWidget(self._make_vsep())
        bar.addWidget(self.btn_run_batch)
        bar.addWidget(self.lbl_batch_filter)
        bar.addWidget(self.combo_batch_filter)
        bar.addWidget(self.lbl_batch_folder)
        bar.addStretch()
        bar.addWidget(self.btn_save)
        bar.addWidget(self.btn_report)
        root.addWidget(toolbar_card)

        # 1b. Thanh tham số Kéo giãn tương phản tùy chỉnh (hiện khi chọn 'Kéo giãn tương phản')
        self.stretch_card = QFrame()
        self.stretch_card.setObjectName("stretchParamCard")
        sbar = QHBoxLayout(self.stretch_card)
        sbar.setContentsMargins(10, 4, 10, 5)
        sbar.setSpacing(8)

        self.chk_batch_auto = QCheckBox("Tự động Min-Max\ntừng ảnh")
        self.chk_batch_auto.setChecked(True)
        self.chk_batch_auto.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_batch_auto.setVisible(False)
        sbar.addWidget(self.chk_batch_auto)

        lbl_rmin, self.sl_rmin, self.sp_rmin = self._make_slider_spin("r_min:", 0, 254, 0)
        lbl_rmax, self.sl_rmax, self.sp_rmax = self._make_slider_spin("r_max:", 1, 255, 255)
        lbl_smin, self.sl_smin, self.sp_smin = self._make_slider_spin("s_min:", 0, 254, 0)
        lbl_smax, self.sl_smax, self.sp_smax = self._make_slider_spin("s_max:", 1, 255, 255)
        self.sl_rmin.setFixedWidth(90)
        self.sl_rmax.setFixedWidth(90)
        self.sl_smin.setFixedWidth(72)
        self.sl_smax.setFixedWidth(72)

        grp_in = self._make_param_group(
            "Khoảng mức xám đầu vào",
            [lbl_rmin, self.sl_rmin, self.sp_rmin, lbl_rmax, self.sl_rmax, self.sp_rmax]
        )
        grp_out = self._make_param_group(
            "Khoảng mức xám đầu ra",
            [lbl_smin, self.sl_smin, self.sp_smin, lbl_smax, self.sl_smax, self.sp_smax]
        )

        sbar.addWidget(grp_in)
        sbar.addWidget(grp_out)
        sbar.addStretch()

        self.btn_auto_minmax = QPushButton("Lấy Min-Max ảnh")
        self.btn_auto_minmax.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_auto_minmax.clicked.connect(self.fill_auto_stretch_bounds)
        sbar.addWidget(self.btn_auto_minmax)

        self.btn_apply_stretch = QPushButton("Áp dụng kéo giãn")
        self.btn_apply_stretch.setObjectName("primary")
        self.btn_apply_stretch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply_stretch.clicked.connect(self.apply_enhancement)
        sbar.addWidget(self.btn_apply_stretch)

        # Đồng bộ 2 chiều Slider <-> SpinBox kèm ràng buộc min < max
        self.sl_rmin.valueChanged.connect(lambda v: self._sync_stretch_pair("rmin", v))
        self.sp_rmin.valueChanged.connect(lambda v: self._sync_stretch_pair("rmin", v))
        self.sl_rmax.valueChanged.connect(lambda v: self._sync_stretch_pair("rmax", v))
        self.sp_rmax.valueChanged.connect(lambda v: self._sync_stretch_pair("rmax", v))
        self.sl_smin.valueChanged.connect(lambda v: self._sync_stretch_pair("smin", v))
        self.sp_smin.valueChanged.connect(lambda v: self._sync_stretch_pair("smin", v))
        self.sl_smax.valueChanged.connect(lambda v: self._sync_stretch_pair("smax", v))
        self.sp_smax.valueChanged.connect(lambda v: self._sync_stretch_pair("smax", v))

        self.stretch_card.setVisible("Kéo giãn" in self.combo_method.currentText())
        root.addWidget(self.stretch_card)

        # 1c. Thanh tham số CLAHE (hiện khi chọn 'CLAHE (Cân bằng cục bộ)')
        self.clahe_card = QFrame()
        self.clahe_card.setObjectName("stretchParamCard")
        cbar = QHBoxLayout(self.clahe_card)
        cbar.setContentsMargins(10, 4, 10, 5)
        cbar.setSpacing(8)

        lbl_cl = QLabel("Clip:")
        lbl_cl.setObjectName("paramLabel")
        self.sl_clip = QSlider(Qt.Orientation.Horizontal)
        self.sl_clip.setRange(10, 100)   # 1.0 .. 10.0 (x10)
        self.sl_clip.setValue(20)
        self.sl_clip.setFixedWidth(120)
        self.sl_clip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sp_clip = QDoubleSpinBox()
        self.sp_clip.setRange(1.0, 10.0)
        self.sp_clip.setSingleStep(0.1)
        self.sp_clip.setDecimals(1)
        self.sp_clip.setValue(2.0)
        self.sp_clip.setFixedWidth(64)

        lbl_tile = QLabel("Lưới ô:")
        lbl_tile.setObjectName("paramLabel")
        self.sl_tile = QSlider(Qt.Orientation.Horizontal)
        self.sl_tile.setRange(2, 16)
        self.sl_tile.setValue(8)
        self.sl_tile.setFixedWidth(110)
        self.sl_tile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sp_tile = QSpinBox()
        self.sp_tile.setRange(2, 16)
        self.sp_tile.setValue(8)
        self.sp_tile.setSuffix("×N")
        self.sp_tile.setFixedWidth(68)

        grp_clip = self._make_param_group("Giới hạn tương phản (Clip Limit)", [lbl_cl, self.sl_clip, self.sp_clip])
        grp_tile = self._make_param_group("Kích thước lưới cục bộ (Tile Grid)", [lbl_tile, self.sl_tile, self.sp_tile])
        cbar.addWidget(grp_clip)
        cbar.addWidget(grp_tile)
        cbar.addStretch()

        btn_clahe_default = QPushButton("Mặc định (2.0 / 8×8)")
        btn_clahe_default.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clahe_default.clicked.connect(self.reset_clahe_defaults)
        cbar.addWidget(btn_clahe_default)

        btn_apply_clahe = QPushButton("Áp dụng CLAHE")
        btn_apply_clahe.setObjectName("primary")
        btn_apply_clahe.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_apply_clahe.clicked.connect(self.apply_enhancement)
        cbar.addWidget(btn_apply_clahe)

        self.sl_clip.valueChanged.connect(lambda v: self._sync_clahe_clip(v / 10.0))
        self.sp_clip.valueChanged.connect(lambda v: self._sync_clahe_clip(v))
        self.sl_tile.valueChanged.connect(lambda v: self._sync_clahe_tile(v))
        self.sp_tile.valueChanged.connect(lambda v: self._sync_clahe_tile(v))

        self.clahe_card.setVisible("CLAHE" in self.combo_method.currentText())
        root.addWidget(self.clahe_card)

        # 1d. Thanh tham số Gamma (hiện khi chọn 'Biến đổi Gamma (γ)')
        self.gamma_card = QFrame()
        self.gamma_card.setObjectName("stretchParamCard")
        gbar = QHBoxLayout(self.gamma_card)
        gbar.setContentsMargins(10, 4, 10, 5)
        gbar.setSpacing(8)

        lbl_g = QLabel("γ:")
        lbl_g.setObjectName("paramLabel")
        self.sl_gamma = QSlider(Qt.Orientation.Horizontal)
        self.sl_gamma.setRange(20, 350)  # 0.20 .. 3.50 (x100)
        self.sl_gamma.setValue(70)
        self.sl_gamma.setFixedWidth(160)
        self.sl_gamma.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sp_gamma = QDoubleSpinBox()
        self.sp_gamma.setRange(0.20, 3.50)
        self.sp_gamma.setSingleStep(0.05)
        self.sp_gamma.setDecimals(2)
        self.sp_gamma.setValue(0.70)
        self.sp_gamma.setFixedWidth(70)

        grp_gamma = self._make_param_group(
            "Hệ số Gamma — s = 255·(r/255)^γ  (<1: tăng sáng, >1: giảm chói)",
            [lbl_g, self.sl_gamma, self.sp_gamma]
        )
        gbar.addWidget(grp_gamma)

        for label_p, g_val in (("Tăng sáng (0.6)", 0.60), ("Gốc (1.0)", 1.00), ("Giảm chói (1.6)", 1.60)):
            bp = QPushButton(label_p)
            bp.setCursor(Qt.CursorShape.PointingHandCursor)
            bp.clicked.connect(lambda _checked=False, gv=g_val: self._sync_gamma(gv))
            gbar.addWidget(bp)

        gbar.addStretch()
        btn_apply_gamma = QPushButton("Áp dụng Gamma")
        btn_apply_gamma.setObjectName("primary")
        btn_apply_gamma.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_apply_gamma.clicked.connect(self.apply_enhancement)
        gbar.addWidget(btn_apply_gamma)

        self.sl_gamma.valueChanged.connect(lambda v: self._sync_gamma(v / 100.0))
        self.sp_gamma.valueChanged.connect(lambda v: self._sync_gamma(v))

        self.gamma_card.setVisible("Gamma" in self.combo_method.currentText())
        root.addWidget(self.gamma_card)

        self.btn_save.setEnabled(False)
        self.btn_report.setEnabled(False)
        self._set_batch_controls_enabled(False)

        # 2. Phân chia giao diện: Phần trên (2 ảnh) & Phần dưới (Tabs phân tích)
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)

        # Phần trên: So sánh Ảnh gốc vs Ảnh sau nâng cao
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.setHandleWidth(10)

        self.canvas_orig = ImageCanvas(
            placeholder="Kéo thả ảnh vào đây để bắt đầu",
            subtext="Hoặc nhấn 'Chọn ảnh/ tệp' (Hỗ trợ đa định dạng: PNG, JPG, WEBP, JFIF, BMP, TIFF, GIF...)",
            accept_drop=True
        )
        self.canvas_proc = ImageCanvas(
            placeholder="Chưa áp dụng nâng cao chất lượng",
            subtext="Chọn phương pháp ở thanh công cụ rồi nhấn 'Nâng cao ảnh'",
            accept_drop=False
        )
        self.canvas_orig.file_dropped.connect(self.load_image)
        self.canvas_orig.double_clicked.connect(lambda: self.zoom("Ảnh gốc", self.original))
        self.canvas_proc.double_clicked.connect(lambda: self.zoom("Ảnh sau nâng cao", self.processed))

        # Di chuyển ô 'Xem ảnh xám' xuống header của từng khung Ảnh gốc & Ảnh sau nâng cao
        self.chk_gray_orig = QCheckBox("Xem ảnh xám")
        self.chk_gray_orig.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_gray_orig.toggled.connect(self.toggle_gray_orig)

        self.chk_gray_proc = QCheckBox("Xem ảnh xám")
        self.chk_gray_proc.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_gray_proc.toggled.connect(self.toggle_gray_proc)

        self.badge_orig = QLabel("Chưa nạp ảnh")
        self.badge_proc = QLabel("Đang chờ xử lý")
        self.lbl_orig = QLabel("Thông số mức xám sẽ hiển thị sau khi nạp ảnh")
        self.lbl_proc = QLabel("Thông số so sánh sẽ hiển thị sau khi xử lý")

        top_splitter.addWidget(
            self.create_card("Ảnh gốc", self.canvas_orig, self.lbl_orig, self.badge_orig,
                             header_extra=self.chk_gray_orig)
        )
        top_splitter.addWidget(
            self.create_card("Ảnh sau nâng cao", self.canvas_proc, self.lbl_proc, self.badge_proc,
                             header_extra=self.chk_gray_proc)
        )
        self.main_splitter.addWidget(top_splitter)

        # Phần dưới: Hệ thống 4 Tab phân tích chuyên sâu
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        self.tabs = QTabWidget()

        # Tab 1: Histogram
        t1 = QWidget()
        l1 = QHBoxLayout(t1)
        l1.setContentsMargins(10, 10, 10, 10)
        l1.setSpacing(12)
        self.plot_h_orig = PlotCanvas()
        self.plot_h_proc = PlotCanvas()
        l1.addWidget(self.plot_h_orig)
        l1.addWidget(self.plot_h_proc)
        self.tabs.addTab(t1, "Biểu đồ Histogram")

        # Tab 2: Phân đoạn Otsu & So sánh chi tiết
        t2 = QWidget()
        l2_main = QVBoxLayout(t2)
        l2_main.setContentsMargins(10, 8, 10, 10)
        l2_main.setSpacing(8)

        # Thanh điều khiển nhỏ trong Tab Otsu
        sub_bar = QHBoxLayout()
        sub_bar.setSpacing(8)
        lbl_view = QLabel("Chế độ hiển thị:")
        lbl_view.setObjectName("toolbarLabel")
        sub_bar.addWidget(lbl_view)

        self.combo_view = QComboBox()
        self.combo_view.addItems(self.VIEW_MODES)
        self.combo_view.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.combo_view.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_view.currentTextChanged.connect(self.on_view_mode_changed)
        sub_bar.addWidget(self.combo_view)

        sub_bar.addSpacing(8)
        self.btn_load_gt = QPushButton("Nạp Ground Truth...")
        self.btn_load_gt.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_gt.clicked.connect(self.load_ground_truth)
        sub_bar.addWidget(self.btn_load_gt)

        self.btn_clear_gt = QPushButton("Xóa GT")
        self.btn_clear_gt.setVisible(False)
        self.btn_clear_gt.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_gt.clicked.connect(self.clear_ground_truth)
        sub_bar.addWidget(self.btn_clear_gt)

        sub_bar.addStretch()
        self.lbl_otsu_summary = QLabel("Chưa thực hiện phân đoạn Otsu")
        self.lbl_otsu_summary.setObjectName("otsuSummaryPill")
        sub_bar.addWidget(self.lbl_otsu_summary)
        l2_main.addLayout(sub_bar)

        # 4 khung hiển thị kết quả phân đoạn & so sánh
        otsu_splitter = QSplitter(Qt.Orientation.Horizontal)
        otsu_splitter.setChildrenCollapsible(False)
        otsu_splitter.setHandleWidth(8)

        self.canvas_mask_orig = ImageCanvas("Chưa phân đoạn")
        self.canvas_mask_proc = ImageCanvas("Chưa phân đoạn")
        self.canvas_diff = ImageCanvas("Chưa có bản đồ sai khác")

        self.canvas_mask_orig.double_clicked.connect(
            lambda: self.zoom("Phân đoạn - Ảnh gốc", self.get_display_image("orig")))
        self.canvas_mask_proc.double_clicked.connect(
            lambda: self.zoom("Phân đoạn - Sau nâng cao", self.get_display_image("proc")))
        self.canvas_diff.double_clicked.connect(
            lambda: self.zoom("Bản đồ khác biệt Otsu", self.diff_map))

        self.lbl_mask_orig = QLabel("")
        self.lbl_mask_proc = QLabel("")
        self.lbl_diff = QLabel("")

        self.plot_var = PlotCanvas()

        otsu_splitter.addWidget(self.create_card("Otsu - Ảnh gốc", self.canvas_mask_orig, self.lbl_mask_orig, compact=True))
        otsu_splitter.addWidget(self.create_card("Otsu - Sau nâng cao", self.canvas_mask_proc, self.lbl_mask_proc, compact=True))
        otsu_splitter.addWidget(self.create_card("Bản đồ sai khác phân đoạn", self.canvas_diff, self.lbl_diff, compact=True))
        otsu_splitter.addWidget(self.create_card("Đường cong phương sai σ²(t)", self.plot_var, QLabel("Ngưỡng tối ưu max σ²_B(t)"), compact=True))
        otsu_splitter.setSizes([280, 280, 280, 380])

        l2_main.addWidget(otsu_splitter, 1)
        self.tabs.addTab(t2, "Phân đoạn Otsu && So sánh")

        # Tab 3: Bảng so sánh chi tiết (Bảng 1 & Bảng 2 ở trên + Bảng thống kê nhiều ảnh tự hiện ở dưới)
        t3 = QWidget()
        l3 = QVBoxLayout(t3)
        l3.setContentsMargins(10, 8, 10, 10)
        l3.setSpacing(8)

        self.t3_vsplitter = QSplitter(Qt.Orientation.Vertical)
        self.t3_vsplitter.setChildrenCollapsible(False)
        self.t3_vsplitter.setHandleWidth(8)

        t3_splitter = QSplitter(Qt.Orientation.Horizontal)
        t3_splitter.setChildrenCollapsible(False)
        t3_splitter.setHandleWidth(10)

        # Bảng 1: Bảng thống kê chỉ số đánh giá trọng tâm tạo sự khác biệt sau nâng cao
        self.table_key = QTableWidget(0, 5)
        self.table_key.setHorizontalHeaderLabels([
            "Chỉ số trọng tâm", "Ảnh gốc", "Sau nâng cao", "Chênh lệch / Giá trị", "Đánh giá khác biệt"
        ])
        self.table_key.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table_key.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_key.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_key.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_key.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table_key.setColumnWidth(0, 275)
        self.table_key.setItemDelegateForColumn(0, MetricSubtitleDelegate(self.table_key))
        self.table_key.verticalHeader().setVisible(False)
        self.table_key.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_key.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_key.setAlternatingRowColors(True)
        self.table_key.setShowGrid(False)

        self.badge_key_verdict = QLabel("Chưa đánh giá")

        # Bảng 2: Bảng thống kê phân bố mức xám & phân đoạn đầy đủ
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            "Chỉ số chi tiết", "Ảnh gốc", "Sau nâng cao", "Mức độ thay đổi (Δ)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 248)
        self.table.setItemDelegateForColumn(0, MetricSubtitleDelegate(self.table))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)

        t3_splitter.addWidget(
            self.create_card("Bảng 1: Chỉ số trọng tâm tạo sự khác biệt", self.table_key,
                             badge=self.badge_key_verdict, compact=True)
        )
        t3_splitter.addWidget(
            self.create_card("Bảng 2: Thông số phân bố & phân đoạn chi tiết", self.table, compact=True)
        )
        t3_splitter.setSizes([780, 480])
        self.t3_vsplitter.addWidget(t3_splitter)

        # Khối thống kê thực nghiệm nhiều ảnh (ẩn mặc định, tự xuất hiện dưới Bảng 1 & Bảng 2 khi ấn 'Chạy thống kê')
        self.batch_panel = QFrame()
        self.batch_panel.setObjectName("cardCompact")
        bp_lay = QVBoxLayout(self.batch_panel)
        bp_lay.setContentsMargins(8, 6, 8, 8)
        bp_lay.setSpacing(6)

        bp_hdr = QHBoxLayout()
        bp_hdr.setSpacing(8)
        lbl_bp_title = QLabel("Bảng 3: Thống kê thực nghiệm nhiều ảnh")
        lbl_bp_title.setObjectName("cardTitleCompact")
        bp_hdr.addWidget(lbl_bp_title)

        self.batch_progress = QProgressBar()
        self.batch_progress.setFixedHeight(16)
        self.batch_progress.setFixedWidth(180)
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("%v/%m (%p%)")
        self.batch_progress.setVisible(False)
        bp_hdr.addWidget(self.batch_progress)

        bp_hdr.addStretch()
        self.btn_batch_charts = QPushButton("Biểu đồ thống kê...")
        self.btn_batch_charts.setObjectName("accentTeal")
        self.btn_batch_charts.setEnabled(False)
        self.btn_batch_charts.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_batch_charts.clicked.connect(self.open_batch_analytics)
        bp_hdr.addWidget(self.btn_batch_charts)

        self.btn_export_batch = QPushButton("Xuất CSV...")
        self.btn_export_batch.setObjectName("btnExport")
        self.btn_export_batch.setEnabled(False)
        self.btn_export_batch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_batch.clicked.connect(self.export_batch_csv)
        bp_hdr.addWidget(self.btn_export_batch)
        bp_lay.addLayout(bp_hdr)

        # 4 thẻ tóm tắt kết quả thực nghiệm toàn tập (trung bình + số lượng tốt hơn / xấu đi)
        bkpi_row = QHBoxLayout()
        bkpi_row.setSpacing(8)
        self.bkpi_count_val, self.bkpi_count_sub, bk1 = self.create_kpi_card(
            "PHÂN LOẠI TỐT HƠN / XẤU ĐI", "#0F766E", compact=True,
            desc="Đánh giá cải thiện / suy giảm chất lượng"
        )
        self.bkpi_mean_val, self.bkpi_mean_sub, bk2 = self.create_kpi_card(
            "TRUNG BÌNH MỨC XÁM (μ)", "#0F172A", compact=True,
            desc="Độ sáng trung bình toàn bộ tập ảnh"
        )
        self.bkpi_std_val, self.bkpi_std_sub, bk3 = self.create_kpi_card(
            "TRUNG BÌNH TƯƠNG PHẢN (σ)", "#1B56B5", compact=True,
            desc="Độ phân tán sáng – tối trung bình"
        )
        self.bkpi_qual_val, self.bkpi_qual_sub, bk4 = self.create_kpi_card(
            "ENTROPY (H) & SSIM", "#B45309", compact=True,
            desc="Độ giàu chi tiết & bảo toàn cấu trúc"
        )
        for bk in (bk1, bk2, bk3, bk4):
            bkpi_row.addWidget(bk, 1)
        bp_lay.addLayout(bkpi_row)

        # Bảng danh sách kết quả từng ảnh (nhấp đúp để xem ảnh trực tiếp phía trên)
        self.table_batch = QTableWidget(0, 10)
        self.table_batch.setHorizontalHeaderLabels([
            "STT", "Tên ảnh",
            "TB xám (μ)\nĐộ sáng",
            "Tương phản (σ)\nTách sáng–tối",
            "Entropy (H)\nGiàu chi tiết",
            "Sắc nét\nRõ biên Δ%",
            "Bão hòa\nCháy/bệt Δ%",
            "SSIM\nGiữ cấu trúc",
            "Kết luận", "Lý do phân loại\n(Nhấp đúp để xem ảnh)"
        ])
        b_hdr = self.table_batch.horizontalHeader()
        for col_idx in (0, 2, 3, 4, 5, 6, 7, 8):
            b_hdr.setSectionResizeMode(col_idx, QHeaderView.ResizeMode.ResizeToContents)
        b_hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        b_hdr.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        self.table_batch.setColumnWidth(1, 155)
        self.table_batch.verticalHeader().setVisible(False)
        self.table_batch.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_batch.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_batch.setAlternatingRowColors(True)
        self.table_batch.setShowGrid(False)
        self.table_batch.cellDoubleClicked.connect(self.on_batch_row_double_clicked)
        bp_lay.addWidget(self.table_batch, 1)

        self.batch_panel.setVisible(False)
        self.t3_vsplitter.addWidget(self.batch_panel)
        l3.addWidget(self.t3_vsplitter, 1)
        self.tabs.addTab(t3, "Bảng so sánh chi tiết")

        # Tab 4: Nhật ký & Lịch sử thao tác (Ghi bền vững vào file log theo ngày)
        t4 = QWidget()
        l4 = QVBoxLayout(t4)
        l4.setContentsMargins(10, 8, 10, 10)
        l4.setSpacing(8)

        log_bar = QHBoxLayout()
        self.lbl_log_info = QLabel("Đang kết nối tệp nhật ký...")
        self.lbl_log_info.setObjectName("logPathBadge")

        self.btn_reload_log = QPushButton("Tải lại lịch sử")
        self.btn_reload_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reload_log.clicked.connect(self.load_today_log_to_ui)

        self.btn_open_log_dir = QPushButton("Mở thư mục Log")
        self.btn_open_log_dir.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_log_dir.clicked.connect(self.open_log_directory)

        self.btn_clear_log_view = QPushButton("Xóa khung xem")
        self.btn_clear_log_view.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_log_view.clicked.connect(self.log_box_clear_view)

        log_bar.addWidget(self.lbl_log_info)
        log_bar.addStretch()
        log_bar.addWidget(self.btn_reload_log)
        log_bar.addWidget(self.btn_open_log_dir)
        log_bar.addWidget(self.btn_clear_log_view)
        l4.addLayout(log_bar)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("logBox")
        l4.addWidget(self.log_box)

        self.tabs.addTab(t4, "Nhật ký xử lý (Lịch sử)")

        bottom_layout.addWidget(self.tabs, 1)
        self.main_splitter.addWidget(bottom_container)
        self.main_splitter.setSizes([380, 410])
        root.addWidget(self.main_splitter, 1)

        self.statusBar().showMessage("Sẵn sàng — Nhấn 'Chọn ảnh/ tệp' hoặc kéo thả tệp ảnh vào khung bên trái.")
        self.refresh_key_table()
        self.refresh_table()
        self.plot_h_orig.draw_histogram(None, "Histogram ảnh gốc")
        self.plot_h_proc.draw_histogram(None, "Histogram ảnh sau nâng cao")
        self.plot_var.draw_variance(None, None, None, None)

        # MenuBar
        m = self.menuBar().addMenu("Tệp")
        for text, slot, key in (
            ("Mở ảnh...", self.open_dialog, "Ctrl+O"),
            ("Mở thư mục ảnh...", self.choose_batch_folder, "Ctrl+Shift+O"),
            ("Lưu ảnh...", self.save_results, "Ctrl+S"),
            ("Xuất báo cáo...", self.export_report, None),
            ("Thoát", self.close, None)
        ):
            a = QAction(text, self)
            if key:
                a.setShortcut(key)
            a.triggered.connect(slot)
            m.addAction(a)

    def create_card(self, title, widget, label=None, badge=None, compact=False, header_extra=None):
        card = QFrame()
        card.setObjectName("cardCompact" if compact else "card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Header bar nổi bật bên trong card
        header = QFrame()
        header.setObjectName("cardHeader")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(12 if not compact else 10, 7 if not compact else 5,
                                 12 if not compact else 10, 7 if not compact else 5)
        h_lay.setSpacing(8)

        lbl_title = QLabel(title)
        lbl_title.setObjectName("cardTitleCompact" if compact else "cardTitle")
        h_lay.addWidget(lbl_title)
        h_lay.addStretch()

        if header_extra is not None:
            h_lay.addWidget(header_extra)

        if badge is not None:
            badge.setObjectName("cardBadge")
            h_lay.addWidget(badge)

        lay.addWidget(header)

        # Nội dung chính (Canvas / Plot)
        body = QWidget()
        b_lay = QVBoxLayout(body)
        pad = 8 if not compact else 6
        b_lay.setContentsMargins(pad, pad, pad, pad)
        b_lay.setSpacing(0)
        b_lay.addWidget(widget, 1)
        lay.addWidget(body, 1)

        # Footer hiển thị thông số đậm rõ
        if label is not None:
            footer = QFrame()
            footer.setObjectName("cardFooter")
            f_lay = QHBoxLayout(footer)
            f_lay.setContentsMargins(10, 5 if not compact else 4, 10, 5 if not compact else 4)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setObjectName("cardFooterTextCompact" if compact else "cardFooterText")
            f_lay.addWidget(label)
            lay.addWidget(footer)

        return card

    def create_kpi_card(self, title: str, accent_color: str, compact: bool = False, desc: str = ""):
        card = QFrame()
        card.setObjectName("kpiCard")
        card.setFixedHeight(74 if (compact and desc) else (64 if compact else 76))
        card.setStyleSheet("""
            QFrame#kpiCard {
                background: #FFFFFF;
                border: 1px solid #D8E0EA;
                border-radius: 9px;
            }
        """)
        outer = QHBoxLayout(card)
        outer.setContentsMargins(10, 5 if compact else 8, 12, 5 if compact else 8)
        outer.setSpacing(10)

        accent_bar = QFrame()
        accent_bar.setFixedWidth(4)
        accent_bar.setStyleSheet(f"background: {accent_color}; border: none; border-radius: 2px;")
        outer.addWidget(accent_bar)

        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0 if desc else 1)

        title_pt = "7.6pt" if compact else "8pt"
        val_pt = "11.2pt" if compact else "13.5pt"
        sub_pt = "7.8pt" if compact else "8.5pt"

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"font-size: {title_pt}; font-weight: 700; color: #334155; letter-spacing: 0.2px;")
        lay.addWidget(lbl_title)

        if desc:
            lbl_desc = QLabel(desc)
            lbl_desc.setStyleSheet("font-size: 7.3pt; font-weight: 400; color: #64748B;")
            lay.addWidget(lbl_desc)

        lbl_val = QLabel("—")
        lbl_val.setStyleSheet(f"font-size: {val_pt}; font-weight: 800; color: {accent_color};")

        lbl_sub = QLabel("Chưa có dữ liệu")
        lbl_sub.setStyleSheet(f"font-size: {sub_pt}; font-weight: 500; color: #64748B;")

        lay.addWidget(lbl_val)
        lay.addWidget(lbl_sub)
        outer.addLayout(lay, 1)
        return lbl_val, lbl_sub, card

    def apply_style(self):
        self.setStyleSheet(f"""
            QWidget {{
                font-family: 'Segoe UI', 'Tahoma', sans-serif;
                font-size: 9.5pt;
                color: #0F172A;
            }}
            QMainWindow {{
                background: #F4F6F9;
            }}
            QMenuBar {{
                background: #FFFFFF;
                border-bottom: 1px solid #DCE3EC;
                font-weight: 600;
                color: #1E293B;
                padding: 2px 6px;
            }}
            QMenuBar::item {{
                padding: 5px 12px;
                border-radius: 6px;
            }}
            QMenuBar::item:selected {{
                background: #EEF2F8;
                color: #1B56B5;
            }}
            QStatusBar {{
                background: #FFFFFF;
                border-top: 1px solid #DCE3EC;
                color: #334155;
                font-size: 9pt;
                font-weight: 500;
                padding: 2px 8px;
            }}

            /* Thanh công cụ chính dạng Card */
            QFrame#toolbarCard {{
                background: #FFFFFF;
                border: 1px solid #D8E0EA;
                border-radius: 10px;
            }}
            QFrame#vSep {{
                background: #E2E8F0;
                border: none;
            }}
            QLabel#toolbarLabel {{
                font-size: 9.5pt;
                font-weight: 700;
                color: #1E293B;
            }}

            /* Thanh tham số kéo giãn tương phản tùy chỉnh */
            QFrame#stretchParamCard {{
                background: #F8FAFD;
                border: 1px solid #C8D9F0;
                border-radius: 9px;
            }}
            QGroupBox#paramGroup {{
                background: transparent;
                border: 1.2px solid #93B5E1;
                border-radius: 7px;
                margin-top: 8px;
                padding-top: 3px;
                font-size: 8pt;
                font-weight: 700;
                color: #1E3A8A;
            }}
            QGroupBox#paramGroup::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                background: #F8FAFD;
                color: #1E3A8A;
            }}
            QLabel#paramLabel {{
                font-size: 9pt;
                font-weight: 700;
                color: #1E3A8A;
            }}
            QSpinBox, QDoubleSpinBox {{
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 3px 6px;
                font-size: 9.5pt;
                font-weight: 700;
                color: #0F172A;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: #1B56B5;
            }}
            QSlider::groove:horizontal {{
                height: 5px;
                background: #DCE4F0;
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: #1B56B5;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: #1B56B5;
                border: 2px solid #FFFFFF;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #144493;
            }}
            QProgressBar {{
                background: #E2E8F0;
                border: none;
                border-radius: 6px;
                text-align: center;
                font-size: 8.5pt;
                font-weight: 700;
                color: #0F172A;
            }}
            QProgressBar::chunk {{
                background: #1B56B5;
                border-radius: 6px;
            }}

            /* Thẻ hiển thị ảnh (Card hiện đại) */
            QFrame#card, QFrame#cardCompact {{
                background: #FFFFFF;
                border: 1px solid #D8E0EA;
                border-radius: 10px;
            }}
            QFrame#cardHeader {{
                background: #FFFFFF;
                border-bottom: 1px solid #EDF2F7;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }}
            QLabel#cardTitle {{
                font-size: 11.5pt;
                font-weight: 700;
                color: #0F172A;
            }}
            QLabel#cardTitleCompact {{
                font-size: 10pt;
                font-weight: 700;
                color: #0F172A;
            }}
            QLabel#cardBadge {{
                background: #EFF4FB;
                color: #1E40AF;
                border: 1px solid #D5E3F8;
                border-radius: 6px;
                padding: 2px 9px;
                font-size: 8.5pt;
                font-weight: 700;
            }}
            QFrame#cardFooter {{
                background: #F8FAFC;
                border-top: 1px solid #EDF2F7;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
            }}
            QLabel#cardFooterText {{
                font-size: 9.5pt;
                font-weight: 600;
                color: #1E293B;
            }}
            QLabel#cardFooterTextCompact {{
                font-size: 8.5pt;
                font-weight: 600;
                color: #334155;
            }}

            /* Hệ thống nút bấm phân cấp rõ ràng */
            QPushButton {{
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                padding: 6px 10px;
                color: #1E293B;
                font-size: 9.2pt;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: #F8FAFC;
                border-color: #94A3B8;
                color: #0F172A;
            }}
            QPushButton:pressed {{
                background: #EDF2F7;
            }}
            QPushButton:disabled {{
                color: #94A3B8;
                background: #F8FAFC;
                border-color: #E2E8F0;
            }}

            /* Nút Chọn ảnh - nổi bật tinh tế */
            QPushButton#btnOpen {{
                background: #EFF6FF;
                color: #1B56B5;
                border: 1.5px solid #93C5FD;
                font-size: 10pt;
                font-weight: 700;
                padding: 7px 16px;
            }}
            QPushButton#btnOpen:hover {{
                background: #DBEAFE;
                border-color: #60A5FA;
                color: #144493;
            }}
            QPushButton#btnOpen:pressed {{
                background: #BFDBFE;
            }}

            /* Nút hành động chính: Nâng cao ảnh */
            QPushButton#primary {{
                background: #1B56B5;
                color: #FFFFFF;
                border: 1px solid #164899;
                font-size: 9.2pt;
                font-weight: 700;
                padding: 6px 12px;
            }}
            QPushButton#primary:hover {{
                background: #144493;
            }}
            QPushButton#primary:pressed {{
                background: #0F3573;
            }}
            QPushButton#primary:disabled {{
                background: #DCE5F2;
                color: #8A9BB3;
                border: 1px solid #CBD5E1;
            }}
            QPushButton::menu-indicator {{
                width: 12px;
                subcontrol-position: right center;
                subcontrol-origin: padding;
                right: 6px;
            }}
            QMenu {{
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                padding: 4px;
                font-size: 9.5pt;
                font-weight: 600;
                color: #0F172A;
            }}
            QMenu::item {{
                padding: 7px 18px;
                border-radius: 5px;
            }}
            QMenu::item:selected {{
                background: #EFF6FF;
                color: #1B56B5;
            }}

            /* Nút hành động chính thứ 2: Phân đoạn Otsu */
            QPushButton#accentTeal {{
                background: #0F766E;
                color: #FFFFFF;
                border: 1px solid #0C5F59;
                font-size: 9.2pt;
                font-weight: 700;
                padding: 6px 12px;
            }}
            QPushButton#accentTeal:hover {{
                background: #0D625C;
            }}
            QPushButton#accentTeal:pressed {{
                background: #0A4D48;
            }}

            /* Nút Xuất kết quả */
            QPushButton#btnExport {{
                background: #F8FAFC;
                border: 1px solid #CBD5E1;
                color: #0F172A;
                font-weight: 600;
            }}
            QPushButton#btnExport:hover {{
                background: #EEF2F8;
                border-color: #94A3B8;
            }}

            /* ComboBox & CheckBox */
            QComboBox {{
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                padding: 5px 8px;
                min-width: 110px;
                font-size: 9.2pt;
                font-weight: 600;
                color: #0F172A;
            }}
            QComboBox:hover {{
                border-color: #64748B;
            }}
            QComboBox:disabled {{
                background: #F1F5F9;
                color: #94A3B8;
                border-color: #E2E8F0;
            }}
            QLabel#toolbarLabel:disabled {{
                color: #94A3B8;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox::down-arrow {{
                image: url({ARROW_PATH});
                width: 14px;
                height: 14px;
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                selection-background-color: #EFF6FF;
                selection-color: #1B56B5;
                background: #FFFFFF;
                padding: 4px;
                font-weight: 600;
            }}
            QCheckBox {{
                spacing: 7px;
                font-size: 9.5pt;
                font-weight: 600;
                color: #1E293B;
            }}

            /* Nhãn tóm tắt Otsu & Log */
            QLabel#otsuSummaryPill {{
                background: #EFF6FF;
                color: #0F172A;
                border: 1px solid #BFDBFE;
                padding: 5px 14px;
                border-radius: 7px;
                font-size: 9.5pt;
                font-weight: 700;
            }}
            QLabel#logPathBadge {{
                background: #F1F5F9;
                color: #1E293B;
                border: 1px solid #DCE3EC;
                padding: 5px 10px;
                border-radius: 6px;
                font-size: 9pt;
                font-weight: 600;
            }}
            QLabel#logPathBadge:disabled {{
                background: #F8FAFC;
                color: #94A3B8;
                border: 1px solid #E2E8F0;
            }}

            /* Hệ thống Tab */
            QTabWidget::pane {{
                background: #FFFFFF;
                border: 1px solid #D8E0EA;
                border-radius: 10px;
                border-top-left-radius: 0px;
                top: -1px;
            }}
            QTabBar::tab {{
                background: #E9EEF5;
                color: #475569;
                padding: 8px 20px;
                margin-right: 4px;
                border: 1px solid transparent;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 10pt;
                font-weight: 600;
            }}
            QTabBar::tab:hover {{
                background: #DFE7F2;
                color: #0F172A;
            }}
            QTabBar::tab:selected {{
                background: #FFFFFF;
                color: #1B56B5;
                font-weight: 700;
                border: 1px solid #D8E0EA;
                border-top: 3px solid #1B56B5;
                border-bottom-color: #FFFFFF;
            }}

            /* Bảng so sánh chi tiết */
            QTableWidget {{
                background: #FFFFFF;
                border: none;
                gridline-color: #E2E8F0;
                alternate-background-color: #F8FAFC;
                selection-background-color: #EFF6FF;
                selection-color: #0F172A;
                font-size: 9.5pt;
            }}
            QTableWidget::item {{
                padding: 6px 10px;
                border-bottom: 1px solid #F1F5F9;
            }}
            QHeaderView::section {{
                background: #F1F5F9;
                border: none;
                border-bottom: 2px solid #CBD5E1;
                padding: 9px 10px;
                font-size: 10pt;
                font-weight: 700;
                color: #0F172A;
            }}

            /* Nhật ký xử lý */
            QPlainTextEdit#logBox {{
                background: #FAFBFD;
                border: 1px solid #DCE3EC;
                border-radius: 8px;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9.5pt;
                color: #1E293B;
            }}

            /* Splitter & Scrollbar */
            QSplitter::handle {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 9px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: #CBD5E1;
                border-radius: 4px;
                min-height: 28px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #94A3B8;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                height: 0;
                width: 0;
            }}
        """)

    # ---------------- Quản lý tệp nhật ký & Lịch sử thao tác ----------------
    def get_log_file(self) -> Path:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now().strftime("%Y-%m-%d")
        return LOGS_DIR / f"history_{today_str}.log"

    def log(self, text: str, write_to_file: bool = True):
        now = datetime.now()
        timestamp_ui = now.strftime("%H:%M:%S")
        timestamp_file = now.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Hiển thị lên giao diện
        ui_line = f"[{timestamp_ui}] {text}"
        self.log_box.appendPlainText(ui_line)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

        # 2. Ghi bền vững vào file log theo ngày (history_YYYY-MM-DD.log)
        if write_to_file:
            log_file = self.get_log_file()
            file_line = f"[{timestamp_file}] {text}\n"
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(file_line)
            except Exception:
                pass

    def init_history_log(self):
        log_file = self.get_log_file()
        try:
            rel_name = log_file.relative_to(BASE_DIR).as_posix()
        except Exception:
            rel_name = log_file.name
        self.lbl_log_info.setText(f"Tệp nhật ký hôm nay: {rel_name}")

        # Tải lại các thao tác đã lưu trước đó trong ngày (nếu có)
        if log_file.exists():
            try:
                content = log_file.read_text(encoding="utf-8").strip()
                if content:
                    self.log_box.appendPlainText("=== [LỊCH SỬ THAO TÁC ĐÃ LƯU TRONG NGÀY] ===")
                    for line in content.splitlines():
                        self.log_box.appendPlainText(line)
                    self.log_box.appendPlainText(f"=== [BẮT ĐẦU PHIÊN LÀM VIỆC MỚI: {datetime.now().strftime('%H:%M:%S')}] ===")
            except Exception:
                pass

        self.log("Khởi động hệ thống xử lý ảnh Đề tài 13.")

    def open_log_directory(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(LOGS_DIR)
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(LOGS_DIR)])

    def load_today_log_to_ui(self):
        log_file = self.get_log_file()
        if not log_file.exists():
            QMessageBox.information(self, "Thông báo", "Chưa có tệp nhật ký nào cho ngày hôm nay.")
            return
        try:
            content = log_file.read_text(encoding="utf-8").strip()
            self.log_box.clear()
            self.log_box.setPlainText(content)
            self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
            line_count = len(content.splitlines()) if content else 0
            self.statusBar().showMessage(f"Đã tải {line_count} dòng lịch sử từ {log_file.name}")
        except Exception as exc:
            QMessageBox.warning(self, "Lỗi", f"Không thể đọc tệp log: {exc}")

    def log_box_clear_view(self):
        self.log_box.clear()
        self.log_box.appendPlainText("--- [Khung hiển thị đã được làm trống. Dữ liệu trên file log vẫn được bảo toàn] ---")

    def closeEvent(self, event):
        self.log("Đóng ứng dụng và kết thúc phiên làm việc.")
        super().closeEvent(event)

    # ---------------- Tham số Kéo giãn / CLAHE / Gamma tùy chỉnh ----------------
    def on_method_changed(self, text: str):
        is_stretch = ("Kéo giãn" in text)
        is_clahe = ("CLAHE" in text)
        is_gamma = ("Gamma" in text)
        self.stretch_card.setVisible(is_stretch)
        self.clahe_card.setVisible(is_clahe)
        self.gamma_card.setVisible(is_gamma)
        if is_stretch and self.original is not None:
            self.fill_auto_stretch_bounds(silent=True)

    def _sync_clahe_clip(self, val: float):
        val = max(1.0, min(10.0, round(float(val), 1)))
        for w in (self.sl_clip, self.sp_clip):
            w.blockSignals(True)
        try:
            self.sl_clip.setValue(int(round(val * 10)))
            self.sp_clip.setValue(val)
        finally:
            for w in (self.sl_clip, self.sp_clip):
                w.blockSignals(False)

    def _sync_clahe_tile(self, val: int):
        val = max(2, min(16, int(val)))
        for w in (self.sl_tile, self.sp_tile):
            w.blockSignals(True)
        try:
            self.sl_tile.setValue(val)
            self.sp_tile.setValue(val)
        finally:
            for w in (self.sl_tile, self.sp_tile):
                w.blockSignals(False)

    def reset_clahe_defaults(self):
        self._sync_clahe_clip(2.0)
        self._sync_clahe_tile(8)
        self.statusBar().showMessage("Đã đặt tham số CLAHE mặc định: Clip Limit = 2.0, Lưới ô = 8×8")

    def _sync_gamma(self, val: float):
        val = max(0.20, min(3.50, round(float(val), 2)))
        for w in (self.sl_gamma, self.sp_gamma):
            w.blockSignals(True)
        try:
            self.sl_gamma.setValue(int(round(val * 100)))
            self.sp_gamma.setValue(val)
        finally:
            for w in (self.sl_gamma, self.sp_gamma):
                w.blockSignals(False)

    def _sync_stretch_pair(self, which: str, val: int, user_edit: bool = True):
        """Đồng bộ hai chiều QSlider <-> QSpinBox và bảo đảm r_min < r_max, s_min < s_max."""
        widgets = (
            self.sl_rmin, self.sp_rmin, self.sl_rmax, self.sp_rmax,
            self.sl_smin, self.sp_smin, self.sl_smax, self.sp_smax,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            if which == "rmin":
                val = max(0, min(254, val))
                self.sl_rmin.setValue(val)
                self.sp_rmin.setValue(val)
                if self.sp_rmax.value() <= val:
                    self.sl_rmax.setValue(val + 1)
                    self.sp_rmax.setValue(val + 1)
            elif which == "rmax":
                val = max(1, min(255, val))
                self.sl_rmax.setValue(val)
                self.sp_rmax.setValue(val)
                if self.sp_rmin.value() >= val:
                    self.sl_rmin.setValue(val - 1)
                    self.sp_rmin.setValue(val - 1)
            elif which == "smin":
                val = max(0, min(254, val))
                self.sl_smin.setValue(val)
                self.sp_smin.setValue(val)
                if self.sp_smax.value() <= val:
                    self.sl_smax.setValue(val + 1)
                    self.sp_smax.setValue(val + 1)
            elif which == "smax":
                val = max(1, min(255, val))
                self.sl_smax.setValue(val)
                self.sp_smax.setValue(val)
                if self.sp_smin.value() >= val:
                    self.sl_smin.setValue(val - 1)
                    self.sp_smin.setValue(val - 1)
        finally:
            for w in widgets:
                w.blockSignals(False)
        if user_edit and self.chk_batch_auto.isVisible() and self.chk_batch_auto.isChecked():
            self.chk_batch_auto.setChecked(False)

    def fill_auto_stretch_bounds(self, silent: bool = False):
        """Điền tự động ngưỡng [min, max] thực tế của ảnh hiện tại vào thanh kéo giãn."""
        if self.original is None:
            if not silent:
                QMessageBox.information(self, "Thông báo", "Vui lòng chọn ảnh trước khi lấy ngưỡng Min-Max.")
            return
        lum = self._orig_lab[0] if self._orig_lab is not None else get_lightness_channel(self.original)
        r_min = int(lum.min())
        r_max = int(lum.max())
        if r_max <= r_min:
            r_min, r_max = 0, 255
        self._sync_stretch_pair("rmin", min(r_min, 254), user_edit=False)
        self._sync_stretch_pair("rmax", max(r_max, min(r_min, 254) + 1), user_edit=False)
        self._sync_stretch_pair("smin", 0, user_edit=False)
        self._sync_stretch_pair("smax", 255, user_edit=False)
        if not silent:
            self.statusBar().showMessage(f"Đã đặt ngưỡng tự động theo ảnh gốc: r=[{r_min}, {r_max}] → s=[0, 255]")

    # ---------------- Các thao tác xử lý ảnh ----------------

    def need_image(self):
        if self.original is None:
            QMessageBox.information(self, "Thông báo", "Vui lòng chọn ảnh trước khi thực hiện.")
            return False
        return True

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh đầu vào", "", IMAGE_FILE_FILTER)
        if path:
            self.load_image(path)

    def _update_proc_cache(self, is_same_as_orig: bool = False):
        self._proc_id += 1
        if is_same_as_orig:
            self._proc_gray = self._orig_gray
            self._proc_hist = self._orig_hist
            self._proc_stats = self._orig_stats
            self._pair_metrics = {
                "mse": 0.0, "psnr": float("inf"), "ssim": 1.0, "ambe": 0.0, "cii": 1.0
            }
        else:
            self._proc_gray = to_gray(self.processed)
            self._proc_hist = calculate_histogram(self._proc_gray)
            self._proc_stats = calculate_image_stats(self._proc_gray, hist=self._proc_hist)
            self._pair_metrics = compute_pair_quality_metrics(
                self._orig_gray, self._proc_gray, s1=self._orig_stats, s2=self._proc_stats
            )

    def _set_batch_controls_enabled(self, enabled: bool):
        """Bật/làm mờ cụm điều khiển thống kê nhiều ảnh trên thanh công cụ."""
        self.btn_run_batch.setEnabled(enabled)
        self.lbl_batch_filter.setEnabled(enabled)
        self.combo_batch_filter.setEnabled(enabled)
        self.lbl_batch_folder.setEnabled(enabled)
        self.chk_batch_auto.setVisible(enabled)

    def load_image(self, path, from_batch: bool = False):
        img = read_image_universal(path, grayscale=False)
        if img is None:
            QMessageBox.warning(self, "Lỗi", "Không thể đọc tệp ảnh này (định dạng không hợp lệ hoặc tệp bị hỏng).")
            return
        if not from_batch:
            self.batch_folder = None
            self.batch_results = []
            self.lbl_batch_folder.setText("Nguồn: Chưa chọn")
            self._set_batch_controls_enabled(False)
            self.btn_export_batch.setEnabled(False)
            self.btn_batch_charts.setEnabled(False)
            self.batch_panel.setVisible(False)
            self.main_splitter.setSizes([380, 410])

        self.path = path
        self.original = img
        self.processed = img.copy()
        self.method_used = None
        self._img_id += 1

        # Tính sẵn cache ảnh gốc 1 lần duy nhất (tăng tốc mọi thao tác sau đó)
        self._orig_gray = to_gray(img)
        self._orig_hist = calculate_histogram(self._orig_gray)
        self._orig_stats = calculate_image_stats(self._orig_gray, hist=self._orig_hist)
        if img.ndim == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            self._orig_lab = cv2.split(lab)
        else:
            self._orig_lab = None
        self._update_proc_cache(is_same_as_orig=True)

        self.clear_otsu()
        self.clear_ground_truth(refresh=False, log_action=False)
        self.fill_auto_stretch_bounds(silent=True)

        self.canvas_orig.set_image(img, self.chk_gray_orig.isChecked())
        self.canvas_proc.set_image(self.processed, self.chk_gray_proc.isChecked())
        h, w = img.shape[:2]
        channels = "RGB (3 kênh)" if img.ndim == 3 else "Ảnh xám (1 kênh)"
        self.badge_orig.setText(f"{w} × {h} px  •  {channels}")
        self.badge_proc.setText("Chưa áp dụng nâng cao")
        self.btn_save.setEnabled(True)
        self.btn_report.setEnabled(True)
        self.refresh_all()
        self.statusBar().showMessage(f"Đã mở: {Path(path).name}  ({w} × {h} px)")
        self.log(f"Mở ảnh thành công: {Path(path).name} ({w} x {h} px).")

    def toggle_gray_orig(self, checked):
        self.canvas_orig.set_image(self.original, checked)

    def toggle_gray_proc(self, checked):
        self.canvas_proc.set_image(self.processed, checked)

    def reset(self):
        if self.original is None:
            return
        self.processed = self.original.copy()
        self.method_used = None
        self._update_proc_cache(is_same_as_orig=True)
        self.clear_otsu()
        self.fill_auto_stretch_bounds(silent=True)
        self.canvas_proc.set_image(self.processed, self.chk_gray_proc.isChecked())
        self.refresh_all()
        self.log("Đặt lại ảnh về trạng thái ban đầu.")
        self.statusBar().showMessage("Đã đặt lại trạng thái ban đầu")

    def apply_enhancement(self):
        if not self.need_image():
            return
        method = self.combo_method.currentText()
        t0 = time.perf_counter()
        if "Kéo giãn" in method:
            r_min = self.sp_rmin.value()
            r_max = self.sp_rmax.value()
            s_min = self.sp_smin.value()
            s_max = self.sp_smax.value()
            self.processed = enhance_color_image(
                self.original, method,
                r_min=r_min, r_max=r_max, s_min=s_min, s_max=s_max,
                lab_cache=self._orig_lab
            )
            method_desc = f"{method} [{r_min}..{r_max} → {s_min}..{s_max}]"
        elif "CLAHE" in method:
            clip_val = self.sp_clip.value()
            tile_val = self.sp_tile.value()
            self.processed = enhance_color_image(
                self.original, method,
                clip_limit=clip_val, tile_grid=tile_val,
                lab_cache=self._orig_lab
            )
            method_desc = f"CLAHE (clip={clip_val:.1f}, lưới={tile_val}×{tile_val})"
        elif "Gamma" in method:
            g_val = self.sp_gamma.value()
            self.processed = enhance_color_image(
                self.original, method,
                gamma=g_val,
                lab_cache=self._orig_lab
            )
            method_desc = f"Gamma (γ={g_val:.2f})"
        else:
            self.processed = enhance_color_image(self.original, method, lab_cache=self._orig_lab)
            method_desc = method
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.method_used = method_desc
        self._update_proc_cache(is_same_as_orig=False)
        self.clear_otsu()
        self.canvas_proc.set_image(self.processed, self.chk_gray_proc.isChecked())
        self.refresh_all()
        self.log(f"Áp dụng {method_desc} thành công trong {elapsed_ms:.1f} ms.")
        self.statusBar().showMessage(f"Đã hoàn thành {method_desc} ({elapsed_ms:.1f} ms)")

    def apply_otsu(self):
        if not self.need_image():
            return
        t0 = time.perf_counter()
        res = {}
        pairs = (
            ("orig", self._orig_gray, self._orig_hist),
            ("proc", self._proc_gray, self._proc_hist),
        )
        for key, gray, hist in pairs:
            t, mask, var = otsu_threshold_full(gray, hist=hist)
            res[key] = {
                "t": t,
                "mask": mask,
                "var": var,
                "white": float(np.count_nonzero(mask)) / mask.size * 100.0
            }

        # Tính các chỉ số so sánh giữa 2 mặt nạ Otsu
        metrics = compute_mask_metrics(res["orig"]["mask"], res["proc"]["mask"])
        res["metrics"] = metrics

        # Tạo bản đồ sai khác
        self.diff_map = create_diff_map(res["orig"]["mask"], res["proc"]["mask"])
        self.otsu = res

        # Cập nhật hiển thị theo View Mode hiện tại
        self.update_otsu_views()

        # Vẽ đồ thị phương sai
        self.plot_var.draw_variance(
            res["orig"]["var"], res["proc"]["var"],
            res["orig"]["t"], res["proc"]["t"]
        )

        # Cập nhật nhãn tóm tắt
        iou_val = metrics["iou"]
        dice_val = metrics["dice"]
        diff_val = metrics["diff_rate"]
        self.lbl_otsu_summary.setText(
            f"IoU: {iou_val:.2f}%   |   Dice: {dice_val:.2f}%   |   Sai khác: {diff_val:.2f}%"
        )

        self.refresh_all()
        self.tabs.setCurrentIndex(1)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.statusBar().showMessage(
            f"Hoàn thành phân đoạn Otsu trong {elapsed_ms:.1f} ms — T gốc={res['orig']['t']}, T sau={res['proc']['t']}"
        )
        self.log(
            f"Phân đoạn Otsu ({elapsed_ms:.1f} ms): Ngưỡng gốc T={res['orig']['t']}, Ngưỡng sau nâng cao T={res['proc']['t']} | "
            f"IoU={iou_val:.2f}%, Dice={dice_val:.2f}%, Sai khác={diff_val:.2f}%."
        )

    def on_view_mode_changed(self, text):
        self.view_mode = text
        if self.otsu:
            self.update_otsu_views()
            self.log(f"Thay đổi chế độ xem phân đoạn Otsu: {text}.")

    def get_display_image(self, key):
        """Trả về ảnh hiển thị cho Otsu tùy theo View Mode."""
        if not self.otsu:
            return None
        mask = self.otsu[key]["mask"]
        img = self.original if key == "orig" else self.processed
        if self.view_mode == "Mặt nạ nhị phân (Binary Mask)":
            return mask
        elif self.view_mode == "Phủ màu đối tượng (Color Overlay)":
            return create_segmented_overlay(img, mask)
        else:  # Tách riêng đối tượng (Cutout)
            return create_cutout(img, mask)

    def update_otsu_views(self):
        if not self.otsu:
            return
        m = self.otsu["metrics"]
        disp_orig = self.get_display_image("orig")
        disp_proc = self.get_display_image("proc")
        self.canvas_mask_orig.set_image(disp_orig)
        self.canvas_mask_proc.set_image(disp_proc)
        self.canvas_diff.set_image(self.diff_map)

        self.lbl_mask_orig.setText(
            f"T = {self.otsu['orig']['t']}  |  Đối tượng: {self.otsu['orig']['white']:.2f}%"
        )
        self.lbl_mask_proc.setText(
            f"T = {self.otsu['proc']['t']}  |  Đối tượng: {self.otsu['proc']['white']:.2f}%"
        )
        self.lbl_diff.setText(
            f"Đồng thuận: {m['common_pixels']:,} px  |  +Mới: {m['added_pixels']:,} px  |  -Giảm: {m['removed_pixels']:,} px"
        )

    def clear_otsu(self):
        if self.otsu is None and self.diff_map is None:
            return
        self.otsu = None
        self.diff_map = None
        self.canvas_mask_orig.set_image(None)
        self.canvas_mask_proc.set_image(None)
        self.canvas_diff.set_image(None)
        self.lbl_mask_orig.setText("")
        self.lbl_mask_proc.setText("")
        self.lbl_diff.setText("")
        self.lbl_otsu_summary.setText("Chưa phân đoạn Otsu")
        self.plot_var.draw_variance(None, None, None, None)

    # ---------------- Xử lý Ground Truth (Tùy chọn) ----------------
    def load_ground_truth(self):
        if not self.need_image():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn mặt nạ chuẩn Ground Truth", "", IMAGE_FILE_FILTER)
        if not path:
            return
        gt = read_image_universal(path, grayscale=True)
        if gt is None:
            QMessageBox.warning(self, "Lỗi", "Không thể đọc tệp Ground Truth.")
            return

        h, w = self.original.shape[:2]
        if gt.shape[:2] != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        self.gt_mask = (gt > 127).astype(np.uint8) * 255
        self.btn_clear_gt.setVisible(True)
        self.log(f"Đã nạp Ground Truth: {Path(path).name}.")
        self.refresh_all()

    def clear_ground_truth(self, refresh: bool = True, log_action: bool = True):
        had_gt = (self.gt_mask is not None)
        self.gt_mask = None
        self.btn_clear_gt.setVisible(False)
        if refresh and had_gt:
            self.refresh_all()
        if log_action and had_gt:
            self.log("Đã xóa dữ liệu Ground Truth khỏi phiên làm việc.")

    # ---------------- Cập nhật biểu đồ & Bảng so sánh ----------------
    def refresh_all(self):
        if self.original is None:
            self.refresh_key_table(None, None, None)
            return
        t1 = self.otsu["orig"]["t"] if self.otsu else None
        t2 = self.otsu["proc"]["t"] if self.otsu else None

        self.plot_h_orig.draw_histogram(
            None, "Histogram ảnh gốc", t1,
            hist=self._orig_hist, cache_key=("orig", self._img_id, t1)
        )
        self.plot_h_proc.draw_histogram(
            None, "Histogram ảnh sau nâng cao", t2,
            hist=self._proc_hist, cache_key=("proc", self._proc_id, t2)
        )

        s1 = self._orig_stats
        s2 = self._proc_stats
        pair_m = self._pair_metrics

        self.badge_proc.setText(self.method_used if self.method_used else "Chưa áp dụng nâng cao")

        self.lbl_orig.setText(
            f"Mức xám: [{s1['min']}..{s1['max']}]   |   Trung bình: {s1['mean']:.1f}   |   Độ lệch chuẩn σ: {s1['std']:.1f}"
        )
        self.lbl_proc.setText(
            f"Mức xám: [{s2['min']}..{s2['max']}]   |   Trung bình: {s2['mean']:.1f}   |   Độ lệch chuẩn σ: {s2['std']:.1f}"
            + (f"   |   {self.method_used}" if self.method_used else "")
        )
        self.refresh_key_table(s1, s2, pair_m)
        self.refresh_table(s1, s2)

    def refresh_key_table(self, s1=None, s2=None, pair_m=None):
        """Điền Bảng 1: Các chỉ số đánh giá trọng tâm tạo sự khác biệt rõ rệt sau nâng cao."""
        rows = []
        if s1 and s2 and pair_m:
            d_std = s2["std"] - s1["std"]
            d_std_pct = d_std / max(s1["std"], 1.0) * 100.0
            d_ent = s2["entropy"] - s1["entropy"]
            d_sh = s2["sharpness"] - s1["sharpness"]
            d_sh_pct = d_sh / max(s1["sharpness"], 0.5) * 100.0
            d_rng = s2["range"] - s1["range"]
            d_clip = s2["clip_rate"] - s1["clip_rate"]
            psnr_str = "∞ dB" if np.isinf(pair_m["psnr"]) else f"{pair_m['psnr']:.2f} dB"

            verdict, score, reason = evaluate_enhancement_outcome(s1, s2, pair_m)

            # Cập nhật badge kết luận ngay trên header của Bảng 1
            if not self.method_used:
                self.badge_key_verdict.setText("Chưa áp dụng nâng cao")
                self.badge_key_verdict.setStyleSheet(
                    "background:#EFF4FB; color:#1E40AF; border:1px solid #D5E3F8; border-radius:6px; padding:2px 9px; font-size:8.5pt; font-weight:700;"
                )
            elif verdict == "Tốt hơn":
                self.badge_key_verdict.setText(f"KẾT LUẬN: TỐT HƠN (ΔQ = {score:+.1f})")
                self.badge_key_verdict.setStyleSheet(
                    "background:#ECFDF5; color:#0F766E; border:1px solid #A7F3D0; border-radius:6px; padding:2px 10px; font-size:8.5pt; font-weight:800;"
                )
            elif verdict == "Xấu đi":
                self.badge_key_verdict.setText(f"KẾT LUẬN: XẤU ĐI (ΔQ = {score:+.1f})")
                self.badge_key_verdict.setStyleSheet(
                    "background:#FEF2F2; color:#B91C1C; border:1px solid #FECACA; border-radius:6px; padding:2px 10px; font-size:8.5pt; font-weight:800;"
                )
            else:
                self.badge_key_verdict.setText(f"KẾT LUẬN: TƯƠNG ĐƯƠNG (ΔQ = {score:+.1f})")
                self.badge_key_verdict.setStyleSheet(
                    "background:#F1F5F9; color:#334155; border:1px solid #CBD5E1; border-radius:6px; padding:2px 10px; font-size:8.5pt; font-weight:700;"
                )

            rows.append((
                "Độ tương phản (σ) & CII",
                f"{s1['std']:.2f}", f"{s2['std']:.2f}",
                f"{d_std:+.2f} ({d_std_pct:+.1f}%) | CII={pair_m['cii']:.2f}x",
                "Tăng tương phản rõ rệt" if d_std_pct > 5 else ("Giảm tương phản" if d_std_pct < -5 else "Ổn định tương phản")
            ))
            rows.append((
                "Độ sắc nét biên (Gradient)",
                f"{s1['sharpness']:.2f}", f"{s2['sharpness']:.2f}",
                f"{d_sh:+.2f} ({d_sh_pct:+.1f}%)",
                "Chi tiết biên rõ hơn" if d_sh_pct > 5 else ("Giảm độ rõ biên" if d_sh_pct < -5 else "Giữ nguyên độ sắc nét")
            ))
            rows.append((
                "Entropy thông tin (H - bit)",
                f"{s1['entropy']:.2f}", f"{s2['entropy']:.2f}",
                f"{d_ent:+.2f} bit",
                "Tăng độ phong phú mức xám" if d_ent > 0.05 else ("Giảm mức xám (gộp dải)" if d_ent < -0.15 else "Bảo toàn thông tin tốt")
            ))
            rows.append((
                "Dải động mức xám [min..max]",
                f"{s1['range']}", f"{s2['range']}",
                f"{d_rng:+d} ([{s2['min']}..{s2['max']}])",
                "Mở rộng dải mức xám" if d_rng > 5 else "Đã khai thác tối đa dải"
            ))
            rows.append((
                "Tỷ lệ bão hòa sáng/tối (Clip)",
                f"{s1['clip_rate']:.2f}%", f"{s2['clip_rate']:.2f}%",
                f"{d_clip:+.2f}%",
                "Cảnh báo mất chi tiết biên sáng/tối" if d_clip > 3.5 else "Kiểm soát bão hòa tốt"
            ))
            rows.append((
                "Độ lệch sáng trung bình (AMBE)",
                f"{s1['mean']:.1f}", f"{s2['mean']:.1f}",
                f"{pair_m['ambe']:.2f}",
                "Dịch chuyển độ sáng mạnh" if pair_m["ambe"] > 28 else "Giữ cân bằng ánh sáng"
            ))
            rows.append((
                "Tương đồng cấu trúc (SSIM / PSNR)",
                "—", "—",
                f"SSIM: {pair_m['ssim']:.3f} | {psnr_str}",
                "Bảo toàn cấu trúc cao" if pair_m["ssim"] >= 0.75 else ("Biến đổi cấu trúc mạnh" if pair_m["ssim"] < 0.58 else "Biến đổi cấu trúc vừa phải")
            ))
            if self.otsu:
                o1, o2, m = self.otsu["orig"], self.otsu["proc"], self.otsu["metrics"]
                rows.append((
                    "Tác động lên phân đoạn Otsu",
                    f"T={o1['t']}", f"T={o2['t']}",
                    f"IoU {m['iou']:.2f}% | Dice {m['dice']:.2f}%",
                    f"Dịch ngưỡng {o2['t'] - o1['t']:+d}, sai khác {m['diff_rate']:.2f}%"
                ))
            else:
                rows.append((
                    "Tác động lên phân đoạn Otsu",
                    "—", "—", "Chưa thực hiện", "Nhấn 'Phân đoạn Otsu' để đánh giá"
                ))

            rows.append((
                "ĐÁNH GIÁ TỔNG HỢP CHẤT LƯỢNG",
                "Gốc", "Sau xử lý",
                f"{verdict.upper()} (ΔQ={score:+.1f})" if self.method_used else "Chưa nâng cao",
                reason if self.method_used else "Chọn phương pháp và nhấn 'Nâng cao ảnh'"
            ))
        else:
            self.badge_key_verdict.setText("Chưa đánh giá")
            default_keys = [
                "Độ tương phản (σ) & CII",
                "Độ sắc nét biên (Gradient)",
                "Entropy thông tin (H - bit)",
                "Dải động mức xám [min..max]",
                "Tỷ lệ bão hòa sáng/tối (Clip)",
                "Độ lệch sáng trung bình (AMBE)",
                "Tương đồng cấu trúc (SSIM / PSNR)",
                "Tác động lên phân đoạn Otsu",
                "ĐÁNH GIÁ TỔNG HỢP CHẤT LƯỢNG",
            ]
            for k in default_keys:
                rows.append((k, "—", "—", "—", "Chưa có dữ liệu"))

        key_desc = {
            "Độ tương phản (σ) & CII": "Mức tách biệt sáng – tối & hệ số tăng tương phản",
            "Độ sắc nét biên (Gradient)": "Độ rõ đường viền, chi tiết biên vật thể",
            "Entropy thông tin (H - bit)": "Độ phong phú sắc độ mức xám (tối đa 8 bit)",
            "Dải động mức xám [min..max]": "Độ rộng khoảng cường độ sáng ảnh sử dụng",
            "Tỷ lệ bão hòa sáng/tối (Clip)": "% pixel đen bệt (≤2) hoặc cháy trắng (≥253)",
            "Độ lệch sáng trung bình (AMBE)": "Mức thay đổi độ sáng trung bình |μ_sau - μ_gốc|",
            "Tương đồng cấu trúc (SSIM / PSNR)": "Mức bảo toàn cấu trúc tự nhiên so với ảnh gốc",
            "Tác động lên phân đoạn Otsu": "Độ dịch ngưỡng T & độ trùng khớp mặt nạ IoU/Dice",
            "ĐÁNH GIÁ TỔNG HỢP CHẤT LƯỢNG": "Kết luận cải thiện / suy giảm theo điểm ΔQ",
        }

        self.table_key.setUpdatesEnabled(False)
        try:
            self.table_key.setRowCount(len(rows))
            for r, row in enumerate(rows):
                is_summary = (r == len(rows) - 1)
                self.table_key.setRowHeight(r, 42 if is_summary else 40)
                for c, val in enumerate(row):
                    item = QTableWidgetItem(val)
                    if is_summary:
                        item.setBackground(QColor("#F1F5F9"))
                    if c == 0:
                        item.setData(Qt.ItemDataRole.UserRole, key_desc.get(val, ""))
                        item.setData(Qt.ItemDataRole.UserRole + 1, True)
                    elif c in (1, 2):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setFont(QFont("Segoe UI", 9.5, QFont.Weight.Bold if c == 2 else QFont.Weight.Medium))
                        item.setForeground(QColor("#1B56B5" if (c == 2 and val not in ("—", "Sau xử lý")) else "#1E293B"))
                    elif c == 3:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setFont(QFont("Segoe UI", 9.5, QFont.Weight.Bold))
                        if "TỐT HƠN" in val:
                            item.setForeground(QColor("#0F766E"))
                        elif "XẤU ĐI" in val:
                            item.setForeground(QColor("#B91C1C"))
                        else:
                            item.setForeground(QColor("#1E3A8A"))
                    else:
                        item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold if is_summary else QFont.Weight.DemiBold))
                        if any(k in val for k in ("rõ rệt", "rõ hơn", "Tăng", "Mở rộng", "Bảo toàn", "Kiểm soát", "cân bằng")):
                            item.setForeground(QColor("#0F766E"))
                        elif any(k in val for k in ("Cảnh báo", "Biến đổi cấu trúc mạnh", "Giảm", "Dịch chuyển độ sáng mạnh")):
                            item.setForeground(QColor("#B91C1C"))
                        else:
                            item.setForeground(QColor("#334155"))
                    self.table_key.setItem(r, c, item)
        finally:
            self.table_key.setUpdatesEnabled(True)

    # ---------------- Thực nghiệm hàng loạt nhiều ảnh ----------------
    def choose_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa ảnh thực nghiệm")
        if not folder:
            return
        self.batch_folder = Path(folder)
        imgs = sorted([
            p for p in self.batch_folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ])
        self.lbl_batch_folder.setText(f"Nguồn: {self.batch_folder.name}/ ({len(imgs)} ảnh)")
        self._set_batch_controls_enabled(True)
        self.statusBar().showMessage(
            f"Đã chọn thư mục {self.batch_folder.name}/ ({len(imgs)} ảnh) — nhấn 'Chạy thống kê' để thực hiện."
        )
        self.log(f"Chọn thư mục thực nghiệm hàng loạt: {self.batch_folder} ({len(imgs)} ảnh).")

    @staticmethod
    def _eval_batch_item(args):
        idx, fpath, method, params, method_desc = args
        im = read_image_universal(fpath, grayscale=False)
        if im is None:
            return None
        kwargs = {}
        if "Kéo giãn" in method and params is not None:
            r_min, r_max, s_min, s_max = params
            kwargs.update(r_min=r_min, r_max=r_max, s_min=s_min, s_max=s_max)
        elif "CLAHE" in method and params is not None:
            clip_val, tile_val = params
            kwargs.update(clip_limit=clip_val, tile_grid=tile_val)
        elif "Gamma" in method and params is not None:
            kwargs.update(gamma=params)
        proc = enhance_color_image(im, method, **kwargs)
        g1, g2 = to_gray(im), to_gray(proc)
        h1, h2 = calculate_histogram(g1), calculate_histogram(g2)
        s1 = calculate_image_stats(g1, hist=h1)
        s2 = calculate_image_stats(g2, hist=h2)
        pm = compute_pair_quality_metrics(g1, g2, s1=s1, s2=s2)
        verdict, score, reason = evaluate_enhancement_outcome(s1, s2, pm)
        d_sh_pct = (s2["sharpness"] - s1["sharpness"]) / max(s1["sharpness"], 0.5) * 100.0
        d_clip = s2["clip_rate"] - s1["clip_rate"]
        return {
            "index": idx,
            "path": str(fpath),
            "name": fpath.name,
            "method": method_desc,
            "base_method": method,
            "stretch_params": params,
            "mean1": s1["mean"], "mean2": s2["mean"],
            "std1": s1["std"], "std2": s2["std"],
            "ent1": s1["entropy"], "ent2": s2["entropy"],
            "sh1": s1["sharpness"], "sh2": s2["sharpness"], "d_sh_pct": d_sh_pct,
            "clip1": s1["clip_rate"], "clip2": s2["clip_rate"], "d_clip": d_clip,
            "ssim": pm["ssim"], "psnr": pm["psnr"], "ambe": pm["ambe"], "cii": pm["cii"],
            "verdict": verdict, "score": score, "reason": reason,
        }

    def run_batch_benchmark(self):
        if self.batch_folder is None or not self.batch_folder.exists():
            QMessageBox.information(self, "Thông báo", "Vui lòng chọn thư mục ảnh từ nút 'Chọn ảnh/ tệp' trước.")
            return

        img_files = sorted([
            p for p in self.batch_folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ])
        if not img_files:
            QMessageBox.information(self, "Thông báo", "Thư mục đã chọn không có tệp ảnh hợp lệ.")
            return

        method = self.combo_method.currentText()
        params = None
        if "Kéo giãn" in method:
            if not self.chk_batch_auto.isChecked():
                r_min = self.sp_rmin.value()
                r_max = self.sp_rmax.value()
                s_min = self.sp_smin.value()
                s_max = self.sp_smax.value()
                params = (r_min, r_max, s_min, s_max)
                method_desc = f"{method} [{r_min}..{r_max} → {s_min}..{s_max}]"
            else:
                method_desc = f"{method} (Tự động Min-Max)"
        elif "CLAHE" in method:
            clip_val = self.sp_clip.value()
            tile_val = self.sp_tile.value()
            params = (clip_val, tile_val)
            method_desc = f"CLAHE (clip={clip_val:.1f}, lưới={tile_val}×{tile_val})"
        elif "Gamma" in method:
            g_val = self.sp_gamma.value()
            params = g_val
            method_desc = f"Gamma (γ={g_val:.2f})"
        else:
            method_desc = method

        # Tự động mở Tab 'Bảng so sánh chi tiết' và hiển thị khối thống kê dưới Bảng 1 & Bảng 2
        self.batch_panel.setVisible(True)
        self.tabs.setCurrentIndex(2)
        self.main_splitter.setSizes([240, 560])
        self.t3_vsplitter.setSizes([165, 355])

        self.batch_progress.setVisible(True)
        self.batch_progress.setMaximum(len(img_files))
        self.batch_progress.setValue(0)
        self.btn_run_batch.setEnabled(False)
        QApplication.processEvents()

        results = []
        t_start = time.perf_counter()
        tasks = [(idx, fpath, method, params, method_desc)
                 for idx, fpath in enumerate(img_files, start=1)]
        workers = max(2, min(8, os.cpu_count() or 4))

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for done, rec in enumerate(pool.map(self._eval_batch_item, tasks), start=1):
                    if rec is not None:
                        results.append(rec)
                    if done % 10 == 0 or done == len(tasks):
                        self.batch_progress.setValue(done)
                        QApplication.processEvents()
        finally:
            self.btn_run_batch.setEnabled(True)
            self.batch_progress.setVisible(False)

        total_sec = time.perf_counter() - t_start
        self.batch_results = results
        self.btn_export_batch.setEnabled(len(results) > 0)
        self.btn_batch_charts.setEnabled(len(results) > 0)
        self._update_batch_summary(results, total_sec)
        self.refresh_batch_table()

        n = len(results)
        n_better = sum(1 for r in results if r["verdict"] == "Tốt hơn")
        n_worse = sum(1 for r in results if r["verdict"] == "Xấu đi")
        n_same = n - n_better - n_worse
        self.log(
            f"Thực nghiệm hàng loạt {n} ảnh ({method_desc}) trong {total_sec:.2f}s: "
            f"Tốt hơn={n_better} ({n_better/max(n,1)*100:.1f}%), "
            f"Xấu đi={n_worse} ({n_worse/max(n,1)*100:.1f}%), Tương đương={n_same}."
        )
        self.statusBar().showMessage(
            f"Hoàn thành thống kê {n} ảnh ({method_desc}) trong {total_sec:.2f}s — "
            f"Tốt hơn: {n_better} | Xấu đi: {n_worse} | Tương đương: {n_same}"
        )

    def _compare_all_methods_on_folder(self) -> list:
        """Chạy đối chiếu nhanh cả 4 phương pháp (HE, Kéo giãn, CLAHE, Gamma) trên cùng thư mục hiện tại."""
        if self.batch_folder is None or not self.batch_folder.exists():
            return []
        img_files = sorted([
            p for p in self.batch_folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ])
        if not img_files:
            return []

        clip_val = self.sp_clip.value()
        tile_val = self.sp_tile.value()
        gamma_val = self.sp_gamma.value()
        configs = [
            ("HE (Toàn cục)", "Cân bằng histogram", None, "Cân bằng histogram"),
            ("Kéo giãn (Auto)", "Kéo giãn tương phản", None, "Kéo giãn tương phản"),
            (f"CLAHE ({clip_val:.1f}/{tile_val}×{tile_val})", "CLAHE (Cân bằng cục bộ)", (clip_val, tile_val), f"CLAHE ({clip_val:.1f})"),
            (f"Gamma (γ={gamma_val:.2f})", "Biến đổi Gamma (γ)", gamma_val, f"Gamma ({gamma_val:.2f})"),
        ]

        summary = []
        workers = max(2, min(8, os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for short_name, method, params, desc in configs:
                tasks = [(i, p, method, params, desc) for i, p in enumerate(img_files, start=1)]
                recs = [r for r in pool.map(self._eval_batch_item, tasks) if r is not None]
                n = max(1, len(recs))
                nb = sum(1 for r in recs if r["verdict"] == "Tốt hơn")
                nw = sum(1 for r in recs if r["verdict"] == "Xấu đi")
                summary.append({
                    "short": short_name,
                    "better_pct": nb / n * 100.0,
                    "worse_pct": nw / n * 100.0,
                    "avg_ssim": float(np.mean([r["ssim"] for r in recs])) if recs else 0.0,
                    "avg_std": float(np.mean([r["std2"] for r in recs])) if recs else 0.0,
                })
        return summary

    def open_batch_analytics(self):
        if not self.batch_results:
            return
        dlg = BatchAnalyticsDialog(
            self.batch_results, self.batch_folder,
            compare_callback=self._compare_all_methods_on_folder,
            parent=self
        )
        dlg.exec()

    def _update_batch_summary(self, results: list, elapsed_sec: float):
        if not results:
            return
        n = len(results)
        n_better = sum(1 for r in results if r["verdict"] == "Tốt hơn")
        n_worse = sum(1 for r in results if r["verdict"] == "Xấu đi")
        n_same = n - n_better - n_worse

        avg_m1 = float(np.mean([r["mean1"] for r in results]))
        avg_m2 = float(np.mean([r["mean2"] for r in results]))
        avg_s1 = float(np.mean([r["std1"] for r in results]))
        avg_s2 = float(np.mean([r["std2"] for r in results]))
        avg_e1 = float(np.mean([r["ent1"] for r in results]))
        avg_e2 = float(np.mean([r["ent2"] for r in results]))
        avg_ssim = float(np.mean([r["ssim"] for r in results]))
        avg_ms = (elapsed_sec / n) * 1000.0
        pct_std = (avg_s2 - avg_s1) / max(avg_s1, 1.0) * 100.0

        self.bkpi_count_val.setText(f"{n_better} tốt hơn  |  {n_worse} xấu đi")
        self.bkpi_count_sub.setText(f"Tổng {n} ảnh  •  Tương đương: {n_same}  •  {avg_ms:.1f} ms/ảnh")

        self.bkpi_mean_val.setText(f"{avg_m1:.1f}  →  {avg_m2:.1f}")
        self.bkpi_mean_sub.setText(f"Chênh lệch TB mức xám: {avg_m2 - avg_m1:+.2f}")

        self.bkpi_std_val.setText(f"{avg_s1:.1f}  →  {avg_s2:.1f} ({pct_std:+.1f}%)")
        self.bkpi_std_sub.setText(f"Tăng tương phản TB: {avg_s2 - avg_s1:+.2f}")

        self.bkpi_qual_val.setText(f"H: {avg_e2:.2f}  |  SSIM: {avg_ssim:.3f}")
        self.bkpi_qual_sub.setText(f"Entropy gốc TB: {avg_e1:.2f} ({avg_e2 - avg_e1:+.2f})")

    def refresh_batch_table(self):
        flt = self.combo_batch_filter.currentText()
        if flt == "Chỉ ảnh tốt hơn":
            shown = [r for r in self.batch_results if r["verdict"] == "Tốt hơn"]
        elif flt == "Chỉ ảnh xấu đi":
            shown = [r for r in self.batch_results if r["verdict"] == "Xấu đi"]
        elif flt == "Chỉ ảnh tương đương":
            shown = [r for r in self.batch_results if r["verdict"] == "Tương đương"]
        else:
            shown = list(self.batch_results)

        f_demi = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        f_bold = QFont("Segoe UI", 9.5, QFont.Weight.Bold)
        f_med = QFont("Segoe UI", 9, QFont.Weight.Medium)
        f_norm = QFont("Segoe UI", 9)

        self.table_batch.setUpdatesEnabled(False)
        try:
            self.table_batch.setRowCount(len(shown))
            for r_idx, item_data in enumerate(shown):
                self.table_batch.setRowHeight(r_idx, 30)
                vals = [
                    str(item_data["index"]),
                    item_data["name"],
                    f"{item_data['mean1']:.1f} → {item_data['mean2']:.1f}",
                    f"{item_data['std1']:.1f} → {item_data['std2']:.1f}",
                    f"{item_data['ent1']:.2f} → {item_data['ent2']:.2f}",
                    f"{item_data['d_sh_pct']:+.1f}%",
                    f"{item_data['d_clip']:+.2f}%",
                    f"{item_data['ssim']:.3f}",
                    item_data["verdict"],
                    item_data["reason"],
                ]
                verdict = item_data["verdict"]
                for c, val in enumerate(vals):
                    cell = QTableWidgetItem(val)
                    if c == 0:
                        cell.setData(Qt.ItemDataRole.UserRole, (
                            item_data["path"],
                            item_data.get("base_method", self.combo_method.currentText()),
                            item_data.get("stretch_params"),
                        ))
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        cell.setFont(f_demi)
                    elif c == 1:
                        cell.setFont(f_demi)
                        cell.setForeground(QColor("#0F172A"))
                    elif c == 8:
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        cell.setFont(f_bold)
                        if verdict == "Tốt hơn":
                            cell.setForeground(QColor("#0F766E"))
                            cell.setBackground(QColor("#ECFDF5"))
                        elif verdict == "Xấu đi":
                            cell.setForeground(QColor("#B91C1C"))
                            cell.setBackground(QColor("#FEF2F2"))
                        else:
                            cell.setForeground(QColor("#475569"))
                            cell.setBackground(QColor("#F1F5F9"))
                    elif c == 9:
                        cell.setFont(f_norm)
                        cell.setForeground(QColor("#334155"))
                    else:
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        cell.setFont(f_med)
                        if c == 6 and item_data["d_clip"] > 3.5:
                            cell.setForeground(QColor("#B91C1C"))
                        elif c == 7 and item_data["ssim"] < 0.58:
                            cell.setForeground(QColor("#B91C1C"))
                    self.table_batch.setItem(r_idx, c, cell)
        finally:
            self.table_batch.setUpdatesEnabled(True)

    def on_batch_row_double_clicked(self, row: int, _col: int):
        """Nhấp đúp vào bất kỳ ảnh nào trong bảng thống kê để nạp và đối chiếu đúng tham số đã chạy."""
        cell0 = self.table_batch.item(row, 0)
        if cell0 is None:
            return
        payload = cell0.data(Qt.ItemDataRole.UserRole)
        if not payload:
            return
        img_path, base_method, stretch_params = payload
        if not img_path or not Path(img_path).exists():
            return
        self.combo_method.setCurrentText(base_method)
        self.load_image(img_path, from_batch=True)
        if "Kéo giãn" in base_method:
            if stretch_params is not None:
                r_min, r_max, s_min, s_max = stretch_params
                self._sync_stretch_pair("rmin", r_min, user_edit=False)
                self._sync_stretch_pair("rmax", r_max, user_edit=False)
                self._sync_stretch_pair("smin", s_min, user_edit=False)
                self._sync_stretch_pair("smax", s_max, user_edit=False)
            else:
                self.fill_auto_stretch_bounds(silent=True)
        elif "CLAHE" in base_method and stretch_params is not None:
            clip_val, tile_val = stretch_params
            self._sync_clahe_clip(clip_val)
            self._sync_clahe_tile(tile_val)
        elif "Gamma" in base_method and stretch_params is not None:
            self._sync_gamma(stretch_params)
        self.apply_enhancement()
        self.statusBar().showMessage(f"Đang xem chi tiết ảnh thực nghiệm: {Path(img_path).name}")

    def export_batch_csv(self):
        if not self.batch_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Xuất bảng thống kê thực nghiệm hàng loạt",
            "ThongKe_100Anh_DeTai13.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "STT", "Ten_Anh", "Phuong_Phap",
                    "Mean_Goc", "Mean_Sau", "Std_Goc", "Std_Sau",
                    "Entropy_Goc", "Entropy_Sau", "Sharpness_Delta_Pct",
                    "Clipping_Delta_Pct", "SSIM", "AMBE", "Diem_Delta_Q",
                    "Ket_Luan", "Ly_Do"
                ])
                for r in self.batch_results:
                    writer.writerow([
                        r["index"], r["name"], r["method"],
                        f"{r['mean1']:.2f}", f"{r['mean2']:.2f}",
                        f"{r['std1']:.2f}", f"{r['std2']:.2f}",
                        f"{r['ent1']:.3f}", f"{r['ent2']:.3f}",
                        f"{r['d_sh_pct']:.2f}", f"{r['d_clip']:.2f}",
                        f"{r['ssim']:.4f}", f"{r['ambe']:.2f}", f"{r['score']:.1f}",
                        r["verdict"], r["reason"]
                    ])
            self.log(f"Đã xuất thống kê {len(self.batch_results)} ảnh ra tệp: {Path(path).name}.")
            QMessageBox.information(self, "Xuất thành công", f"Đã xuất kết quả thống kê ra:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi xuất CSV", str(exc))

    def refresh_table(self, s1=None, s2=None):
        # Thiết kế bảng so sánh toàn diện khoa học
        data_rows = []

        if s1 and s2:
            # 1. Thống kê mức xám & độ tương phản
            data_rows.append(("Mức xám nhỏ nhất (min)", str(s1["min"]), str(s2["min"]),
                              f"{s2['min'] - s1['min']:+d}"))
            data_rows.append(("Mức xám lớn nhất (max)", str(s1["max"]), str(s2["max"]),
                              f"{s2['max'] - s1['max']:+d}"))
            delta_range = s2["range"] - s1["range"]
            data_rows.append(("Dải động (max - min)", str(s1["range"]), str(s2["range"]),
                              f"{delta_range:+d} ({'Mở rộng' if delta_range > 0 else 'Không đổi'})"))
            delta_mean = s2["mean"] - s1["mean"]
            data_rows.append(("Giá trị xám trung bình", f"{s1['mean']:.2f}", f"{s2['mean']:.2f}",
                              f"{delta_mean:+.2f}"))
            delta_std = s2["std"] - s1["std"]
            data_rows.append(("Độ lệch chuẩn (Tương phản)", f"{s1['std']:.2f}", f"{s2['std']:.2f}",
                              f"{delta_std:+.2f} ({'Tăng tương phản' if delta_std > 0 else 'Giảm'})"))
            delta_ent = s2["entropy"] - s1["entropy"]
            data_rows.append(("Entropy (Độ phong phú chi tiết)", f"{s1['entropy']:.2f}", f"{s2['entropy']:.2f}",
                              f"{delta_ent:+.2f} ({'Tăng chi tiết' if delta_ent > 0 else 'Giảm'})"))

            # 2. Phân đoạn Otsu
            if self.otsu:
                o1 = self.otsu["orig"]
                o2 = self.otsu["proc"]
                m = self.otsu["metrics"]
                data_rows.append(("Ngưỡng phân đoạn Otsu (T)", str(o1["t"]), str(o2["t"]),
                                  f"Dịch ngưỡng: {o2['t'] - o1['t']:+d}"))
                data_rows.append(("Tỷ lệ diện tích đối tượng (%)", f"{o1['white']:.2f}%", f"{o2['white']:.2f}%",
                                  f"{o2['white'] - o1['white']:+.2f}%"))
                data_rows.append(("Chỉ số tương đồng IoU (Jaccard)", "-", "-",
                                  f"{m['iou']:.2f}%"))
                data_rows.append(("Chỉ số tương đồng Dice (F1)", "-", "-",
                                  f"{m['dice']:.2f}%"))
                data_rows.append(("Tỷ lệ pixel sai khác (%)", "-", "-",
                                  f"{m['diff_rate']:.2f}% ({m['diff_pixels']:,} px)"))
                data_rows.append(("Pixel đối tượng mới bổ sung (+)", "-", "-",
                                  f"+{m['added_pixels']:,} px"))
                data_rows.append(("Pixel đối tượng bị loại trừ (-)", "-", "-",
                                  f"-{m['removed_pixels']:,} px"))
                data_rows.append(("Pixel đối tượng đồng thuận", "-", "-",
                                  f"{m['common_pixels']:,} px"))

                # 3. Đánh giá Ground Truth nếu có
                if self.gt_mask is not None:
                    m_gt1 = compute_mask_metrics(o1["mask"], self.gt_mask)
                    m_gt2 = compute_mask_metrics(o2["mask"], self.gt_mask)
                    d_iou = m_gt2["iou"] - m_gt1["iou"]
                    data_rows.append(("IoU so với Ground Truth",
                                      f"{m_gt1['iou']:.2f}%", f"{m_gt2['iou']:.2f}%",
                                      f"Δ: {d_iou:+.2f}% ({'Cải thiện độ chính xác' if d_iou > 0 else 'Giảm'})"))
                    d_dice = m_gt2["dice"] - m_gt1["dice"]
                    data_rows.append(("Dice so với Ground Truth",
                                      f"{m_gt1['dice']:.2f}%", f"{m_gt2['dice']:.2f}%",
                                      f"Δ: {d_dice:+.2f}%"))
            else:
                data_rows.append(("Phân đoạn Otsu", "Chưa thực hiện", "Chưa thực hiện", "Nhấn 'Phân đoạn Otsu'"))
        else:
            default_labels = [
                "Mức xám nhỏ nhất (min)", "Mức xám lớn nhất (max)", "Dải động (max - min)",
                "Giá trị xám trung bình", "Độ lệch chuẩn (Tương phản)", "Entropy",
                "Ngưỡng Otsu (T)", "Tỷ lệ diện tích đối tượng (%)", "Chỉ số tương đồng IoU",
                "Chỉ số tương đồng Dice", "Tỷ lệ pixel sai khác (%)"
            ]
            for lab in default_labels:
                data_rows.append((lab, "-", "-", "-"))

        key_metrics = {
            "Độ lệch chuẩn (Tương phản)",
            "Ngưỡng phân đoạn Otsu (T)",
            "Chỉ số tương đồng IoU (Jaccard)",
            "Chỉ số tương đồng Dice (F1)",
            "IoU so với Ground Truth",
            "Dice so với Ground Truth",
        }

        detail_desc = {
            "Mức xám nhỏ nhất (min)": "Điểm ảnh tối nhất trong ảnh (0..255)",
            "Mức xám lớn nhất (max)": "Điểm ảnh sáng nhất trong ảnh (0..255)",
            "Dải động (max - min)": "Độ rộng dải mức xám thực tế của ảnh",
            "Giá trị xám trung bình": "Độ sáng trung bình toàn ảnh (μ)",
            "Độ lệch chuẩn (Tương phản)": "Độ phân tán mức xám quanh trung bình (σ)",
            "Entropy (Độ phong phú chi tiết)": "Lượng thông tin mức xám Shannon (bit)",
            "Entropy": "Lượng thông tin mức xám Shannon (bit)",
            "Ngưỡng phân đoạn Otsu (T)": "Ngưỡng tự động tách đối tượng và nền",
            "Ngưỡng Otsu (T)": "Ngưỡng tự động tách đối tượng và nền",
            "Tỷ lệ diện tích đối tượng (%)": "% pixel phân loại là đối tượng (vùng trắng)",
            "Chỉ số tương đồng IoU (Jaccard)": "Tỷ lệ giao / hợp giữa 2 mặt nạ Otsu",
            "Chỉ số tương đồng IoU": "Tỷ lệ giao / hợp giữa 2 mặt nạ Otsu",
            "Chỉ số tương đồng Dice (F1)": "Độ trùng khớp 2×giao / tổng diện tích",
            "Chỉ số tương đồng Dice": "Độ trùng khớp 2×giao / tổng diện tích",
            "Tỷ lệ pixel sai khác (%)": "% pixel đổi nhãn nền ↔ đối tượng",
            "Pixel đối tượng mới bổ sung (+)": "Vùng đối tượng phát hiện thêm (xanh lục)",
            "Pixel đối tượng bị loại trừ (-)": "Vùng đối tượng bị loại bỏ (đỏ cam)",
            "Pixel đối tượng đồng thuận": "Vùng đối tượng trùng khớp cả 2 lần (trắng)",
            "IoU so với Ground Truth": "Độ chính xác phân đoạn so với mặt nạ chuẩn",
            "Dice so với Ground Truth": "Hệ số F1 phân đoạn so với mặt nạ chuẩn",
            "Phân đoạn Otsu": "Tách đối tượng và nền tự động theo σ²_B(t)",
        }

        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(data_rows))
            for r, row in enumerate(data_rows):
                self.table.setRowHeight(r, 38)
                metric_name = row[0]
                is_key = metric_name in key_metrics
                for c, val in enumerate(row):
                    item = QTableWidgetItem(val)
                    if c == 0:
                        item.setData(Qt.ItemDataRole.UserRole, detail_desc.get(val, ""))
                        item.setData(Qt.ItemDataRole.UserRole + 1, is_key)
                    elif c in (1, 2):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        weight = QFont.Weight.Bold if (is_key and c == 2) else QFont.Weight.Medium
                        item.setFont(QFont("Segoe UI", 9.5, weight))
                        item.setForeground(QColor("#1B56B5" if (is_key and c == 2 and val != "-") else "#1E293B"))
                    else:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setFont(QFont("Segoe UI", 9.5, QFont.Weight.Bold if is_key else QFont.Weight.DemiBold))
                        if any(k in val for k in ("Tăng", "Mở rộng", "Cải thiện")):
                            item.setForeground(QColor("#0F766E"))
                        elif "IoU" in metric_name or "Dice" in metric_name:
                            item.setForeground(QColor("#1B56B5"))
                        elif "Giảm" in val or val.startswith("-"):
                            item.setForeground(QColor("#B45309"))
                        else:
                            item.setForeground(QColor("#334155"))
                    self.table.setItem(r, c, item)
        finally:
            self.table.setUpdatesEnabled(True)

    def zoom(self, title, image):
        if image is not None:
            ZoomDialog(title, image, self).exec()

    # ---------------- Lưu ảnh & Xuất báo cáo ----------------
    def save_results(self):
        if not self.need_image():
            return
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả")
        if not folder:
            return
        folder = Path(folder)
        outputs = {
            "01_anh_goc.png": self.original,
            "02_anh_sau_nang_cao.png": self.processed,
            "03_anh_goc_xam.png": to_gray(self.original),
            "04_anh_sau_nang_cao_xam.png": to_gray(self.processed),
        }
        if self.otsu:
            outputs["05_otsu_mat_na_goc.png"] = self.otsu["orig"]["mask"]
            outputs["06_otsu_mat_na_sau_nang_cao.png"] = self.otsu["proc"]["mask"]
            if self.diff_map is not None:
                outputs["07_otsu_bando_sai_khac.png"] = self.diff_map
            outputs["08_otsu_tach_doi_tuong_goc.png"] = create_cutout(self.original, self.otsu["orig"]["mask"])
            outputs["09_otsu_tach_doi_tuong_sau_nang_cao.png"] = create_cutout(self.processed, self.otsu["proc"]["mask"])

        try:
            for name, img in outputs.items():
                ok, buf = cv2.imencode(".png", img)
                if ok:
                    buf.tofile(str(folder / name))
            QMessageBox.information(self, "Lưu thành công", f"Đã lưu {len(outputs)} tệp ảnh vào:\n{folder}")
            self.log(f"Đã lưu {len(outputs)} tệp ảnh kết quả vào {folder}.")
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi khi lưu", str(exc))

    def export_report(self):
        if not self.need_image():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Xuất báo cáo kết quả", "BaoCao_DeTai13.txt", "Text Files (*.txt)")
        if not path:
            return

        g1, g2 = to_gray(self.original), to_gray(self.processed)
        s1, s2 = calculate_image_stats(g1), calculate_image_stats(g2)
        pm = compute_pair_quality_metrics(g1, g2)
        verdict, score, reason = evaluate_enhancement_outcome(s1, s2, pm)
        psnr_str = "Inf dB" if np.isinf(pm["psnr"]) else f"{pm['psnr']:.2f} dB"
        h, w = self.original.shape[:2]

        lines = [
            "================================================================================",
            "              BÁO CÁO KẾT QUẢ THỰC NGHIỆM - ĐỀ TÀI 13",
            "  TÍNH HISTOGRAM VÀ NÂNG CAO CHẤT LƯỢNG ẢNH BẰNG HISTOGRAM & PHÂN ĐOẠN OTSU",
            "================================================================================",
            f"Thời gian xuất: {datetime.now():%d/%m/%Y %H:%M:%S}",
            f"Tệp ảnh xử lý: {Path(self.path).name if self.path else 'Ảnh đầu vào'} ({w} x {h} px, {self.original.size:,} pixels)",
            f"Phương pháp nâng cao đã dùng: {self.method_used or 'Chưa áp dụng'}",
            "",
            "1. SO SÁNH THỐNG KÊ PHÂN BỐ MỨC XÁM & CHỈ SỐ TRỌNG TÂM TẠO KHÁC BIỆT",
            "-" * 80,
            f"{'Chỉ số':<34}{'Ảnh gốc':>14}{'Sau nâng cao':>15}{'Thay đổi':>16}",
            f"{'Mức xám nhỏ nhất (min)':<34}{s1['min']:>14}{s2['min']:>15}{s2['min']-s1['min']:>+16d}",
            f"{'Mức xám lớn nhất (max)':<34}{s1['max']:>14}{s2['max']:>15}{s2['max']-s1['max']:>+16d}",
            f"{'Dải động (max - min)':<34}{s1['range']:>14}{s2['range']:>15}{s2['range']-s1['range']:>+16d}",
            f"{'Giá trị xám trung bình':<34}{s1['mean']:>14.2f}{s2['mean']:>15.2f}{s2['mean']-s1['mean']:>+16.2f}",
            f"{'Độ lệch chuẩn (Tương phản)':<34}{s1['std']:>14.2f}{s2['std']:>15.2f}{s2['std']-s1['std']:>+16.2f}",
            f"{'Entropy (Độ phong phú chi tiết)':<34}{s1['entropy']:>14.2f}{s2['entropy']:>15.2f}{s2['entropy']-s1['entropy']:>+16.2f}",
            f"{'Độ sắc nét biên (Gradient)':<34}{s1['sharpness']:>14.2f}{s2['sharpness']:>15.2f}{s2['sharpness']-s1['sharpness']:>+16.2f}",
            f"{'Tỷ lệ bão hòa sáng/tối (%)':<34}{s1['clip_rate']:>13.2f}%{s2['clip_rate']:>14.2f}%{s2['clip_rate']-s1['clip_rate']:>+15.2f}%",
            "",
            "Chỉ số đánh giá chất lượng cặp ảnh trước & sau nâng cao:",
            f"  - Chỉ số cải thiện tương phản (CII)     : {pm['cii']:.2f}x",
            f"  - Độ lệch sáng trung bình (AMBE)        : {pm['ambe']:.2f}",
            f"  - Độ tương đồng cấu trúc (SSIM)         : {pm['ssim']:.4f}",
            f"  - Tỷ số tín hiệu trên nhiễu (PSNR)      : {psnr_str}",
            f"  - Đánh giá tổng hợp                     : {verdict} (ΔQ = {score:+.1f}) — {reason}",
            "",
            "2. SO SÁNH KẾT QUẢ PHÂN ĐOẠN OTSU TRƯỚC VÀ SAU NÂNG CAO",
            "-" * 80,
        ]

        if self.otsu:
            o1 = self.otsu["orig"]
            o2 = self.otsu["proc"]
            m = self.otsu["metrics"]
            lines += [
                f"{'Ngưỡng phân đoạn Otsu (T)':<32}{o1['t']:>15}{o2['t']:>16}{o2['t']-o1['t']:>+16d}",
                f"{'Tỷ lệ diện tích đối tượng (%)':<32}{o1['white']:>14.2f}%{o2['white']:>15.2f}%{o2['white']-o1['white']:>+15.2f}%",
                "",
                "Các chỉ số định lượng đánh giá sự sai khác mặt nạ:",
                f"  - Chỉ số tương đồng IoU (Jaccard Index) : {m['iou']:.2f}%",
                f"  - Chỉ số tương đồng Dice (F1-Score)     : {m['dice']:.2f}%",
                f"  - Tỷ lệ pixel sai khác                  : {m['diff_rate']:.2f}% ({m['diff_pixels']:,} px)",
                f"  - Pixel đối tượng mới được phát hiện (+) : +{m['added_pixels']:,} px",
                f"  - Pixel đối tượng bị loại trừ (-)       : -{m['removed_pixels']:,} px",
                f"  - Pixel đối tượng đồng thuận            : {m['common_pixels']:,} px",
            ]
            if self.gt_mask is not None:
                m_gt1 = compute_mask_metrics(o1["mask"], self.gt_mask)
                m_gt2 = compute_mask_metrics(o2["mask"], self.gt_mask)
                lines += [
                    "",
                    "Đánh giá so với Ground Truth (Mặt nạ chuẩn):",
                    f"  - IoU ảnh gốc vs GT         : {m_gt1['iou']:.2f}%",
                    f"  - IoU sau nâng cao vs GT    : {m_gt2['iou']:.2f}%",
                    f"  - Mức cải thiện IoU (Δ)     : {m_gt2['iou'] - m_gt1['iou']:+.2f}%",
                    f"  - Dice ảnh gốc vs GT        : {m_gt1['dice']:.2f}%",
                    f"  - Dice sau nâng cao vs GT   : {m_gt2['dice']:.2f}%",
                ]
        else:
            lines.append("Chưa thực hiện phân đoạn Otsu.")

        lines += [
            "",
            "3. KẾT LUẬN & NHẬN XÉT HỌC THUẬT",
            "-" * 80,
            "- Kéo giãn tương phản giúp đưa mức xám đạt cực đại [0, 255] một cách tuyến tính.",
            "- Cân bằng histogram tối ưu hóa mật độ phân bố tích lũy CDF, làm nổi rõ chi tiết vùng tối và sáng.",
            "- Sau khi nâng cao chất lượng ảnh, phân đoạn Otsu tính toán ngưỡng tách lớp chính xác hơn,",
            "  giúp phân biệt đối tượng và nền rõ ràng, thể hiện qua sự thay đổi phương sai giữa hai lớp sigma_B^2(t).",
            "================================================================================"
        ]

        try:
            Path(path).write_text("\n".join(lines), encoding="utf-8")
            self.log(f"Đã xuất báo cáo chi tiết: {Path(path).name}.")
            QMessageBox.information(self, "Báo cáo", f"Đã xuất báo cáo thành công ra tệp:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi khi xuất báo cáo", str(exc))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    icon = ASSETS_DIR / "logo.ico"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    win = ImageProcessingStudio()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
