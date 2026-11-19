import sys
import os
import time
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
    QScrollArea, QAbstractItemView, QGroupBox, QCheckBox, QPlainTextEdit
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
from matplotlib.figure import Figure

# Cấu hình đường dẫn
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
ARROW_PATH = (ASSETS_DIR / "arrow_down.png").as_posix()


# ============================================================
# CÁC THUẬT TOÁN CỐT LÕI - CÀI ĐẶT BẰNG NUMPY
# (Bám sát yêu cầu Đề tài 13 - Không dùng cv2.calcHist,
#  cv2.equalizeHist, cv2.threshold(..., THRESH_OTSU))
# ============================================================

def calculate_histogram(gray_image: np.ndarray) -> np.ndarray:
    """
    Tính histogram 256 mức xám bằng NumPy:
    h(r_k) = n_k với k = 0..255
    """
    hist = np.zeros(256, dtype=np.int64)
    values, counts = np.unique(gray_image, return_counts=True)
    hist[values] = counts
    return hist


def contrast_stretch(gray_image: np.ndarray) -> np.ndarray:
    """
    Kéo giãn tương phản tuyến tính (Contrast Stretching):
    s = (r - min) * 255 / (max - min)
    Ánh xạ khoảng mức xám thực tế [min, max] về toàn bộ dải [0, 255].
    """
    img = gray_image.astype(np.float32)
    low, high = float(img.min()), float(img.max())
    if high == low:
        return gray_image.copy()
    result = (img - low) * 255.0 / (high - low)
    return np.clip(result, 0, 255).astype(np.uint8)


def histogram_equalization(gray_image: np.ndarray) -> np.ndarray:
    """
    Cân bằng histogram (Histogram Equalization) bằng hàm phân bố tích lũy CDF:
    s_k = round((CDF(k) - CDF_min) * 255 / (N - CDF_min))
    """
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


def otsu_threshold_full(gray_image: np.ndarray):
    """
    Phân đoạn Otsu tự cài đặt:
    Tối đa hóa phương sai giữa hai lớp (Between-Class Variance):
    sigma_B^2(t) = omega_0(t) * omega_1(t) * [mu_0(t) - mu_1(t)]^2
    Trả về: (ngưỡng tối ưu t, mặt nạ nhị phân 0-255, mảng phương sai 256 phần tử).
    """
    hist = calculate_histogram(gray_image).astype(np.float64)
    total = float(gray_image.size)
    levels = np.arange(256, dtype=np.float64)
    probability = hist / total
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * levels)
    mu_total = mu[-1]
    denominator = omega * (1.0 - omega)
    numerator = (mu_total * omega - mu) ** 2
    variance = np.zeros(256, dtype=np.float64)
    valid = denominator > 1e-7
    variance[valid] = numerator[valid] / denominator[valid]
    threshold = int(np.argmax(variance))
    mask = np.where(gray_image > threshold, 255, 0).astype(np.uint8)
    return threshold, mask, variance


def to_gray(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh sang mức xám nếu là ảnh màu BGR."""
    if image is None:
        return None
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_color_image(image: np.ndarray, method: str) -> np.ndarray:
    """
    Nâng cao chất lượng ảnh:
    - Nếu là ảnh xám: áp dụng trực tiếp thuật toán.
    - Nếu là ảnh màu BGR: chuyển sang không gian màu CIE LAB,
      chỉ áp dụng nâng cao trên kênh độ sáng L (Lightness), sau đó
      ghép lại kênh A và B để bảo toàn màu sắc tự nhiên không bị sai màu.
    """
    if image.ndim == 2:
        if "Kéo giãn" in method:
            return contrast_stretch(image)
        return histogram_equalization(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    if "Kéo giãn" in method:
        new_lightness = contrast_stretch(lightness)
    else:
        new_lightness = histogram_equalization(lightness)
    merged = cv2.merge((new_lightness, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def calculate_image_stats(gray_image: np.ndarray) -> dict:
    """Tính các chỉ số thống kê mô tả phân bố mức xám và entropy."""
    if gray_image is None or gray_image.size == 0:
        return {}
    hist = calculate_histogram(gray_image)
    total = float(gray_image.size)
    prob = hist / total
    levels = np.arange(256)
    mean_val = float(np.sum(prob * levels))
    variance_val = float(np.sum(prob * ((levels - mean_val) ** 2)))
    std_val = float(np.sqrt(variance_val))
    min_val = int(gray_image.min())
    max_val = int(gray_image.max())
    median_val = float(np.median(gray_image))

    nz_prob = prob[prob > 0]
    entropy_val = float(-np.sum(nz_prob * np.log2(nz_prob)))

    return {
        "min": min_val,
        "max": max_val,
        "range": max_val - min_val,
        "mean": mean_val,
        "variance": variance_val,
        "std": std_val,
        "median": median_val,
        "entropy": entropy_val
    }


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
        dw, dh = iw * s, ih * s
        target = QRectF(self.width() / 2 - dw / 2, self.height() / 2 - dh / 2, dw, dh)

        img_clip = QPainterPath()
        img_clip.addRoundedRect(target, 4, 4)
        p.setClipPath(img_clip)
        p.drawImage(target, self.qimage)
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
            if Path(path).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
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


class PlotCanvas(FigureCanvas):
    """Widget vẽ đồ thị matplotlib sắc nét, đồng bộ phông chữ hệ thống."""
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(4, 2.2), dpi=100, facecolor="#FFFFFF")
        plt_rc = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Tahoma", "DejaVu Sans"],
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
        matplotlib.rcParams.update(plt_rc)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(110)

    def _axes(self, title):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#FFFFFF")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#0F172A", pad=8)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Mức xám (0 - 255)", fontsize=8.5, fontweight="semibold", color="#334155")
        ax.set_ylabel("Số pixel", fontsize=8.5, fontweight="semibold", color="#334155")
        ax.tick_params(labelsize=8, colors="#475569")
        ax.grid(True, linewidth=0.6, linestyle="--", color="#E2E8F0")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color("#CBD5E1")
            ax.spines[sp].set_linewidth(1.0)
        return ax

    def draw_histogram(self, gray, title, threshold=None):
        ax = self._axes(title)
        if gray is not None:
            hist = calculate_histogram(gray)
            ax.bar(np.arange(256), hist, width=1.0, color="#2563EB", alpha=0.82)
            if threshold is not None:
                ax.axvline(threshold, color="#DC2626", linewidth=1.6, linestyle="--")
                y_max = ax.get_ylim()[1]
                x_pos = threshold + 5 if threshold < 195 else threshold - 48
                ax.text(
                    x_pos, y_max * 0.86,
                    f"T = {threshold}",
                    color="#B91C1C", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#FCA5A5", linewidth=0.9)
                )
        self.figure.tight_layout(pad=0.8)
        self.draw_idle()

    def draw_variance(self, var_before, var_after, t_before, t_after):
        ax = self._axes("")
        ax.set_title("")
        ax.set_ylabel("σ²_B(t)", fontsize=8, fontweight="semibold", color="#334155")
        ax.set_xlabel("Ngưỡng t", fontsize=8, fontweight="semibold", color="#334155")
        ax.tick_params(labelsize=7.5, colors="#475569")
        if var_before is not None:
            ax.plot(var_before, color="#64748B", linewidth=1.3, linestyle="--",
                    label=f"Gốc (T={t_before})")
            ax.plot(var_after, color="#1B56B5", linewidth=1.8,
                    label=f"Nâng cao (T={t_after})")
            ax.axvline(t_before, color="#94A3B8", linestyle=":", linewidth=1.0)
            ax.axvline(t_after, color="#1B56B5", linestyle=":", linewidth=1.1)
            ax.scatter([t_before], [var_before[t_before]], color="#475569", s=22, zorder=5)
            ax.scatter([t_after], [var_after[t_after]], color="#1B56B5", s=28, zorder=6)
            ax.legend(fontsize=7.2, loc="lower center", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
        self.figure.tight_layout(pad=0.35)
        self.draw_idle()


# ============================================================
# CỬA SỔ CHÍNH CỦA CHƯƠNG TRÌNH
# ============================================================

class ImageProcessingStudio(QMainWindow):
    METHODS = ["Cân bằng histogram", "Kéo giãn tương phản"]
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

        # Trạng thái dữ liệu
        self.path = None
        self.original = None
        self.processed = None
        self.method_used = None
        self.otsu = None          # Kết quả phân đoạn Otsu
        self.diff_map = None      # Bản đồ khác biệt
        self.gt_mask = None       # Mặt nạ chuẩn Ground Truth (tùy chọn)
        self.view_mode = self.VIEW_MODES[0]

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

    def build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(10)
        self.setCentralWidget(central)

        # 1. Thanh công cụ ngang cao cấp (Card Toolbar phân 3 nhóm rõ ràng)
        toolbar_card = QFrame()
        toolbar_card.setObjectName("toolbarCard")
        bar = QHBoxLayout(toolbar_card)
        bar.setContentsMargins(12, 8, 12, 8)
        bar.setSpacing(8)

        # Nhóm 1: Nguồn ảnh
        self.btn_open = QPushButton("Chọn ảnh...")
        self.btn_open.setObjectName("btnOpen")
        self.btn_open.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open.clicked.connect(self.open_dialog)

        self.btn_reset = QPushButton("Đặt lại")
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset)

        # Nhóm 2: Xử lý cốt lõi
        lbl_method = QLabel("Phương pháp:")
        lbl_method.setObjectName("toolbarLabel")

        self.combo_method = QComboBox()
        self.combo_method.addItems(self.METHODS)
        self.combo_method.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.combo_method.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_apply = QPushButton("Nâng cao ảnh")
        self.btn_apply.setObjectName("primary")
        self.btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply.clicked.connect(self.apply_enhancement)

        self.btn_otsu = QPushButton("Phân đoạn Otsu")
        self.btn_otsu.setObjectName("accentTeal")
        self.btn_otsu.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_otsu.clicked.connect(self.apply_otsu)

        self.chk_gray = QCheckBox("Xem ảnh xám")
        self.chk_gray.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_gray.toggled.connect(self.toggle_gray)

        # Nhóm 3: Xuất kết quả
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
        bar.addSpacing(4)
        bar.addWidget(self._make_vsep())
        bar.addSpacing(4)
        bar.addWidget(lbl_method)
        bar.addWidget(self.combo_method)
        bar.addWidget(self.btn_apply)
        bar.addWidget(self.btn_otsu)
        bar.addSpacing(4)
        bar.addWidget(self._make_vsep())
        bar.addSpacing(4)
        bar.addWidget(self.chk_gray)
        bar.addStretch()
        bar.addWidget(self.btn_save)
        bar.addWidget(self.btn_report)
        root.addWidget(toolbar_card)

        self.btn_save.setEnabled(False)
        self.btn_report.setEnabled(False)

        # 2. Phân chia giao diện: Phần trên (2 ảnh) & Phần dưới (KPI + Tabs phân tích)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(8)

        # Phần trên: So sánh Ảnh gốc vs Ảnh sau nâng cao
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.setHandleWidth(10)

        self.canvas_orig = ImageCanvas(
            placeholder="Kéo thả ảnh vào đây để bắt đầu",
            subtext="Hoặc nhấn nút 'Chọn ảnh...' (Hỗ trợ PNG, JPG, BMP, TIFF)",
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

        self.badge_orig = QLabel("Chưa nạp ảnh")
        self.badge_proc = QLabel("Đang chờ xử lý")
        self.lbl_orig = QLabel("Thông số mức xám sẽ hiển thị sau khi nạp ảnh")
        self.lbl_proc = QLabel("Thông số so sánh sẽ hiển thị sau khi xử lý")

        top_splitter.addWidget(self.create_card("Ảnh gốc", self.canvas_orig, self.lbl_orig, self.badge_orig))
        top_splitter.addWidget(self.create_card("Ảnh sau nâng cao", self.canvas_proc, self.lbl_proc, self.badge_proc))
        main_splitter.addWidget(top_splitter)

        # Phần dưới: Dải 4 thẻ KPI tóm tắt + Hệ thống 4 Tab phân tích
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)

        # Dải 4 thẻ KPI tóm tắt chỉ số quan trọng
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(10)

        self.kpi_contrast_val, self.kpi_contrast_sub, card_k1 = self.create_kpi_card(
            "ĐỘ TƯƠNG PHẢN (ĐỘ LỆCH CHUẨN σ)", "#1B56B5"
        )
        self.kpi_otsu_val, self.kpi_otsu_sub, card_k2 = self.create_kpi_card(
            "NGƯỠNG PHÂN ĐOẠN OTSU (T)", "#0F172A"
        )
        self.kpi_iou_val, self.kpi_iou_sub, card_k3 = self.create_kpi_card(
            "ĐỘ TƯƠNG ĐỒNG IoU (JACCARD)", "#0F766E"
        )
        self.kpi_dice_val, self.kpi_dice_sub, card_k4 = self.create_kpi_card(
            "HỆ SỐ DICE (F1) & SAI KHÁC", "#B45309"
        )

        for kc in (card_k1, card_k2, card_k3, card_k4):
            kpi_row.addWidget(kc, 1)
        bottom_layout.addLayout(kpi_row)

        # Hệ thống các Tab phân tích chuyên sâu
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

        # Tab 3: Bảng so sánh chỉ số chi tiết
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            "Chỉ số đánh giá", "Ảnh gốc", "Ảnh sau nâng cao", "Mức độ thay đổi (Δ / Đánh giá)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.tabs.addTab(self.table, "Bảng so sánh chi tiết")

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
        main_splitter.addWidget(bottom_container)
        main_splitter.setSizes([380, 390])
        root.addWidget(main_splitter, 1)

        self.statusBar().showMessage("Sẵn sàng — Nhấn 'Chọn ảnh...' hoặc kéo thả tệp ảnh vào khung bên trái.")
        self.refresh_table()
        self.plot_h_orig.draw_histogram(None, "Histogram ảnh gốc")
        self.plot_h_proc.draw_histogram(None, "Histogram ảnh sau nâng cao")
        self.plot_var.draw_variance(None, None, None, None)

        # MenuBar
        m = self.menuBar().addMenu("Tệp")
        for text, slot, key in (
            ("Mở ảnh...", self.open_dialog, "Ctrl+O"),
            ("Lưu ảnh...", self.save_results, "Ctrl+S"),
            ("Xuất báo cáo...", self.export_report, None),
            ("Thoát", self.close, None)
        ):
            a = QAction(text, self)
            if key:
                a.setShortcut(key)
            a.triggered.connect(slot)
            m.addAction(a)

    def create_card(self, title, widget, label=None, badge=None, compact=False):
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

    def create_kpi_card(self, title: str, accent_color: str):
        card = QFrame()
        card.setObjectName("kpiCard")
        card.setFixedHeight(76)
        card.setStyleSheet("""
            QFrame#kpiCard {
                background: #FFFFFF;
                border: 1px solid #D8E0EA;
                border-radius: 9px;
            }
        """)
        outer = QHBoxLayout(card)
        outer.setContentsMargins(10, 8, 12, 8)
        outer.setSpacing(10)

        accent_bar = QFrame()
        accent_bar.setFixedWidth(4)
        accent_bar.setStyleSheet(f"background: {accent_color}; border: none; border-radius: 2px;")
        outer.addWidget(accent_bar)

        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 8pt; font-weight: 700; color: #475569; letter-spacing: 0.2px;")

        lbl_val = QLabel("—")
        lbl_val.setStyleSheet(f"font-size: 13.5pt; font-weight: 800; color: {accent_color};")

        lbl_sub = QLabel("Chưa có dữ liệu")
        lbl_sub.setStyleSheet("font-size: 8.5pt; font-weight: 500; color: #64748B;")

        lay.addWidget(lbl_title)
        lay.addWidget(lbl_val)
        lay.addWidget(lbl_sub)
        outer.addLayout(lay, 1)
        return lbl_val, lbl_sub, card

    def update_kpi_cards(self, s1=None, s2=None):
        """Cập nhật đồng bộ 4 thẻ chỉ số KPI nổi bật."""
        if s1 is None or s2 is None:
            self.kpi_contrast_val.setText("—")
            self.kpi_contrast_sub.setText("Chưa nạp ảnh đầu vào")
            self.kpi_otsu_val.setText("—")
            self.kpi_otsu_sub.setText("Chưa phân đoạn Otsu")
            self.kpi_iou_val.setText("—")
            self.kpi_iou_sub.setText("Chưa phân đoạn Otsu")
            self.kpi_dice_val.setText("—")
            self.kpi_dice_sub.setText("Chưa phân đoạn Otsu")
            return

        # 1. Độ tương phản (Độ lệch chuẩn σ)
        std1, std2 = s1["std"], s2["std"]
        if self.method_used and std1 > 0:
            pct = (std2 - std1) / std1 * 100.0
            self.kpi_contrast_val.setText(f"{std2:.2f}  ({pct:+.1f}%)")
            self.kpi_contrast_sub.setText(f"Ảnh gốc: {std1:.2f}  •  Dải động: [{s2['min']}..{s2['max']}]")
        else:
            self.kpi_contrast_val.setText(f"{std1:.2f}")
            self.kpi_contrast_sub.setText(f"Chưa nâng cao  •  Dải động: [{s1['min']}..{s1['max']}]")

        # 2-4. Chỉ số Phân đoạn Otsu
        if self.otsu:
            o1 = self.otsu["orig"]
            o2 = self.otsu["proc"]
            m = self.otsu["metrics"]
            dt = o2["t"] - o1["t"]

            self.kpi_otsu_val.setText(f"T = {o2['t']}")
            self.kpi_otsu_sub.setText(f"Ảnh gốc: T = {o1['t']}  •  Dịch ngưỡng: {dt:+d}")

            self.kpi_iou_val.setText(f"{m['iou']:.2f}%")
            self.kpi_iou_sub.setText(f"Đối tượng đồng thuận: {m['common_pixels']:,} px")

            self.kpi_dice_val.setText(f"{m['dice']:.2f}%")
            self.kpi_dice_sub.setText(f"Tỷ lệ sai khác: {m['diff_rate']:.2f}% ({m['diff_pixels']:,} px)")
        else:
            self.kpi_otsu_val.setText("—")
            self.kpi_otsu_sub.setText("Nhấn 'Phân đoạn Otsu' để tính ngưỡng")
            self.kpi_iou_val.setText("—")
            self.kpi_iou_sub.setText("Đo độ trùng khớp trước & sau nâng cao")
            self.kpi_dice_val.setText("—")
            self.kpi_dice_sub.setText("Đánh giá tương đồng vùng đối tượng")

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
                padding: 7px 14px;
                color: #1E293B;
                font-size: 9.5pt;
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
                font-size: 10pt;
                font-weight: 700;
                padding: 7px 18px;
            }}
            QPushButton#primary:hover {{
                background: #144493;
            }}
            QPushButton#primary:pressed {{
                background: #0F3573;
            }}

            /* Nút hành động chính thứ 2: Phân đoạn Otsu */
            QPushButton#accentTeal {{
                background: #0F766E;
                color: #FFFFFF;
                border: 1px solid #0C5F59;
                font-size: 10pt;
                font-weight: 700;
                padding: 7px 18px;
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
                padding: 6px 12px;
                min-width: 175px;
                font-size: 9.5pt;
                font-weight: 600;
                color: #0F172A;
            }}
            QComboBox:hover {{
                border-color: #64748B;
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
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 9pt;
                font-weight: 600;
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

    # ---------------- Các thao tác xử lý ảnh ----------------

    def need_image(self):
        if self.original is None:
            QMessageBox.information(self, "Thông báo", "Vui lòng chọn ảnh trước khi thực hiện.")
            return False
        return True

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh đầu vào", "", "Ảnh (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Lỗi", "Không thể đọc tệp ảnh này.")
            return
        self.path = path
        self.original = img
        self.processed = img.copy()
        self.method_used = None
        self.clear_otsu()
        self.clear_ground_truth()

        self.canvas_orig.set_image(img, self.chk_gray.isChecked())
        self.canvas_proc.set_image(self.processed, self.chk_gray.isChecked())
        h, w = img.shape[:2]
        channels = "RGB (3 kênh)" if img.ndim == 3 else "Ảnh xám (1 kênh)"
        self.badge_orig.setText(f"{w} × {h} px  •  {channels}")
        self.badge_proc.setText("Chưa áp dụng nâng cao")
        self.btn_save.setEnabled(True)
        self.btn_report.setEnabled(True)
        self.refresh_all()
        self.statusBar().showMessage(f"Đã mở: {Path(path).name}  ({w} × {h} px)")
        self.log(f"Mở ảnh thành công: {Path(path).name} ({w} x {h} px).")

    def toggle_gray(self, checked):
        self.canvas_orig.set_image(self.original, checked)
        self.canvas_proc.set_image(self.processed, checked)

    def reset(self):
        if self.original is None:
            return
        self.processed = self.original.copy()
        self.method_used = None
        self.clear_otsu()
        self.canvas_proc.set_image(self.processed, self.chk_gray.isChecked())
        self.refresh_all()
        self.log("Đặt lại ảnh về trạng thái ban đầu.")
        self.statusBar().showMessage("Đã đặt lại trạng thái ban đầu")

    def apply_enhancement(self):
        if not self.need_image():
            return
        method = self.combo_method.currentText()
        t0 = time.perf_counter()
        self.processed = enhance_color_image(self.original, method)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.method_used = method
        self.clear_otsu()
        self.canvas_proc.set_image(self.processed, self.chk_gray.isChecked())
        self.refresh_all()
        self.log(f"Áp dụng {method} thành công trong {elapsed_ms:.1f} ms.")
        self.statusBar().showMessage(f"Đã hoàn thành {method} ({elapsed_ms:.1f} ms)")

    def apply_otsu(self):
        if not self.need_image():
            return
        res = {}
        for key, img in (("orig", self.original), ("proc", self.processed)):
            gray = to_gray(img)
            t, mask, var = otsu_threshold_full(gray)
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
        self.log(
            f"Phân đoạn Otsu: Ngưỡng gốc T={res['orig']['t']}, Ngưỡng sau nâng cao T={res['proc']['t']} | "
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
            self, "Chọn mặt nạ chuẩn Ground Truth", "", "Ảnh (*.png *.jpg *.jpeg *.bmp *.tif)")
        if not path:
            return
        gt = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            QMessageBox.warning(self, "Lỗi", "Không thể đọc tệp Ground Truth.")
            return

        h, w = self.original.shape[:2]
        if gt.shape[:2] != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        self.gt_mask = np.where(gt > 127, 255, 0).astype(np.uint8)
        self.btn_clear_gt.setVisible(True)
        self.log(f"Đã nạp Ground Truth: {Path(path).name}.")
        self.refresh_all()

    def clear_ground_truth(self):
        self.gt_mask = None
        self.btn_clear_gt.setVisible(False)
        self.refresh_all()
        self.log("Đã xóa dữ liệu Ground Truth khỏi phiên làm việc.")

    # ---------------- Cập nhật biểu đồ & Bảng so sánh ----------------
    def refresh_all(self):
        if self.original is None:
            self.update_kpi_cards(None, None)
            return
        g1, g2 = to_gray(self.original), to_gray(self.processed)
        t1 = self.otsu["orig"]["t"] if self.otsu else None
        t2 = self.otsu["proc"]["t"] if self.otsu else None

        self.plot_h_orig.draw_histogram(g1, "Histogram ảnh gốc", t1)
        self.plot_h_proc.draw_histogram(g2, "Histogram ảnh sau nâng cao", t2)

        s1 = calculate_image_stats(g1)
        s2 = calculate_image_stats(g2)

        self.badge_proc.setText(self.method_used if self.method_used else "Chưa áp dụng nâng cao")

        self.lbl_orig.setText(
            f"Mức xám: [{s1['min']}..{s1['max']}]   |   Trung bình: {s1['mean']:.1f}   |   Độ lệch chuẩn σ: {s1['std']:.1f}"
        )
        self.lbl_proc.setText(
            f"Mức xám: [{s2['min']}..{s2['max']}]   |   Trung bình: {s2['mean']:.1f}   |   Độ lệch chuẩn σ: {s2['std']:.1f}"
            + (f"   |   {self.method_used}" if self.method_used else "")
        )
        self.update_kpi_cards(s1, s2)
        self.refresh_table(s1, s2)

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

        self.table.setRowCount(len(data_rows))
        for r, row in enumerate(data_rows):
            self.table.setRowHeight(r, 32)
            metric_name = row[0]
            is_key = metric_name in key_metrics
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                if c == 0:
                    weight = QFont.Weight.Bold if is_key else QFont.Weight.DemiBold
                    item.setFont(QFont("Segoe UI", 9.5, weight))
                    item.setForeground(QColor("#0F172A"))
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
            "1. SO SÁNH THỐNG KÊ PHÂN BỐ MỨC XÁM & ĐỘ TƯƠNG PHẢN",
            "-" * 80,
            f"{'Chỉ số':<32}{'Ảnh gốc':>15}{'Sau nâng cao':>16}{'Thay đổi':>16}",
            f"{'Mức xám nhỏ nhất (min)':<32}{s1['min']:>15}{s2['min']:>16}{s2['min']-s1['min']:>+16d}",
            f"{'Mức xám lớn nhất (max)':<32}{s1['max']:>15}{s2['max']:>16}{s2['max']-s1['max']:>+16d}",
            f"{'Dải động (max - min)':<32}{s1['range']:>15}{s2['range']:>16}{s2['range']-s1['range']:>+16d}",
            f"{'Giá trị xám trung bình':<32}{s1['mean']:>15.2f}{s2['mean']:>16.2f}{s2['mean']-s1['mean']:>+16.2f}",
            f"{'Độ lệch chuẩn (Tương phản)':<32}{s1['std']:>15.2f}{s2['std']:>16.2f}{s2['std']-s1['std']:>+16.2f}",
            f"{'Entropy (Độ phong phú chi tiết)':<32}{s1['entropy']:>15.2f}{s2['entropy']:>16.2f}{s2['entropy']-s1['entropy']:>+16.2f}",
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
