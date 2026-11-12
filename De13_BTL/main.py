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
    - Hỗ trợ kéo thả tệp ảnh trực tiếp và nhấp đúp để phóng to.
    """
    file_dropped = Signal(str)
    double_clicked = Signal()

    def __init__(self, placeholder="Chưa có ảnh", parent=None):
        super().__init__(parent)
        self.image = None
        self.qimage = None
        self.gray_view = False
        self.placeholder = placeholder
        self.setAcceptDrops(True)
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
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        clip = QPainterPath()
        clip.addRoundedRect(rect, 8, 8)
        p.setClipPath(clip)
        p.fillRect(self.rect(), QColor("#F4F7FB" if self.qimage is None else "#242B33"))
        p.setClipping(False)

        p.setPen(QPen(QColor("#D5DCE6"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect, 8, 8)

        if self.qimage is None:
            p.setPen(QColor("#7B8794"))
            p.drawText(self.rect().adjusted(12, 0, -12, 0),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       self.placeholder)
            return

        aw, ah = self.width() - 14, self.height() - 14
        iw, ih = self.qimage.width(), self.qimage.height()
        if aw <= 0 or ah <= 0:
            return

        s = min(aw / iw, ah / ih)
        dw, dh = iw * s, ih * s
        target = QRectF(self.width() / 2 - dw / 2, self.height() / 2 - dh / 2, dw, dh)

        img_clip = QPainterPath()
        img_clip.addRoundedRect(target, 5, 5)
        p.setClipPath(img_clip)
        p.drawImage(target, self.qimage)

    def mouseDoubleClickEvent(self, event):
        if self.image is not None:
            self.double_clicked.emit()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
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
        self.setWindowTitle(f"{title} - {w} x {h} px")
        self.resize(920, 680)
        lay = QVBoxLayout(self)
        canvas = ImageCanvas()
        canvas.set_image(image)
        lay.addWidget(canvas)


class PlotCanvas(FigureCanvas):
    """Widget vẽ đồ thị matplotlib trang nhã, sắc nét."""
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(4, 2.2), dpi=100, facecolor="white")
        plt_rc = {
            "axes.edgecolor": "#C5CED9",
            "axes.labelcolor": "#52606D",
            "xtick.color": "#52606D",
            "ytick.color": "#52606D"
        }
        matplotlib.rcParams.update(plt_rc)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(110)

    def _axes(self, title):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title(title, fontsize=8.5, fontweight="bold", color="#243B53")
        ax.set_xlim(0, 255)
        ax.set_xlabel("Mức xám", fontsize=8)
        ax.set_ylabel("Số pixel", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.5, color="#E4E9F0")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        return ax

    def draw_histogram(self, gray, title, threshold=None):
        ax = self._axes(title)
        if gray is not None:
            hist = calculate_histogram(gray)
            ax.bar(np.arange(256), hist, width=1.0, color="#5B8DD6")
            if threshold is not None:
                ax.axvline(threshold, color="#E03131", linewidth=1.2, linestyle="--")
                ax.text(min(threshold + 3, 205), ax.get_ylim()[1] * 0.88,
                        f"T = {threshold}", color="#E03131", fontsize=8, fontweight="bold")
        self.figure.tight_layout(pad=0.6)
        self.draw_idle()

    def draw_variance(self, var_before, var_after, t_before, t_after):
        ax = self._axes("Phương sai giữa hai lớp theo ngưỡng t (Otsu)")
        ax.set_ylabel("Phương sai σ²", fontsize=8)
        if var_before is not None:
            ax.plot(var_before, color="#7B8794", linewidth=1.1, label=f"Ảnh gốc (T = {t_before})")
            ax.plot(var_after, color="#2F6DB5", linewidth=1.4, label=f"Sau nâng cao (T = {t_after})")
            ax.axvline(t_before, color="#7B8794", linestyle=":", linewidth=0.9)
            ax.axvline(t_after, color="#2F6DB5", linestyle=":", linewidth=0.9)
            ax.legend(fontsize=7, frameon=False)
        self.figure.tight_layout(pad=0.6)
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
        self.resize(1260, 800)
        self.setMinimumSize(920, 620)

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
    def build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)
        self.setCentralWidget(central)

        # 1. Thanh công cụ chính
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self.btn_open = QPushButton("Chọn ảnh...")
        self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(self.open_dialog)

        self.btn_reset = QPushButton("Đặt lại")
        self.btn_reset.clicked.connect(self.reset)

        self.combo_method = QComboBox()
        self.combo_method.addItems(self.METHODS)

        self.btn_apply = QPushButton("Nâng cao ảnh")
        self.btn_apply.setObjectName("primary")
        self.btn_apply.clicked.connect(self.apply_enhancement)

        self.btn_otsu = QPushButton("Phân đoạn Otsu")
        self.btn_otsu.setObjectName("primary")
        self.btn_otsu.clicked.connect(self.apply_otsu)

        self.chk_gray = QCheckBox("Xem ảnh xám")
        self.chk_gray.toggled.connect(self.toggle_gray)

        self.btn_save = QPushButton("Lưu ảnh...")
        self.btn_save.clicked.connect(self.save_results)

        self.btn_report = QPushButton("Xuất báo cáo...")
        self.btn_report.clicked.connect(self.export_report)

        for w in (self.btn_open, self.btn_reset):
            bar.addWidget(w)
        bar.addSpacing(10)
        bar.addWidget(QLabel("Phương pháp:"))
        bar.addWidget(self.combo_method)
        bar.addWidget(self.btn_apply)
        bar.addWidget(self.btn_otsu)
        bar.addSpacing(10)
        bar.addWidget(self.chk_gray)
        bar.addStretch()
        bar.addWidget(self.btn_save)
        bar.addWidget(self.btn_report)
        root.addLayout(bar)

        self.btn_save.setEnabled(False)
        self.btn_report.setEnabled(False)

        # 2. Phân chia giao diện: Phần trên (2 ảnh) & Phần dưới (Tabs phân tích)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)

        # Phần trên: So sánh Ảnh gốc vs Ảnh sau nâng cao
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setChildrenCollapsible(False)

        self.canvas_orig = ImageCanvas("Chưa có ảnh (nhấn 'Chọn ảnh' hoặc kéo thả ảnh vào đây)")
        self.canvas_proc = ImageCanvas("Chưa áp dụng nâng cao chất lượng")
        self.canvas_orig.file_dropped.connect(self.load_image)
        self.canvas_orig.double_clicked.connect(lambda: self.zoom("Ảnh gốc", self.original))
        self.canvas_proc.double_clicked.connect(lambda: self.zoom("Ảnh sau nâng cao", self.processed))

        self.lbl_orig = QLabel("")
        self.lbl_proc = QLabel("")

        top_splitter.addWidget(self.create_card("Ảnh gốc", self.canvas_orig, self.lbl_orig))
        top_splitter.addWidget(self.create_card("Ảnh sau nâng cao", self.canvas_proc, self.lbl_proc))
        main_splitter.addWidget(top_splitter)

        # Phần dưới: Hệ thống các Tab phân tích chuyên sâu
        self.tabs = QTabWidget()

        # Tab 1: Histogram
        t1 = QWidget()
        l1 = QHBoxLayout(t1)
        l1.setContentsMargins(8, 8, 8, 8)
        self.plot_h_orig = PlotCanvas()
        self.plot_h_proc = PlotCanvas()
        l1.addWidget(self.plot_h_orig)
        l1.addWidget(self.plot_h_proc)
        self.tabs.addTab(t1, "Biểu đồ Histogram")

        # Tab 2: Phân đoạn Otsu & So sánh chi tiết
        t2 = QWidget()
        l2_main = QVBoxLayout(t2)
        l2_main.setContentsMargins(8, 6, 8, 6)
        l2_main.setSpacing(6)

        # Thanh điều khiển nhỏ trong Tab Otsu
        sub_bar = QHBoxLayout()
        sub_bar.setSpacing(8)
        sub_bar.addWidget(QLabel("Chế độ hiển thị:"))

        self.combo_view = QComboBox()
        self.combo_view.addItems(self.VIEW_MODES)
        self.combo_view.currentTextChanged.connect(self.on_view_mode_changed)
        sub_bar.addWidget(self.combo_view)

        sub_bar.addSpacing(10)
        self.btn_load_gt = QPushButton("Nạp Ground Truth...")
        self.btn_load_gt.clicked.connect(self.load_ground_truth)
        sub_bar.addWidget(self.btn_load_gt)

        self.btn_clear_gt = QPushButton("Xóa GT")
        self.btn_clear_gt.setVisible(False)
        self.btn_clear_gt.clicked.connect(self.clear_ground_truth)
        sub_bar.addWidget(self.btn_clear_gt)

        sub_bar.addStretch()
        self.lbl_otsu_summary = QLabel("Chưa phân đoạn Otsu")
        self.lbl_otsu_summary.setStyleSheet(
            "background: #EEF3F9; color: #243B53; padding: 4px 12px; border-radius: 6px; font-weight: 600;"
        )
        sub_bar.addWidget(self.lbl_otsu_summary)
        l2_main.addLayout(sub_bar)

        # 4 khung hiển thị kết quả phân đoạn & so sánh
        otsu_splitter = QSplitter(Qt.Orientation.Horizontal)
        otsu_splitter.setChildrenCollapsible(False)

        self.canvas_mask_orig = ImageCanvas("Chưa phân đoạn")
        self.canvas_mask_proc = ImageCanvas("Chưa phân đoạn")
        self.canvas_diff = ImageCanvas("Chưa có bản đồ khác biệt")

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

        otsu_splitter.addWidget(self.create_card("Otsu - Ảnh gốc", self.canvas_mask_orig, self.lbl_mask_orig))
        otsu_splitter.addWidget(self.create_card("Otsu - Sau nâng cao", self.canvas_mask_proc, self.lbl_mask_proc))
        otsu_splitter.addWidget(self.create_card("Bản đồ sai khác phân đoạn", self.canvas_diff, self.lbl_diff))
        otsu_splitter.addWidget(self.create_card("Đường cong phương sai σ²(t)", self.plot_var, QLabel("Ngưỡng tối ưu max σ²_B(t)")))

        l2_main.addWidget(otsu_splitter, 1)
        self.tabs.addTab(t2, "Phân đoạn Otsu & So sánh")

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
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.tabs.addTab(self.table, "Bảng so sánh chi tiết")

        # Tab 4: Nhật ký & Lịch sử thao tác (Ghi bền vững vào file log theo ngày)
        t4 = QWidget()
        l4 = QVBoxLayout(t4)
        l4.setContentsMargins(8, 8, 8, 8)
        l4.setSpacing(6)

        log_bar = QHBoxLayout()
        self.lbl_log_info = QLabel("📁 Đang kết nối tệp nhật ký...")
        self.lbl_log_info.setStyleSheet("color: #52606D; font-size: 8.5pt;")

        self.btn_reload_log = QPushButton("Tải lại lịch sử")
        self.btn_reload_log.clicked.connect(self.load_today_log_to_ui)

        self.btn_open_log_dir = QPushButton("Mở thư mục Log")
        self.btn_open_log_dir.clicked.connect(self.open_log_directory)

        self.btn_clear_log_view = QPushButton("Xóa khung xem")
        self.btn_clear_log_view.clicked.connect(self.log_box_clear_view)

        log_bar.addWidget(self.lbl_log_info)
        log_bar.addStretch()
        log_bar.addWidget(self.btn_reload_log)
        log_bar.addWidget(self.btn_open_log_dir)
        log_bar.addWidget(self.btn_clear_log_view)
        l4.addLayout(log_bar)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; font-size: 9pt;")
        l4.addWidget(self.log_box)

        self.tabs.addTab(t4, "Nhật ký xử lý (Lịch sử)")

        main_splitter.addWidget(self.tabs)
        main_splitter.setSizes([400, 350])
        root.addWidget(main_splitter, 1)

        self.statusBar().showMessage("Sẵn sàng")
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

    def create_card(self, title, widget, label=None):
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 10, 8, 8)
        lay.setSpacing(4)
        lay.addWidget(widget, 1)
        if label is not None:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-size: 8.5pt; color: #52606D; padding-top: 2px;")
            lay.addWidget(label)
        return box

    def apply_style(self):
        self.setStyleSheet(f"""
            QWidget {{ font-family: 'Segoe UI'; font-size: 9.5pt; color: #1F2933; }}
            QMainWindow {{ background: #EAEFF5; }}
            QMenuBar {{ background: #FFFFFF; border-bottom: 1px solid #D5DCE6; }}
            QMenuBar::item {{ padding: 5px 12px; border-radius: 6px; }}
            QMenuBar::item:selected {{ background: #E3ECF8; }}
            QStatusBar {{ background: #FFFFFF; border-top: 1px solid #D5DCE6; color: #52606D; }}

            QGroupBox {{ background: #FFFFFF; border: 1px solid #D5DCE6; border-radius: 10px;
                        margin-top: 12px; padding-top: 10px; font-weight: 600; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px;
                               color: #243B53; background: #EAEFF5; border-radius: 4px; }}

            QPushButton {{ background: #FFFFFF; border: 1px solid #C5CED9; border-radius: 8px;
                          padding: 6px 15px; color: #243B53; }}
            QPushButton:hover {{ background: #F0F5FB; border-color: #9FB3C8; }}
            QPushButton:pressed {{ background: #E3ECF8; }}
            QPushButton:disabled {{ color: #9AA5B1; background: #F2F4F7; border-color: #E1E6EC; }}
            QPushButton#primary {{ background: #2F6DB5; color: #FFFFFF; border: 1px solid #2F6DB5;
                                  font-weight: 600; }}
            QPushButton#primary:hover {{ background: #265C9C; }}
            QPushButton#primary:pressed {{ background: #1F4D84; }}

            QComboBox {{ background: #FFFFFF; border: 1px solid #C5CED9; border-radius: 8px;
                        padding: 5px 12px; min-width: 140px; }}
            QComboBox:hover {{ border-color: #9FB3C8; }}
            QComboBox::drop-down {{ border: none; width: 22px; }}
            QComboBox::down-arrow {{ image: url({ARROW_PATH}); width: 14px; height: 14px; }}
            QComboBox QAbstractItemView {{ border: 1px solid #C5CED9; border-radius: 6px;
                                          selection-background-color: #E3ECF8;
                                          selection-color: #1F2933; background: white; }}
            QCheckBox {{ spacing: 6px; }}

            QTabWidget::pane {{ background: #FFFFFF; border: 1px solid #D5DCE6; border-radius: 10px;
                               top: -1px; }}
            QTabBar::tab {{ background: transparent; padding: 7px 18px; margin-right: 4px;
                           border-top-left-radius: 8px; border-top-right-radius: 8px;
                           color: #52606D; }}
            QTabBar::tab:hover {{ background: #DDE6F0; }}
            QTabBar::tab:selected {{ background: #FFFFFF; color: #2F6DB5; font-weight: 600;
                                    border: 1px solid #D5DCE6; border-bottom-color: #FFFFFF; }}

            QTableWidget {{ background: #FFFFFF; border: none; gridline-color: #E4E9F0;
                           alternate-background-color: #F8FAFC; }}
            QHeaderView::section {{ background: #EEF3F9; border: none; border-bottom: 1px solid #D5DCE6;
                                   padding: 7px; font-weight: 600; color: #243B53; }}
            QPlainTextEdit {{ background: #FFFFFF; border: none; border-radius: 10px; padding: 6px; }}

            QSplitter::handle {{ background: transparent; }}
            QScrollBar:vertical {{ background: transparent; width: 10px; }}
            QScrollBar::handle:vertical {{ background: #C5CED9; border-radius: 5px; min-height: 24px; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
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
        self.lbl_log_info.setText(f"📁 Tệp nhật ký hôm nay: {rel_name}")

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
        self.btn_save.setEnabled(True)
        self.btn_report.setEnabled(True)
        self.refresh_all()
        self.statusBar().showMessage(f"Đã mở: {Path(path).name}  ({w} x {h} px)")
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
            return
        g1, g2 = to_gray(self.original), to_gray(self.processed)
        t1 = self.otsu["orig"]["t"] if self.otsu else None
        t2 = self.otsu["proc"]["t"] if self.otsu else None

        self.plot_h_orig.draw_histogram(g1, "Histogram ảnh gốc", t1)
        self.plot_h_proc.draw_histogram(g2, "Histogram ảnh sau nâng cao", t2)

        s1 = calculate_image_stats(g1)
        s2 = calculate_image_stats(g2)

        self.lbl_orig.setText(
            f"Mức xám [{s1['min']}..{s1['max']}]  |  Trung bình {s1['mean']:.1f}  |  Độ lệch chuẩn {s1['std']:.1f}"
        )
        self.lbl_proc.setText(
            f"Mức xám [{s2['min']}..{s2['max']}]  |  Trung bình {s2['mean']:.1f}  |  Độ lệch chuẩn {s2['std']:.1f}"
            + (f"  |  {self.method_used}" if self.method_used else "")
        )
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

        self.table.setRowCount(len(data_rows))
        for r, row in enumerate(data_rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                if c == 0:
                    item.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
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
