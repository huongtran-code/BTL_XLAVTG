# Đề tài 13: Tính Histogram và Nâng cao Chất lượng Ảnh bằng Histogram

Phần mềm máy tính chuyên dụng phục vụ học tập, nghiên cứu thực nghiệm và báo cáo Bài tập lớn môn **Xử lý ảnh và Thị giác máy tính**.

Hệ thống được xây dựng với kiến trúc chuẩn, giao diện máy tính hiện đại (PySide6), tối ưu hóa trải nghiệm người dùng với bố cục liền mạch, hiển thị sắc nét không vỡ tỷ lệ và bám sát tuyệt đối 100% yêu cầu học thuật của đề tài.

---

## 1. Yêu cầu đề tài & Thuật toán cốt lõi

Toàn bộ các thuật toán cốt lõi đều được **tự cài đặt bằng thư viện toán học NumPy**, hoàn toàn không sử dụng các hàm có sẵn như `cv2.calcHist`, `cv2.equalizeHist`, `cv2.threshold(..., THRESH_OTSU)`:

1. **Tính Histogram 256 mức xám**:
   $$h(r_k) = n_k, \quad k = 0, 1, \dots, 255$$
   Đếm tần suất xuất hiện của từng giá trị pixel trong ảnh mức xám bằng `np.unique(..., return_counts=True)`.

2. **Kéo giãn tương phản tuyến tính (Contrast Stretching)**:
   $$s = \frac{r - r_{\min}}{r_{\max} - r_{\min}} \times 255$$
   Ánh xạ dải mức xám thực tế hẹp $[r_{\min}, r_{\max}]$ về trọn vẹn toàn dải $[0, 255]$, cải thiện độ tương phản tổng thể.

3. **Cân bằng Histogram (Histogram Equalization)**:
   $$s_k = \text{round}\left( \frac{\text{CDF}(k) - \text{CDF}_{\min}}{N - \text{CDF}_{\min}} \times 255 \right)$$
   Sử dụng hàm phân bố tích lũy CDF để tái phân bố đều mật độ mức xám, làm nổi bật chi tiết ở cả vùng tối lẫn vùng sáng.
   - *Đối với ảnh màu BGR*: Tách kênh trong không gian màu **CIE LAB**, chỉ nâng cao kênh độ sáng $L$ (Lightness) rồi ghép lại với kênh $A, B$ để giữ trọn vẹn màu sắc tự nhiên, tránh bị sai lệch màu.

4. **Phân đoạn ngưỡng Otsu (Otsu's Thresholding)**:
   Tối đa hóa phương sai giữa 2 lớp (Between-Class Variance):
   $$\sigma_B^2(t) = \omega_0(t) \cdot \omega_1(t) \cdot [\mu_0(t) - \mu_1(t)]^2 = \frac{[\mu_T \cdot \omega_0(t) - \mu(t)]^2}{\omega_0(t) \cdot (1 - \omega_0(t))}$$
   Tự động quét tìm ngưỡng tối ưu $T = \arg\max_t \sigma_B^2(t)$ để tách đối tượng khỏi nền.

5. **So sánh định lượng & Trực quan hóa kết quả phân đoạn**:
   - **Chỉ số IoU (Jaccard Index)**: Đo mức độ trùng khớp vùng đối tượng:
     $$\text{IoU} = \frac{|M_{\text{gốc}} \cap M_{\text{nâng cao}}|}{|M_{\text{gốc}} \cup M_{\text{nâng cao}}|}$$
   - **Chỉ số Dice (F1-Score)**:
     $$\text{Dice} = \frac{2 \cdot |M_{\text{gốc}} \cap M_{\text{nâng cao}}|}{|M_{\text{gốc}}| + |M_{\text{nâng cao}}|}$$
   - **Tỷ lệ pixel sai khác**:
     $$\text{Diff Rate} = \frac{\sum (M_{\text{gốc}} \ne M_{\text{nâng cao}})}{N} \times 100\%$$
   - **Bản đồ sai khác trực quan (Difference Map)**:
     - *Màu trắng*: Vùng đối tượng đồng thuận ở cả 2 ảnh.
     - *Màu xanh lục*: Pixel đối tượng mới được phục hồi thêm sau nâng cao chất lượng.
     - *Màu đỏ*: Pixel bị loại bỏ sau nâng cao chất lượng (loại bỏ nhiễu/bóng mờ).
     - *Màu tối*: Vùng nền chung.
   - **Chế độ xem linh hoạt**:
     - Mặt nạ nhị phân (Binary Mask)
     - Phủ màu bán trong suốt lên đối tượng (Color Overlay) kèm đường viền (Contour)
     - Tách riêng đối tượng (Foreground Cutout)
   - **Hỗ trợ nạp Ground Truth (tùy chọn)**: Đối chiếu với mặt nạ chuẩn để kiểm chứng sự cải thiện độ chính xác phân đoạn ($\Delta \text{IoU}$).

---

## 2. Cấu trúc thư mục

```text
De13_BTL/
├── assets/                     # Tài nguyên biểu tượng, đồ họa và ảnh mẫu
│   ├── arrow_down.png          # Mũi tên dropdown đồng bộ giao diện
│   ├── logo.ico                # Biểu tượng ứng dụng Windows
│   ├── logo.png                # Logo độ phân giải cao
│   ├── logo_64.png             # Logo 64x64
│   └── sample_test.png         # Ảnh mẫu thử nghiệm
├── logs/                       # Tệp nhật ký lịch sử thao tác theo ngày (history_YYYY-MM-DD.log)
├── main.py                     # Chương trình chính (UI & toàn bộ thuật toán)
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Hướng dẫn chi tiết
```

---

## 3. Cài đặt & Hướng dẫn sử dụng

### Bước 1: Kích hoạt môi trường ảo & cài đặt thư viện
Tại thư mục gốc của dự án:
```powershell
# Kích hoạt môi trường ảo (nếu đã có sẵn .venv)
.\.venv\Scripts\Activate.ps1

# Hoặc tạo mới và cài đặt thư viện
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Bước 2: Khởi chạy phần mềm
```powershell
python De13_BTL\main.py
```

### Bước 3: Các thao tác trên giao diện
1. Nhấn **Chọn ảnh...** (hoặc kéo thả tệp ảnh trực tiếp vào khung ảnh gốc).
2. Chọn phương pháp: **Cân bằng histogram** hoặc **Kéo giãn tương phản**.
3. Nhấn **Nâng cao ảnh**: Ảnh sau nâng cao và biểu đồ histogram được cập nhật tức thì.
4. Nhấn **Phân đoạn Otsu**: Hệ thống tự động chuyển sang tab phân đoạn, tính toán ngưỡng $T$ cho cả 2 ảnh, vẽ đường cong phương sai $\sigma_B^2(t)$, sinh bản đồ sai khác, và đo lường các chỉ số tương đồng (IoU, Dice, % khác biệt).
5. Thay đổi tùy chọn xem: Nhấp đúp chuột vào bất kỳ ô ảnh nào để phóng to toàn màn hình; chuyển đổi giữa *Mặt nạ nhị phân*, *Phủ màu đối tượng* và *Tách riêng đối tượng*.
6. Xem **Bảng so sánh chi tiết** để đối chiếu các chỉ số thống kê (min, max, dải động, mean, std, entropy, IoU, Dice).
7. Nhấn **Lưu ảnh...** để xuất toàn bộ các ảnh kết quả và **Xuất báo cáo...** để nhận tệp văn bản tổng hợp phục vụ viết báo cáo tiểu luận.
