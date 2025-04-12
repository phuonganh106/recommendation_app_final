# 🚀 Hệ thống gợi ý sản phẩm Shopee

**Người thực hiện:** Phạm Thị Mai Linh  
**Ngày báo cáo:** 13/04/2025

## 📌 Giới thiệu

Dự án xây dựng hệ thống gợi ý sản phẩm thời trang nam trên Shopee, bao gồm:

- **Gợi ý theo nội dung sản phẩm (Product-based):**
  - Sử dụng mô hình Cosine Similarity
  - Người dùng nhập mô tả sản phẩm để tìm sản phẩm tương tự.

- **Gợi ý theo người dùng (User-based):**
  - Sử dụng thuật toán Surprise SVD
  - Hệ thống gợi ý sản phẩm dựa trên lịch sử đánh giá của người dùng.

Ứng dụng phù hợp để demo báo cáo, hoặc mở rộng thành ứng dụng thực tế.

---

## 📂 Cấu trúc project

├── app.py # App chính ├── product_cosine.pkl # Model Cosine đã train ├── surprise_model.pkl # Model Surprise đã train ├── requirements.txt # Thư viện cần thiết └── README.md # Hướng dẫn sử dụng

yaml
Copy
Edit

---

## 🚀 Hướng dẫn cài đặt Local

### 1. Tạo thư mục project và copy file vào

Copy đầy đủ các file:  
`app.py`, `requirements.txt`, `README.md`, `product_cosine.pkl`, `surprise_model.pkl`

### 2. Cài đặt thư viện

Mở terminal tại thư mục project, chạy lệnh:

```bash
pip install -r requirements.txt