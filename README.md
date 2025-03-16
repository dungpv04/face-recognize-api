# Hướng dẫn cài đặt và chạy dự án với UV, PostgreSQL và pgvector

## 1. Cài đặt UV cho Python
UV có thể được cài đặt bằng `pip` hoặc `pipx`:

- Cài bằng pip:
  ```sh
  pip install uv
  ```
- Cài bằng pipx:
  ```sh
  pipx install uv
  ```

## 2. Sử dụng UV để tạo và cài đặt venv Python 3.12.5

### Tạo môi trường ảo với Python 3.12.5:
```sh
uv venv -p 3.12.5 .venv
```

### Kích hoạt môi trường ảo:
- Trên Windows:
  ```sh
  .venv\Scripts\activate
  ```
- Trên macOS/Linux:
  ```sh
  source .venv/bin/activate
  ```

## 3. Cài đặt dependencies từ file TOML
Nếu đã có file TOML chứa danh sách dependencies, sử dụng UV để tải và cài đặt:
```sh
uv pip install -r pyproject.toml
```

## 4. Cài đặt PostgreSQL và pgvector (Nên sử dụng Docker)

### Cài đặt pgvector
Làm theo hướng dẫn tại: [pgvector GitHub](https://github.com/pgvector/pgvector)

## 5. Thiết lập Environment Variables
Environment variables có thể được thiết lập theo nhiều cách, tùy chọn sử dụng:
- File `.env`
- Docker environment variables
- Environment variables mặc định của hệ điều hành

Danh sách biến:
```env
db_name=<Tên database>
username=<Tên người dùng>
password=<Mật khẩu>
hostname=<Tên host>
port=5432  # Mặc định của PostgreSQL
```
*Lưu ý: Biến môi trường có phân biệt chữ hoa và chữ thường (case-sensitive).*

## 6. Chạy dự án

```sh
uv run main.py
```

Mặc định, **Uvicorn** sẽ chạy trên địa chỉ:
```
http://127.0.0.1:8000/
```

## 7. Đọc tài liệu API
Truy cập vào đường dẫn sau để xem tài liệu API từ Swagger:
```
http://127.0.0.1:8000/docs
```

