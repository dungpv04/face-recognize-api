# Hướng Dẫn Cài Đặt Môi Trường Python với UV

## 1. Cài đặt UV cho Python

UV có thể được cài đặt bằng **pip** hoặc **pipx**:

- Cài bằng pip:
  ```sh
  pip install uv
  ```
- Cài bằng pipx:
  ```sh
  pipx install uv
  ```

## 2. Sử dụng UV để tạo và cài đặt venv Python 3.12.5

Tạo môi trường ảo với **Python 3.12.5**:

```sh
uv venv -p 3.12.5 .venv
```

Kích hoạt môi trường ảo:

- **Trên Windows**:
  ```sh
  .venv\Scripts\activate
  ```
- **Trên macOS/Linux**:
  ```sh
  source .venv/bin/activate
  ```

## 3. Cài đặt dependencies từ file TOML

Nếu đã có file TOML chứa danh sách dependencies, sử dụng UV để tải và cài đặt:

```sh
uv pip install -r pyproject.toml
```

## 4. Di chuyển vào thư mục dự án

```sh
cd src/face-recognize-api/
```

## 5. Chạy ứng dụng với Uvicorn

Chạy server bằng **Uvicorn**:

```sh
uvicorn app:app --reload
```

Server sẽ khởi động và có thể truy cập qua trình duyệt hoặc API client.

