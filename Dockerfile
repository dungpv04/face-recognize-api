FROM python:3.12.5
# Thiết lập thư mục làm việc
WORKDIR /app
COPY /src /app
COPY main.py /app
COPY pyproject.toml /app
# Cập nhật pip và cài đặt UV
RUN pip install --no-cache-dir uv

# Tạo virtual environment
RUN uv venv --python 3.12.5

# Cài đặt dependencies từ pyproject.toml
RUN uv pip install -r pyproject.toml || uv lock

# Chạy ứng dụng
CMD ["uv", "run", "main.py"]
