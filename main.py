import sys
import os

sys.path.append(os.path.abspath("src"))  # Đảm bảo Python tìm thấy module

import uvicorn
from face_recoginze_api.app import app  # Đúng tên module

if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8000, reload=False)
