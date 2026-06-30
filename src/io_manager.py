import os
import uuid
import shutil
from src.config import TMP_BASE_DIR

class IOManager:
    def __init__(self):
        if not os.path.exists(TMP_BASE_DIR):
            os.makedirs(TMP_BASE_DIR, exist_ok=True)
            
    def create_job_dir(self) -> str:
        job_id = str(uuid.uuid4())
        job_dir = os.path.join(TMP_BASE_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        return job_dir
        
    def cleanup_job_dir(self, job_dir: str):
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)
