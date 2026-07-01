import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.orchestrator import Orchestrator
from src.io_manager import IOManager

app = FastAPI(title="Interview Anonymizer")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory status store
tasks = {}

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

def background_process(task_id: str, input_path: str, output_path: str):
    tasks[task_id] = {"status": "processing", "message": "Initializing...", "logs": ["Initializing..."]}
    
    def status_callback(msg: str):
        tasks[task_id]["message"] = msg
        tasks[task_id]["logs"].append(msg)
        
    io_manager = IOManager()
    job_dir = io_manager.create_job_dir()
    
    try:
        Orchestrator.process_video_job(input_path, output_path, job_dir, status_cb=status_callback)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = "Done!"
        tasks[task_id]["logs"].append("Done!")
    except Exception as e:
        print(f"Error in background_process: {e}")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["message"] = str(e)
        tasks[task_id]["logs"].append(str(e))
    finally:
        io_manager.cleanup_job_dir(job_dir)
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")
        
    task_id = str(uuid.uuid4())
    input_path = os.path.join("uploads", f"{task_id}.mp4")
    output_path = os.path.join("output", f"{task_id}_anonymized.mp4")
    
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
        
    tasks[task_id] = {"status": "queued", "message": "Waiting in queue...", "logs": ["Waiting in queue..."]}
    
    background_tasks.add_task(background_process, task_id, input_path, output_path)
    
    return {"task_id": task_id, "status": "queued", "message": "Waiting in queue..."}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, **tasks[task_id]}

@app.get("/api/download/{task_id}")
async def download_video(task_id: str):
    if task_id not in tasks or tasks[task_id].get("status") != "completed":
        raise HTTPException(status_code=400, detail="File not ready or task failed")
        
    output_path = os.path.join("output", f"{task_id}_anonymized.mp4")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        output_path, 
        media_type="video/mp4", 
        filename=f"anonymized_{task_id}.mp4"
    )

if __name__ == "__main__":
    import uvicorn
    # Important on Windows for multiprocessing
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
