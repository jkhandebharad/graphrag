from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .cosmos_client import init_cosmos_db
from .cosmos_crud import save_document, index_documents_for_case, get_answer
from .cosmos_input import InputManager
from dotenv import load_dotenv
import os

load_dotenv()

# Validate CosmosDB configuration
if not os.getenv("COSMOS_ENDPOINT") or not os.getenv("COSMOS_KEY"):
    print("[WARNING] COSMOS_ENDPOINT and COSMOS_KEY not set. Using local file storage.")
    print("[INFO] To use CosmosDB, set these environment variables in .env file")
    USE_COSMOS = False
else:
    USE_COSMOS = True
    print("[SUCCESS] CosmosDB configuration found")

# Pydantic models
class QueryRequest(BaseModel):
    firm_id: str
    case_id: str
    question: str
    user_id: str = ""  # Optional if using session/auth

class QueryResponse(BaseModel):
    answer: str
    references: list = []
    chunks: list = []
    graph_paths: list = []

# Initialize app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
def on_startup():
    if USE_COSMOS:
        print("[INFO] CosmosDB will be initialized per firm as needed...")
        print("[SUCCESS] CosmosDB ready for firm-specific initialization!")
    else:
        print("[WARNING] Running in local file mode (CosmosDB not configured)")

# Serve frontend
@app.get("/")
async def serve_index():
    return FileResponse("index.html", media_type="text/html")

# Upload endpoint
@app.post("/upload")
async def upload_files(
    case_id: str = Form(...),
    firm_id: str = Form(...),  # Required firm_id from frontend
    files: list[UploadFile] = File(...),  # Accept multiple files
    background_tasks: BackgroundTasks = None
):
    print(f"\n[INFO] ===== NEW UPLOAD REQUEST =====")
    print(f"[INFO] Firm ID: {firm_id}")
    print(f"[INFO] Case ID: {case_id}")
    print(f"[INFO] Number of files received: {len(files) if files else 0}")
    
    try:
        if not case_id:
            raise HTTPException(status_code=400, detail="No case_id provided")

        uploaded_documents = []

        for idx, file in enumerate(files):
            print(f"\n[INFO] Processing file {idx+1}/{len(files)}: {file.filename}")
            try:
                # Save document to CosmosDB (or local storage if not configured)
                document = await save_document(file, case_id, firm_id)
                uploaded_documents.append({
                    "firm_id": document.firm_id,
                    "case_id": document.case_id,
                    "document_id": document.document_id,
                    "filename": document.filename
                })
            except Exception as e:
                import traceback
                error_msg = f"Failed to upload {file.filename}: {str(e)}"
                traceback_msg = traceback.format_exc()
                print(f"[ERROR] {error_msg}")
                print(f"[ERROR] Traceback:\n{traceback_msg}")
                # Also log to file for debugging
                with open("upload_errors.log", "a") as log_file:
                    log_file.write(f"\n{'='*60}\n")
                    log_file.write(f"Time: {__import__('datetime').datetime.now()}\n")
                    log_file.write(f"File: {file.filename}\n")
                    log_file.write(f"Error: {error_msg}\n")
                    log_file.write(f"Traceback:\n{traceback_msg}\n")
                continue

        # Run indexing in the background (once per case, not per file)
        if background_tasks:
            background_tasks.add_task(index_documents_for_case, case_id,firm_id)

        if not uploaded_documents:
            raise HTTPException(status_code=500, detail="No files were successfully uploaded.")

        return {
            "message": f"{len(uploaded_documents)} file(s) uploaded successfully. Indexing is running in background.",
            "documents_info": uploaded_documents
        }

    except Exception as e:
        print(f"Upload endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def ask(payload: QueryRequest) -> QueryResponse:
    print(f"Received query for firm_id: {payload.firm_id}, case_id: {payload.case_id} with question: {payload.question}")

    # Get answer from GraphRAG
    result = await get_answer(query_text=payload.question, case_id=payload.case_id, firm_id=payload.firm_id)
    # Debug print the result
    
    return QueryResponse(
        answer=result.get("response", ""),
        references=[],
        chunks=[],
        graph_paths=[]
    )
