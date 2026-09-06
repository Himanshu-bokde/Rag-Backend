import uuid
from pathlib import Path

from fastapi import APIRouter,File,HTTPException,UploadFile,status

from app.core.config import settings
from app.schemas.document import UploadResponse
from app.sevices.document.processor import process_documnt
from app.workers.rq_worer import decument_queue

route = APIRouter(
    prefix='/document',
    tags=["Documents"]
)


@route.post('/upload',response_model=UploadResponse,status_code=status.HTTP_200_OK)
async def upload_document(file:UploadFile=File(...)):
    # Validate Extension
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    #Generate document ID
    document_id = str(uuid.uuid4())

    #3 Create upload directory
    upload_dir = Path(settings.UPLOAD_DIR).resolve()
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # 4.Save File
    file_path = upload_dir / f"{document_id}.pdf"

    content = await file.read()


    #5 . File Size Validation 
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail="File too large"
        )


    file_path.write_bytes(content)


    #6 . Add job to RQ
    job = decument_queue.enqueue(
        process_documnt,
        document_id,
        str(file_path)
    )

    print("JON",job)

    #. Return immediately

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="queued"
    )