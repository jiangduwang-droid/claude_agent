import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import select, delete

from app.core.auth import get_current_user
from app.models.models import KnowledgeBase, Document, DocumentChunk, User

router = APIRouter()


class CreateKBRequest(BaseModel):
    name: str
    description: str | None = None


class SearchRequest(BaseModel):
    query: str
    kb_ids: list[str] | None = None
    top_k: int = 5


@router.get("/")
async def list_knowledge_bases(
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    async with engine.begin() as conn:
        result = await conn.execute(
            select(KnowledgeBase)
            .where(KnowledgeBase.user_id == user.id)
            .order_by(KnowledgeBase.id.desc())
        )
        kbs = result.scalars().all()
    return [
        {
            "id": str(kb.id),
            "name": kb.name,
            "description": kb.description,
        }
        for kb in kbs
    ]


@router.post("/")
async def create_knowledge_base(
    req: CreateKBRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    kb = KnowledgeBase(
        id=uuid.uuid4(),
        user_id=user.id,
        name=req.name,
        description=req.description,
    )
    async with engine.begin() as conn:
        await conn.execute(
            KnowledgeBase.__table__.insert().values(
                id=kb.id,
                user_id=kb.user_id,
                name=kb.name,
                description=kb.description,
            )
        )
    return {"id": str(kb.id), "name": kb.name}


@router.get("/{kb_id}")
async def get_knowledge_base(
    kb_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    kb_uuid = uuid.UUID(kb_id)

    async with engine.begin() as conn:
        result = await conn.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == kb_uuid)
        )
        kb = result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if kb.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your knowledge base")

    async with engine.begin() as conn:
        result = await conn.execute(
            select(Document).where(Document.kb_id == kb_uuid)
            .order_by(Document.id.desc())
        )
        docs = result.scalars().all()

    return {
        "id": str(kb.id),
        "name": kb.name,
        "description": kb.description,
        "documents": [
            {
                "id": str(d.id),
                "filename": d.filename,
                "file_type": d.file_type,
                "chunk_count": 0,
            }
            for d in docs
        ],
    }


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    kb_uuid = uuid.UUID(kb_id)

    async with engine.begin() as conn:
        result = await conn.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == kb_uuid)
        )
        kb = result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if kb.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your knowledge base")

    async with engine.begin() as conn:
        await conn.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id.in_(
                    select(Document.id).where(Document.kb_id == kb_uuid)
                )
            )
        )
        await conn.execute(
            delete(Document).where(Document.kb_id == kb_uuid)
        )
        await conn.execute(
            delete(KnowledgeBase).where(KnowledgeBase.id == kb_uuid)
        )

    return {"status": "deleted"}


@router.post("/{kb_id}/documents")
async def upload_document(
    kb_id: str,
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    kb_uuid = uuid.UUID(kb_id)

    async with engine.begin() as conn:
        result = await conn.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == kb_uuid)
        )
        kb = result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if kb.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your knowledge base")

    content = await file.read()
    content_str = content.decode("utf-8", errors="replace")

    file_type = file.filename.rsplit(".", 1)[-1].lower() if file.filename else "txt"
    if file_type == "pdf":
        try:
            import fitz
            import io
            pdf_doc = fitz.open(stream=content, filetype="pdf")
            content_str = "\n\n".join(page.get_text() for page in pdf_doc)
            pdf_doc.close()
        except Exception as e:
            print(f"[Knowledge] PDF parsing failed: {e}")

    doc = Document(
        id=uuid.uuid4(),
        kb_id=kb_uuid,
        filename=file.filename or "unknown",
        file_type=file_type,
        content=content_str,
    )

    async with engine.begin() as conn:
        await conn.execute(
            Document.__table__.insert().values(
                id=doc.id,
                kb_id=doc.kb_id,
                filename=doc.filename,
                file_type=doc.file_type,
                content=doc.content,
            )
        )

    try:
        from app.services.rag_service import RAGService
        config = request.app.state.config
        rag = RAGService(engine, config)
        await rag.process_document(doc.id, content_str, file.filename)
    except Exception as e:
        print(f"[Knowledge] Document processing failed: {e}")

    return {"id": str(doc.id), "filename": doc.filename, "status": "processed"}


@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_document(
    kb_id: str,
    doc_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    doc_uuid = uuid.UUID(doc_id)

    async with engine.begin() as conn:
        await conn.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == doc_uuid)
        )
        await conn.execute(
            delete(Document).where(Document.id == doc_uuid)
        )

    return {"status": "deleted"}


@router.post("/search")
async def search_knowledge(
    req: SearchRequest,
    request: Request,
    user: User = Depends(get_current_user),
):
    engine = request.app.state.engine
    config = request.app.state.config

    try:
        from app.services.rag_service import RAGService
        rag = RAGService(engine, config)
        results = await rag.search(req.query, str(user.id), req.kb_ids, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return {"results": results, "query": req.query}
