import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Config
from app.models.models import Conversation, Message


class CompressionService:
    def __init__(self, engine, config=None):
        self._engine = engine
        self._config = config or Config.get()

    async def compress_conversation(self, conversation_id: str) -> dict:
        conv_uuid = uuid.UUID(conversation_id)

        async with self._engine.begin() as conn:
            result = await conn.execute(
                select(Message)
                .where(Message.conversation_id == conv_uuid)
                .order_by(Message.created_at)
            )
            messages = result.scalars().all()

        if len(messages) < 10:
            return {"status": "not_enough_messages", "count": len(messages)}

        old_messages = [m for m in messages if m.role in ("user", "assistant")]
        recent = old_messages[-6:]
        to_compress = old_messages[:-6]

        if not to_compress:
            return {"status": "nothing_to_compress"}

        transcript = "\n\n".join(
            f"[{m.role}] {m.content}" for m in to_compress if m.content
        )

        try:
            from app.core.agent import _get_llm
            llm = _get_llm(self._config)
            summary_response = await llm.ainvoke([
                SystemMessage(content="将以下对话历史总结为300-600字的摘要,保留关键信息和上下文。"),
                HumanMessage(content=transcript),
            ])
            summary = summary_response.content
        except Exception as e:
            return {"status": "error", "error": str(e)}

        async with self._engine.begin() as conn:
            result = await conn.execute(
                select(Conversation).where(Conversation.id == conv_uuid)
            )
            conv = result.scalar_one_or_none()
            metadata = conv.metadata_ or {}
            metadata["compressed_summary"] = summary
            metadata["recent_summary"] = "\n".join(
                f"[{m.role}] {m.content[:200]}" for m in recent if m.content
            )
            await conn.execute(
                Conversation.__table__.update()
                .where(Conversation.id == conv_uuid)
                .values(metadata=metadata)
            )

        return {"status": "compressed", "summary_length": len(summary)}
