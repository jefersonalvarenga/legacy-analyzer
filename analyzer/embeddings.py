"""
embeddings.py
-------------
Generates and stores OpenAI embeddings for:
  1. Individual messages  (for per-message RAG retrieval)
  2. Whole conversations  (for conversation-level clustering / similarity)

Uses text-embedding-3-small (1536 dimensions) by default.
Batches requests to stay within OpenAI rate limits.
"""

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from analyzer.parser import Message

logger = logging.getLogger(__name__)

# OpenAI embedding limits
EMBED_BATCH_SIZE = 100          # max items per embedding request
EMBED_MAX_TOKENS = 8191         # max tokens per text for text-embedding-3-small
# pt-BR averages ~4 chars/token; 8191 tokens ≈ 32k chars.
# We use 20k as a conservative limit to stay well within the token budget.
EMBED_MAX_CHARS = 20_000


def _truncate(text: str, max_chars: int = EMBED_MAX_CHARS) -> str:
    """
    Truncate text to max_chars. For long conversations, keep the first 60%
    and last 40% so we preserve both the opening context and the most recent
    exchanges (which tend to be the most relevant for RAG).
    """
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + "\n...[conversa truncada]...\n" + text[-tail:]


def _messages_to_text(messages: list[Message]) -> str:
    """Serialise a conversation into a single string for embedding."""
    parts = []
    for msg in messages:
        parts.append(f"{msg.sender_type}: {msg.content}")
    return "\n".join(parts)


class EmbeddingClient:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts, batching as needed.
        Returns a list of embedding vectors in the same order as input.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = [_truncate(t) for t in texts[i: i + EMBED_BATCH_SIZE]]
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error("Embedding batch %d failed: %s", i // EMBED_BATCH_SIZE, e)
                # Fill with None placeholders to keep alignment
                all_embeddings.extend([None] * len(batch))

        return all_embeddings

    async def embed_messages(
        self, messages: list[Message]
    ) -> list[Optional[list[float]]]:
        """
        Embed each message individually.
        Returns embeddings in the same order as the input messages list.
        """
        texts = [msg.content for msg in messages]
        return await self.embed_texts(texts)

    async def embed_conversation(
        self, messages: list[Message]
    ) -> Optional[list[float]]:
        """
        Embed an entire conversation as a single vector.
        Returns one embedding vector or None on failure.
        """
        text = _messages_to_text(messages)
        results = await self.embed_texts([text])
        return results[0] if results else None
