"""Conversation context and history management."""
import logging
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
from collections import OrderedDict
from ...schemas.llm import ChatMessage
from .models import ConversationContext, ConversationTurn


logger = logging.getLogger(__name__)


class ContextManager:
    """Manages conversation history and state for each user."""

    def __init__(self, storage_dir: str = "transcriptions", max_turns: int = 15, session_ttl_hours: int = 24):
        self.storage_dir = Path(storage_dir)
        self.max_turns = max_turns
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self._in_memory_cache: OrderedDict = OrderedDict()
        self._cache_max_size = 1000

    def get_context(self, user_id: str) -> ConversationContext:
        """Get or create conversation context for user."""
        # Check in-memory cache first
        if user_id in self._in_memory_cache:
            context = self._in_memory_cache[user_id]
            self._in_memory_cache.move_to_end(user_id)
            return context

        # Try to load from persistent storage
        context = self._load_from_storage(user_id)

        if context is None:
            context = ConversationContext(
                user_id=user_id,
                session_start=datetime.now(),
                last_interaction=datetime.now()
            )

        # Cache it
        self._cache_context(user_id, context)
        return context

    def add_turn(self, user_id: str, user_text: str, assistant_response: str,
                intent: Optional[str] = None, quality_score: Optional[float] = None):
        """Add a conversation turn to history."""
        context = self.get_context(user_id)

        turn = ConversationTurn(
            user_input=user_text,
            assistant_response=assistant_response,
            intent=intent,
            timestamp=datetime.now(),
            quality_score=quality_score
        )

        context.conversation_history.append(turn)
        context.turn_count += 1
        context.last_interaction = datetime.now()

        # Keep only recent turns
        if len(context.conversation_history) > self.max_turns:
            context.conversation_history = context.conversation_history[-self.max_turns:]

        # Save to persistent storage
        self._save_to_storage(user_id, context)

        logger.debug(f"Added turn for {user_id}, total turns: {context.turn_count}")

    def get_history(self, user_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get formatted chat history for LLM."""
        context = self.get_context(user_id)

        if limit is None:
            limit = self.max_turns

        messages = []
        for turn in context.conversation_history[-limit:]:
            messages.append(ChatMessage(role="user", content=turn.user_input))
            messages.append(ChatMessage(role="assistant", content=turn.assistant_response))

        return messages

    def update_topic(self, user_id: str, topic: str):
        """Update current conversation topic."""
        context = self.get_context(user_id)
        context.current_topic = topic
        self._save_to_storage(user_id, context)

    def set_user_preference(self, user_id: str, key: str, value):
        """Set user preference."""
        context = self.get_context(user_id)
        context.user_preferences[key] = value
        self._save_to_storage(user_id, context)

    def get_user_preference(self, user_id: str, key: str, default=None):
        """Get user preference."""
        context = self.get_context(user_id)
        return context.user_preferences.get(key, default)

    def clear_history(self, user_id: str):
        """Clear conversation history for user."""
        context = self.get_context(user_id)
        context.conversation_history = []
        context.current_topic = None
        self._save_to_storage(user_id, context)

    def _cache_context(self, user_id: str, context: ConversationContext):
        """Cache context in memory (LRU)."""
        if len(self._in_memory_cache) >= self._cache_max_size:
            self._in_memory_cache.popitem(last=False)

        self._in_memory_cache[user_id] = context

    def _save_to_storage(self, user_id: str, context: ConversationContext):
        """Save context to persistent storage."""
        try:
            user_dir = self.storage_dir / user_id
            user_dir.mkdir(parents=True, exist_ok=True)

            session_file = user_dir / "session_metadata.json"

            data = {
                "user_id": context.user_id,
                "session_start": context.session_start.isoformat(),
                "last_interaction": context.last_interaction.isoformat(),
                "turn_count": context.turn_count,
                "current_topic": context.current_topic,
                "user_preferences": context.user_preferences,
                "conversation_history": [
                    {
                        "user_input": turn.user_input,
                        "assistant_response": turn.assistant_response,
                        "intent": turn.intent,
                        "timestamp": turn.timestamp.isoformat(),
                        "quality_score": turn.quality_score
                    }
                    for turn in context.conversation_history
                ]
            }

            with open(session_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save context for {user_id}: {e}")

    def _load_from_storage(self, user_id: str) -> Optional[ConversationContext]:
        """Load context from persistent storage."""
        try:
            session_file = self.storage_dir / user_id / "session_metadata.json"

            if not session_file.exists():
                return None

            with open(session_file, "r") as f:
                data = json.load(f)

            session_start = datetime.fromisoformat(data["session_start"])
            last_interaction = datetime.fromisoformat(data["last_interaction"])

            # Check if session has expired
            if datetime.now() - last_interaction > self.session_ttl:
                logger.debug(f"Session for {user_id} expired")
                return None

            context = ConversationContext(
                user_id=data["user_id"],
                session_start=session_start,
                last_interaction=last_interaction,
                turn_count=data["turn_count"],
                current_topic=data.get("current_topic"),
                user_preferences=data.get("user_preferences", {})
            )

            for turn_data in data.get("conversation_history", []):
                turn = ConversationTurn(
                    user_input=turn_data["user_input"],
                    assistant_response=turn_data["assistant_response"],
                    intent=turn_data.get("intent"),
                    timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                    quality_score=turn_data.get("quality_score")
                )
                context.conversation_history.append(turn)

            return context

        except Exception as e:
            logger.error(f"Failed to load context for {user_id}: {e}")
            return None
