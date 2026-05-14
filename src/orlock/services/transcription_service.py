"""
transcription_service.py
------------------------
Service for storing and retrieving transcription data.
Stores transcriptions as JSON files with metadata.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Directory where transcriptions are stored
TRANSCRIPTIONS_DIR = Path("transcriptions")


class TranscriptionService:
    """Service for managing transcription storage and retrieval."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize the service with a storage directory."""
        self.storage_dir = storage_dir or TRANSCRIPTIONS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TranscriptionService initialized - storage: {self.storage_dir}")

    def save_transcription(
        self,
        user_id: str,
        audio_path: str,
        transcript_text: str,
        llm_response: str,
        intent: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Save a transcription record as a JSON file.

        Args:
            user_id: User identifier
            audio_path: Path to the audio file
            transcript_text: Transcribed text
            llm_response: Response from LLM
            intent: Classified intent (e.g., "WhereIs", "Greeting")
            temperature: LLM temperature used

        Returns:
            dict with transcription record and file path
        """
        try:
            # Create user-specific subdirectory
            user_dir = self.storage_dir / user_id
            user_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().isoformat()
            timestamp_clean = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"transcription_{timestamp_clean}.json"
            file_path = user_dir / filename

            # Build transcription record
            record = {
                "user_id": user_id,
                "timestamp": timestamp,
                "audio_path": str(audio_path),
                "transcript_text": transcript_text,
                "llm_response": llm_response,
                "intent": intent,
                "temperature": temperature,
            }

            # Save to JSON file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

            logger.info(f"Transcription saved: {file_path}")

            return {
                "success": True,
                "file_path": str(file_path),
                "record": record,
            }

        except Exception as e:
            logger.error(f"Failed to save transcription: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_user_transcriptions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the latest transcriptions for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of transcriptions to return

        Returns:
            List of transcription records
        """
        try:
            user_dir = self.storage_dir / user_id
            if not user_dir.exists():
                logger.info(f"No transcriptions found for user: {user_id}")
                return []

            # Get all JSON files, sorted by modification time (newest first)
            json_files = sorted(user_dir.glob("transcription_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            json_files = json_files[:limit]

            transcriptions = []
            for file_path in json_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        record = json.load(f)
                        transcriptions.append(record)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

            logger.info(f"Retrieved {len(transcriptions)} transcriptions for user {user_id}")
            return transcriptions

        except Exception as e:
            logger.error(f"Failed to get user transcriptions: {e}")
            return []

    def get_all_transcriptions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve the latest transcriptions from all users.

        Args:
            limit: Maximum number of transcriptions to return

        Returns:
            List of transcription records
        """
        try:
            all_transcriptions = []

            # Iterate through all user directories
            if self.storage_dir.exists():
                for user_dir in self.storage_dir.iterdir():
                    if user_dir.is_dir():
                        json_files = list(user_dir.glob("transcription_*.json"))
                        for file_path in json_files:
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    record = json.load(f)
                                    all_transcriptions.append(record)
                            except Exception as e:
                                logger.error(f"Error reading {file_path}: {e}")

            # Sort by timestamp (newest first) and limit
            all_transcriptions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            all_transcriptions = all_transcriptions[:limit]

            logger.info(f"Retrieved {len(all_transcriptions)} total transcriptions")
            return all_transcriptions

        except Exception as e:
            logger.error(f"Failed to get all transcriptions: {e}")
            return []

    def get_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about transcriptions.

        Args:
            user_id: Optional user ID to get stats for a specific user

        Returns:
            Statistics dictionary
        """
        try:
            if user_id:
                user_dir = self.storage_dir / user_id
                if not user_dir.exists():
                    return {"user_id": user_id, "count": 0}

                json_files = list(user_dir.glob("transcription_*.json"))
                return {
                    "user_id": user_id,
                    "count": len(json_files),
                    "directory": str(user_dir),
                }
            else:
                # Get stats for all users
                total_count = 0
                user_count = 0

                if self.storage_dir.exists():
                    for user_dir in self.storage_dir.iterdir():
                        if user_dir.is_dir():
                            user_count += 1
                            json_files = list(user_dir.glob("transcription_*.json"))
                            total_count += len(json_files)

                return {
                    "total_transcriptions": total_count,
                    "total_users": user_count,
                    "storage_directory": str(self.storage_dir),
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
