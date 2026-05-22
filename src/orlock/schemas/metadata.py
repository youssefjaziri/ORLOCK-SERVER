from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SpeechMetadata(BaseModel):
    """Speech quality and context metadata from the ROS2 pipeline."""
    vad_confidence: float = Field(..., ge=0.0, le=1.0, description="Silero VAD confidence [0.0-1.0]")
    speech_duration: float = Field(..., ge=0.0, description="Duration of detected speech in seconds")
    silence_duration: float = Field(..., ge=0.0, description="Duration of silence before segment end")
    prebuffer_duration: float = Field(default=0.0, ge=0.0, description="Pre-buffered audio duration in seconds")

    rms_energy_min: Optional[float] = Field(None, description="Minimum RMS energy during segment")
    rms_energy_max: Optional[float] = Field(None, description="Maximum RMS energy during segment")
    rms_energy_avg: Optional[float] = Field(None, description="Average RMS energy during segment")

    amplitude_acceptance_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Amplitude filter acceptance rate")

    segment_start_time: Optional[str] = None
    segment_end_time: Optional[str] = None

    transcription_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Whisper transcription confidence")

    class Config:
        json_schema_extra = {
            "example": {
                "vad_confidence": 0.85,
                "speech_duration": 2.5,
                "silence_duration": 0.8,
                "prebuffer_duration": 0.5,
                "rms_energy_min": 0.1,
                "rms_energy_max": 0.8,
                "rms_energy_avg": 0.45,
                "amplitude_acceptance_rate": 0.95
            }
        }


class SpeechQualityScore(BaseModel):
    """Overall speech quality assessment."""
    quality_level: str  # "excellent", "good", "fair", "poor", "unintelligible"
    overall_score: float = Field(..., ge=0.0, le=1.0)
    vad_score: float = Field(..., ge=0.0, le=1.0)
    duration_score: float = Field(..., ge=0.0, le=1.0)
    energy_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
