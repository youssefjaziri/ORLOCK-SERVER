"""Speech quality analysis from VAD confidence and audio metadata."""
import logging
from typing import Optional
from ..schemas.metadata import SpeechMetadata, SpeechQualityScore


logger = logging.getLogger(__name__)


class SpeechQualityAnalyzer:
    """Analyzes speech quality from metadata."""

    QUALITY_THRESHOLDS = {
        "excellent": (0.8, 3.0),     # (min_vad_conf, min_duration)
        "good": (0.6, 1.0),
        "fair": (0.4, 0.5),
        "poor": (0.2, 0.2),
        "unintelligible": (0.0, 0.0),
    }

    def analyze(self, metadata: SpeechMetadata) -> SpeechQualityScore:
        """Analyze overall speech quality from metadata."""
        vad_score = self._score_vad_confidence(metadata.vad_confidence)
        duration_score = self._score_duration(metadata.speech_duration)
        energy_score = self._score_energy(metadata) if metadata.rms_energy_avg else 0.5
        confidence_score = metadata.transcription_confidence or 0.7

        overall_score = (vad_score * 0.4 + duration_score * 0.3 +
                        energy_score * 0.15 + confidence_score * 0.15)

        quality_level = self._determine_quality_level(metadata.vad_confidence, metadata.speech_duration)

        reasoning = self._build_reasoning(vad_score, duration_score, energy_score, confidence_score, quality_level)

        return SpeechQualityScore(
            quality_level=quality_level,
            overall_score=overall_score,
            vad_score=vad_score,
            duration_score=duration_score,
            energy_score=energy_score,
            confidence_score=confidence_score,
            reasoning=reasoning
        )

    def _score_vad_confidence(self, confidence: float) -> float:
        """Score VAD confidence [0.0, 1.0] on scale [0.0, 1.0]."""
        if confidence >= 0.9:
            return 1.0
        elif confidence >= 0.7:
            return 0.9
        elif confidence >= 0.5:
            return 0.7
        elif confidence >= 0.3:
            return 0.4
        else:
            return max(0.0, confidence)

    def _score_duration(self, duration: float) -> float:
        """Score speech duration. Optimal: 1-5s."""
        if duration >= 5.0:
            return 0.95
        elif duration >= 3.0:
            return 1.0
        elif duration >= 1.0:
            return 0.85
        elif duration >= 0.5:
            return 0.5
        else:
            return 0.2

    def _score_energy(self, metadata: SpeechMetadata) -> float:
        """Score RMS energy consistency. Optimal: avg in middle range."""
        if not metadata.rms_energy_avg:
            return 0.5

        avg_energy = metadata.rms_energy_avg
        if 0.3 <= avg_energy <= 0.7:
            return 0.9
        elif 0.2 <= avg_energy <= 0.8:
            return 0.75
        else:
            return 0.5

    def _determine_quality_level(self, vad_confidence: float, duration: float) -> str:
        """Determine quality level from VAD and duration."""
        if vad_confidence > 0.8 and duration > 3.0:
            return "excellent"
        elif vad_confidence > 0.6 and duration > 1.0:
            return "good"
        elif vad_confidence > 0.4:
            return "fair"
        elif vad_confidence > 0.2:
            return "poor"
        else:
            return "unintelligible"

    def _build_reasoning(self, vad_score: float, duration_score: float,
                        energy_score: float, confidence_score: float,
                        quality_level: str) -> str:
        """Build human-readable reasoning for quality score."""
        reasons = []

        if vad_score < 0.5:
            reasons.append("Low VAD confidence")
        if duration_score < 0.5:
            reasons.append("Short speech duration")
        if energy_score < 0.5:
            reasons.append("Inconsistent energy levels")
        if confidence_score < 0.5:
            reasons.append("Low transcription confidence")

        if not reasons:
            reasons.append("High quality speech detected")

        return "; ".join(reasons)

    def should_ask_for_clarification(self, quality: SpeechQualityScore) -> bool:
        """Determine if we should ask user to repeat."""
        return quality.quality_level in ["poor", "unintelligible"]

    def get_response_confidence_level(self, quality: SpeechQualityScore) -> str:
        """Get confidence level for response generation."""
        if quality.quality_level == "excellent":
            return "high"
        elif quality.quality_level == "good":
            return "normal"
        elif quality.quality_level == "fair":
            return "cautious"
        else:
            return "minimal"
