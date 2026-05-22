# Orchestration Layer Documentation

## Overview

The Orchestration Layer is an intelligent multi-stage pipeline that transforms simple transcription→LLM calls into sophisticated, context-aware response generation. It combines:

- **Intent Detection**: Classifies user input into 13 intent categories
- **Context Management**: Maintains conversation history and user state
- **Speech Quality Analysis**: Evaluates audio quality from VAD metrics
- **Dynamic Prompt Engineering**: Generates optimized prompts based on intent and context
- **Response Validation**: Checks response quality and applies fallbacks
- **Intelligent LLM Calling**: Selects appropriate temperature and parameters

## Architecture

```
User Audio/Text
    ↓
[Intent Detector] → Classify into 13 categories (technical, greeting, command, etc.)
    ↓
[Context Manager] → Retrieve conversation history and user state
    ↓
[Quality Analyzer] → Score speech quality (excellent/good/fair/poor)
    ↓
[Prompt Builder] → Generate dynamic system + user prompts based on all above
    ↓
[LLM Generator] → Call Ollama with optimized parameters
    ↓
[Response Validator] → Check quality, apply fallbacks if needed
    ↓
[Context Updater] → Save turn to history
    ↓
Intelligent Response
```

## Intent Categories

The system recognizes 13 user intents, each with optimized response strategies:

| Intent | Description | Temperature | Response Length |
|--------|-------------|-------------|-----------------|
| CONVERSATIONAL | Small talk, social | 0.3 | 50-300 tokens |
| QUESTION | Factual questions | 0.2 | 75-400 tokens |
| COMMAND | Action requests | 0.1 | 30-150 tokens |
| TECHNICAL | Technical questions | 0.15 | 100-500 tokens |
| EXPLANATION | Detailed breakdowns | 0.25 | 150-600 tokens |
| GREETING | Hello, goodbye | 0.35 | 20-100 tokens |
| EMERGENCY | Urgent requests | 0.1 | 20-150 tokens |
| CLARIFICATION | Clarification requests | 0.2 | 50-200 tokens |
| ACKNOWLEDGEMENT | Confirmations | 0.0 | 10-30 tokens |
| SYSTEM_CONTROL | Settings, config | 0.1 | 50-200 tokens |
| SUMMARIZATION | Summarize info | 0.15 | 75-300 tokens |
| NAVIGATION | Directions | 0.15 | 50-250 tokens |
| TASK_EXECUTION | Multi-step tasks | 0.2 | 200-800 tokens |

## API Endpoints

### 1. Orchestrated Audio (Intelligent)

**POST** `/api/v1/orchestrated/audio`

Send audio with metadata for intelligent processing.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/orchestrated/audio \
  -F "user_id=user_123" \
  -F "audio=@audio.wav" \
  -F 'metadata_json={"vad_confidence": 0.85, "speech_duration": 2.5, "silence_duration": 0.5, "prebuffer_duration": 0.2}'
```

**Response:**
```json
{
  "user_text": "What is voice activity detection?",
  "llm_response": "Voice Activity Detection (VAD) is...",
  "intent": "technical",
  "intent_confidence": 0.92,
  "speech_quality_score": 0.87,
  "speech_quality_level": "good",
  "processing_time_ms": 1250,
  "pipeline_stages": {
    "intent_detection": 150,
    "context_retrieval": 50,
    "quality_analysis": 75,
    "prompt_building": 100,
    "llm_generation": 850,
    "response_validation": 25
  }
}
```

### 2. Orchestrated Message (Text Only)

**POST** `/api/v1/orchestrated/message`

Send text for intelligent processing (no audio).

**Request:**
```json
{
  "user_id": "user_123",
  "text": "Tell me about VAD",
  "system_prompt": null
}
```

**Response:** Same as audio endpoint (with perfect quality score)

### 3. Debug Intent Detection

**POST** `/api/v1/debug/intent`

Test intent detection for any text.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/debug/intent?text=What%20is%20VAD?"
```

**Response:**
```json
{
  "category": "technical",
  "confidence": 0.92,
  "reasoning": "User asked a technical question",
  "related_categories": [["question", 0.8], ["explanation", 0.75]],
  "suggested_response_style": "professional, code-focused, structured"
}
```

## ROS2 Integration

The ROS2 API client has been updated to support the orchestrated endpoints:

```python
from vad_component.orlock_api_client import OrlockAPIClient

client = OrlockAPIClient(use_orchestrated=True)

# With metadata from VAD pipeline
metadata = {
    "vad_confidence": 0.85,
    "speech_duration": 2.5,
    "silence_duration": 0.5,
    "prebuffer_duration": 0.2,
    "rms_energy_avg": 0.45,
    "amplitude_acceptance_rate": 0.95,
    "transcription_confidence": 0.92
}

response = client.send_audio(
    audio_data=audio_array,
    user_id="user_123",
    metadata=metadata
)

# Or send text directly
response = client.send_orchestrated_text(
    text="Tell me about VAD",
    user_id="user_123"
)
```

## Components

### Intent Detector (`intent/detector.py`)

Classifies user input into one of 13 intent categories using the LLM.

**Features:**
- LLM-based classification for high accuracy
- Intent caching (1000 entries, LRU)
- Confidence scoring (0.0-1.0)
- Related intent suggestions

```python
from orlock.intent.detector import IntentDetector

detector = IntentDetector()
result = await detector.detect("What is VAD?")
# Returns IntentResult with category, confidence, reasoning
```

### Context Manager (`orchestration/context/manager.py`)

Manages conversation history and user state.

**Features:**
- In-memory cache (LRU, 1000 users)
- Persistent JSON storage per user
- Session TTL (24 hours default)
- Conversation history limit (15 turns default)

```python
from orlock.orchestration.context.manager import ContextManager

manager = ContextManager()
context = manager.get_context(user_id)
manager.add_turn(user_id, user_text, response_text)
history = manager.get_history(user_id, limit=10)
```

### Quality Analyzer (`quality/analyzer.py`)

Scores speech quality from VAD confidence and audio metrics.

**Quality Levels:**
- `excellent`: >0.8 VAD confidence, >3s duration
- `good`: >0.6 VAD confidence, >1s duration
- `fair`: >0.4 VAD confidence
- `poor`: >0.2 VAD confidence
- `unintelligible`: <0.2 VAD confidence

```python
from orlock.quality.analyzer import SpeechQualityAnalyzer

analyzer = SpeechQualityAnalyzer()
quality = analyzer.analyze(metadata)
# Returns SpeechQualityScore with level, overall_score, reasoning
```

### Prompt Builder (`prompts/builder.py`)

Generates dynamic system prompts based on intent, context, and quality.

**Features:**
- Intent-specific templates (13 templates)
- Quality-aware modifications
- Context-aware adjustments
- Speech metadata modifiers

```python
from orlock.prompts.builder import PromptBuilder

builder = PromptBuilder()
system_prompt, user_prompt = builder.build_full_prompt(
    user_input, intent, metadata, context, quality
)
```

### Response Validator (`validation/validator.py`)

Validates response quality and provides fallbacks.

**Validation Checks:**
- Length check (too short/too long)
- Coherence check (sentence structure, formatting)
- Hallucination check (detecting made-up details)
- Safety check (harmful content)
- Format check (intent-specific structure)

```python
from orlock.validation.validator import ResponseValidator, FallbackHandler

validator = ResponseValidator()
result = validator.validate(response, intent)

if not result.is_valid:
    fallback = FallbackHandler.get_fallback(reason)
```

### Orchestration Service (`services/orchestration_service.py`)

Main service that coordinates all components.

```python
from orlock.services.orchestration_service import OrchestrationService

service = OrchestrationService()

# Process transcription with metadata
result = await service.process_transcription(
    user_id="user_123",
    transcription="What is VAD?",
    metadata=speech_metadata
)

# Or process text message
result = await service.process_message(
    user_id="user_123",
    user_text="Hello"
)
```

## Configuration

### Context Manager Configuration

```python
manager = ContextManager(
    storage_dir="transcriptions",  # Where to persist conversations
    max_turns=15,                   # Keep last N turns in history
    session_ttl_hours=24           # Session timeout
)
```

### Prompt Builder Configuration

```python
builder = PromptBuilder(
    templates_dir="src/orlock/prompts/templates"  # Path to templates
)
```

## Speech Metadata Schema

The orchestrated audio endpoint expects metadata in this format:

```json
{
  "vad_confidence": 0.85,
  "speech_duration": 2.5,
  "silence_duration": 0.5,
  "prebuffer_duration": 0.2,
  "rms_energy_min": 0.1,
  "rms_energy_max": 0.8,
  "rms_energy_avg": 0.45,
  "amplitude_acceptance_rate": 0.95,
  "segment_start_time": "2025-01-15T10:00:00Z",
  "segment_end_time": "2025-01-15T10:00:02.5Z",
  "transcription_confidence": 0.92
}
```

All fields are optional except `vad_confidence`, `speech_duration`, `silence_duration`, and `prebuffer_duration`.

## Performance Characteristics

- **Intent Detection**: ~150ms (cached after first call)
- **Context Retrieval**: ~50ms
- **Quality Analysis**: ~75ms
- **Prompt Building**: ~100ms
- **LLM Generation**: 800-1500ms (varies by model/response length)
- **Response Validation**: ~25ms
- **Total Pipeline**: 1000-2000ms (mostly LLM time)

## Optimization Tips

1. **Reuse Services**: Create service instances once, reuse across requests
2. **Cache Intent Results**: Intent cache is automatic, 1000 entry LRU
3. **Limit History**: Keep max_turns reasonable (15 default)
4. **Template Caching**: Templates are cached on first load
5. **Async Processing**: All I/O is async for concurrency
6. **Temperature Tuning**: Pre-configured per intent, can override

## Backward Compatibility

The existing endpoints (`/api/v1/userAudio`, `/api/v1/userMessage`) remain unchanged. New orchestrated endpoints are additions:

- New: `/api/v1/orchestrated/audio`
- New: `/api/v1/orchestrated/message`
- New: `/api/v1/debug/intent`

Gradually migrate to orchestrated endpoints for enhanced capabilities.

## Testing

Run the test suite:

```bash
# Unit tests
pytest tests/test_orchestration.py -v

# Integration tests
pytest tests/test_orchestration_integration.py -v

# Full test suite
pytest tests/ -v
```

## Future Enhancements

- Multi-language intent detection
- Emotion detection from speech
- User preference learning
- A/B testing for prompt variations
- Response streaming
- Real-time quality feedback
- Conversation summarization for long histories
