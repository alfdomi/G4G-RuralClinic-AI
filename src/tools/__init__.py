"""
Tool schemas for Gemma 4 native function calling.

Three schemas are defined:
  ROUTING_TOOL     — triage_and_route: Phase 1 routing only.
                     Gemma 4 uses this to produce a structured JSON routing
                     decision (domain, triage_level, extracted_symptoms,
                     recommended_workers, reasoning) before any worker is called.
  DERMATOLOGY_TOOL — analyze_skin_lesion: Phase 2 image analysis worker.
  RAG_TOOL         — retrieve_dermatology_knowledge: Phase 2 knowledge Q&A.

Schemas follow the OpenAI-compatible tool spec supported by Ollama, which
Gemma 4 uses for structured function-call outputs.

Project: G4G RuralClinic AI — offline dermatology assistant.
"""

# ─── Phase 1: Routing / Triage Tool ──────────────────────────────────────────
# Used exclusively by the orchestrator's _triage_query() phase.
# Gemma 4 returns one call to this tool containing the routing decision JSON.
ROUTING_TOOL = {
    "type": "function",
    "function": {
        "name": "triage_and_route",
        "description": (
            "Analyse the user's query and any attached image to produce a structured "
            "triage and routing decision. Call this tool for EVERY user message. "
            "Determine the clinical domain, urgency level, key symptoms mentioned, "
            "and which specialist workers should be invoked next. "
            "Use your reasoning capability (<|think|> mode when available) to justify "
            "the routing decision before producing the JSON output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "enum": ["dermatology", "general", "unclear"],
                    "description": (
                        "Clinical domain of the query. "
                        "'dermatology' if the user mentions skin, moles, lesions, rashes, "
                        "or provides a skin image. 'general' for unrelated health or "
                        "non-health questions. 'unclear' when ambiguous."
                    ),
                },
                "triage_level": {
                    "type": "string",
                    "enum": ["urgent", "moderate", "low", "unclear"],
                    "description": (
                        "'urgent': features suggesting melanoma, rapidly growing lesion, "
                        "bleeding, immunocompromised patient, or systemic alarm signs. "
                        "'moderate': chronic or changing but not immediately alarming. "
                        "'low': stable, likely benign, or educational question. "
                        "'unclear': insufficient information."
                    ),
                },
                "extracted_symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of specific symptoms or visual features mentioned by the user "
                        "or visible in the image. E.g. ['itchy', 'dark mole', 'irregular border', "
                        "'bleeding', 'growing for 3 months']. Empty list if none mentioned."
                    ),
                },
                "recommended_workers": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["analyze_skin_lesion", "retrieve_dermatology_knowledge"],
                    },
                    "description": (
                        "Ordered list of worker tools to invoke. "
                        "Use 'analyze_skin_lesion' when an image is provided or "
                        "the user describes personal skin symptoms. "
                        "Use 'retrieve_dermatology_knowledge' when the user asks a "
                        "knowledge question ('What is...?', 'How do I...?', etc.). "
                        "Both can be listed for image queries with an educational question. "
                        "Empty list for general non-dermatology queries."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Brief explanation of the routing decision (1–2 sentences). "
                        "Reference specific ABCDE features, keywords, or image properties "
                        "that informed the domain and triage level choices."
                    ),
                },
            },
            "required": [
                "domain",
                "triage_level",
                "extracted_symptoms",
                "recommended_workers",
                "reasoning",
            ],
        },
    },
}

# ─── Phase 2: Dermatology Analysis Tool ───────────────────────────────────────
DERMATOLOGY_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_skin_lesion",
        "description": (
            "Analyze a skin lesion or skin condition image to provide an informational "
            "assessment. Use this tool when the user provides an image of skin, a mole, "
            "rash, lesion, growth, or any dermatological concern, OR when they describe "
            "symptoms related to skin conditions. "
            "Returns a structured assessment including classification, confidence level, "
            "visual description, risk level, and plain-language explanation. "
            "IMPORTANT: Never provides a definitive diagnosis — always recommends "
            "professional consultation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "string",
                    "description": (
                        "Patient-reported symptoms or concerns related to the skin lesion. "
                        "Examples: 'itchy mole that has been growing for 3 months', "
                        "'red rash on forearm for 2 weeks', 'dark spot that bleeds'. "
                        "Provide as much detail as the user gave."
                    ),
                },
                "image_location": {
                    "type": "string",
                    "description": (
                        "Optional: Body location of the skin lesion as described by the user."
                    ),
                },
                "duration": {
                    "type": "string",
                    "description": (
                        "Optional: How long the skin issue has been present."
                    ),
                },
            },
            "required": [],
        },
    },
}

# ─── Phase 2: RAG Knowledge Retrieval Tool ────────────────────────────────────
RAG_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_dermatology_knowledge",
        "description": (
            "Retrieve relevant dermatology knowledge to answer educational questions "
            "about skin conditions, treatments, prevention, or general skin health. "
            "Use this tool when the user asks a knowledge or information question — "
            "e.g. 'What is melanoma?', 'How do I check a mole?', 'What causes psoriasis?', "
            "'When should I see a dermatologist?'. "
            "Returns relevant excerpts from an offline dermatology knowledge base that "
            "covers ABCDE criteria, skin-of-color presentations, Fitzpatrick types, "
            "rural/tropical conditions, and WHO referral guidelines. "
            "Does NOT require or analyse an image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The user's question or topic to look up in the knowledge base."
                    ),
                },
                "condition": {
                    "type": "string",
                    "description": (
                        "Optional: specific skin condition to focus the search on. "
                        "Examples: 'melanoma', 'eczema', 'actinic_keratosis'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

# ─── Registries ───────────────────────────────────────────────────────────────
# ROUTING_SCHEMAS: used in Phase 1 (triage call) — only the routing tool.
ROUTING_SCHEMAS: list[dict] = [ROUTING_TOOL]

# WORKER_SCHEMAS: used in Phase 2 (worker dispatch descriptions for Gemma 4).
WORKER_SCHEMAS: list[dict] = [DERMATOLOGY_TOOL, RAG_TOOL]

# TOOL_SCHEMAS: full list exposed for backward compatibility.
TOOL_SCHEMAS: list[dict] = [DERMATOLOGY_TOOL, RAG_TOOL]
TOOL_NAMES: set[str] = {t["function"]["name"] for t in TOOL_SCHEMAS}
