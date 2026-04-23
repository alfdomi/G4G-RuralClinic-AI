"""
Tool schemas for Gemma 4 native function calling.

These JSON schemas define the tools available to the orchestrator.
Gemma 4 (via Ollama) reads these at inference time and decides which
tool to invoke based on the user's query and attached image.

Schema format follows the OpenAI-compatible tool spec supported by Ollama,
which Gemma 4 uses for structured function-call outputs.

Project: G4G RuralClinic AI — offline dermatology assistant.
"""

# ─── Dermatology Analysis Tool ────────────────────────────────────────────────
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
                        "Optional: Body location of the skin lesion as described by the user. "
                        "Examples: 'left forearm', 'upper back', 'face near nose'. "
                        "Helps contextualise the analysis."
                    ),
                },
                "duration": {
                    "type": "string",
                    "description": (
                        "Optional: How long the skin issue has been present. "
                        "Examples: '2 weeks', 'since childhood', 'appeared last month'."
                    ),
                },
            },
            "required": [],
        },
    },
}

# ─── RAG Knowledge Retrieval Tool ─────────────────────────────────────────────
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
            "Returns relevant excerpts from an offline dermatology knowledge base. "
            "Does NOT require or analyse an image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The user's question or topic to look up. "
                        "Examples: 'What is the ABCDE rule?', 'basal cell carcinoma treatment', "
                        "'how to do a skin self-exam'."
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

# ─── Tool Registry ────────────────────────────────────────────────────────────
TOOL_SCHEMAS: list[dict] = [DERMATOLOGY_TOOL, RAG_TOOL]
TOOL_NAMES: set[str] = {t["function"]["name"] for t in TOOL_SCHEMAS}
