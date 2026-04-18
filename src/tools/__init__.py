"""
Tool schemas for Gemma 4 native function calling.

These JSON schemas define the tools available to the orchestrator.
Gemma 4 (via Ollama) reads these at inference time and decides which
tool to invoke based on the user's query and attached image.

Schema format follows the OpenAI-compatible tool spec supported by Ollama,
which Gemma 4 uses for structured function-call outputs.

Project: G4G RuralClinic AI — offline dermatology assistant.
"""

# ─── Dermatology Tool ─────────────────────────────────────────────────────────
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
                        "Helps contextualize the analysis."
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
            "required": [],  # Image is passed separately via multimodal message content
        },
    },
}

# ─── Tool Registry ────────────────────────────────────────────────────────────
# All tools available to the orchestrator for routing decisions.
TOOL_SCHEMAS: list[dict] = [DERMATOLOGY_TOOL]
TOOL_NAMES: set[str] = {t["function"]["name"] for t in TOOL_SCHEMAS}
