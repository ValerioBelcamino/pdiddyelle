# Auto-generated tool definitions for domain: maintenance
# Edit with care; regenerate when the domain changes.
DOMAIN_NAME = "maintenance"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "workat",
            "description": "Perform maintenance work at an airport on a specific day. Precondition: the day must be 'today'. Effects: marks the day as no longer 'today', marks all planes that are at the specified airport on that day as 'done', and increases total-cost by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {
                        "type": "string",
                        "description": "The day on which to work (must be 'today')."
                    },
                    "airport": {
                        "type": "string",
                        "description": "The airport where maintenance is performed."
                    }
                },
                "required": [
                    "day",
                    "airport"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end",
            "description": "Signal the plan is complete and stop emitting further actions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]
__all__ = ['DOMAIN_NAME', 'TOOLS']
