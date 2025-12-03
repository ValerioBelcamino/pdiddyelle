# Auto-generated tool definitions for domain: spanner
# Edit with care; regenerate when the domain changes.
DOMAIN_NAME = "spanner"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "walk",
            "description": "Move a man from one location to another along a link. Preconditions: man must be at start location, and there must be a link between start and end locations. Effects: man is no longer at start location, man is now at end location, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Starting location (must have link to end)."
                    },
                    "end": {
                        "type": "string",
                        "description": "Destination location (must have link from start)."
                    },
                    "m": {
                        "type": "string",
                        "description": "Man identifier."
                    }
                },
                "required": [
                    "start",
                    "end",
                    "m"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pickup_spanner",
            "description": "Pick up a spanner from a location. Preconditions: man must be at the location, spanner must be at that location. Effects: spanner is no longer at the location, man is now carrying the spanner, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "l": {
                        "type": "string",
                        "description": "Location where spanner is picked up."
                    },
                    "s": {
                        "type": "string",
                        "description": "Spanner identifier."
                    },
                    "m": {
                        "type": "string",
                        "description": "Man identifier."
                    }
                },
                "required": [
                    "l",
                    "s",
                    "m"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tighten_nut",
            "description": "Tighten a loose nut using a usable spanner. Preconditions: man must be at the location, nut must be at that location, man must be carrying the spanner, spanner must be usable, nut must be loose. Effects: nut is no longer loose, spanner becomes unusable, nut becomes tightened, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "l": {
                        "type": "string",
                        "description": "Location where nut is tightened."
                    },
                    "s": {
                        "type": "string",
                        "description": "Spanner identifier (must be usable and carried)."
                    },
                    "m": {
                        "type": "string",
                        "description": "Man identifier."
                    },
                    "n": {
                        "type": "string",
                        "description": "Nut identifier (must be loose)."
                    }
                },
                "required": [
                    "l",
                    "s",
                    "m",
                    "n"
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
