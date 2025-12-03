# Auto-generated tool definitions for domain: ferry
# Edit with care; regenerate when the domain changes.
DOMAIN_NAME = "ferry"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sail",
            "description": "Move the ferry from one location to another. Preconditions: ferry must be at 'from' location and not at 'to' location. Effects: ferry moves to 'to' location, leaves 'from' location, and total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from": {
                        "type": "string",
                        "description": "Starting location of the ferry"
                    },
                    "to": {
                        "type": "string",
                        "description": "Destination location for the ferry"
                    }
                },
                "required": [
                    "from",
                    "to"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "board",
            "description": "Load a car onto the ferry. Preconditions: car must be at the same location as the ferry, and ferry must be empty. Effects: car is on ferry, no longer at the location, ferry becomes non-empty, and total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "car": {
                        "type": "string",
                        "description": "Car to be loaded onto the ferry"
                    },
                    "loc": {
                        "type": "string",
                        "description": "Location where boarding occurs (must match ferry location)"
                    }
                },
                "required": [
                    "car",
                    "loc"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "debark",
            "description": "Unload a car from the ferry. Preconditions: car must be on the ferry, and ferry must be at the destination location. Effects: car is placed at the location, ferry becomes empty, car is no longer on ferry, and total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "car": {
                        "type": "string",
                        "description": "Car to be unloaded from the ferry"
                    },
                    "loc": {
                        "type": "string",
                        "description": "Destination location for the car (must match ferry location)"
                    }
                },
                "required": [
                    "car",
                    "loc"
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
