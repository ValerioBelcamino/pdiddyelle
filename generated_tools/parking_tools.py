# Auto-generated tool definitions for domain: parking
# Edit with care; regenerate when the domain changes.
DOMAIN_NAME = "parking"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move-curb-to-curb",
            "description": "Move a car from one curb to another. Preconditions: car must be clear, destination curb must be clear, and car must be at source curb. Effects: destination curb becomes occupied, source curb becomes clear, car moves to destination curb, and total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "car": {
                        "type": "string",
                        "description": "The car to move"
                    },
                    "curbsrc": {
                        "type": "string",
                        "description": "Source curb where car is currently parked"
                    },
                    "curbdest": {
                        "type": "string",
                        "description": "Destination curb where car will move to"
                    }
                },
                "required": [
                    "car",
                    "curbsrc",
                    "curbdest"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move-curb-to-car",
            "description": "Move a car from a curb to behind another car. Preconditions: both cars must be clear, moving car must be at source curb, and destination car must be at a curb. Effects: destination car becomes blocked, source curb becomes clear, moving car is now behind destination car, moving car leaves curb, and total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "car": {
                        "type": "string",
                        "description": "The car to move"
                    },
                    "curbsrc": {
                        "type": "string",
                        "description": "Source curb where car is currently parked"
                    },
                    "cardest": {
                        "type": "string",
                        "description": "Destination car to park behind"
                    }
                },
                "required": [
                    "car",
                    "curbsrc",
                    "cardest"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move-car-to-curb",
            "description": "Move a car from behind another car to a curb. Preconditions: moving car must be clear, destination curb must be clear, and moving car must be behind source car. Effects: destination curb becomes occupied, source car becomes clear, moving car parks at destination curb, moving car leaves behind source car, moving car is now at curb, and total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "car": {
                        "type": "string",
                        "description": "The car to move"
                    },
                    "carsrc": {
                        "type": "string",
                        "description": "Source car that the moving car is currently behind"
                    },
                    "curbdest": {
                        "type": "string",
                        "description": "Destination curb where car will move to"
                    }
                },
                "required": [
                    "car",
                    "carsrc",
                    "curbdest"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move-car-to-car",
            "description": "Move a car from behind one car to behind another car. Preconditions: both moving car and destination car must be clear, moving car must be behind source car, and destination car must be at a curb. Effects: destination car becomes blocked, source car becomes clear, moving car is now behind destination car, moving car leaves behind source car, and total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "car": {
                        "type": "string",
                        "description": "The car to move"
                    },
                    "carsrc": {
                        "type": "string",
                        "description": "Source car that the moving car is currently behind"
                    },
                    "cardest": {
                        "type": "string",
                        "description": "Destination car to park behind"
                    }
                },
                "required": [
                    "car",
                    "carsrc",
                    "cardest"
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
