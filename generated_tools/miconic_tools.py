# Auto-generated tool definitions for domain: miconic
# Edit with care; regenerate when the domain changes.
DOMAIN_NAME = "miconic"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "board",
            "description": "Board a passenger onto the lift. Preconditions: lift is at floor ?f, passenger ?p has origin floor ?f. Effects: passenger becomes boarded, origin relation removed, total-cost increased by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "f": {
                        "type": "string",
                        "description": "Floor where boarding occurs"
                    },
                    "p": {
                        "type": "string",
                        "description": "Passenger to board"
                    }
                },
                "required": [
                    "f",
                    "p"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "depart",
            "description": "Depart a passenger from the lift at their destination floor. Preconditions: lift is at floor ?f, passenger ?p has destination floor ?f, passenger is boarded. Effects: passenger becomes served, boarded relation removed, total-cost increased by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "f": {
                        "type": "string",
                        "description": "Floor where departing occurs"
                    },
                    "p": {
                        "type": "string",
                        "description": "Passenger to depart"
                    }
                },
                "required": [
                    "f",
                    "p"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "up",
            "description": "Move lift up from floor ?f1 to floor ?f2. Preconditions: lift is at floor ?f1, floor ?f1 is above floor ?f2 (i.e., ?f2 is lower). Effects: lift moves to ?f2, leaves ?f1, total-cost increased by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "f1": {
                        "type": "string",
                        "description": "Current floor (higher)"
                    },
                    "f2": {
                        "type": "string",
                        "description": "Target floor (lower)"
                    }
                },
                "required": [
                    "f1",
                    "f2"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "down",
            "description": "Move lift down from floor ?f1 to floor ?f2. Preconditions: lift is at floor ?f1, floor ?f2 is above floor ?f1 (i.e., ?f2 is higher). Effects: lift moves to ?f2, leaves ?f1, total-cost increased by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "f1": {
                        "type": "string",
                        "description": "Current floor (lower)"
                    },
                    "f2": {
                        "type": "string",
                        "description": "Target floor (higher)"
                    }
                },
                "required": [
                    "f1",
                    "f2"
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
