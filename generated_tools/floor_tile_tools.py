# Auto-generated tool definitions for domain: floor_tile
# Edit with care; regenerate when the domain changes.
DOMAIN_NAME = "floor_tile"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "change-color",
            "description": "Change the robot's current color to a new available color. Preconditions: robot must have current color ?c, and new color ?c2 must be available. Effects: robot loses old color ?c, gains new color ?c2, total cost increases by 5.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r": {
                        "type": "string",
                        "description": "Robot identifier"
                    },
                    "c": {
                        "type": "string",
                        "description": "Current color the robot has"
                    },
                    "c2": {
                        "type": "string",
                        "description": "New color to change to (must be available)"
                    }
                },
                "required": [
                    "r",
                    "c",
                    "c2"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "paint-up",
            "description": "Paint the tile directly above the robot's current position. Preconditions: robot has color ?c, is at tile ?x, tile ?y is above ?x (up relation), and ?y is clear. Effects: tile ?y becomes painted with color ?c and is no longer clear, total cost increases by 2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r": {
                        "type": "string",
                        "description": "Robot identifier"
                    },
                    "y": {
                        "type": "string",
                        "description": "Tile above to paint (must be clear and up from current tile)"
                    },
                    "x": {
                        "type": "string",
                        "description": "Current tile where robot is located"
                    },
                    "c": {
                        "type": "string",
                        "description": "Color to paint with (robot must have this color)"
                    }
                },
                "required": [
                    "r",
                    "y",
                    "x",
                    "c"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "paint-down",
            "description": "Paint the tile directly below the robot's current position. Preconditions: robot has color ?c, is at tile ?x, tile ?y is below ?x (down relation), and ?y is clear. Effects: tile ?y becomes painted with color ?c and is no longer clear, total cost increases by 2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r": {
                        "type": "string",
                        "description": "Robot identifier"
                    },
                    "y": {
                        "type": "string",
                        "description": "Tile below to paint (must be clear and down from current tile)"
                    },
                    "x": {
                        "type": "string",
                        "description": "Current tile where robot is located"
                    },
                    "c": {
                        "type": "string",
                        "description": "Color to paint with (robot must have this color)"
                    }
                },
                "required": [
                    "r",
                    "y",
                    "x",
                    "c"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "up",
            "description": "Move the robot upward to an adjacent tile. Preconditions: robot is at tile ?x, tile ?y is above ?x (up relation), and ?y is clear. Effects: robot moves to ?y, leaves ?x, ?x becomes clear, ?y becomes occupied (not clear), total cost increases by 3.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r": {
                        "type": "string",
                        "description": "Robot identifier"
                    },
                    "x": {
                        "type": "string",
                        "description": "Current tile where robot is located"
                    },
                    "y": {
                        "type": "string",
                        "description": "Tile above to move to (must be clear and up from current tile)"
                    }
                },
                "required": [
                    "r",
                    "x",
                    "y"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "down",
            "description": "Move the robot downward to an adjacent tile. Preconditions: robot is at tile ?x, tile ?y is below ?x (down relation), and ?y is clear. Effects: robot moves to ?y, leaves ?x, ?x becomes clear, ?y becomes occupied (not clear), total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r": {
                        "type": "string",
                        "description": "Robot identifier"
                    },
                    "x": {
                        "type": "string",
                        "description": "Current tile where robot is located"
                    },
                    "y": {
                        "type": "string",
                        "description": "Tile below to move to (must be clear and down from current tile)"
                    }
                },
                "required": [
                    "r",
                    "x",
                    "y"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "right",
            "description": "Move the robot rightward to an adjacent tile. Preconditions: robot is at tile ?x, tile ?y is right of ?x (right relation), and ?y is clear. Effects: robot moves to ?y, leaves ?x, ?x becomes clear, ?y becomes occupied (not clear), total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r": {
                        "type": "string",
                        "description": "Robot identifier"
                    },
                    "x": {
                        "type": "string",
                        "description": "Current tile where robot is located"
                    },
                    "y": {
                        "type": "string",
                        "description": "Tile to the right to move to (must be clear and right from current tile)"
                    }
                },
                "required": [
                    "r",
                    "x",
                    "y"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "left",
            "description": "Move the robot leftward to an adjacent tile. Preconditions: robot is at tile ?x, tile ?y is left of ?x (left relation), and ?y is clear. Effects: robot moves to ?y, leaves ?x, ?x becomes clear, ?y becomes occupied (not clear), total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r": {
                        "type": "string",
                        "description": "Robot identifier"
                    },
                    "x": {
                        "type": "string",
                        "description": "Current tile where robot is located"
                    },
                    "y": {
                        "type": "string",
                        "description": "Tile to the left to move to (must be clear and left from current tile)"
                    }
                },
                "required": [
                    "r",
                    "x",
                    "y"
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
