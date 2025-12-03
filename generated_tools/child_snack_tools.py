# Auto-generated tool definitions for domain: child_snack
# Edit with care; regenerate when the domain changes.
DOMAIN_NAME = "child_snack"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "make_sandwich_no_gluten",
            "description": "Create a gluten-free sandwich using specified bread and content portions. Preconditions: bread portion must be at kitchen and gluten-free, content portion must be at kitchen and gluten-free, sandwich must not exist. Effects: bread and content portions removed from kitchen, sandwich appears at kitchen and marked gluten-free, sandwich existence established, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "Sandwich identifier to create"
                    },
                    "b": {
                        "type": "string",
                        "description": "Gluten-free bread portion identifier"
                    },
                    "c": {
                        "type": "string",
                        "description": "Gluten-free content portion identifier"
                    }
                },
                "required": [
                    "s",
                    "b",
                    "c"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "make_sandwich",
            "description": "Create a regular sandwich using specified bread and content portions. Preconditions: bread portion must be at kitchen, content portion must be at kitchen, sandwich must not exist. Effects: bread and content portions removed from kitchen, sandwich appears at kitchen, sandwich existence established, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "Sandwich identifier to create"
                    },
                    "b": {
                        "type": "string",
                        "description": "Bread portion identifier"
                    },
                    "c": {
                        "type": "string",
                        "description": "Content portion identifier"
                    }
                },
                "required": [
                    "s",
                    "b",
                    "c"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "put_on_tray",
            "description": "Place a sandwich from kitchen onto a tray. Preconditions: sandwich must be at kitchen, tray must be at kitchen. Effects: sandwich removed from kitchen, placed on tray, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "Sandwich identifier"
                    },
                    "t": {
                        "type": "string",
                        "description": "Tray identifier"
                    }
                },
                "required": [
                    "s",
                    "t"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "serve_sandwich_no_gluten",
            "description": "Serve a gluten-free sandwich to a gluten-allergic child. Preconditions: child must be allergic to gluten, sandwich must be on tray, child must be waiting at place, sandwich must be gluten-free, tray must be at same place as child. Effects: sandwich removed from tray, child marked as served, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "Gluten-free sandwich identifier"
                    },
                    "c": {
                        "type": "string",
                        "description": "Gluten-allergic child identifier"
                    },
                    "t": {
                        "type": "string",
                        "description": "Tray identifier"
                    },
                    "p": {
                        "type": "string",
                        "description": "Place identifier where child is waiting"
                    }
                },
                "required": [
                    "s",
                    "c",
                    "t",
                    "p"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "serve_sandwich",
            "description": "Serve a regular sandwich to a non-allergic child. Preconditions: child must not be allergic to gluten, child must be waiting at place, sandwich must be on tray, tray must be at same place as child. Effects: sandwich removed from tray, child marked as served, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "Sandwich identifier"
                    },
                    "c": {
                        "type": "string",
                        "description": "Non-allergic child identifier"
                    },
                    "t": {
                        "type": "string",
                        "description": "Tray identifier"
                    },
                    "p": {
                        "type": "string",
                        "description": "Place identifier where child is waiting"
                    }
                },
                "required": [
                    "s",
                    "c",
                    "t",
                    "p"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_tray",
            "description": "Move a tray from one place to another. Preconditions: tray must be at starting place. Effects: tray removed from starting place, placed at destination place, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "t": {
                        "type": "string",
                        "description": "Tray identifier"
                    },
                    "p1": {
                        "type": "string",
                        "description": "Starting place identifier"
                    },
                    "p2": {
                        "type": "string",
                        "description": "Destination place identifier"
                    }
                },
                "required": [
                    "t",
                    "p1",
                    "p2"
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
