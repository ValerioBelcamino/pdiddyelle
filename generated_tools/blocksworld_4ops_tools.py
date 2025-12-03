# Auto-generated tool definitions for domain: blocksworld_4ops
# Edit with care; regenerate when the domain changes.
DOMAIN_NAME = "blocksworld_4ops"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pickup",
            "description": "Pick up a block from the table. Preconditions: block must be clear, on the table, and the arm must be empty. Effects: holding the block, block no longer clear or on table, arm no longer empty, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ob": {
                        "type": "string",
                        "description": "The block to pick up."
                    }
                },
                "required": [
                    "ob"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "putdown",
            "description": "Put down a held block onto the table. Preconditions: holding the block. Effects: block becomes clear and on table, arm becomes empty, no longer holding the block, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ob": {
                        "type": "string",
                        "description": "The block to put down."
                    }
                },
                "required": [
                    "ob"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stack",
            "description": "Stack a held block onto another clear block. Preconditions: holding the top block, the bottom block must be clear. Effects: arm becomes empty, top block becomes clear and on bottom block, bottom block no longer clear, no longer holding top block, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ob": {
                        "type": "string",
                        "description": "The block to stack (top block)."
                    },
                    "underob": {
                        "type": "string",
                        "description": "The block to stack onto (bottom block)."
                    }
                },
                "required": [
                    "ob",
                    "underob"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unstack",
            "description": "Unstack a block from on top of another block. Preconditions: top block is on bottom block, top block is clear, arm is empty. Effects: holding top block, bottom block becomes clear, top block no longer on bottom block or clear, arm no longer empty, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ob": {
                        "type": "string",
                        "description": "The block to unstack (top block)."
                    },
                    "underob": {
                        "type": "string",
                        "description": "The block underneath (bottom block)."
                    }
                },
                "required": [
                    "ob",
                    "underob"
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
