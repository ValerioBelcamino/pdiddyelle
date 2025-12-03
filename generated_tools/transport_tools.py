# Auto-generated tool definitions for domain: transport
# Edit with care; regenerate when the domain changes.
DOMAIN_NAME = "transport"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "drive",
            "description": "Move a vehicle from one location to another along a road. Preconditions: vehicle must be at the starting location, and there must be a road connecting the two locations. Effects: vehicle is no longer at the starting location, vehicle is at the destination location, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "v": {
                        "type": "string",
                        "description": "The vehicle to drive."
                    },
                    "l1": {
                        "type": "string",
                        "description": "The starting location (must have road to l2)."
                    },
                    "l2": {
                        "type": "string",
                        "description": "The destination location (must have road from l1)."
                    }
                },
                "required": [
                    "v",
                    "l1",
                    "l2"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pick-up",
            "description": "Load a package onto a vehicle at a location. Preconditions: vehicle and package must be at the same location, vehicle must have capacity size s2, and s1 must be the predecessor size of s2 (capacity-predecessor s1 s2). Effects: package is no longer at the location, package is in the vehicle, vehicle capacity changes from s2 to s1, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "v": {
                        "type": "string",
                        "description": "The vehicle picking up the package."
                    },
                    "l": {
                        "type": "string",
                        "description": "The location where pickup occurs."
                    },
                    "p": {
                        "type": "string",
                        "description": "The package to pick up."
                    },
                    "s1": {
                        "type": "string",
                        "description": "The smaller size (predecessor of s2)."
                    },
                    "s2": {
                        "type": "string",
                        "description": "The larger size (successor of s1)."
                    }
                },
                "required": [
                    "v",
                    "l",
                    "p",
                    "s1",
                    "s2"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drop",
            "description": "Unload a package from a vehicle at a location. Preconditions: vehicle must be at the location, package must be in the vehicle, vehicle must have capacity size s1, and s1 must be the predecessor size of s2 (capacity-predecessor s1 s2). Effects: package is no longer in the vehicle, package is at the location, vehicle capacity changes from s1 to s2, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "v": {
                        "type": "string",
                        "description": "The vehicle dropping the package."
                    },
                    "l": {
                        "type": "string",
                        "description": "The location where drop occurs."
                    },
                    "p": {
                        "type": "string",
                        "description": "The package to drop."
                    },
                    "s1": {
                        "type": "string",
                        "description": "The smaller size (predecessor of s2)."
                    },
                    "s2": {
                        "type": "string",
                        "description": "The larger size (successor of s1)."
                    }
                },
                "required": [
                    "v",
                    "l",
                    "p",
                    "s1",
                    "s2"
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
