# Auto-generated tool definitions for domain: satellite
# Edit with care; regenerate when the domain changes.
DOMAIN_NAME = "satellite"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "turn_to",
            "description": "Change satellite pointing direction. Preconditions: satellite must be pointing at previous direction and NOT pointing at new direction. Effects: satellite points at new direction, no longer points at previous direction, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "Satellite identifier"
                    },
                    "d_new": {
                        "type": "string",
                        "description": "New direction to point at"
                    },
                    "d_prev": {
                        "type": "string",
                        "description": "Current direction satellite is pointing at"
                    }
                },
                "required": [
                    "s",
                    "d_new",
                    "d_prev"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "switch_on",
            "description": "Power on an instrument on a satellite. Preconditions: instrument must be on board the satellite, satellite must have power available. Effects: instrument is powered on, instrument becomes uncalibrated, satellite loses power availability, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "i": {
                        "type": "string",
                        "description": "Instrument identifier"
                    },
                    "s": {
                        "type": "string",
                        "description": "Satellite identifier"
                    }
                },
                "required": [
                    "i",
                    "s"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "switch_off",
            "description": "Power off an instrument on a satellite. Preconditions: instrument must be on board the satellite and currently powered on. Effects: instrument is powered off, satellite regains power availability, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "i": {
                        "type": "string",
                        "description": "Instrument identifier"
                    },
                    "s": {
                        "type": "string",
                        "description": "Satellite identifier"
                    }
                },
                "required": [
                    "i",
                    "s"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calibrate",
            "description": "Calibrate an instrument on a satellite pointing at a specific direction. Preconditions: instrument must be on board the satellite, instrument has calibration target at the direction, satellite is pointing at that direction, instrument is powered on. Effects: instrument becomes calibrated, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "Satellite identifier"
                    },
                    "i": {
                        "type": "string",
                        "description": "Instrument identifier"
                    },
                    "d": {
                        "type": "string",
                        "description": "Direction to calibrate at (must match calibration target)"
                    }
                },
                "required": [
                    "s",
                    "i",
                    "d"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_image",
            "description": "Capture an image of a direction using an instrument in a specific mode. Preconditions: instrument must be calibrated, on board the satellite, supports the mode, powered on, and satellite must be pointing at the direction. Effects: image of direction in mode is obtained, total cost increases by 1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "Satellite identifier"
                    },
                    "d": {
                        "type": "string",
                        "description": "Direction to image"
                    },
                    "i": {
                        "type": "string",
                        "description": "Instrument identifier"
                    },
                    "m": {
                        "type": "string",
                        "description": "Mode to use for imaging"
                    }
                },
                "required": [
                    "s",
                    "d",
                    "i",
                    "m"
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
