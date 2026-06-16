import json
import os

DEFAULT_PRESETS = {
    "Natural Glow": {
        "skin_smoothing": 0.35,
        "skin_texture_recovery": 0.30,
        "blush_warmth": 0.20,
        "skin_brightening": 0.15,
        "eye_enhancement": 0.30,
        "undereye_lighten": 0.25,
        "nose_reduce": 0.00,
        "cheeks_reduce": 0.00,
        "forehead_reduce": 0.00,
        "eye_enlarge": 0.00,
        "lips_plump": 0.00,
        "lipstick_shade": "None",
        "lipstick_strength": 0.00,
        "eye_color_shade": "Natural",
        "eye_color_strength": 0.00,
        "color_look": "None",
        "look_intensity": 1.00,
        "eyeliner_strength": 0.00,
        "eyeshadow_shade": "None",
        "eyeshadow_strength": 0.00,
        "lip_gloss_strength": 0.20,
        "facial_highlighter_strength": 0.15
    },
    "Hollywood Smooth": {
        "skin_smoothing": 0.65,
        "skin_texture_recovery": 0.35,
        "blush_warmth": 0.30,
        "skin_brightening": 0.25,
        "eye_enhancement": 0.50,
        "undereye_lighten": 0.45,
        "nose_reduce": 0.15,
        "cheeks_reduce": 0.25,
        "forehead_reduce": 0.00,
        "eye_enlarge": 0.25,
        "lips_plump": 0.15,
        "lipstick_shade": "Soft Pink",
        "lipstick_strength": 0.30,
        "eye_color_shade": "Natural",
        "eye_color_strength": 0.00,
        "color_look": "Warm Sunset",
        "look_intensity": 0.60,
        "eyeliner_strength": 0.35,
        "eyeshadow_shade": "Rose Gold",
        "eyeshadow_strength": 0.30,
        "lip_gloss_strength": 0.40,
        "facial_highlighter_strength": 0.30
    },
    "High Glamour": {
        "skin_smoothing": 0.80,
        "skin_texture_recovery": 0.40,
        "blush_warmth": 0.50,
        "skin_brightening": 0.40,
        "eye_enhancement": 0.65,
        "undereye_lighten": 0.60,
        "nose_reduce": 0.35,
        "cheeks_reduce": 0.40,
        "forehead_reduce": 0.20,
        "eye_enlarge": 0.50,
        "lips_plump": 0.40,
        "lipstick_shade": "Rose Red",
        "lipstick_strength": 0.50,
        "eye_color_shade": "Ocean Blue",
        "eye_color_strength": 0.45,
        "color_look": "Teal & Orange",
        "look_intensity": 0.75,
        "eyeliner_strength": 0.60,
        "eyeshadow_shade": "Royal Purple",
        "eyeshadow_strength": 0.50,
        "lip_gloss_strength": 0.65,
        "facial_highlighter_strength": 0.55
    },
    "Subtle Polish": {
        "skin_smoothing": 0.15,
        "skin_texture_recovery": 0.20,
        "blush_warmth": 0.10,
        "skin_brightening": 0.05,
        "eye_enhancement": 0.15,
        "undereye_lighten": 0.10,
        "nose_reduce": 0.00,
        "cheeks_reduce": 0.00,
        "forehead_reduce": 0.00,
        "eye_enlarge": 0.00,
        "lips_plump": 0.00,
        "lipstick_shade": "None",
        "lipstick_strength": 0.00,
        "eye_color_shade": "Natural",
        "eye_color_strength": 0.00,
        "color_look": "None",
        "look_intensity": 1.00,
        "eyeliner_strength": 0.00,
        "eyeshadow_shade": "None",
        "eyeshadow_strength": 0.00,
        "lip_gloss_strength": 0.10,
        "facial_highlighter_strength": 0.05
    },
    "Default Reset": {
        "skin_smoothing": 0.00,
        "skin_texture_recovery": 0.00,
        "blush_warmth": 0.00,
        "skin_brightening": 0.00,
        "eye_enhancement": 0.00,
        "undereye_lighten": 0.00,
        "nose_reduce": 0.00,
        "cheeks_reduce": 0.00,
        "forehead_reduce": 0.00,
        "eye_enlarge": 0.00,
        "lips_plump": 0.00,
        "lipstick_shade": "None",
        "lipstick_strength": 0.00,
        "eye_color_shade": "Natural",
        "eye_color_strength": 0.00,
        "color_look": "None",
        "look_intensity": 1.00,
        "eyeliner_strength": 0.00,
        "eyeshadow_shade": "None",
        "eyeshadow_strength": 0.00,
        "lip_gloss_strength": 0.00,
        "facial_highlighter_strength": 0.00
    }
}

def save_preset_to_file(filepath, params):
    """
    Save the given parameter dictionary to a JSON file.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving preset to {filepath}: {e}")
        return False

def load_preset_from_file(filepath):
    """
    Load a parameter dictionary from a JSON file.
    """
    try:
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        required_keys = {
            "skin_smoothing": 0.0,
            "skin_texture_recovery": 0.0,
            "blush_warmth": 0.0,
            "skin_brightening": 0.0,
            "eye_enhancement": 0.0,
            "undereye_lighten": 0.0,
            "nose_reduce": 0.0,
            "cheeks_reduce": 0.0,
            "forehead_reduce": 0.0,
            "eye_enlarge": 0.0,
            "lips_plump": 0.0,
            "lipstick_shade": "None",
            "lipstick_strength": 0.0,
            "eye_color_shade": "Natural",
            "eye_color_strength": 0.0,
            "color_look": "None",
            "look_intensity": 1.0,
            "eyeliner_strength": 0.0,
            "eyeshadow_shade": "None",
            "eyeshadow_strength": 0.0,
            "lip_gloss_strength": 0.0,
            "facial_highlighter_strength": 0.0,
            "enable_body_retouching": False,
            "body_sensitivity": 1.5
        }
        for key, default in required_keys.items():
            if key not in params:
                params[key] = default
        return params
    except Exception as e:
        print(f"Error loading preset from {filepath}: {e}")
        return None
