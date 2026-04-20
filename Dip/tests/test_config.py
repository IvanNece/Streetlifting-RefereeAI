import os
import yaml
from dip_validator.cli import load_config

def test_load_config(tmp_path):
    config_content = {
        "pose": {"model": "rtmpose-s", "device": "cuda", "confidence_threshold": 0.5},
        "phases": {"smoothing_window": 11, "smoothing_polyorder": 3, "bottom_window": 3},
        "landmarks": {"elbow_offset_ratio": 0.2, "deltoid_offset_ratio": 0.25, "ema_alpha": 0.5},
        "decision": {"min_confidence": 0.4},
        "output": {"save_landmarks_trace": False, "overlay_show_margin": False}
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)
        
    loaded_config = load_config(str(config_file))
    
    assert loaded_config["pose"]["model"] == "rtmpose-s"
    assert loaded_config["pose"]["device"] == "cuda"
    assert loaded_config["phases"]["bottom_window"] == 3
    assert loaded_config["landmarks"]["ema_alpha"] == 0.5
