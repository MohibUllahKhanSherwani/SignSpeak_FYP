import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import json

def strip_keras3_metadata(h5_path, output_path):
    """
    Manually strips Keras 3 specific metadata ('batch_shape', 'optional') 
    from the model configuration inside the H5 file.
    """
    print(f"Processing {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        # Get the model configuration
        model_config_raw = f.attrs.get('model_config')
        if model_config_raw is None:
            print("No model_config found in attributes. This might not be a standard Keras H5 file.")
            return
        
        if isinstance(model_config_raw, bytes):
            model_config_raw = model_config_raw.decode('utf-8')
            
        model_config = json.loads(model_config_raw)

    def clean_config(obj):
        if isinstance(obj, dict):
            # Remove Keras 3 specific keys that break Keras 2
            if 'batch_shape' in obj:
                # Keras 2 uses 'batch_input_shape' for the first layer
                # but 'batch_shape' is often injected by Keras 3 into many layers
                del obj['batch_shape']
            if 'optional' in obj:
                del obj['optional']
            
            # Recurse
            for key, value in obj.items():
                clean_config(value)
        elif isinstance(obj, list):
            for item in obj:
                clean_config(item)

    clean_config(model_config)
    new_config_raw = json.dumps(model_config).encode('utf-8')

    # Copy the file and update the attribute
    import shutil
    shutil.copy(h5_path, output_path)
    
    with h5py.File(output_path, 'a') as f:
        f.attrs['model_config'] = new_config_raw
    
    print(f"Legacy-compatible model saved to: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "all_models")
    
    models_to_convert = [
        "action_model_baseline_new.h5",
        "action_model_augmented_new.h5"
    ]
    
    for model_name in models_to_convert:
        input_path = os.path.join(model_dir, model_name)
        output_name = model_name.replace("_new.h5", "_legacy.h5")
        output_path = os.path.join(model_dir, output_name)
        
        if os.path.exists(input_path):
            try:
                strip_keras3_metadata(input_path, output_path)
            except Exception as e:
                print(f"Failed to convert {model_name}: {e}")
        else:
            print(f"Skipping {model_name}, not found.")

if __name__ == "__main__":
    main()
