import sys
import torch
import os
from pprint import pprint

def show_checkpoint_meta(ckpt_path):
    """Display metadata information from a PyTorch checkpoint file."""
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file '{ckpt_path}' does not exist.")
        return
    
    try:
        # Add safe globals for omegaconf if needed
        try:
            import omegaconf
            torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
        except ImportError:
            pass
        
        # Try loading with weights_only=True first (safer)
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        except Exception as e:
            print(f"Warning: weights_only=True failed ({str(e)[:100]}...)")
            print("Falling back to weights_only=False (less secure)")
            # Fallback to weights_only=False for complex checkpoints
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        print(f"Checkpoint: {ckpt_path}")
        print("=" * 50)
        
        # Show basic info
        print(f"Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Keys: {list(checkpoint.keys())}")
            print("\nDetailed Information:")
            print("-" * 30)
            
            for key, value in checkpoint.items():
                print(f"\n[{key}]")
                if key == 'state_dict' and isinstance(value, dict):
                    print(f"  Number of parameters: {len(value)}")
                    print(f"  Sample keys: {list(value.keys())[:5]}")
                    if len(value) > 5:
                        print(f"  ... and {len(value) - 5} more")
                elif key == 'optimizer' and isinstance(value, dict):
                    print(f"  Optimizer keys: {list(value.keys())}")
                elif key in ['epoch', 'step', 'best_acc', 'best_loss']:
                    print(f"  Value: {value}")
                elif key == 'cfg':
                    print(f"  Config type: {type(value)}")
                    print(f"  Config content:")
                    pprint(value, indent=4)
                elif isinstance(value, (int, float, str, bool)):
                    print(f"  Value: {value}")
                elif isinstance(value, (list, tuple)):
                    print(f"  Length: {len(value)}")
                    if len(value) > 0:
                        print(f"  First item type: {type(value[0])}")
                elif hasattr(value, 'shape'):
                    print(f"  Shape: {value.shape}")
                    print(f"  Type: {type(value)}")
                else:
                    print(f"  Type: {type(value)}")
                    if hasattr(value, '__len__'):
                        try:
                            print(f"  Length: {len(value)}")
                        except:
                            pass
        else:
            print("\nCheckpoint content:")
            if hasattr(checkpoint, 'shape'):
                print(f"Shape: {checkpoint.shape}")
            pprint(checkpoint)
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python show_checkpoints_meta.py <checkpoint_path>")
        print("Example: python show_checkpoints_meta.py model.pth")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    show_checkpoint_meta(ckpt_path)

if __name__ == "__main__":
    main()
