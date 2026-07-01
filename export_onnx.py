import os
import glob
import torch as th
from stable_baselines3 import PPO

def get_latest_checkpoint(folder):
    zip_files = glob.glob(os.path.join(folder, "*.zip"))
    if not zip_files:
        return None
    zip_files.sort(key=os.path.getmtime, reverse=True)
    return zip_files[0]

# 1. Create a PyTorch wrapper to extract only the deterministic actor
class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # Extract features (handles CNN/MLP preprocessing if any)
        features = self.policy.extract_features(observation)
        # Pass through the MLP latent actor network
        latent_pi, _ = self.policy.mlp_extractor(features)
        # Output the deterministic action (the mean of the distribution)
        action = self.policy.action_net(latent_pi)
        return action

def export_to_onnx():
    checkpoint_dir = "./results/krti_single_rl/"
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        print("No checkpoint found to export!")
        return
        
    print(f"Loading model for export: {latest_checkpoint}")
    
    # Load the trained PPO model
    model = PPO.load(latest_checkpoint, device="cpu")
    
    # Wrap the policy
    onnxable_model = OnnxableSB3Policy(model.policy)
    onnxable_model.eval()
    
    # Create a dummy observation (batch_size=1, obs_dim=13)
    dummy_input = th.randn(1, 13, dtype=th.float32)
    
    # Export path
    onnx_path = os.path.join(checkpoint_dir, "krti_single_brain.onnx")
    
    print(f"Exporting to ONNX: {onnx_path}...")
    th.onnx.export(
        onnxable_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
    )
    
    print(f"Export successful! ONNX model saved at: {onnx_path}")
    print("You can now deploy this .onnx file using ONNX Runtime on your real drone.")

if __name__ == "__main__":
    export_to_onnx()
