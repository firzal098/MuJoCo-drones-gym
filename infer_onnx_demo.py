import numpy as np
import onnxruntime as ort
import time

def run_inference_demo():
    onnx_path = "./results/krti_single_rl/krti_single_brain.onnx"
    
    print(f"Loading ONNX model from {onnx_path}...")
    # 1. Initialize the ONNX Runtime Session
    # On a real drone (like Jetson Nano), you might use providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
    # For CPU inference (like Raspberry Pi), use providers=['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 2. Get the input name that the model expects (we set this to 'observation' during export)
    input_name = session.get_inputs()[0].name
    print(f"Model expects input named: '{input_name}'")
    
    # 3. Create a dummy observation (simulate what your real drone's sensors would output)
    # Shape must be (1, 13) because we have 13 features. Data type must be float32.
    dummy_obs = np.array([[
        0.50, 0.50, 0.20, 0.20,  # Fake YOLO bounding box
        12.0, 0.5, -0.2,         # Fake relative gate position (X, Y, Z)
        3.0, 0.0, 0.0,           # Fake drone velocity (Vx, Vy, Vz)
        0.0, 0.0, 0.0            # Fake drone attitude (Roll, Pitch, Yaw)
    ]], dtype=np.float32)
    
    print("\nRunning inference...")
    
    # Measure inference speed
    start_time = time.time()
    
    # 4. Run the inference!
    # The first argument is None (returns all outputs). 
    # The second argument is a dictionary mapping the input name to your numpy array.
    outputs = session.run(None, {input_name: dummy_obs})
    
    end_time = time.time()
    
    # 5. Extract the action array
    action = outputs[0][0]
    
    print(f"Inference took: {(end_time - start_time) * 1000:.2f} milliseconds")
    print(f"Network Output Action: {action}")
    
    print("\n--- How to use these actions ---")
    print(f"Vx (Forward) Command:  {action[0]:.2f} (Scale by max_speed)")
    print(f"Vy (Lateral) Command:  {action[1]:.2f} (Scale by max_speed)")
    print(f"Vz (Vertical) Command: {action[2]:.2f} (Scale by max_speed)")
    print(f"Yaw Rate Command:      {action[3]:.2f} (Scale by max_yaw_rate)")

if __name__ == "__main__":
    run_inference_demo()
