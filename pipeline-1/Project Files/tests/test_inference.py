import os
import sys

# Ensure backend imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.pipeline1 import PipelineCNN
from inference.pipeline2 import PipelineMultimodal
from inference.merge import merge_outputs

def main():
    print("Loading CNN Pipeline...")
    pipeline_cnn = PipelineCNN("models/disaster_cnn.h5")
    
    print("Loading Multimodal Pipeline...")
    pipeline_mm = PipelineMultimodal("models/multimodal.pth", device='cpu')
    
    # Find a test image in the root directory
    test_image = "../disaster1.jpg"
    if not os.path.exists(test_image):
        print(f"Error: Could not find test image at {test_image}")
        return
        
    print(f"\nUsing test image: {test_image}")
        
    print("\n--- Test 1: Image Only ---")
    res_cnn = None
    res_mm = None
    try:
        res_cnn = pipeline_cnn.predict(test_image)
        print("CNN output:", res_cnn)
    except Exception as e:
        print("CNN failed:", e)
        
    final_output_1 = merge_outputs(res_cnn, res_mm)
    print("Merged JSON:\n", final_output_1)
    
    print("\n--- Test 2: Image + Text ---")
    test_text = "Breaking: A massive flood has struck the city! Infrastructure damaged."
    try:
        res_mm = pipeline_mm.predict(test_image, test_text)
        print("Multimodal output:", res_mm)
    except Exception as e:
        print("Multimodal failed:", e)
        
    final_output_2 = merge_outputs(res_cnn, res_mm)
    print("Merged JSON:\n", final_output_2)

if __name__ == "__main__":
    main()
