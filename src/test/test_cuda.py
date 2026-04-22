import torch
import sys

def test_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")
        
        current_device = torch.cuda.current_device()
        print(f"Current device ID: {current_device}")
        
        device_name = torch.cuda.get_device_name(0)
        print(f"Device 0 name: {device_name}")
        
        # Test tensor on GPU 0
        try:
            x = torch.randn(1, 1).to('cuda:0')
            print("Successfully allocated tensor on cuda:0")
        except Exception as e:
            print(f"Failed to allocate on cuda:0: {e}")
    else:
        print("CUDA is NOT available. Check your installation.")

if __name__ == '__main__':
    test_cuda()
