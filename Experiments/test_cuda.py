import torch
import time

# Check if CUDA is available and set PyTorch to use it
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("CUDA is not available. Exiting.")
    exit()

# Create two large tensors
tensor1 = torch.randn(5000, 5000, device=device)
tensor2 = torch.randn(5000, 5000, device=device)

start_time = time.time()
end_time = start_time + 20*60  # 20 minutes from now

# Perform tensor multiplication until approximately 20 minutes have passed
while time.time() < end_time:
    result = torch.matmul(tensor1, tensor2)

# Print the result
print(result)