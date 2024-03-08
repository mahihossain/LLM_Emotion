import torch
import time

# Check if CUDA is available and set PyTorch to use it
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Setting device to use CUDA.")
else:
    print("CUDA is not available. Exiting.")
    exit()

start_time = time.time()
end_time = start_time + 20*60  # 20 minutes from now
print_time = start_time + 1*60  # 1 minutes from now

# Perform tensor multiplication until approximately 20 minutes have passed
while time.time() < end_time:
    # Create two large tensors
    tensor1 = torch.randn(5000, 5000, device=device)
    tensor2 = torch.randn(5000, 5000, device=device)

    result = torch.matmul(tensor1, tensor2)

    # Print the result every 1 minute
    if time.time() > print_time:
        print("1 minute has passed.\n")
        print(result)
        print_time += 1*60  # Set the next time to print