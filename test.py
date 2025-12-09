import torch

# Check if CUDA (GPU support) is available
print("CUDA available:", torch.cuda.is_available())

# Check the number of GPUs detected
print("Number of GPUs:", torch.cuda.device_count())

# Get the name of the GPU
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Optional: Test if a tensor can be moved to the GPU
x = torch.tensor([1.0, 2.0, 3.0]).cuda()  # move tensor to GPU
print("Tensor device:", x.device)

print("lol")