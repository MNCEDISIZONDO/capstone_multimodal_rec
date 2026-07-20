import torch
from src.device_utils import get_device, run_with_fallback, vram_report

device = get_device()
print("Device:", device)
print(vram_report())

# Cap PyTorch's allocator at 25% of VRAM (~1.5 GB) so OOM is raised
# deterministically, above the driver's system-memory fallback.
torch.cuda.set_per_process_memory_fraction(0.25, 0)

n = 12000  # a and b fit inside the cap; the matmul output does not
a = torch.randn(n, n, device=device)
b = torch.randn(n, n, device=device)
print("Allocated inputs. Now forcing the matmul to exceed the cap...")

result = run_with_fallback(lambda x, y: (x @ y).sum(), a, b)
print("Result:", result.item())
print("Survived. Fallback works.")