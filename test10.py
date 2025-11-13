import torch
from utils import make_custom_pool

pool1 = make_custom_pool(1)

x1 = torch.randn(8, device="cuda")
with torch.cuda.use_mem_pool(pool1):
    print("Pool 1 ctx start")
    del x1
    x2 = torch.randn(8, device="cuda")
    del x2
    print("Pool 1 ctx end")

# Expected output:
# Pool 1 ctx start
# Pool 1 alloc
# Pool 1 ctx end
# Pool 1 free
