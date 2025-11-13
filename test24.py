import torch
from utils import make_custom_pool

pool1 = torch.cuda.MemPool(make_custom_pool(1))
pool2 = torch.cuda.MemPool(make_custom_pool(2))

with torch.cuda.use_mem_pool(pool1):
    print("Pool 1 ctx start")
    with torch.cuda.use_mem_pool(pool2):
        print("Pool 2 ctx start")
        print("Pool 2 ctx end")
    print("Pool 1 ctx end")
x1 = torch.randn(8, device="cuda")
del x1

# Expected output:
# Pool 1 ctx start
# Pool 2 ctx start
# Pool 2 ctx end
# Pool 1 ctx end
