import torch
from utils import make_custom_pool

pool1 = make_custom_pool(1)
pool2 = make_custom_pool(2)

with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool1)):
    print("Pool 1 ctx start")
    with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool2)):
        print("Pool 2 ctx start")
        x1 = torch.randn(8, device="cuda")
        print("Pool 2 ctx end")
    print("Pool 1 ctx end")
del x1

# Expected output:
# Pool 1 ctx start
# Pool 2 ctx start
# Pool 2 alloc
# Pool 2 ctx end
# Pool 1 ctx end
# Pool 2 free
