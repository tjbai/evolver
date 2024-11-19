import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.layers(x)

class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

@record
def main():
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f'training on rank {rank} out of {world_size} processes')

    device = rank
    torch.cuda.set_device(device)

    model = TinyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=sampler)
    criterion = nn.MSELoss()

    # Single forward and backward pass
    for (data, target) in tqdm(loader):
        data, target = data.to(device), target.to(device)

        output = ddp_model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'loss: {loss}')

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
