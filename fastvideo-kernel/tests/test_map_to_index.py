"""Stable sparse-route compaction across chunk boundaries and strided inputs."""
import torch
from fastvideo_kernel.triton_kernels.index import map_to_index


def test_map_to_index():
    torch.manual_seed(42)
    for width in (1, 7, 128, 256, 577, 1051):
        for transposed in (False, True):
            shape = (2, 3, width, 5) if transposed else (2, 3, 5, width)
            mask = torch.rand(shape, device="cuda") < 0.15
            if transposed:
                mask = mask.transpose(-1, -2)
            mask[:, :, 0] = False
            mask[:, :, 1] = True
            actual, counts = map_to_index(mask)
            expected = torch.arange(width, device="cuda").expand(mask.shape)
            expected = expected.masked_fill(~mask, width).sort(-1).values
            expected = expected.masked_fill(expected == width, -1).to(torch.int32)
            assert torch.equal(actual, expected), (width, transposed)
            assert torch.equal(counts, mask.sum(-1).to(torch.int32))
    print("PASS sparse index: stable order, counts, empty/full rows, strided input, chunk tails")


if __name__ == "__main__":
    test_map_to_index()
