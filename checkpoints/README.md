# checkpoints/

Place trained PyTorch checkpoint files here before starting the server.

## Expected Files

| Filename | Model | Best Validation |
|----------|-------|----------------|
| `GroupDRO_best.pt` | GroupDRO (Best Research Model) | Round 2 |
| `ERM_best.pt` | ERM (Centralized Baseline) | Epoch 1 |
| `FedAvg_best.pt` | FedAvg (Federated Baseline) | Round 5 |
| `FedProx_best.pt` | FedProx (Proximal FL) | Round 1 |
| `DP-WHFedDG_best.pt` | DP-WHFedDG (Privacy + DG) | Round 5 |
| `DP-FedAvg_best.pt` | DP-FedAvg | Reserved (Coming Soon) |

## Checkpoint Format

Each `.pt` file is a `torch.save` dictionary containing:

```python
{
    "model_state_dict": ...,   # ResNet18 weights (FC: 512 → 2)
    "best_round": ...,         # Best validation round/epoch
    ...                        # Training metadata
}
```

## Note

Checkpoint files (~43 MB each) are not tracked in git due to size constraints.
Retrain using the provided `Camylon (2).ipynb` notebook or contact the author.
