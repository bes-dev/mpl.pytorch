# Max-Pooling Loss

[**Loss Max-Pooling for Semantic Image Segmentation**](https://arxiv.org/abs/1704.02966)

## Installation


### Requirements

To install PyTorch, please refer to https://github.com/pytorch/pytorch#installation.


### Compiling

Some parts of Max-Pooling Loss have a native C++ implementation, which must be compiled with the following commands:
```bash
cd mpl
python build.py
```

## Using

```python
import mpl
import torch

max_pooling_loss = mpl.MaxPoolingLoss(ratio=0.3, p=1.7, reduce=True)
loss = torch.Tensor(1, 3, 3, 3).uniform_(0, 1)
loss = max_pooling_loss(loss)
```
