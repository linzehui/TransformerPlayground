import torch

class Config:

    # data config
    max_sent_len=50

    # model config
    d_model=512
    d_inner_hid=2048
    d_k=64
    d_v=64
    n_head=8
    n_layers=6

    dropout=0.1

    # training config
    epoch=10
    n_warmup_steps=4000
    use_cuda=torch.cuda.is_available()
    CUDA_VISIBLE_DEVICES = 5
    torch.cuda.set_device(CUDA_VISIBLE_DEVICES)