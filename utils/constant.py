from torch.cuda import is_available
DEVICE = "cuda" if is_available() else "cpu"
TQDM_BAR_FORMAT = "{desc}: {n_fmt}/{total_fmt} |{percentage:3.2f}% {elapsed}<{remaining} {postfix} {rate_fmt}"