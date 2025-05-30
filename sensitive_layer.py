import torch
import numpy as np

class SensitiveLayerFinder:
    def __init__(self, model, dataloader, device):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.activations = {}

    def _hook(self, name):
        def fn(_, __, output):
            # flatten each feature-map and compute its rank
            fmap = output.detach()
            per_filter_ranks = []
            # for each channel j, compute rank per sample and then average
            for j in range(fmap.size(1)):
                # fmap[:, j] has shape [batch, H, W]
                # ranks_j = torch.linalg.matrix_rank(fmap[:, j], rtol=1e-3)   # -> tensor of shape [batch]
                ranks_j = torch.linalg.svdvals(fmap[:, j]).gt(1e-3).sum()
                avg_rank_j = ranks_j.float().mean().item()        # one scalar
                per_filter_ranks.append(avg_rank_j)
        
            self.activations[name].append(per_filter_ranks)
        return fn

    def compute_rank_expectations(self, max_batches=1):
        self.activations = {name: [] for name, module in self.model.named_modules() if isinstance(module, torch.nn.Conv2d)}

        hook_handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                handle = module.register_forward_hook(self._hook(name))
                hook_handles.append(handle)

        with torch.no_grad():
            for i, (imgs, _) in enumerate(self.dataloader):
                if i >= max_batches: break
                imgs = imgs.to(self.device)
                self.model(imgs)

        for handle in hook_handles:
            handle.remove()

        rank_exps = {}
        for name, lists in self.activations.items():
            mean_ranks = torch.tensor(lists).mean(dim=0).tolist()
            rank_exps[name] = mean_ranks
        return rank_exps

    def identify_sensitive_layers(self, rank_exps, gamma=1e-5, sensitive_ratio_threshold=0.5):
        sensitive = []

        for name, ranks in rank_exps.items():
            ranks_t = torch.tensor(ranks)
            if ranks_t.numel() == 0:
                continue

            median = ranks_t.median()
            important_mask = ranks_t >= median
            unimportant_mask = ranks_t < median

            l = ranks_t[important_mask].mean()
            l_ = ranks_t[unimportant_mask].mean() if unimportant_mask.any() else torch.tensor(0.0)

            delta2 = ranks_t[unimportant_mask].var(unbiased=False) if unimportant_mask.sum() > 1 else torch.tensor(0.0)

            T_i = torch.sqrt(l**2 + l_**2) / torch.sqrt(delta2 + gamma)

            high = (ranks_t >= T_i).float().mean().item()

            if high >= sensitive_ratio_threshold:
                sensitive.append(name)

        return sensitive
