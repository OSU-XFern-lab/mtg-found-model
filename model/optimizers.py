"""The tested Muon encoder-matrix / AdamW remainder recipe."""
import torch


class MuonAdamW:
    def __init__(self, model, lr, weight_decay):
        base = model.module if hasattr(model, "module") else model
        matrices = [p for p in base.encoder.parameters() if p.requires_grad and p.ndim == 2]
        matrix_ids = {id(p) for p in matrices}
        remainder = [p for p in base.parameters() if p.requires_grad and id(p) not in matrix_ids]
        self.muon = torch.optim.Muon(matrices, lr=lr, weight_decay=weight_decay,
                                     adjust_lr_fn="match_rms_adamw")
        self.adamw = torch.optim.AdamW(remainder, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
