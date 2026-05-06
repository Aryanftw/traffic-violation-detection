"""
training/arch/custom_modules.py
================================
Custom modules for the enhanced YOLOv8 architecture.

Modifications from Chen et al. (ISoIRS 2025):
  1. PConv + C2f_FasterNet  — lightweight PConv-based backbone block
  2. LSCD_Detect             — shared-weight head with GroupNorm

Note on ADown
-------------
ADown (from YOLOv9) is already shipped inside Ultralytics >= 8.x — no
re-implementation needed.

Registration
------------
Call register_custom_modules() ONCE before any YOLO() call that uses an
enhanced YAML or loads an enhanced .pt file.

Why parse_model is monkey-patched (not frozenset-patched)
---------------------------------------------------------
parse_model() defines `base_modules` and `repeat_modules` as LOCAL variables.
Patching parse_model.__globals__ only touches the *module namespace*, which
parse_model never reads for those names — it only reads its own locals.
The only reliable fix without forking Ultralytics is to wrap parse_model so
C2f_FasterNet rows look like C2f rows (base+repeat branch) and LSCD_Detect
rows look like Detect rows (Detect branch). After parse_model builds the
layers, the wrapper swaps the stand-in instances back to our real classes.

Why LSCD_Detect overrides forward_head (not forward)
-----------------------------------------------------
Detect.forward calls forward_head() and returns its dict in training mode.
DetectionModel.__init__ calls self.forward() with training=True to compute
strides — it expects the dict key "feats". Overriding only forward_head
keeps all that plumbing (stride init, bias_init, export, _inference) intact.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, DFL, Detect
from ultralytics.utils.tal import dist2bbox, make_anchors


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Partial Convolution  (PConv)
# ─────────────────────────────────────────────────────────────────────────────

class PConv(nn.Module):
    """
    Partial Convolution (FasterNet building block).

    Applies a full k×k convolution to the first `cp` channels and passes the
    remaining channels through unchanged:

        Y = concat( Conv_kxk(X[:cp]),  X[cp:] )

    Reduces FLOPs to ~1/16 and memory access to ~1/4 vs a full convolution.

    Args:
        c_in : input channels
        k    : kernel size (default 3)
        cp   : partial channel count (default c_in // 4)
    """

    def __init__(self, c_in: int, k: int = 3, cp: int = None):
        super().__init__()
        c_in      = int(c_in)
        self.cp   = int(cp) if cp is not None else max(1, c_in // 4)
        self.conv = nn.Conv2d(self.cp, self.cp, k, 1, k // 2, bias=False)
        self.bn   = nn.BatchNorm2d(self.cp)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.bn(self.conv(x[:, :self.cp])))
        return torch.cat([x1, x[:, self.cp:]], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FasterNet Block
# ─────────────────────────────────────────────────────────────────────────────

class FasterNetBlock(nn.Module):
    """
    FasterNet residual block: PConv → pw 1×1 → pw 1×1 → add.

    Args:
        c : channels (in = out)
        k : PConv kernel size (default 3)
    """

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        c = int(c)
        self.pconv = PConv(c, k)
        self.pw1   = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.pw2   = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(self.pw1(self.pconv(x)))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  C2f_FasterNet
# ─────────────────────────────────────────────────────────────────────────────

class C2f_FasterNet(nn.Module):
    """
    C2f variant using FasterNet blocks instead of standard Bottleneck.

    parse_model calls this as C2f_FasterNet(c1, c2, n, shortcut, ...)
    after the monkey-patch routes it through the base_modules + repeat_modules
    branch (same path as the native C2f).

    Args:
        c1       : input channels   (injected by parse_model)
        c2       : output channels  (injected by parse_model, width-scaled)
        n        : number of blocks (injected by parse_model, depth-scaled)
        shortcut : unused, kept for interface parity with C2f
        e        : channel expansion ratio (default 0.5)
    """

    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = False, e: float = 0.5):
        super().__init__()
        c1 = int(c1)
        c2 = int(c2)
        n  = max(1, int(round(float(n))))
        e  = float(e)

        self.c      = int(c2 * e)
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.blocks = nn.ModuleList(FasterNetBlock(self.c) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(blk(y[-1]) for blk in self.blocks)
        return self.cv2(torch.cat(y, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LSCD_Detect
# ─────────────────────────────────────────────────────────────────────────────

class LSCD_Detect(Detect):
    """
    Lightweight Shared Convolutional Detection Head (Chen et al., ISoIRS 2025).

    Two modifications over the standard YOLOv8 Detect head:
      1. Shared 3×3 conv weights across all scales (P3/P4/P5).
      2. GroupNorm instead of BatchNorm — stable at batch-size 1.

    Only forward_head() is overridden so that Detect.forward, stride init,
    bias_init, _inference and export all continue to work unchanged.

    forward_head contract (must match Detect.forward_head return format):
        {"boxes": Tensor[bs, 4*reg_max, total_anchors],
         "scores": Tensor[bs, nc, total_anchors],
         "feats": list of per-scale Tensors}

    Args: same as Detect — (nc, reg_max, end2end, ch)
    parse_model injects these automatically.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16,
                 end2end: bool = False, ch: tuple = ()):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)

        c2 = max(16, ch[0] // 4, self.reg_max * 4)   # regression inner ch
        c3 = max(ch[0], min(self.nc, 100))            # classification inner ch

        def _gn_block(cin: int, cout: int) -> nn.Sequential:
            # num_groups must divide cout evenly; clamp to valid value
            g = min(32, cout)
            while cout % g != 0:
                g //= 2
            g = max(g, 1)
            return nn.Sequential(
                nn.Conv2d(cin,  cout, 3, 1, 1, bias=False),
                nn.GroupNorm(g, cout), nn.SiLU(),
                nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                nn.GroupNorm(g, cout), nn.SiLU(),
            )

        # ONE shared feature-extraction block per branch
        self.shared_reg = _gn_block(ch[0], c2)
        self.shared_cls = _gn_block(ch[0], c3)

        # Per-scale alignment: each scale has a different channel count,
        # but the shared blocks need fixed-size input → align to ch[0].
        self.align_reg = nn.ModuleList(Conv(x, ch[0], 1) for x in ch)
        self.align_cls = nn.ModuleList(Conv(x, ch[0], 1) for x in ch)

        # Per-scale output projections (replace parent's cv2 / cv3)
        self.cv2 = nn.ModuleList(nn.Sequential(nn.Conv2d(c2, 4 * self.reg_max, 1)) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for _ in ch)

    def forward_head(self, x, box_head=None, cls_head=None):
        """
        Apply shared GN conv stacks and return the dict that Detect.forward
        expects:  {"boxes": ..., "scores": ..., "feats": x}

        Detect.forward passes self.cv2 / self.cv3 as box_head / cls_head via
        self.one2many — we ignore those and use our shared blocks instead.
        """
        bs = x[0].shape[0]
        boxes_list, scores_list = [], []

        for i in range(self.nl):
            reg = self.cv2[i](self.shared_reg(self.align_reg[i](x[i])))
            cls = self.cv3[i](self.shared_cls(self.align_cls[i](x[i])))
            boxes_list.append(reg.view(bs, 4 * self.reg_max, -1))
            scores_list.append(cls.view(bs, self.nc, -1))

        return dict(
            boxes  = torch.cat(boxes_list,  dim=-1),
            scores = torch.cat(scores_list, dim=-1),
            feats  = x,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def register_custom_modules():
    """
    Register C2f_FasterNet and LSCD_Detect with Ultralytics.

    Must be called once before any YOLO() call that references these classes.
    Idempotent — safe to call multiple times.
    """
    import ultralytics.nn.tasks   as tasks
    import ultralytics.nn.modules as modules_pkg
    from ultralytics.nn.modules   import C2f, Detect as _Detect

    _custom = [PConv, FasterNetBlock, C2f_FasterNet, LSCD_Detect]

    # ── Step 1: inject names so globals()[m] inside parse_model resolves them
    for cls in _custom:
        setattr(tasks,       cls.__name__, cls)
        setattr(modules_pkg, cls.__name__, cls)
        tasks.__dict__[cls.__name__] = cls

    # ── Step 2: monkey-patch parse_model (idempotent guard)
    if getattr(tasks.parse_model, '_custom_patched', False):
        print("[arch] (already registered) PConv, FasterNetBlock, "
              "C2f_FasterNet, LSCD_Detect")
        return

    _real_parse_model = tasks.parse_model

    def _patched_parse_model(d, ch, verbose=True):
        import copy

        d2 = copy.deepcopy(d)

        _remap = {
            "C2f_FasterNet": (C2f_FasterNet, "C2f"),
            "LSCD_Detect":   (LSCD_Detect,   "Detect"),
        }

        # Rewrite custom module names → stand-in names in the YAML dict,
        # and record which layers need swapping after parse_model returns.
        all_layers  = d2.get("backbone", []) + d2.get("head", [])
        replacements = []   # (abs_layer_index, orig_name, real_cls)
        for abs_i, layer in enumerate(all_layers):
            mname = layer[2]
            if mname in _remap:
                real_cls, standin_name = _remap[mname]
                layer[2] = standin_name
                replacements.append((abs_i, mname, real_cls))

        model_seq, save = _real_parse_model(d2, ch, verbose=verbose)

        if not replacements:
            return model_seq, save

        layers = list(model_seq)

        for (abs_i, orig_name, real_cls) in replacements:
            old_m = layers[abs_i]
            # Unwrap nn.Sequential wrapper only when it's NOT already one of
            # our custom types (parse_model wraps in Sequential when n > 1,
            # but repeat_modules sets n = 1 after inserting into args).
            inner = (old_m[0]
                     if isinstance(old_m, nn.Sequential)
                     and not isinstance(old_m, (C2f_FasterNet, LSCD_Detect))
                     else old_m)

            if orig_name == "C2f_FasterNet":
                # Recover channel dims from the C2f stand-in
                c1_actual = inner.cv1.conv.in_channels
                c2_actual = inner.cv2.conv.out_channels
                # C2f stores its repeat blocks in .m (ModuleList)
                n_actual  = len(inner.m) if hasattr(inner, 'm') else 1
                new_m = C2f_FasterNet(c1_actual, c2_actual, n_actual)

            elif orig_name == "LSCD_Detect":
                # Recover config from the Detect stand-in
                nc_actual      = inner.nc
                reg_max_actual = inner.reg_max
                end2end_actual = getattr(inner, 'end2end', False)
                # cv2[i] is Sequential; [0] is Ultralytics Conv; .conv is nn.Conv2d
                ch_actual      = tuple(cv[0].conv.in_channels for cv in inner.cv2)
                new_m = LSCD_Detect(nc_actual, reg_max_actual,
                                    end2end_actual, ch_actual)
            else:
                continue

            # Copy parse_model metadata
            new_m.i    = old_m.i
            new_m.f    = old_m.f
            new_m.type = f"arch.custom_modules.{orig_name}"
            new_m.np   = sum(x.numel() for x in new_m.parameters())
            if hasattr(old_m, 'stride'):
                new_m.stride = old_m.stride

            layers[abs_i] = new_m

        new_seq = nn.Sequential(*layers)
        for attr in ('i', 'f', 'type', 'stride'):
            if hasattr(model_seq, attr):
                setattr(new_seq, attr, getattr(model_seq, attr))

        return new_seq, save

    _patched_parse_model._custom_patched = True
    tasks.parse_model = _patched_parse_model

    print("[arch] Registered: PConv, FasterNetBlock, C2f_FasterNet, LSCD_Detect")
    print("[arch] ADown is built-in to Ultralytics — no registration needed.")
    print("[arch] parse_model patched — custom modules routed correctly.")