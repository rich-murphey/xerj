# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0",
#   "transformers==5.3.0",
#   "safetensors",
#   "onnx",
#   "onnxscript",
#   "onnxruntime==1.31.1",
# ]
# ///
"""Re-export answerdotai/answerai-colbert-small-v1 to ONNX so ONNX Runtime
fuses attention, for xerj-cmem-embed's `onnx-experimental` ColBERT backend.

The hub's `onnx/model.onnx` was traced from an SDPA BertModel. Its split
softmax scale defeats ORT's Attention fusion (`fuse_attention: failed to
match qk path`). This exports BertModel with `attn_implementation="eager"`,
runs `onnxruntime.transformers.optimizer` (bert, 12 heads, hidden 384) and
counts the fused ops.

    uv run --script colbert_export.py <out_dir> [--head]

Without `--head`, the output is `last_hidden_state` [batch, seq, 384]
(the Rust head applies `linear.weight`). With `--head`, the 384 -> 96
`linear` and the L2 norm are in the graph, and the output is
`token_vectors` [batch, seq, 96].

Writes <out_dir>/colbert-eager{-head}.onnx (raw export) and
<out_dir>/colbert-eager{-head}-opt.onnx (optimised), and prints op counts
and sha256.
"""
import collections
import hashlib
import sys
from pathlib import Path

import onnx
import torch
import transformers
from huggingface_hub import hf_hub_download
from onnxruntime.transformers import optimizer
from safetensors.torch import load_file
from transformers import BertModel

MODEL = 'answerdotai/answerai-colbert-small-v1'
REVISION = '934fa8bb4ce2284f4c2baa232d81aca4d076fa5e'


class Encoder(torch.nn.Module):
    def __init__(self, bert, linear):
        super().__init__()
        self.bert = bert
        self.linear = linear

    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        if self.linear is None:
            return h
        return torch.nn.functional.normalize(h @ self.linear.T, p=2, dim=-1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1])
    head = '--head' in sys.argv[2:]
    out.mkdir(parents=True, exist_ok=True)
    stem = 'colbert-eager' + ('-head' if head else '')
    bert = BertModel.from_pretrained(MODEL, revision=REVISION, attn_implementation='eager', add_pooling_layer=False).eval()
    linear = load_file(hf_hub_download(MODEL, 'model.safetensors', revision=REVISION))['linear.weight'] if head else None
    model = Encoder(bert, linear).eval()
    ids = torch.tensor([[101, 2, 7592, 2088, 102, 103]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    raw = out / f'{stem}.onnx'
    output = 'token_vectors' if head else 'last_hidden_state'
    with torch.inference_mode():
        torch.onnx.export(
            model, (ids, mask, torch.zeros_like(ids)), str(raw),
            input_names=['input_ids', 'attention_mask', 'token_type_ids'], output_names=[output],
            dynamic_axes={n: {0: 'batch', 1: 'seq'} for n in ('input_ids', 'attention_mask', 'token_type_ids', output)},
            opset_version=17, dynamo=False,
        )
    opt = optimizer.optimize_model(str(raw), model_type='bert', num_heads=12, hidden_size=384, opt_level=0, use_gpu=False)
    print('fused:', opt.get_fused_operator_statistics())
    dst = out / f'{stem}-opt.onnx'
    opt.save_model_to_file(str(dst))
    for p in (raw, dst):
        g = onnx.load(str(p))
        print(p.name, dict(collections.Counter(n.op_type for n in g.graph.node).most_common(30)))
        print(p.name, 'sha256', sha256(p))
    print('torch', torch.__version__, 'transformers', transformers.__version__, 'onnx', onnx.__version__)


if __name__ == '__main__':
    main()
