"""Performance test for the two entanglement methods + downstream completion logits.

For ``--n`` (default 1000) random tokens this script:

  (a) computes entangled tokens by *both* methods (``unembedding`` and
      ``output_distribution``); and
  (b) for each of the top-10 tokens found by the *unembedding* method, prepends
      that token to ``". I'd like to say:"`` and decodes the top-20 next-token
      logit indices.

The step-(b) result has shape ``(n, 10, 20)`` = (random tokens × entangled
tokens × output tokens). tqdm bars track each stage and per-step timings are
printed at the end.

Run (defaults to Llama-3.1-8B-Instruct):

    uv run scripts/perf_entanglement.py
    uv run scripts/perf_entanglement.py --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import contextlib
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminality import compute_entanglements, get_device

# --- Experiment constants -----------------------------------------------------
N_RANDOM = 1000  # random query tokens
TOP_ENTANGLED = 10  # entangled tokens kept per query token
TOP_OUTPUT = 20  # downstream next-token candidates kept per entangled token
SUFFIX = ". I'd like to say:"  # prepend the entangled token, then read top-20
SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="HF model id (default: %(default)s).")
    p.add_argument("--device", default=None,
                   help="Device preference passed to get_device (cuda/mps/cpu). "
                        "Default: auto.")
    p.add_argument("--n", type=int, default=N_RANDOM,
                   help="Number of random query tokens (default: %(default)s).")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Random seed for token sampling (default: %(default)s).")
    p.add_argument("--od-batch", type=int, default=16,
                   help="Chunk size for the output_distribution sweep "
                        "(one forward pass per token; default: %(default)s).")
    p.add_argument("--b-batch", type=int, default=64,
                   help="Chunk size for the downstream top-20 sweep "
                        "(default: %(default)s).")
    return p.parse_args()


def sync(device: torch.device) -> None:
    "Block until queued device work finishes, so timings are accurate (CUDA/MPS)."
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@contextlib.contextmanager
def timed(label: str, device: torch.device, timings: dict[str, float]):
    "Record wall-clock seconds for a block into ``timings`` (device-synced)."
    sync(device)
    t0 = time.perf_counter()
    yield
    sync(device)
    timings[label] = time.perf_counter() - t0


def seed_everything(seed: int, device: torch.device) -> None:
    "Device-aware seeding for reproducible token sampling."
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def chunks(n: int, size: int):
    "Yield (start, stop) index ranges covering [0, n) in steps of ``size``."
    for start in range(0, n, size):
        yield start, min(start + size, n)


def main() -> None:
    args = parse_args()
    timings: dict[str, float] = {}

    device = get_device(args.device)
    print(f"Model:  {args.model}")
    print(f"Device: {device}")

    # --- load -----------------------------------------------------------------
    with timed("model load", device, timings):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = (
            AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
            .to(device)
            .eval()
        )

    vocab = model.get_output_embeddings().weight.shape[0]
    seed_everything(args.seed, device)
    random_tokens = torch.randint(0, vocab, (args.n,))  # [n], on CPU
    print(f"Vocab:  {vocab}   |   query tokens: {args.n}\n")

    # --- (a1) unembedding -----------------------------------------------------
    # A single matmul that normalizes the whole unembedding matrix internally, so
    # we run it in one call (chunking would redo that normalization each chunk and
    # misrepresent the cost). The bar is therefore a single-step bar.
    with timed("entanglement: unembedding", device, timings):
        with tqdm(total=1, desc="unembedding") as bar:
            ent_unemb = compute_entanglements(model, random_tokens, method="unembedding")
            bar.update(1)
    unemb_top = ent_unemb.topk(TOP_ENTANGLED, dim=-1).indices.cpu()  # [n, 10]
    del ent_unemb

    # --- (a2) output_distribution ---------------------------------------------
    # One forward pass per token: chunk it (batch=n would materialize a
    # [n, L, vocab] logits tensor and likely OOM) and keep the top-10 per token.
    od_top = torch.empty(args.n, TOP_ENTANGLED, dtype=torch.long)
    with timed("entanglement: output_distribution", device, timings):
        for lo, hi in tqdm(list(chunks(args.n, args.od_batch)), desc="output_distribution"):
            ratio = compute_entanglements(
                model, random_tokens[lo:hi], method="output_distribution",
                tokenizer=tokenizer)  # [chunk, vocab]
            od_top[lo:hi] = ratio.topk(TOP_ENTANGLED, dim=-1).indices.cpu()
            del ratio

    # --- (b) downstream top-20 ------------------------------------------------
    # Prepend each entangled token (exact id splice -- never decode->re-encode)
    # to SUFFIX and read the top-20 next-token logits. All inputs are the same
    # length, so batching needs no padding.
    suffix_ids = tokenizer(SUFFIX, add_special_tokens=False).input_ids
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    flat = unemb_top.reshape(-1)  # [n*10]
    out_top = torch.empty(flat.numel(), TOP_OUTPUT, dtype=torch.long)
    with timed("downstream top-20", device, timings):
        for lo, hi in tqdm(list(chunks(flat.numel(), args.b_batch)), desc="downstream"):
            ids = torch.tensor(
                [bos + [int(t)] + suffix_ids for t in flat[lo:hi].tolist()],
                device=device)  # [chunk, L]
            with torch.no_grad():
                logits = model(ids).logits[:, -1, :]  # [chunk, vocab]
            out_top[lo:hi] = logits.topk(TOP_OUTPUT, dim=-1).indices.cpu()
            del logits
    result = out_top.reshape(args.n, TOP_ENTANGLED, TOP_OUTPUT)  # [n, 10, 20]

    # --- report ---------------------------------------------------------------
    print(f"\nResult shape: {tuple(result.shape)} "
          f"(random tokens x entangled tokens x output tokens)\n")

    print("Timings (seconds):")
    for label, secs in timings.items():
        print(f"  {label:<34} {secs:8.2f}")
    print(f"  {'TOTAL':<34} {sum(timings.values()):8.2f}\n")

    # Small decoded sample: the first query token, its 10 unembedding-entangled
    # tokens, and the 20 outputs for the first of those.
    decode = tokenizer.convert_ids_to_tokens
    q = int(random_tokens[0])
    print(f"Sample (first query token {q!r} = {decode([q])[0]!r}):")
    ent_ids = unemb_top[0].tolist()
    print(f"  unembedding top-{TOP_ENTANGLED}: {decode(ent_ids)}")
    print(f"  output_distribution top-{TOP_ENTANGLED}: {decode(od_top[0].tolist())}")
    print(f"  downstream top-{TOP_OUTPUT} after {decode([ent_ids[0]])[0]!r} + {SUFFIX!r}:")
    print(f"    {decode(result[0, 0].tolist())}")


if __name__ == "__main__":
    main()
