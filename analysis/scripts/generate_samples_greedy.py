import json
import os
import pickle
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import sentencepiece as spm
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

CUSTOM_OBJECTS: Dict[str, object] = {}
CUSTOM_OBJECTS: Dict[str, object] = {}
BUILD_V5_MODEL = None
try:
    # Ensure custom layers/schedules from v5 are available if needed
    from scripts.llm_transformer_v5 import (  # noqa: F401
        OutputProjection,
        WarmupCosineRestartSchedule,
        WarmupCosineSchedule,
        build_transformer_model,
    )

    CUSTOM_OBJECTS["WarmupCosineSchedule"] = WarmupCosineSchedule
    CUSTOM_OBJECTS["WarmupCosineRestartSchedule"] = WarmupCosineRestartSchedule
    CUSTOM_OBJECTS["OutputProjection"] = OutputProjection
    BUILD_V5_MODEL = build_transformer_model
except Exception as exc:
    print(f"[WARN] Falha ao importar componentes v5: {exc}")
    BUILD_V5_MODEL = None
    CUSTOM_OBJECTS = {}

OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "analysis/artifacts/samples/samples_brwac_20k.json"))

MODELS = {
    "v1_brwac_20k": {
        "model": Path('versions/v1-char-rnn/models/modelo_brwac_v1_20k.keras'),
        "mapping": Path('versions/v1-char-rnn/mappings/mapeamentos_brwac_v1_20k.pkl'),
    },
    "v2_brwac_20k": {
        "model": Path('versions/v2-char-lm/models/modelo_brwac_v2_20k.keras'),
        "mapping": Path('versions/v2-char-lm/mappings/mapeamentos_brwac_v2_20k.pkl'),
    },
    "v3_brwac_20k": {
        "model": Path('versions/v3-stacked-lstm/models/modelo_brwac_v3_20k.keras'),
        "mapping": Path('versions/v3-stacked-lstm/mappings/mapeamentos_brwac_v3_20k.pkl'),
    },
    "v4_brwac_subword": {
        "model": Path('versions/v4-subword-lstm/models/modelo_brwac_v4_subword.keras'),
        "mapping": Path('versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4.pkl'),
    },
    "v4_brwac_subword_ep6": {
        "model": Path('versions/v4-subword-lstm/models/modelo_brwac_v4_subword_ep6.keras'),
        "mapping": Path('versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4_ep6.pkl'),
    },
    "v4_brwac_subword_v4k_bf_ep2": {
        "model": Path('versions/v4-subword-lstm/models/modelo_brwac_v4_subword_v4k_bf_ep2.keras'),
        "mapping": Path('versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4k_bf_ep2.pkl'),
    },
    "v4_brwac_subword_v4k_bf_lstm_ep2": {
        "model": Path('versions/v4-subword-lstm/models/modelo_brwac_v4_lstm_v4k_bf_ep2.keras'),
        "mapping": Path('versions/v4-subword-lstm/mappings/mapeamentos_brwac_v4_lstm_v4k_bf_ep2.pkl'),
    },
    "v5_brwac_transformer": {
        "model": Path('versions/v5-transformer/models/modelo_v5_ready_saved.keras'),
        "mapping": Path('versions/v5-transformer/mappings/mapeamentos_v5.pkl'),
    },
    "v5_brwac_transformer_fixed": {
        "model": Path('versions/v5-transformer/models/modelo_v5_fixed.keras'),
        "mapping": Path('versions/v5-transformer/mappings/mapeamentos_v5_fixed.pkl'),
    },
    "v5_brwac_transformer_main": {
        "model": Path("versions/v5-transformer/models/modelo_v5_main.keras"),
        "mapping": Path("versions/v5-transformer/mappings/mapeamentos_v5_main.pkl"),
    },
    "v5_brwac_transformer_v5b_dryrun": {
        "model": Path("versions/v5b-transformer/models/modelo_v5b_dryrun.keras"),
        "mapping": Path("versions/v5b-transformer/mappings/mapeamentos_v5b_dryrun.pkl"),
    },
    "v5_brwac_transformer_opt2_8k_long": {
        "model": Path("versions/v5-option2/models/modelo_v5_opt2_8k_long.keras"),
        "mapping": Path("versions/v5-option2/mappings/mapeamentos_v5_opt2_8k_long.pkl"),
    },
    "v5_brwac_transformer_opt2_8k_archA": {
        "model": Path("versions/v5-option2/models/modelo_v5_opt2_8k_archA.keras"),
        "mapping": Path("versions/v5-option2/mappings/mapeamentos_v5_opt2_8k_archA.pkl"),
    },
    "v5_brwac_transformer_opt2_8k_trial": {
        "model": Path("versions/v5-option2/models/modelo_v5_opt2_8k_trial.keras"),
        "mapping": Path("versions/v5-option2/mappings/mapeamentos_v5_opt2_8k_trial.pkl"),
    },
    "v5_brwac_transformer_opt2_16k_trial": {
        "model": Path("versions/v5-option2/models/modelo_v5_opt2_16k_trial.keras"),
        "mapping": Path("versions/v5-option2/mappings/mapeamentos_v5_opt2_16k_trial.pkl"),
    },
}

for entry in MODELS.values():
    entry["model"] = (ROOT_DIR / entry["model"]).resolve()
    entry["mapping"] = (ROOT_DIR / entry["mapping"]).resolve()

PROMPTS = [
    (
        101,
        "O setor de infraestrutura logistica brasileira debate novas concessoes ferroviarias, cronogramas de duplicacao, metas de produtividade e integracao com portos para desafogar corredores de exportacao.",
    ),
    (
        102,
        "Analistas do mercado financeiro reavaliam previsoes trimestrais para bancos listados, discutem juros, carteira de credito, inadimplencia corporativa e estrategias de hedge diante de volatilidade externa.",
    ),
    (
        103,
        "Engenheiros de software de uma fintech planejam migracao para microservicos, definem SLAs, monitoramento proativo, politicas de rollback continuo e treinamentos para times de suporte e compliance.",
    ),
    (
        104,
        "A diretoria de um clube de futebol negocia patrocinio master, reforcos na janela de transferencia, metas de bilheteria, programa de socios e auditoria das contas apos uma temporada irregular nos gramados.",
    ),
    (
        105,
        "Uma rede de hospitais privados avalia expansao para telemedicina, protocolos integrados de prontuario, parcerias com seguradoras, capacidade de leitos, gestao de estoque farmacologico e contratacao medica.",
    ),
]

TEMPERATURE = 0.3
TOP_K = 80
TOP_P = 0.90
LENGTH = 280
MIN_OUTPUT_TOKENS = 120
MAX_SKIP_EOS = 16
FREQUENCY_PENALTY = 0.5


@dataclass
class TokenizerAssets:
    mode: str
    seq_len: int
    vocab_size: int
    char_to_idx: Dict[str, int] | None = None
    idx_to_token: Dict[int, str] | None = None
    unk_idx: int | None = None
    unk_token: str | None = None
    processor: spm.SentencePieceProcessor | None = None
    add_bos: bool = False
    add_eos: bool = False
    tokenizer_model: Path | None = None


def load_mapping(path: Path) -> TokenizerAssets:
    with path.open('rb') as fh:
        data = pickle.load(fh)

    seq_len = data.get('tamanho_sequencia') or data.get('sequence_length') or 160
    tokenization = data.get('tokenization', 'char')

    if tokenization == 'sentencepiece':
        model_raw = data.get('tokenizer_model')
        if not model_raw:
            raise ValueError(f'tokenizer_model ausente no mapeamento {path}')
        model_path = Path(model_raw)
        if not model_path.exists():
            try:
                parts = list(model_path.parts)
                if "versions" in parts:
                    idx = parts.index("versions")
                    candidate = ROOT_DIR / Path(*parts[idx:])
                    if candidate.exists():
                        model_path = candidate
            except Exception:
                pass
        if not model_path.exists():
            raise ValueError(f'Arquivo SentencePiece nao encontrado: {model_path}')
        processor = spm.SentencePieceProcessor()
        if not processor.Load(str(model_path)):
            raise ValueError(f'Falha ao carregar SentencePiece: {model_path}')
        vocab_size = int(data.get('tokenizer_vocab_size') or processor.GetPieceSize())
        add_bos = bool(data.get('add_bos', False))
        add_eos = bool(data.get('add_eos', False))
        return TokenizerAssets(
            mode='sentencepiece',
            seq_len=int(seq_len),
            vocab_size=vocab_size,
            processor=processor,
            add_bos=add_bos,
            add_eos=add_eos,
            tokenizer_model=model_path,
        )

    if 'char_to_idx' in data:
        c2i = data['char_to_idx']
        i2c = data['idx_to_char']
    elif 'char_to_id' in data:
        c2i = data['char_to_id']
        i2c = data['id_to_char']
    else:
        c2i = data['char_para_int']
        i2c = data['int_para_char']
    try:
        c2i = {str(k): int(v) for k, v in c2i.items()}
        i2c = {int(k): str(v) for k, v in i2c.items()}
    except Exception:
        pass
    unk = data.get('unk_token')
    if isinstance(unk, bytes):
        unk = unk.decode('utf-8', errors='ignore') or None
    if isinstance(unk, str) and unk in c2i:
        unk_idx = c2i[unk]
    else:
        unk_idx = c2i.get(' ', 0)
    return TokenizerAssets(
        mode='char',
        seq_len=int(seq_len),
        vocab_size=len(c2i),
        char_to_idx=c2i,
        idx_to_token=i2c,
        unk_idx=unk_idx,
        unk_token=unk if isinstance(unk, str) else None,
    )


def encode_prompt(prompt: str, assets: TokenizerAssets) -> List[int]:
    if assets.mode == 'char':
        assert assets.char_to_idx is not None
        unk_idx = assets.unk_idx if assets.unk_idx is not None else 0
        seq = [assets.char_to_idx.get(ch, unk_idx) for ch in prompt][-assets.seq_len:]
        if len(seq) < assets.seq_len:
            seq = [unk_idx] * (assets.seq_len - len(seq)) + seq
        return seq

    assert assets.processor is not None
    # CRITICAL: Use add_bos/add_eos from assets to match training configuration
    ids = assets.processor.EncodeAsIds(prompt, add_bos=assets.add_bos, add_eos=False)
    if not ids:
        bos_id = assets.processor.bos_id()
        if bos_id is not None and bos_id >= 0:
            ids = [bos_id]
        else:
            unk_id = assets.processor.unk_id()
            ids = [unk_id if unk_id is not None and unk_id >= 0 else 0]
    if len(ids) > assets.seq_len:
        ids = ids[-assets.seq_len:]
    return ids


def decode_tokens(tokens: Sequence[int], assets: TokenizerAssets) -> str:
    if assets.mode == 'char':
        assert assets.idx_to_token is not None
        text = ''.join(assets.idx_to_token.get(int(idx), '?') for idx in tokens)
        if assets.unk_token:
            text = text.replace(assets.unk_token, ' ')
        return text

    assert assets.processor is not None
    pad_id = assets.processor.pad_id()
    filtered = [int(idx) for idx in tokens if idx != pad_id]
    if not filtered:
        return ''
    return assets.processor.DecodeIds(filtered)


def sample_topk_topp(logits, temperature, top_k, top_p, rng):
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        temperature = 1e-6
    logits = logits / temperature

    if top_k and 0 < top_k < logits.shape[-1]:
        kth = np.argpartition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf

    if top_p and 0 < top_p < 1:
        sorted_idx = np.argsort(-logits)
        sorted_logits = logits[sorted_idx]
        probs = tf.nn.softmax(sorted_logits).numpy()
        cumulative = np.cumsum(probs)
        mask = cumulative > top_p
        if np.all(mask):
            mask[0] = False
        probs[mask] = 0.0
        logits[sorted_idx] = np.log(probs + 1e-9)

    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs_sum = probs.sum()
    if not np.isfinite(probs_sum) or probs_sum == 0.0:
        probs = np.ones_like(probs) / probs.size
    else:
        probs = probs / probs_sum
    return int(rng.choice(len(probs), p=probs))


def prepare_input(seq, model, vocab_size):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if len(input_shape) == 3:
        # modelo espera one-hot
        x = np.zeros((1, len(seq), vocab_size), dtype=np.float32)
        for t, idx in enumerate(seq):
            if 0 <= idx < vocab_size:
                x[0, t, idx] = 1.0
        return x, True
    return np.array(seq, dtype=np.int32)[None, :], False


def extract_logits(output, was_one_hot):
    arr = np.asarray(output[0])
    if arr.ndim > 1:
        arr = arr[-1]
    if was_one_hot:
        arr = np.log(np.clip(arr, 1e-9, 1.0))
    return arr


def generate(
    model: tf.keras.Model,
    assets: TokenizerAssets,
    prompt: str,
    seed: int,
    temperature: float,
    top_k: int,
    top_p: float,
    length: int,
) -> str:
    rng = np.random.default_rng(seed)
    out = []
    seq = encode_prompt(prompt, assets)
    vocab_size = assets.vocab_size
    eos_id = None
    pad_id = None
    min_tokens = max(0, MIN_OUTPUT_TOKENS)
    if assets.mode == 'sentencepiece' and assets.processor is not None:
        eos_id = assets.processor.eos_id() if assets.add_eos else None
        pad_id = assets.processor.pad_id()
        if pad_id is None or pad_id < 0:
            alt = assets.processor.unk_id()
            pad_id = alt if alt is not None and alt >= 0 else 0
    skip_eos = 0
    token_counts = Counter()
    for _ in range(length):
        current_seq = seq[-assets.seq_len:]
        x, one_hot = prepare_input(current_seq, model, vocab_size)
        preds = model(x, training=False).numpy()
        logits = extract_logits(preds, one_hot)
        if FREQUENCY_PENALTY > 0.0:
            penalties = np.array([token_counts[token] for token in range(len(logits))], dtype=np.float64)
            logits = logits - FREQUENCY_PENALTY * penalties
        idx = sample_topk_topp(logits, temperature, top_k, top_p, rng)
        if eos_id is not None and idx == eos_id:
            if len(out) >= min_tokens:
                break
            skip_eos += 1
            if skip_eos >= MAX_SKIP_EOS:
                break
            replacement = pad_id if pad_id is not None and pad_id >= 0 else idx
            seq.append(int(replacement))
            if len(seq) > assets.seq_len:
                seq = seq[-assets.seq_len:]
            continue
        out.append(idx)
        token_counts[idx] += 1
        seq.append(int(idx))
        if len(seq) > assets.seq_len:
            seq = seq[-assets.seq_len:]
    decoded = decode_tokens(out, assets)
    return clean_output(decoded)


def clean_output(text: str) -> str:
    text = re.sub(r'__+', '_', text)
    return ''.join(ch for ch in text if ch.isprintable() or ch in '\n\t')

results = {}
filter_raw = os.environ.get('MODEL_KEYS')
allowed_models = {name.strip() for name in filter_raw.split(',')} if filter_raw else None
for key, paths in MODELS.items():
    if allowed_models and key not in allowed_models:
        continue
    model_path = paths['model']
    mapping_path = paths['mapping']
    if not model_path.exists() or not mapping_path.exists():
        print(f"[WARN] Pulando {key}: artefatos ausentes ({model_path} / {mapping_path}).")
        continue
    if key.startswith("v5_brwac_transformer"):
        mapping_json = mapping_path.with_suffix(".json")
        cfg = json.loads(mapping_json.read_text(encoding="utf-8"))
        model_cfg = cfg["model"]
        seq_len = cfg["sequence_length"]
        if BUILD_V5_MODEL:
            model = BUILD_V5_MODEL(
                vocab_size=cfg["tokenizer_vocab_size"],
                seq_len=seq_len,
                d_model=model_cfg["d_model"],
                num_layers=model_cfg["num_layers"],
                num_heads=model_cfg["num_heads"],
                d_ff=model_cfg["d_ff"],
                dropout=model_cfg["dropout"],
            )
            ckpt_candidates = [
                model_path.with_name(f"{model_path.stem}_checkpoint.keras"),
                model_path,
            ]
            stage_name = model_path.stem.split("_")[-1]
            stage_dir = ROOT_DIR / "versions" / "v5-transformer" / "checkpoints" / stage_name
            if stage_dir.is_dir():
                ckpt_candidates.extend(sorted(stage_dir.glob("epoch_*.weights.h5"), reverse=True))

            weights_loaded = False
            for candidate in ckpt_candidates:
                if not candidate.exists():
                    continue
                try:
                    model.load_weights(str(candidate))
                    print(f"[INFO] Pesos carregados de {candidate}")
                    weights_loaded = True
                    break
                except Exception as exc:
                    print(f"[WARN] Falha ao carregar pesos de {candidate}: {exc}")
            if not weights_loaded:
                print(f"[WARN] Nenhum checkpoint aplicável encontrado para {key}; pulando.")
                continue
        embed_layer = model.get_layer("token_embedding")
        proj_layer = model.get_layer("logits")
        proj_layer.kernel.assign(tf.cast(embed_layer.get_weights()[0], proj_layer.kernel.dtype))
    else:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,
        )
    try:
        assets = load_mapping(mapping_path)
    except ValueError as exc:
        print(f"[WARN] Pulando {key}: {exc}")
        continue
    model_results = []
    for idx, (seed, prompt) in enumerate(PROMPTS, 1):
        generated = generate(model, assets, prompt, seed, TEMPERATURE, TOP_K, TOP_P, LENGTH)
        model_results.append({"seed": seed, "prompt": prompt, "output": generated})
        total = len(PROMPTS)
        bar = "#" * idx + "-" * (total - idx)
        print(f"[{bar}] {key}: {idx}/{total}")
    results[key] = model_results

with OUTPUT_PATH.open('w', encoding='utf-8') as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2)

