# -*- coding: utf-8 -*-
"""Treina modelos char/subword (v1-v4) no BrWaC para comparacao justa."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from datasets import load_dataset

_ROOT = Path(__file__).resolve().parents[0].parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from tcc_llm.llm_generico import LLMSimplesGenerico
from tcc_llm.llm_generico_v2 import LLMSimplesGenericoV2
from tcc_llm.llm_generico_v3 import LLMSimplesGenericoV3
from tcc_llm.llm_subword_v4 import LLMSubwordV4
from scripts.brwac_preprocess import preparar_texto, resolve_end_inline_sep


def _process_documents(
    dataset_split,
    limite: int,
    *,
    min_len: int,
    clean_kwargs: Dict[str, object],
) -> Tuple[int, str, int]:
    count = 0
    total_chars = 0
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        caminho = tmp.name

    with open(caminho, "w", encoding="utf-8") as handle:
        for exemplo in dataset_split:
            if count >= limite:
                break
            texto_p = preparar_texto(exemplo["text"], **clean_kwargs)
            if len(texto_p) >= min_len:
                handle.write(texto_p + "\n\n")
                total_chars += len(texto_p)
                count += 1
                if count % 5000 == 0:
                    print(f"[INFO] Processados {count} textos...")

    return count, caminho, total_chars


def _parse_units_arg(raw: str) -> list[int]:
    """Extrai lista de unidades a partir de string separada por virgula."""
    if not raw:
        return []
    unidades: list[int] = []
    for parte in raw.split(","):
        parte = parte.strip()
        if not parte:
            continue
        unidades.append(int(parte))
    return unidades


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar modelos char/subword (v1-v4) com BrWaC")
    parser.add_argument("--arch", choices=["v1", "v2", "v3", "v4"], default="v1", help="Arquitetura alvo")
    parser.add_argument("--tokenization", choices=["char", "subword"], default="char", help="Modo de tokenizacao")
    parser.add_argument("--max_textos", type=int, default=20000, help="Numero maximo de textos")
    parser.add_argument("--min_len", type=int, default=200, help="Comprimento minimo por documento")
    parser.add_argument("--seed", type=int, default=42, help="Seed de reproducao")
    parser.add_argument("--no_lowercase", action="store_true", help="Nao converter para minusculas")
    parser.add_argument("--keep_numbers", action="store_true", help="Nao normalizar numeros")
    parser.add_argument("--keep_upper_metadata", action="store_true", help="Preservar linhas em caixa alta")
    parser.add_argument("--end_inline_sep", choices=["space", "newline"], default="newline")
    parser.add_argument("--epocas", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tamanho_sequencia", type=int, default=160, help="Tamanho de janela para char/subword")
    parser.add_argument("--tamanho_lstm", type=int, default=256, help="Usado em v1 e v2")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Usado nas arquiteturas com embedding (v2/v4)")
    parser.add_argument("--stride", type=int, default=4, help="Usado nas arquiteturas baseadas em tf.data (v2/v4)")
    parser.add_argument("--shuffle_buffer", type=int, default=200000, help="Usado nas arquiteturas com tf.data")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout em LSTM/GRU")
    parser.add_argument("--recurrent_dropout", type=float, default=0.0, help="Recurrent dropout em LSTM/GRU")
    parser.add_argument("--clipnorm", type=float, default=1.0, help="Norm clipping nas arquiteturas tf.data")
    parser.add_argument("--stack_units", type=str, default="", help="Lista (ex: 512,512,256) para LSTMs empilhadas (v3)")
    parser.add_argument("--final_dropout", type=float, default=0.0, help="Dropout final no topo da pilha (v3/v4)")
    parser.add_argument("--layer_norm", action="store_true", help="Aplica LayerNorm apos cada LSTM (v3)")
    parser.add_argument("--layer_norm_epsilon", type=float, default=1e-5, help="Epsilon da LayerNorm (v3)")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Caminho do modelo SentencePiece (.model) para subword")
    parser.add_argument("--vocab_size", type=int, default=0, help="Tamanho esperado do vocabulario subword (validacao)")
    parser.add_argument("--rnn_type", choices=["gru", "lstm"], default="gru", help="Tipo de recorrencia para v4")
    parser.add_argument("--subword_add_bos", action="store_true", help="Adicionar token BOS em cada documento")
    parser.add_argument("--subword_add_eos", action="store_true", help="Adicionar token EOS em cada documento")
    parser.add_argument("--validacao_split", type=float, default=0.1)
    parser.add_argument("--batch_log_freq", type=int, default=500)
    parser.add_argument(
        "--modelo_saida",
        type=str,
        default="",
    )
    parser.add_argument(
        "--mapeamentos_saida",
        type=str,
        default="",
    )
    parser.add_argument(
        "--log_json_saida",
        type=str,
        default="",
    )
    args = parser.parse_args()

    if args.arch == "v4":
        if args.tokenization != "subword":
            print("[WARN] Forcando modo de tokenizacao 'subword' para a arquitetura v4.")
        args.tokenization = "subword"
    elif args.tokenization == "subword":
        parser.error("Somente a arquitetura v4 suporta tokenizacao subword neste script.")

    if args.tokenization == "subword":
        if not args.tokenizer_path:
            parser.error("--tokenizer_path e obrigatorio para tokenizacao subword.")
        tokenizer_path = Path(args.tokenizer_path)
        if not tokenizer_path.exists():
            parser.error(f"Arquivo do tokenizer nao encontrado: {args.tokenizer_path}")
        args.tokenizer_path = str(tokenizer_path.resolve())
        if args.vocab_size and args.vocab_size <= 0:
            parser.error("--vocab_size deve ser positivo quando fornecido.")

    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    base_dir_map = {
        "v1": Path("versions") / "v1-char-rnn",
        "v2": Path("versions") / "v2-char-lm",
        "v3": Path("versions") / "v3-stacked-lstm",
        "v4": Path("versions") / "v4-subword-lstm",
    }
    base_dir = base_dir_map[args.arch]
    base_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    default_model_path = {
        "v1": base_dir / "models" / "modelo_brwac_v1.keras",
        "v2": base_dir / "models" / "modelo_brwac_v2.keras",
        "v3": base_dir / "models" / "modelo_brwac_v3.keras",
        "v4": base_dir / "models" / "modelo_brwac_v4_subword.keras",
    }
    default_map_path = {
        "v1": base_dir / "mappings" / "mapeamentos_brwac_v1.pkl",
        "v2": base_dir / "mappings" / "mapeamentos_brwac_v2.pkl",
        "v3": base_dir / "mappings" / "mapeamentos_brwac_v3.pkl",
        "v4": base_dir / "mappings" / "mapeamentos_brwac_v4.pkl",
    }
    args.modelo_saida = args.modelo_saida or str(default_model_path[args.arch])
    args.mapeamentos_saida = args.mapeamentos_saida or str(default_map_path[args.arch])

    log_json_path = Path(args.log_json_saida) if args.log_json_saida else logs_dir / f"train_{ts}.json"
    csv_log_path = logs_dir / f"history_{ts}.csv"
    batch_log_path = logs_dir / f"batches_{ts}.log"

    print("[INFO] Carregando dataset BrWaC (split 'train')...")
    dataset = load_dataset("nlpufg/brwac")
    ds_train_full = dataset["train"].shuffle(seed=args.seed)
    print(f"[INFO] Dataset carregado com {len(ds_train_full)} exemplos")

    split = ds_train_full.train_test_split(test_size=args.validacao_split, seed=args.seed)
    ds_train = split["train"]
    ds_val = split["test"]
    print(f"[INFO] Split -> train: {len(ds_train)} | val: {len(ds_val)}")

    end_inline_sep = resolve_end_inline_sep(args.end_inline_sep)
    clean_kwargs = dict(
        lowercase=not args.no_lowercase,
        end_inline_sep=end_inline_sep,
        min_line_chars=40,
        min_alpha_ratio=0.4,
        normalize_numbers=not args.keep_numbers,
        drop_uppercase_metadata=not args.keep_upper_metadata,
    )

    max_train = min(args.max_textos, len(ds_train))
    max_val = min(int(args.max_textos * args.validacao_split), len(ds_val))

    count_t, caminho_texto_train, train_chars = _process_documents(
        ds_train,
        max_train,
        min_len=args.min_len,
        clean_kwargs=clean_kwargs,
    )
    count_v = 0
    caminho_texto_val = ""
    val_chars = 0
    if args.arch != "v1":
        count_v, caminho_texto_val, val_chars = _process_documents(
            ds_val,
            max_val,
            min_len=args.min_len,
            clean_kwargs=clean_kwargs,
        )

    print(f"[INFO] Treino: {count_t} docs | Val: {count_v} docs")

    started_at = time.time()
    try:
        stride = args.stride if args.arch in {"v2", "v3", "v4"} else 1
        embedding_dim = 64 if args.arch == "v1" else (args.embedding_dim if args.embedding_dim > 0 else 256)
        dropout = args.dropout if args.arch in {"v2", "v3", "v4"} else 0.0
        recurrent_dropout = args.recurrent_dropout if args.arch in {"v2", "v3", "v4"} else 0.0
        clipnorm = (args.clipnorm if args.clipnorm > 0 else None) if args.arch in {"v2", "v3", "v4"} else None
        final_dropout = args.final_dropout if args.arch in {"v3", "v4"} else 0.0
        stack_units = _parse_units_arg(args.stack_units)
        if args.arch == "v3":
            if not stack_units:
                stack_units = [args.tamanho_lstm, args.tamanho_lstm]
            elif len(stack_units) == 1:
                stack_units = stack_units * 2
        elif args.arch == "v4":
            if not stack_units:
                stack_units = [embedding_dim, embedding_dim]
        else:
            stack_units = [args.tamanho_lstm]

        if args.arch == "v1":
            llm = LLMSimplesGenerico(
                tamanho_sequencia=args.tamanho_sequencia,
                tamanho_lstm=args.tamanho_lstm,
                epocas_treino=args.epocas,
                batch_size=args.batch_size,
                learning_rate=1e-3,
                validacao_split=args.validacao_split,
            )
        elif args.arch == "v2":
            llm = LLMSimplesGenericoV2(
                tamanho_sequencia=args.tamanho_sequencia,
                tamanho_lstm=args.tamanho_lstm,
                embedding_dim=embedding_dim,
                epocas_treino=args.epocas,
                batch_size=args.batch_size,
                learning_rate=1e-3,
                stride=stride,
                shuffle_buffer=args.shuffle_buffer,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                clipnorm=clipnorm,
                unk_token="~",
            )
        elif args.arch == "v3":
            llm = LLMSimplesGenericoV3(
                tamanho_sequencia=args.tamanho_sequencia,
                lstm_units=stack_units,
                embedding_dim=embedding_dim,
                epocas_treino=args.epocas,
                batch_size=args.batch_size,
                learning_rate=1e-3,
                stride=stride,
                shuffle_buffer=args.shuffle_buffer,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                final_dropout=final_dropout,
                layer_norm=args.layer_norm,
                layer_norm_epsilon=args.layer_norm_epsilon,
                clipnorm=clipnorm,
                unk_token="~",
            )
        else:
            llm = LLMSubwordV4(
                sequence_length=args.tamanho_sequencia,
                embedding_dim=embedding_dim,
                rnn_units=stack_units,
                rnn_type=args.rnn_type,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                final_dropout=final_dropout,
                stride=stride,
                shuffle_buffer=args.shuffle_buffer,
                batch_size=args.batch_size,
                epochs=args.epocas,
                learning_rate=1e-3,
                clipnorm=clipnorm,
                add_bos=args.subword_add_bos,
                add_eos=args.subword_add_eos,
                tokenizer_path=args.tokenizer_path,
            )

        has_external_val = args.arch != "v1" and count_v > 0
        monitor = "val_loss" if has_external_val else "loss"

        def _batch_logger_factory(path: Path, freq: int) -> tf.keras.callbacks.Callback:
            if freq <= 0:
                return tf.keras.callbacks.LambdaCallback()

            def _on_batch_end(batch, logs=None):
                if logs is None or batch % freq != 0:
                    return
                loss = logs.get("loss")
                msg = f"batch {batch} - loss {loss:.4f}" if isinstance(loss, float) else f"batch {batch} - loss {loss}"
                print(msg, flush=True)
                try:
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(f"{int(time.time())},{msg}\n")
                except Exception:
                    pass

            return tf.keras.callbacks.LambdaCallback(on_batch_end=_on_batch_end)

        callbacks: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=args.modelo_saida,
                save_best_only=True,
                monitor=monitor,
                mode="min",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=5e-5),
            tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(csv_log_path),
        ]
        if args.batch_log_freq > 0:
            callbacks.append(_batch_logger_factory(batch_log_path, args.batch_log_freq))

        treinar_kwargs: Dict[str, object] = dict(
            caminho_texto=caminho_texto_train,
            nome_arquivo_modelo=args.modelo_saida,
            nome_arquivo_maps=args.mapeamentos_saida,
            callbacks=callbacks,
        )
        if args.arch != "v1":
            treinar_kwargs["caminho_texto_validacao"] = caminho_texto_val
        llm.treinar(**treinar_kwargs)

        ended_at = time.time()
        elapsed = ended_at - started_at
        print(f"[INFO] Treinamento concluido! Modelo em {args.modelo_saida}")

        stats = llm.stats() if hasattr(llm, "stats") else {}
        stride_eff = stride
        if args.tokenization == "char":
            train_windows_est = max(0, (train_chars - args.tamanho_sequencia) // max(1, stride_eff))
            val_windows_est = max(0, (val_chars - args.tamanho_sequencia) // max(1, stride_eff)) if val_chars else 0
        else:
            train_windows_est = stats["train_windows"] if isinstance(stats, dict) and "train_windows" in stats else 0
            val_windows_est = stats["val_windows"] if isinstance(stats, dict) and "val_windows" in stats else 0
        if isinstance(stats, dict):
            stats.setdefault("train_windows", stats.get("train_windows_est", train_windows_est))
            stats.setdefault("val_windows", stats.get("val_windows_est", val_windows_est))
            stats["train_windows_est"] = train_windows_est
            stats["val_windows_est"] = val_windows_est
            stats.setdefault("tokens_per_epoch_est", train_windows_est * args.tamanho_sequencia)
        else:
            stats = {}

        train_windows_actual = stats.get("train_windows", train_windows_est)
        tokens_per_epoch = stats.get("tokens_per_epoch_est", train_windows_actual * args.tamanho_sequencia)
        total_tokens = tokens_per_epoch * args.epocas
        vocab_size = stats.get("vocab_size", 0)
        train_tokens_total = stats.get(
            "train_tokens", train_windows_actual * args.tamanho_sequencia
        )
        val_tokens_total = stats.get(
            "val_tokens",
            (stats.get("val_windows", val_windows_est) if isinstance(stats, dict) else val_windows_est)
            * args.tamanho_sequencia,
        )
        if args.tokenization == "subword" and args.vocab_size:
            if vocab_size and vocab_size != args.vocab_size:
                print(
                    f"[WARN] Vocabulario SentencePiece ({vocab_size}) difere do esperado ({args.vocab_size})."
                )
        def _estimate_rnn_stack_flops(input_dim: int, units_stack: list[int], cell_type: str) -> float:
            current = input_dim
            total = 0.0
            for units in units_stack:
                if cell_type == "gru":
                    total += 6 * (current * units + units * units + units)
                else:
                    total += 8 * (current * units + units * units + units)
                current = units
            return total

        if args.arch == "v1":
            stack_for_flops = [args.tamanho_lstm]
            lstm_input_dim = 64
            cell_type = "lstm"
        else:
            stack_for_flops = list(stack_units)
            lstm_input_dim = embedding_dim
            cell_type = "gru" if args.arch == "v4" and args.rnn_type == "gru" else "lstm"
        # Estimativa grosseira de FLOPs por token (forward)
        flops_lstm = _estimate_rnn_stack_flops(lstm_input_dim, stack_for_flops, cell_type)
        last_units = stack_for_flops[-1] if stack_for_flops else args.tamanho_lstm
        flops_dense = 2 * last_units * max(vocab_size, 1) + 3 * max(vocab_size, 1)
        flops_forward = flops_lstm + flops_dense
        # Aproximamos treino completo como ~3x o custo do forward (forward + backward + update)
        flops_per_token_train = flops_forward * 3
        approx_total_flops = flops_per_token_train * total_tokens
        elapsed_per_epoch = elapsed / max(args.epocas, 1)
        windows_total = train_windows_actual * args.epocas
        windows_per_sec = windows_total / elapsed if elapsed > 0 else 0.0
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
        tflops = approx_total_flops / 1e12 if approx_total_flops else 0.0

        gpu_devices = []
        for device in tf.config.list_physical_devices("GPU"):
            try:
                details = tf.config.experimental.get_device_details(device)
                if isinstance(details, dict):
                    gpu_devices.append(details.get("device_name", str(device)))
                else:
                    gpu_devices.append(str(device))
            except Exception:
                gpu_devices.append(str(device))

        train_config: Dict[str, object] = {
            "arch": args.arch,
            "tokenization": args.tokenization,
            "epocas": args.epocas,
            "batch_size": args.batch_size,
            "tamanho_sequencia": args.tamanho_sequencia,
            "tamanho_lstm": args.tamanho_lstm,
            "batch_log_freq": args.batch_log_freq,
            "seed": args.seed,
        }
        if args.arch == "v2":
            train_config.update(
                {
                    "embedding_dim": embedding_dim,
                    "stride": stride,
                    "shuffle_buffer": args.shuffle_buffer,
                    "dropout": args.dropout,
                    "recurrent_dropout": args.recurrent_dropout,
                    "clipnorm": args.clipnorm,
                }
            )
        elif args.arch == "v3":
            train_config.update(
                {
                    "embedding_dim": embedding_dim,
                    "stride": stride,
                    "shuffle_buffer": args.shuffle_buffer,
                    "dropout": args.dropout,
                    "recurrent_dropout": args.recurrent_dropout,
                    "clipnorm": args.clipnorm,
                    "lstm_units": list(stack_units),
                    "final_dropout": final_dropout,
                    "layer_norm": args.layer_norm,
                    "layer_norm_epsilon": args.layer_norm_epsilon,
                }
            )
        elif args.arch == "v4":
            train_config.update(
                {
                    "embedding_dim": embedding_dim,
                    "stride": stride,
                    "shuffle_buffer": args.shuffle_buffer,
                    "dropout": args.dropout,
                    "recurrent_dropout": args.recurrent_dropout,
                    "clipnorm": args.clipnorm,
                    "rnn_units": list(stack_units),
                    "rnn_type": args.rnn_type,
                    "final_dropout": final_dropout,
                    "tokenizer_path": args.tokenizer_path,
                    "vocab_size_expected": args.vocab_size,
                    "subword_add_bos": args.subword_add_bos,
                    "subword_add_eos": args.subword_add_eos,
                }
            )

        dataset_info = {
            "tokenization": args.tokenization,
            "train_docs": count_t,
            "val_docs": count_v,
            "min_len": args.min_len,
            "train_chars": train_chars,
            "val_chars": val_chars,
            "avg_train_chars_per_doc": train_chars / count_t if count_t else 0,
            "avg_val_chars_per_doc": val_chars / count_v if count_v else 0,
        }
        if args.tokenization == "subword":
            dataset_info.update(
                {
                    "train_tokens": int(train_tokens_total),
                    "val_tokens": int(val_tokens_total),
                    "avg_train_tokens_per_doc": float(train_tokens_total / count_t) if count_t else 0.0,
                    "avg_val_tokens_per_doc": float(val_tokens_total / count_v) if count_v else 0.0,
                }
            )

        cleaning_info = {
            "lowercase": not args.no_lowercase,
            "end_inline_sep": args.end_inline_sep,
            "min_line_chars": clean_kwargs["min_line_chars"],
            "min_alpha_ratio": clean_kwargs["min_alpha_ratio"],
            "normalize_numbers": not args.keep_numbers,
            "drop_uppercase_metadata": not args.keep_upper_metadata,
        }

        payload = {
            "timestamps": {
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)),
                "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ended_at)),
                "elapsed_sec": elapsed,
            },
            "env": {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "os": platform.platform(),
                "cpu": platform.processor(),
                "cpu_count": os.cpu_count(),
                "gpu_devices": gpu_devices,
            },
            "dataset": dataset_info,
            "cleaning": cleaning_info,
            "train_config": train_config,
            "model": stats,
            "artifacts": {
                "modelo": args.modelo_saida,
                "mapeamentos": args.mapeamentos_saida,
                "csv_history": str(csv_log_path),
                "batch_log": str(batch_log_path) if args.batch_log_freq > 0 else "",
            },
            "performance": {
                "elapsed_sec": float(elapsed),
                "elapsed_per_epoch_sec": float(elapsed_per_epoch),
                "train_windows_total": int(windows_total),
                "tokens_per_epoch_est": int(tokens_per_epoch),
                "tokens_total_est": int(total_tokens),
                "windows_per_sec": float(windows_per_sec),
                "tokens_per_sec": float(tokens_per_sec),
                "train_tokens_total": int(train_tokens_total),
                "val_tokens_total": int(val_tokens_total),
                "approx_flops_per_token_train": float(flops_per_token_train),
                "approx_total_flops_train": float(approx_total_flops),
                "approx_total_tflops_train": float(tflops),
            },
        }
        try:
            payload["git"] = {"commit": os.popen("git rev-parse HEAD").read().strip()}
        except Exception:
            pass
        try:
            with open(log_json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            print(f"[INFO] Log salvo em {log_json_path}")
        except Exception as exc:
            print(f"[WARN] Nao foi possivel salvar log JSON: {exc}")

    finally:
        for caminho in [caminho_texto_train, caminho_texto_val]:
            if caminho:
                try:
                    os.unlink(caminho)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
