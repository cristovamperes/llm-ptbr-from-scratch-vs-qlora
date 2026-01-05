from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Set


INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
REPO_BASE = "https://github.com/cristovamperes/llm-ptbr-from-scratch-vs-qlora"


def _resolve_tex_path(latex_root: Path, token: str) -> Path | None:
    raw = token.strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".tex")

    probes = [
        latex_root / candidate,
        latex_root / "tex" / candidate,
        (latex_root / candidate).resolve(),
        (latex_root / "tex" / candidate).resolve(),
    ]
    for probe in probes:
        if probe.exists():
            return probe
    return None


def _strip_comments(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        if not line:
            out_lines.append(line)
            continue
        buf: list[str] = []
        escaped = False
        for ch in line:
            if escaped:
                buf.append(ch)
                escaped = False
                continue
            if ch == "\\":
                buf.append(ch)
                escaped = True
                continue
            if ch == "%":
                break
            buf.append(ch)
        out_lines.append("".join(buf).rstrip())
    return "\n".join(out_lines)


def _expand_file(latex_root: Path, path: Path, *, seen: Set[Path]) -> str:
    resolved = path.resolve()
    if resolved in seen:
        return f"\n\n% [omitido: include repetido] {path.as_posix()}\n\n"
    seen.add(resolved)

    raw = path.read_text(encoding="utf-8", errors="ignore")

    def replacer(match: re.Match[str]) -> str:
        token = match.group(1)
        child = _resolve_tex_path(latex_root, token)
        if child is None:
            return f"\n\n% [omitido: include não encontrado] {token}\n\n"
        return _expand_file(latex_root, child, seen=seen)

    return INPUT_RE.sub(replacer, raw)


def _format_citation_keys(keys: str) -> str:
    parts = [part.strip() for part in keys.split(",") if part.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return f"[{parts[0]}]"
    return "[" + "; ".join(parts) + "]"


def _convert_to_markdown(text: str) -> str:
    text = _strip_comments(text)

    # Remover definições de macros/commands que não fazem sentido no Markdown.
    # (ex.: \newcommand{\comando}[1]{...})
    text = re.sub(
        r"(?m)^[ \t]*\\(?:re)?newcommand\*?\s*\{[^}]+\}(?:\[[^\]]+\])?\s*\{.*\}\s*$",
        "",
        text,
    )
    text = re.sub(r"(?m)^[ \t]*\\providecommand\*?\s*\{[^}]+\}.*$", "", text)

    # Citações: converte \cite{chave} -> [chave] (mantém rastreabilidade sem LaTeX).
    text = CITE_RE.sub(lambda m: _format_citation_keys(m.group(1)), text)

    # Headings
    text = re.sub(r"\\chapter\*?\{([^}]+)\}", r"# \1", text)
    text = re.sub(r"\\section\*?\{([^}]+)\}", r"## \1", text)
    text = re.sub(r"\\subsection\*?\{([^}]+)\}", r"### \1", text)
    text = re.sub(r"\\subsubsection\*?\{([^}]+)\}", r"#### \1", text)
    text = re.sub(r"\\paragraph\*?\{([^}]+)\}", r"**\1**\n", text)

    # Links
    # Macros do projeto para apontar ao repositório público.
    text = re.sub(
        r"\\repofile\{([^}]+)\}",
        lambda m: f"[{m.group(1)}]({REPO_BASE}/blob/main/{m.group(1)})",
        text,
    )
    text = re.sub(
        r"\\repodir\{([^}]+)\}",
        lambda m: f"[{m.group(1)}]({REPO_BASE}/tree/main/{m.group(1)})",
        text,
    )
    text = re.sub(r"\\href\{([^}]+)\}\{([^}]+)\}", r"[\2](\1)", text)
    text = re.sub(r"\\url\{([^}]+)\}", r"<\1>", text)
    text = re.sub(r"\\path\{([^}]+)\}", r"`\1`", text)

    # Inline formatting (best-effort, non-nested)
    text = re.sub(r"\\textbf\{([^{}]+)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{([^{}]+)\}", r"*\1*", text)
    text = re.sub(r"\\emph\{([^{}]+)\}", r"*\1*", text)
    text = re.sub(r"\\texttt\{([^{}]+)\}", r"`\1`", text)
    text = re.sub(r"\\nolinkurl\{([^}]+)\}", r"`\1`", text)

    # Lists
    text = re.sub(r"\\begin\{itemize\}", "", text)
    text = re.sub(r"\\end\{itemize\}", "", text)
    text = re.sub(r"\\begin\{enumerate\}", "", text)
    text = re.sub(r"\\end\{enumerate\}", "", text)
    text = re.sub(r"(?m)^[ \t]*\\\\item[ \t]*", "- ", text)

    # Verbatim-like environments -> code fences
    text = re.sub(r"\\begin\{verbatim\}", "```", text)
    text = re.sub(r"\\end\{verbatim\}", "```", text)
    text = re.sub(r"\\begin\{lstlisting\}", "```", text)
    text = re.sub(r"\\end\{lstlisting\}", "```", text)
    text = re.sub(r"\\begin\{minted\}\{([^}]+)\}", r"```\1", text)
    text = re.sub(r"\\end\{minted\}", "```", text)

    # Remove some common layout commands that don't translate well.
    text = re.sub(r"\\(label|ref|pageref)\{[^}]+\}", "", text)
    text = re.sub(r"\\(caption|centering)\b", "", text)
    text = re.sub(r"\\(toprule|midrule|bottomrule)\b", "", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta uma versão Markdown do TCC (a partir do LaTeX interno).")
    parser.add_argument("--latex-root", type=Path, default=Path("latex_document"), help="Pasta raiz do LaTeX.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("repositorio_publico/documento/markdown/tcc.md"),
        help="Arquivo Markdown de saída.",
    )
    parser.add_argument(
        "--entrypoints",
        type=Path,
        nargs="*",
        default=[
            Path("tex/introducao.tex"),
            Path("tex/revisao-historica.tex"),
            Path("tex/fundtecnicos.tex"),
            Path("tex/projeto-pratico.tex"),
            Path("tex/resultados_trilha2.tex"),
            Path("tex/conclusao.tex"),
        ],
        help="Arquivos .tex (relativos ao latex-root) que compõem o corpo do texto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latex_root = args.latex_root
    seen: Set[Path] = set()
    parts: list[str] = []
    for ep in args.entrypoints:
        path = (latex_root / ep).resolve()
        if not path.exists():
            raise SystemExit(f"Entrypoint não encontrado: {path}")
        parts.append(_expand_file(latex_root, path, seen=seen))

    markdown = _convert_to_markdown("\n\n".join(parts))
    note = (
        "> Nota: este arquivo é uma conversão automática do LaTeX. "
        "Citações foram convertidas para chaves BibTeX entre colchetes (ex.: [ross2023lost]). "
        "Para referências completas e formatação final, consulte o PDF.\n\n"
    )
    markdown = note + markdown
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"[OK] Markdown gerado em: {args.output}")


if __name__ == "__main__":
    main()
