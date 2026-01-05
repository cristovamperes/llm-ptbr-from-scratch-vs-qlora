param(
  [string]$OutputDir = "repositorio_publico",
  [int]$MaxFileMB = 25,
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
  $root = Resolve-Path (Join-Path $PSScriptRoot "..")
  return $root.Path
}

function Ensure-EmptyDir([string]$Path, [switch]$Clean) {
  if (Test-Path $Path) {
    $items = Get-ChildItem -Force $Path
    if ($items.Count -gt 0) {
      if (-not $Clean) {
        throw "Diretório de saída já existe e não está vazio: $Path (use -Clean para recriar)"
      }

      # Se o diretório já for um repositório Git (ex.: `repositorio_publico/.git`),
      # preserva o histórico e remove apenas o conteúdo versionado (mantém `.git`).
      # Também preserva PDFs já exportados em `documento/pdf/*.pdf`.
      $gitDir = Join-Path $Path ".git"
      $pdfDir = Join-Path $Path "documento/pdf"
      $pdfBackupDir = $null
      if (Test-Path $pdfDir) {
        $pdfBackupDir = Join-Path $env:TEMP ("tcc_public_pdf_backup_" + [guid]::NewGuid().ToString())
        New-Item -ItemType Directory -Force -Path $pdfBackupDir | Out-Null
        Copy-Item -Force -ErrorAction SilentlyContinue (Join-Path $pdfDir "*.pdf") $pdfBackupDir
      }

      if (Test-Path $gitDir) {
        Get-ChildItem -Force $Path | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force
      } else {
        Remove-Item -Recurse -Force $Path
      }

      if ($pdfBackupDir) {
        $restorePdfDir = Join-Path $Path "documento/pdf"
        New-Item -ItemType Directory -Force -Path $restorePdfDir | Out-Null
        Copy-Item -Force -ErrorAction SilentlyContinue (Join-Path $pdfBackupDir "*.pdf") $restorePdfDir
        Remove-Item -Recurse -Force $pdfBackupDir
      }
    }
  }
  New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function To-ForwardSlash([string]$Path) {
  return ($Path -replace "\\", "/")
}

function Get-RelativePath([string]$Root, [string]$FullPath) {
  $rootFull = (Resolve-Path $Root).Path.TrimEnd("\") + "\"
  $full = (Resolve-Path $FullPath).Path
  if ($full.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $full.Substring($rootFull.Length)
  }
  throw "Caminho fora do root: $full"
}

function Get-SkipReason([string]$RelativePath, [System.IO.FileInfo]$FileInfo, [int64]$MaxBytes) {
  $rel = To-ForwardSlash $RelativePath
  $lower = $rel.ToLowerInvariant()

  $denyPrefixes = @(
    "latex_document/",
    "secrets/",
    "archives/",
    ".git/",
    ".claude/",
    "data/",
    "artigos/"
  )
  foreach ($prefix in $denyPrefixes) {
    if ($lower.StartsWith($prefix)) {
      return "denylist: $prefix"
    }
  }

  if ($lower.Contains("/__pycache__/") -or $lower.EndsWith(".pyc")) {
    return "denylist: __pycache__"
  }

  # No weights/checkpoints
  $denyExt = @(".keras", ".h5", ".safetensors", ".bin", ".pt", ".ckpt")
  foreach ($ext in $denyExt) {
    if ($lower.EndsWith($ext)) {
      return "denylist: weights ($ext)"
    }
  }
  if ($lower.EndsWith(".weights.h5")) {
    return "denylist: weights (.weights.h5)"
  }

  # Deny common large/derived experiment outputs
  if ($lower.Contains("/models/")) {
    return "denylist: models/"
  }
  if ($lower.Contains("/outputs/")) {
    return "denylist: outputs/"
  }
  if ($lower.Contains("/checkpoints")) {
    return "denylist: checkpoints/"
  }

  if ($FileInfo.Length -gt $MaxBytes) {
    return "size>$($MaxBytes)"
  }

  return $null
}

function Copy-Curated([string]$Root, [string]$Dest, [string[]]$RootsToInclude, [int64]$MaxBytes) {
  $copied = New-Object System.Collections.Generic.List[object]
  $skipped = New-Object System.Collections.Generic.List[object]

  foreach ($entry in $RootsToInclude) {
    $src = Join-Path $Root $entry
    if (-not (Test-Path $src)) {
      continue
    }

    $files = Get-ChildItem -Recurse -File -Force $src
    foreach ($f in $files) {
      $rel = Get-RelativePath $Root $f.FullName
      $reason = Get-SkipReason $rel $f $MaxBytes
      if ($null -ne $reason) {
        $skipped.Add([pscustomobject]@{ path = (To-ForwardSlash $rel); size_bytes = $f.Length; reason = $reason }) | Out-Null
        continue
      }

      $dstPath = Join-Path $Dest $rel
      $dstDir = Split-Path -Parent $dstPath
      New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
      Copy-Item -Force $f.FullName $dstPath
      $copied.Add([pscustomobject]@{ path = (To-ForwardSlash $rel); size_bytes = $f.Length }) | Out-Null
    }
  }

  return @{ copied = $copied; skipped = $skipped }
}

function Write-TextFileUtf8([string]$Path, [string]$Content) {
  $utf8WithBom = New-Object System.Text.UTF8Encoding($true)
  [System.IO.File]::WriteAllText($Path, $Content, $utf8WithBom)
}

$root = Get-RepoRoot
$dest = $OutputDir
if (-not [System.IO.Path]::IsPathRooted($dest)) {
  $dest = Join-Path $root $dest
}

Ensure-EmptyDir -Path $dest -Clean:$Clean

# Documento (PDF + Markdown)
New-Item -ItemType Directory -Force -Path (Join-Path $dest "documento/pdf") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $dest "documento/markdown") | Out-Null

# Copiar snapshot curado
$maxBytes = [int64]$MaxFileMB * 1024 * 1024
$rootsToInclude = @(
  "README.md",
  "requirements.txt",
  "analysis",
  "docs",
  "scripts",
  "configs",
  "tcc_llm",
  "comparisons",
  "versions"
)

$result = Copy-Curated -Root $root -Dest $dest -RootsToInclude $rootsToInclude -MaxBytes $maxBytes

# Ajuste do snapshot público: garantir que alguns artefatos pequenos (ex.: datasets JSONL da Trilha 2)
# não fiquem ignorados por `.gitignore`, pois o documento referencia esses arquivos via links no GitHub.
$trilha2GitignorePath = Join-Path $dest "versions/trilha2-lora/.gitignore"
if (Test-Path $trilha2GitignorePath) {
  $trilha2Gitignore = @"
# Snapshot público (Trilha 2)
# - Inclui datasets pequenos (JSONL) e estatísticas (JSON) para auditoria/reprodução.
# - Mantém fora do controle de versão outputs/adapters e logs verbosos.
outputs/
logs/*.log
logs/*.out
__pycache__/
*.pyc
"@
  # Em alguns ambientes (shares/ACLs), o arquivo copiado pode vir sem permissão de escrita.
  # Remover e recriar evita falhas de "access denied".
  try { Remove-Item -Force $trilha2GitignorePath } catch { }
  Write-TextFileUtf8 -Path $trilha2GitignorePath -Content $trilha2Gitignore
}

# Gerar Markdown do TCC (a partir do LaTeX interno) para o snapshot público.
$markdownOut = Join-Path $dest "documento/markdown/tcc.md"
try {
  python (Join-Path $root "scripts/export_tcc_markdown.py") --output $markdownOut | Out-Null
  if (Test-Path $markdownOut) {
    $contentUtf8 = [System.IO.File]::ReadAllText($markdownOut, [System.Text.Encoding]::UTF8)
    Write-TextFileUtf8 -Path $markdownOut -Content $contentUtf8
  }
} catch {
  Write-Warning "Falha ao gerar Markdown do TCC automaticamente: $($_.Exception.Message)"
}

# README/REPRODUCIBILITY básicos (público)
$publicReadme = @'
# LLM PT-BR: treino do zero (v1–v6) vs QLoRA (CPT/SFT)

Este repositório reúne os artefatos do meu Trabalho de Conclusão de Curso (TCC) do **MBA em Ciência de Dados (ICMC/USP)**.

- **Título:** Um caminho estratégico para a soberania digital através de LLMs nacionais
- **Autor:** Cristovam Belizário Peres
- **Ano:** 2026

## O que foi investigado

Sob restrições realistas (GPU única), o trabalho compara duas estratégias para obter um modelo em PT-BR:

- **Trilha 1 — Treinar do zero (v1–v6):** protótipos em escala reduzida para evidenciar custos, limitações e descobertas (com foco em limpeza/tokenização e trade-offs de arquitetura).
- **Trilha 2 — Pós-treinamento eficiente (QLoRA):** (i) *continued pretraining* (CPT) em PT-BR (BrWaC 10k) e (ii) *supervised fine-tuning* (SFT) instrucional (Canarim 10k) sobre **LLaMA 3.1-8B**, com avaliação extrínseca (QA/sumarização/reescrita) repetida em múltiplas seeds.

## Documento

- PDF (versão oficial): [documento/pdf/tcc.pdf](documento/pdf/tcc.pdf) (exportado do Overleaf)
- Markdown (conversão best-effort): [documento/markdown/tcc.md](documento/markdown/tcc.md)

> Observação: o Markdown é uma conversão automática para facilitar leitura rápida e busca; o PDF é a referência.

## Onde estão as evidências

- [versions/](versions/) — versões/experimentos com `README.md`, logs e artefatos.
- [analysis/](analysis/) — métricas agregadas, guardrails, amostras e relatórios.
- [scripts/](scripts/) — scripts de preparo, treino e avaliação.
- [docs/](docs/) — notas e documentação auxiliar.

No PDF, as referências a artefatos e logs são links clicáveis que apontam para estes caminhos no repositório.

## Reprodutibilidade (alto nível)

Veja [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Em resumo:

- este snapshot contém logs, análises e scripts suficientes para auditoria e replicação dos números reportados;
- **não** inclui pesos/checkpoints nem segredos/credenciais;
- alguns arquivos grandes são omitidos (limite de {0} MB por arquivo); consulte [MANIFEST.md](MANIFEST.md).

## O que não está incluído

- O **fonte LaTeX** do TCC (o PDF é exportado do Overleaf).
- Pesos/checkpoints (TensorFlow/Keras e adapters LoRA/QLoRA).
- Arquivos temporários, credenciais e corpora grandes fora do limite de tamanho.
'@ -f $MaxFileMB
Write-TextFileUtf8 -Path (Join-Path $dest "README.md") -Content $publicReadme

$repro = @'
# Reprodutibilidade (visão geral)

## O que este snapshot contém

- Logs, análises e scripts suficientes para auditoria dos números reportados no texto.
- Datasets pequenos/splits usados em Trilha 2 quando couberem no limite de tamanho.

## O que este snapshot NÃO contém

- Pesos/checkpoints (TensorFlow/Keras, adapters LoRA/QLoRA, etc.).
- Segredos/credenciais.
- Corpora grandes exportados (ex.: `corpus_v6.txt`), quando excedem o limite de tamanho.

## Como reproduzir (alto nível)

1. Crie um ambiente Python e instale dependências:
   - `pip install -r requirements.txt`
2. Execute os scripts em `scripts/` e use os artefatos em `analysis/` e `versions/` como referência.

Detalhes por versão e trilha estão documentados nos `README.md` dentro de `versions/`.
'@
Write-TextFileUtf8 -Path (Join-Path $dest "REPRODUCIBILITY.md") -Content $repro

# .gitignore (público)
$publicGitignore = @"
__pycache__/
*.pyc
.venv/
.env
.DS_Store
"@
Write-TextFileUtf8 -Path (Join-Path $dest ".gitignore") -Content $publicGitignore

# MANIFEST (MD + JSON)
$timestamp = (Get-Date).ToString("s")
$commit = ""
try {
  $commit = (git -C $root rev-parse HEAD 2>$null)
} catch {
  $commit = ""
}

$copied = $result.copied
$skipped = $result.skipped

$copiedBytes = ($copied | Measure-Object -Property size_bytes -Sum).Sum
$skippedBytes = ($skipped | Measure-Object -Property size_bytes -Sum).Sum

$manifestJson = @{
  generated_at = $timestamp
  source_commit = $commit
  max_file_mb = $MaxFileMB
  summary = @{
    copied_files = $copied.Count
    copied_bytes = $copiedBytes
    skipped_files = $skipped.Count
    skipped_bytes = $skippedBytes
  }
  copied = $copied
  skipped = $skipped
}
$manifestJsonPath = Join-Path $dest "MANIFEST.json"
$manifestJson | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 $manifestJsonPath

$skippedBySize = $skipped | Where-Object { $_.reason -like "size>*" } | Sort-Object -Property size_bytes -Descending
$skippedByRule = $skipped | Where-Object { $_.reason -notlike "size>*" } | Sort-Object -Property reason, path

function Format-Bytes([int64]$Bytes) {
  if ($Bytes -ge 1GB) { return "{0:N2} GB" -f ($Bytes / 1GB) }
  if ($Bytes -ge 1MB) { return "{0:N2} MB" -f ($Bytes / 1MB) }
  if ($Bytes -ge 1KB) { return "{0:N2} KB" -f ($Bytes / 1KB) }
  return "$Bytes B"
}

$manifestMd = New-Object System.Text.StringBuilder
$mdBacktick = [char]96
$null = $manifestMd.AppendLine("# Manifest do snapshot público")
$null = $manifestMd.AppendLine()
$null = $manifestMd.AppendLine("- Gerado em: $timestamp")
if ($commit) { $null = $manifestMd.AppendLine(("- Commit fonte: " + $mdBacktick + $commit + $mdBacktick)) }
$null = $manifestMd.AppendLine("- Limite de tamanho por arquivo: **$MaxFileMB MB**")
$null = $manifestMd.AppendLine("- Copiados: **$($copied.Count)** arquivos (**$(Format-Bytes $copiedBytes)**)")
$null = $manifestMd.AppendLine("- Ignorados: **$($skipped.Count)** arquivos (**$(Format-Bytes $skippedBytes)**)")
$null = $manifestMd.AppendLine()
$null = $manifestMd.AppendLine(("Arquivo detalhado: " + $mdBacktick + "MANIFEST.json" + $mdBacktick + "."))
$null = $manifestMd.AppendLine()

$null = $manifestMd.AppendLine("## Ignorados por tamanho")
if ($skippedBySize.Count -eq 0) {
  $null = $manifestMd.AppendLine("- (nenhum)")
} else {
  $null = $manifestMd.AppendLine("| Caminho | Tamanho |")
  $null = $manifestMd.AppendLine("|---|---:|")
  foreach ($item in ($skippedBySize | Select-Object -First 50)) {
    $null = $manifestMd.AppendLine(("| " + $mdBacktick + $item.path + $mdBacktick + " | " + (Format-Bytes $item.size_bytes) + " |"))
  }
  if ($skippedBySize.Count -gt 50) {
    $null = $manifestMd.AppendLine()
    $null = $manifestMd.AppendLine(("> Lista completa em " + $mdBacktick + "MANIFEST.json" + $mdBacktick + "."))
  }
}
$null = $manifestMd.AppendLine()

$null = $manifestMd.AppendLine("## Ignorados por regra (denylist)")
if ($skippedByRule.Count -eq 0) {
  $null = $manifestMd.AppendLine("- (nenhum)")
} else {
  $null = $manifestMd.AppendLine("| Caminho | Motivo |")
  $null = $manifestMd.AppendLine("|---|---|")
  foreach ($item in ($skippedByRule | Select-Object -First 80)) {
    $null = $manifestMd.AppendLine(("| " + $mdBacktick + $item.path + $mdBacktick + " | " + $item.reason + " |"))
  }
  if ($skippedByRule.Count -gt 80) {
    $null = $manifestMd.AppendLine()
    $null = $manifestMd.AppendLine(("> Lista completa em " + $mdBacktick + "MANIFEST.json" + $mdBacktick + "."))
  }
}
$null = $manifestMd.AppendLine()

Write-TextFileUtf8 -Path (Join-Path $dest "MANIFEST.md") -Content $manifestMd.ToString()

Write-Host "[OK] Snapshot gerado em: $dest"
