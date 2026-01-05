# Scripts de infraestrutura

Arquivos desta pasta auxiliam no provisionamento e diagnóstico de máquinas remotas
usadas nos experimentos (ex.: verificar GPU, instalar dependências ou inspecionar
downloads do HuggingFace). Eles **não** fazem parte do pipeline principal e podem
ser executados conforme necessidade manual.

Scripts principais:

- `remote_install.sh`: instala dependências no ambiente remoto (pip + requirements).
- `install_tf_cuda.sh`: garante a instalação do pacote `tensorflow[and-cuda]`.
- `check_tf_gpu.sh`, `test_lstm_gpu.sh`: verificações rápidas de TensorFlow/GPU.
- `check_download.sh`, `find_json.sh`, `list_files.sh`: utilitários para inspecionar arquivos/logs.

Como rodar:

```bash
bash scripts/infra/remote_install.sh
```

Sinta-se à vontade para remover ou adaptar esses arquivos conforme a automação
da sua infraestrutura evoluir.
