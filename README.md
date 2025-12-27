# Tradutor Offline - Inglês para Português

Este projeto traduz arquivos de texto de inglês para português usando a biblioteca Transformers da Hugging Face e modelos pré-treinados.

## Requisitos

- Python 3.9+
- uv (gerenciador de pacotes Python)
- **GPU NVIDIA (recomendado)** - Para tradução muito mais rápida

## Instalação

1. Instale o uv (se ainda não tiver):
```bash
pip install uv
```

2. Instale as dependências do projeto:
```bash
uv sync
```

### Instalação com suporte GPU (Recomendado)

Para aproveitar a aceleração por GPU (até 10x mais rápido):

```bash
# Desinstalar PyTorch CPU
uv pip uninstall torch

# Instalar PyTorch com CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Requisitos para GPU:**
- NVIDIA GPU com suporte CUDA
- Drivers NVIDIA atualizados
- CUDA 12.1 ou superior

## Uso

### Traduzir o arquivo texto.txt (padrão)

```bash
uv run tradutor.py
```

Isso irá traduzir o arquivo `texto.txt` e criar `texto_traduzido.txt` com a tradução.

### Traduzir arquivo customizado

```bash
uv run tradutor.py arquivo_entrada.txt arquivo_saida.txt
```

## Como funciona

O script usa o modelo `unicamp-dl/translation-en-pt-t5` da Hugging Face, que é um modelo T5 treinado especificamente para inglês → português.

**Otimizações para GPU:**
- Detecta automaticamente se GPU está disponível
- Usa mixed precision (AMP) para acelerar processamento
- Limpa cache da GPU entre lotes para evitar estouro de memória
- Exibe estatísticas de uso de memória GPU
- Velocidade típica: **50-200 linhas/segundo** com GPU vs **5-20 linhas/segundo** com CPU

**Funcionalidades:**
- Processa o texto em lotes para melhor performance
- Preserva linhas vazias do arquivo original
- Utiliza GPU automaticamente se disponível
- Suporta arquivos grandes através do processamento em lotes
- Barra de progresso em tempo real
- Estatísticas detalhadas de performance

## Dependências

- **transformers**: Biblioteca da Hugging Face para modelos de linguagem
- **torch**: PyTorch para execução dos modelos (com ou sem GPU)
- **sentencepiece**: Tokenização
- **sacremoses**: Pré-processamento de texto

## Nota

Na primeira execução, o modelo será baixado automaticamente (aproximadamente 900MB).

## Performance

| Dispositivo | Velocidade Típica | Tempo para 2779 linhas |
|-------------|-------------------|------------------------|
| GPU (RTX 3060) | 100-150 linhas/s | ~20-30 segundos |
| GPU (RTX 4090) | 200-300 linhas/s | ~10-15 segundos |
| CPU (i7) | 5-10 linhas/s | ~5-10 minutos |

## Solução de Problemas

### GPU não detectada
```bash
# Verifique se CUDA está disponível
python -c "import torch; print(torch.cuda.is_available())"

# Se retornar False, instale PyTorch com CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Erro de memória GPU
- Reduza o tamanho do lote: edite `batch_size=4` no código
- Feche outros programas que usam GPU
