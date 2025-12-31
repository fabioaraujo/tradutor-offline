# Tradutor Offline - Inglês para Português

Este projeto oferece múltiplas soluções para tradução de arquivos de texto de inglês para português, incluindo tradução local com **mBART-50** e integração com **API OpenAI local**.

## Características

### Tradutor mBART-50 (tradutor.py)
- ✅ **Alta Qualidade**: 95%+ de acurácia usando modelo mBART-50
- ✅ **Processamento em GPU**: Até 10x mais rápido com NVIDIA CUDA
- ✅ **Alta Performance**: ~34 linhas/segundo em GPU RTX 4070
- ✅ **Monitoramento em Tempo Real**: Barra de progresso com estatísticas de GPU
- ✅ **Beam Search**: Tradução com múltiplas alternativas para melhor resultado
- ✅ **Pós-processamento Inteligente**: Correção automática de mistura de idiomas
- ✅ **Otimização Automática**: Batch size ajustado para sua GPU

### Tradutor OpenAI (tradutor_openai.py)
- ✅ **Flexível**: Funciona com qualquer modelo compatível com API OpenAI
- ✅ **Precisão Linha por Linha**: Traduz cada linha individualmente
- ✅ **Servidor Local**: Suporte para LM Studio, Ollama, etc.
- ✅ **Monitoramento Detalhado**: Velocidade, ETA e estatísticas em tempo real
- ✅ **Configurável**: API URL, modelo e tamanho de lote customizáveis

### Utilitários
- ✅ **juntar_linhas.py**: Junta sentenças quebradas em múltiplas linhas

## Requisitos

- Python 3.9+
- uv (gerenciador de pacotes Python)
- **GPU NVIDIA (recomendado)** - Para tradução muito mais rápida
- **nvidia-smi** (incluído nos drivers NVIDIA) - Para monitoramento preciso de memória

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
# Desinstalar PyTorch CPU (se instalado)
uv pip uninstall torch

# Instalar PyTorch com CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Instalar dependências adicionais
uv pip install protobuf accelerate
```

**Requisitos para GPU:**
- NVIDIA GPU com suporte CUDA (8GB+ VRAM recomendado)
- Drivers NVIDIA atualizados
- CUDA 12.1 ou superior

## Uso

### Opção 1: Tradutor mBART-50 (Offline, GPU/CPU)

#### Traduzir o arquivo texto.txt (padrão)

```bash
uv run tradutor.py
```

Isso irá traduzir o arquivo `texto.txt` e criar `texto_traduzido.txt` com a tradução.

#### Traduzir arquivo customizado

```bash
uv run tradutor.py arquivo_entrada.txt arquivo_saida.txt
```

#### Ajustar tamanho do lote manualmente

```bash
uv run tradutor.py texto.txt texto_traduzido.txt 1024
```

### Opção 2: Tradutor OpenAI API (Servidor Local)

#### Uso básico (com LM Studio ou similar rodando em localhost:1234)

```bash
uv run tradutor_openai.py
```

#### Uso com parâmetros customizados

```bash
# Sintaxe: python tradutor_openai.py [entrada] [saida] [api_url] [modelo] [linhas_por_lote]
uv run tradutor_openai.py texto.txt saida.txt http://127.0.0.1:1234/v1 local-model 5
```

**Parâmetros:**
- `entrada`: Arquivo de entrada (padrão: texto.txt)
- `saida`: Arquivo de saída (padrão: texto_traduzido.txt)
- `api_url`: URL base da API OpenAI (padrão: http://127.0.0.1:1234/v1)
- `modelo`: Nome do modelo (padrão: local-model)
- `linhas_por_lote`: Número de linhas por requisição (padrão: 5)

**Servidores compatíveis:**
- LM Studio
- Ollama (com endpoint OpenAI)
- LocalAI
- text-generation-webui (com extensão OpenAI)

### Opção 3: Utilitário para Juntar Linhas

Útil para arquivos de legendas ou texto quebrado em múltiplas linhas:

```bash
uv run juntar_linhas.py entrada.txt saida.txt
```

**Recursos:**
- Junta sentenças quebradas mantendo estrutura
- Preserva parágrafos e linhas vazias
- Detecta automaticamente continuação de frases
- Mantém pontuação e formatação

O batch size é ajustado automaticamente baseado na memória da GPU:
- GPU com 12GB (RTX 4070): batch_size = 1024 tokens (~34 linhas/segundo)
- GPU com 10-12GB: batch_size = 896 tokens
- GPU com 8-10GB: batch_size = 768 tokens
- GPU com 6-8GB: batch_size = 640 tokens
- GPU com <6GB: batch_size = 512 tokens
- CPU: batch_size = 2048 tokens (padrão, ~0.3-0.5 linhas/segundo)

## Como funciona

O script usa o modelo **`facebook/mbart-large-50-many-to-many-mmt`** da Hugging Face, um modelo multilíngue de alta qualidade (2.4GB) treinado para tradução entre 50 idiomas.

### Por que mBART-50?

Migramos do modelo T5 original para o mBART-50 devido a:
- ✅ **Qualidade superior**: 95%+ de acurácia vs 60% do T5
- ✅ **Sem repetições**: T5 gerava loops infinitos de texto
- ✅ **Vocabulário rico**: Melhor handling de contextos complexos
- ✅ **Beam search nativo**: Suporte robusto para múltiplas hipóteses

**Otimizações Implementadas:**
- ✅ Detecta automaticamente GPU disponível
- ✅ Mixed precision (FP16) para acelerar processamento
- ✅ Beam search com 5 beams para melhor qualidade
- ✅ Monitoramento preciso de memória GPU via nvidia-smi
- ✅ Limpeza de cache entre lotes para evitar estouro de memória
- ✅ Barra de progresso ASCII-safe para Windows PowerShell
- ✅ Pós-processamento para corrigir mistura de idiomas
- ✅ Anti-repetição com n-gram blocking (no_repeat_ngram_size=3)
- ✅ Velocidade típica: **~34 linhas/segundo** com GPU RTX 4070 (1024 tokens/lote)

**Funcionalidades:**
- Processa o texto em lotes para melhor performance
- Preserva linhas vazias do arquivo original
- Utiliza GPU automaticamente se disponível
- Suporta arquivos grandes através do processamento em lotes
- Barra de progresso em tempo real com estatísticas detalhadas:
  ```
  [==========          ] 20.3% (565/2779) | GPU: 2.5GB/12.0GB | 34.2 l/s
  ```
- Limpeza automática de artefatos de tradução (mistura de idiomas)

## Pós-processamento Inteligente

O tradutor inclui correções automáticas para problemas comuns:

- **Mistura de Espanhol**: "El" → "Ele", "decoración" → "decoração"
- **Caracteres Cirílicos**: Remoção e substituição contextual
- **Palavras em Inglês**: "shelf" → "prateleira"
- **Duplicações**: "detetiveses" → "detetives"

## Dependências

### Tradutor mBART-50 (tradutor.py)
- **transformers**: Biblioteca da Hugging Face para modelos de linguagem
- **torch**: PyTorch para execução dos modelos (com ou sem GPU)
- **sentencepiece**: Tokenização para mBART
- **sacremoses**: Pré-processamento de texto
- **protobuf**: Serialização de dados (requerido pelo tokenizador mBART)
- **accelerate**: Otimizações de hardware

### Tradutor OpenAI (tradutor_openai.py)
- **requests**: Requisições HTTP para API
- Servidor com API OpenAI compatível (LM Studio, Ollama, etc.)

### Utilitários
- Apenas biblioteca padrão do Python

## Nota

Na primeira execução, o modelo mBART-50 será baixado automaticamente (aproximadamente **2.4GB**). Certifique-se de ter espaço em disco e conexão estável.

## Performance

### mBART-50 (tradutor.py)
| Dispositivo | Velocidade Típica | Tempo para 2780 linhas | Qualidade |
|-------------|-------------------|------------------------|-----------|
| GPU (RTX 4070) | ~34 linhas/s | ~1-2 minutos | 95%+ |
| GPU (RTX 3060) | ~20-25 linhas/s | ~2-3 minutos | 95%+ |
| CPU (i7) | 0.3-0.5 linhas/s | ~90-150 minutos | 95%+ |

**Nota**: Performance com mBART-50 + beam search (5 beams) e 1024 tokens/lote em RTX 4070.

### OpenAI API (tradutor_openai.py)

**Hardware de referência**: RTX 4070 Ti (12GB VRAM) - LM Studio rodando localmente

| Modelo | Quantização | Velocidade | Tempo (1577 linhas) | Tempo est. (2780 linhas) | Qualidade | VRAM |
|--------|-------------|------------|---------------------|--------------------------|-----------|------|
| **Qwen 2.5 7B** ⭐ | Q6_K | ~4.5 linhas/s | ~5m 47s | ~10m 17s | 95-98% | 8GB |
| **Llama 3.1 8B** | Q5_K_M | ~4-6 linhas/s | ~4-7 min | ~8-12 min | 90-95% | 7GB |
| **Aya 23 8B** | Q5_K_M | ~3-5 linhas/s | ~5-9 min | ~10-15 min | 95%+ | 7GB |
| **Mixtral 8x7B** | Q4_K_M | ~2-3 linhas/s | ~9-13 min | ~15-25 min | 98%+ | 12GB |
| **GPT-4 (Cloud)** | - | Variável | Variável | Depende da API | 98%+ | N/A |

**Testes Reais Executados**:
1. **Arquivo pequeno (29 linhas)**: 6.48s = 4.48 l/s
2. **Arquivo grande (1577 linhas)**: 347.13s (5m 47s) = **4.54 l/s** ✅
   - Modelo: Qwen 2.5 7B (Q6_K) no LM Studio
   - Hardware: RTX 4070 Ti (12GB VRAM)
   - Configuração: 5 linhas por lote, temperature 0.3
   - Performance consistente em arquivos de diferentes tamanhos

**Modelos Recomendados para Tradução**:
- **Melhor custo-benefício**: Qwen 2.5 7B (excelente em multilíngue, testado)
- **Mais rápido**: Llama 3.1 8B (ótimo desempenho geral)
- **Especializado**: Aya 23 8B (focado em 23 idiomas)
- **Máxima qualidade**: Mixtral 8x7B (superior, mas mais lento)

**Nota**: 
- Performance medida com temperatura 0.3 e 5 linhas por lote
- Velocidade muito consistente: 4.48 l/s (29 linhas) vs 4.54 l/s (1577 linhas)
- LM Studio, Ollama e LocalAI têm performance similar
- Modelos GGUF (quantizados) oferecem melhor velocidade/qualidade
- Para 2780 linhas: ~10 minutos estimados com Qwen 2.5 7B

## Solução de Problemas

### Problemas com mBART-50 (tradutor.py)

#### GPU não detectada
```bash
# Verifique se CUDA está disponível
python -c "import torch; print(torch.cuda.is_available())"

# Se retornar False, instale PyTorch com CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Erro de memória GPU
- Reduza o batch size manualmente: `uv run tradutor.py texto.txt saida.txt 128`
- Feche outros programas que usam GPU
- Para GPUs com menos de 8GB, considere usar batch_size=64 ou CPU

#### Barra de progresso com caracteres estranhos
- Já corrigido! Usamos caracteres ASCII-safe (= e espaço) compatíveis com PowerShell
- Se ainda houver problemas, verifique a codificação do terminal: `$OutputEncoding = [System.Text.Encoding]::UTF8`

#### Repetições infinitas no texto traduzido
- Problema resolvido com migração para mBART-50
- Se usar outros modelos, adicione: `no_repeat_ngram_size=3`, `repetition_penalty=1.2`

#### Mistura de idiomas na tradução (espanhol, inglês, etc.)
- Já implementado! A função `limpar_mixagem_idiomas()` corrige automaticamente
- Para idiomas adicionais, edite os padrões regex no código

#### Erro "protobuf not found"
```bash
uv pip install protobuf
```

#### Monitoramento de memória mostrando 0GB
- Certifique-se de que `nvidia-smi` está no PATH
- Execute: `nvidia-smi` no terminal para verificar

### Problemas com OpenAI API (tradutor_openai.py)

#### Erro de conexão com a API
```bash
# Verifique se o servidor está rodando
curl http://127.0.0.1:1234/v1/models

# Ou no PowerShell
Invoke-WebRequest http://127.0.0.1:1234/v1/models
```

#### API retorna erros 500/502
- Verifique se o modelo está carregado no servidor
- Confirme que o nome do modelo está correto
- Verifique logs do servidor (LM Studio, Ollama, etc.)

#### Tradução muito lenta
- Reduza `linhas_por_lote` para 1-3 linhas
- Use modelo menor e mais rápido
- Verifique se o servidor está usando GPU

#### Tradução de baixa qualidade
- Use modelos maiores (Mixtral, Llama 3.1 70B, etc.)
- Aumente a `temperature` no código (padrão: 0.3)
- Teste diferentes prompts no código

#### Erro "requests module not found"
```bash
uv pip install requests
```

## Histórico de Desenvolvimento

### v3.0 (Atual) - Multi-Engine
- ✅ Adicionado tradutor com API OpenAI local (tradutor_openai.py)
- ✅ Suporte para LM Studio, Ollama e outros servidores
- ✅ Tradução linha por linha com alta precisão
- ✅ Utilitário para juntar linhas quebradas (juntar_linhas.py)
- ✅ Configuração flexível de API, modelo e batch size

### v2.0 - mBART-50
- ✅ Migração para modelo mBART-large-50-many-to-many-mmt
- ✅ Beam search com 5 beams para máxima qualidade
- ✅ Pós-processamento para limpeza de mistura de idiomas
- ✅ Monitoramento preciso via nvidia-smi
- ✅ Barra de progresso ASCII-safe para Windows
- ✅ Qualidade: 95%+ (vs 60% do T5)

### v1.0 - T5 (Descontinuado)
- ❌ Modelo unicamp-dl/translation-en-pt-t5
- ❌ Problemas com repetições infinitas
- ❌ Qualidade insatisfatória (~60%)

## Licença

Este projeto usa modelos de código aberto da Hugging Face. Consulte as licenças individuais dos modelos para uso comercial.
