"""
Tradutor de inglês para português usando Transformers
"""
import torch
import time
import sys
import subprocess
import os
from datetime import datetime

# Configurar encoding UTF-8 para o terminal Windows
if sys.platform == 'win32':
    try:
        # Tentar configurar UTF-8 no Windows
        os.system('chcp 65001 > nul')
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def obter_memoria_gpu_real():
    """
    Obtém a memória GPU real usando nvidia-smi (mais preciso).
    Retorna (memória_usada_GB, memória_total_GB).
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(',')
            return float(used) / 1024, float(total) / 1024
    except Exception:
        pass
    # Fallback para PyTorch se nvidia-smi falhar
    return (
        torch.cuda.memory_reserved(0) / 1024**3,
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )


def limpar_mixagem_idiomas(texto):
    """Remove palavras em outros idiomas que vazam do mBART."""
    import re
    
    # Primeiro, substituir palavras específicas com cirílico
    # (детективes -> detetives)
    texto = re.sub(r'\b[а-яА-ЯёЁ]+(?:es|os|as)?\b', 'detetives', texto)
    
    # Depois, substituições de espanhol/outros idiomas
    substituicoes = {
        r'\bEl\b': 'Ele',  # Espanhol
        r'\bLa\b': 'A',
        r'\bLos\b': 'Os',
        r'\bLas\b': 'As',
        r'\bUn\b': 'Um',
        r'\bUna\b': 'Uma',
        r'decoración': 'decoração',
        r'\bshelf\b': 'prateleira',
    }
    
    for padrao, substituto in substituicoes.items():
        texto = re.sub(padrao, substituto, texto)
    
    # Remover "es" duplicado que pode ter sobrado
    texto = re.sub(r'\b(detetives)es\b', r'\1', texto)
    
    return texto.strip()


def imprimir_barra_progresso(atual, total, largura=50, info_extra=""):
    """
    Imprime uma barra de progresso no terminal com fallback ASCII.
    """
    percentual = (atual / total) * 100
    blocos_preenchidos = int(largura * atual / total)
    blocos_vazios = largura - blocos_preenchidos
    
    # Usar caracteres mais seguros que funcionam em todos os terminais
    barra = "=" * blocos_preenchidos + " " * blocos_vazios
    
    print(
        f"\r[{barra}] {percentual:.1f}% ({atual}/{total}) {info_extra}",
        end="",
        flush=True
    )


def traduzir_arquivo(arquivo_entrada: str, arquivo_saida: str, max_tokens: int = 2048):
    """
    Traduz um arquivo de texto de inglês para português.
    
    Args:
        arquivo_entrada: Caminho do arquivo em inglês
        arquivo_saida: Caminho do arquivo de saída em português
        max_tokens: Número máximo de tokens por lote (padrão: 2048)
    """
    tempo_inicio = time.time()
    
    print("=" * 70)
    print("🌐 TRADUTOR INGLÊS → PORTUGUÊS")
    print("=" * 70)
    print(f"📅 Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    
    print("📥 Carregando modelo de tradução...")
    # Modelo mBART-50 (Meta) - melhor qualidade
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    print(f"   Modelo: {model_name}")
    print("   ⚠️  Primeira execução: Baixando modelo (~2.4GB)...")
    print("   Isso pode levar alguns minutos dependendo da conexão.\n")
    
    tempo_modelo_inicio = time.time()
    try:
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        
        tokenizer = MBart50TokenizerFast.from_pretrained(
            model_name,
            src_lang="en_XX",
            tgt_lang="pt_XX"
        )
        
        # Forçar tokenizer a usar português explicitamente
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "pt_XX"
        
        model = MBartForConditionalGeneration.from_pretrained(
            model_name
        )
        
        tempo_modelo = time.time() - tempo_modelo_inicio
        print(f"   ✓ Modelo carregado em {tempo_modelo:.2f}s")
    except Exception as e:
        print("\n❌ ERRO ao carregar o modelo:")
        print(f"   {str(e)}\n")
        print("💡 Soluções possíveis:")
        print("   1. Verifique sua conexão com a internet")
        print("   2. O modelo será baixado automaticamente na primeira vez")
        print("   3. Certifique-se de ter espaço em disco (~1GB)")
        print("   4. Tente: pip install --upgrade transformers")
        raise
    
    # Verificar e configurar GPU
    print("\n🔍 Verificando dispositivos disponíveis...")
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   🚀 GPU detectada: {gpu_name}")
        print(f"   💾 Memória GPU: {gpu_memory:.2f} GB")
        print("   ⚡ Usando CUDA para aceleração")
        print("   ℹ️  Nota: Memória GPU sobe/desce durante processamento")
        print("           (PyTorch aloca/libera cache dinamicamente)")
        
        # Otimizações para GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = model.to(device)
        model.eval()  # Modo de inferência
        
        # Limpar cache da GPU
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        print("   ⚠️  GPU não detectada")
        print("   💻 Usando CPU (será mais lento)")
        print("\n   💡 Para usar GPU:")
        print("      - Instale PyTorch com suporte CUDA")
        cuda_url = "https://download.pytorch.org/whl/cu121"
        print(f"      - Comando: pip install torch --index-url {cuda_url}")
        model = model.to(device)
    
    # Ler o arquivo
    print(f"📄 Lendo arquivo: {arquivo_entrada}")
    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
    
    linhas_nao_vazias = sum(1 for linha in linhas if linha.strip())
    print(f"   Total de linhas: {len(linhas)}")
    print(f"   Linhas com texto: {linhas_nao_vazias}")
    print(f"   Linhas vazias: {len(linhas) - linhas_nao_vazias}\n")
    
    # Traduzir em lotes dinâmicos baseados em tokens
    print("🔄 Iniciando tradução...")
    print(f"   Agrupando linhas dinamicamente (max {max_tokens} tokens)\n")
    
    linhas_traduzidas = 0
    tempo_traducao_inicio = time.time()
    
    # Criar lotes dinâmicos baseados em número de caracteres/tokens
    lotes = []
    lote_atual = []
    lote_indices = []
    tamanho_lote_atual = 0
    
    for i, linha in enumerate(linhas):
        linha_limpa = linha.strip()
        if linha_limpa:
            # Estimar tokens (~4 chars por token em média)
            tokens_estimados = len(linha_limpa) // 4
            
            # Se adicionar essa linha ultrapassar o limite, fechar lote atual
            if lote_atual and (tamanho_lote_atual + tokens_estimados > max_tokens):
                lotes.append((lote_atual[:], lote_indices[:]))
                lote_atual = [linha_limpa]
                lote_indices = [i]
                tamanho_lote_atual = tokens_estimados
            else:
                lote_atual.append(linha_limpa)
                lote_indices.append(i)
                tamanho_lote_atual += tokens_estimados
        else:
            # Linha vazia - adicionar ao lote como marcador
            if lote_atual:
                lotes.append((lote_atual[:], lote_indices[:]))
                lote_atual = []
                lote_indices = []
                tamanho_lote_atual = 0
    
    # Adicionar último lote se houver
    if lote_atual:
        lotes.append((lote_atual, lote_indices))
    
    total_batches = len(lotes)
    print(f"   📦 Total de lotes criados: {total_batches}")
    print(f"   📊 Média de linhas por lote: {len(linhas) / total_batches:.1f}\n")
    
    # Processar cada lote
    resultado_ordenado = {}
    ultimo_update = time.time()
    update_interval = 10  # Atualizar a cada 10 segundos no máximo
    
    # Mostrar barra de progresso inicial
    agora_inicial = datetime.now().strftime('%H:%M:%S')
    info_inicial = f"| Iniciando... | {agora_inicial}"
    imprimir_barra_progresso(
        0, linhas_nao_vazias, largura=40, info_extra=info_inicial
    )
    print()  # Nova linha para começar o progresso
    sys.stdout.flush()  # Forçar exibição imediata
    
    for batch_num, (linhas_para_traduzir, indices) in enumerate(lotes, 1):
        # Traduzir linhas não vazias
        if linhas_para_traduzir:
            inputs = tokenizer(
                linhas_para_traduzir,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            # Mover inputs para GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Gerar traduções (sem calcular gradientes para economizar memória)
            with torch.no_grad():
                # Usar mixed precision se GPU disponível
                if device == "cuda":
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        translated = model.generate(
                            **inputs,
                            forced_bos_token_id=tokenizer.lang_code_to_id["pt_XX"],
                            max_length=200,
                            num_beams=5,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                            length_penalty=1.0
                        )
                else:
                    translated = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.lang_code_to_id["pt_XX"],
                        max_length=200,
                        num_beams=5,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        length_penalty=1.0
                    )
            
            traducoes_batch = [
                limpar_mixagem_idiomas(
                    tokenizer.decode(t, skip_special_tokens=True)
                )
                for t in translated
            ]
            
            # Armazenar traduções com seus índices originais
            for idx_orig, traducao in zip(indices, traducoes_batch):
                resultado_ordenado[idx_orig] = traducao
            
            # Limpar memória GPU mais agressivamente
            if device == "cuda":
                # Deletar tensores explicitamente
                del inputs, translated
                # Limpar cache do CUDA
                torch.cuda.empty_cache()
                # Sincronizar para forçar limpeza
                torch.cuda.synchronize()
        
        linhas_traduzidas += len(linhas_para_traduzir)
        
        # Calcular estatísticas
        tempo_atual = time.time()
        tempo_total_ate_agora = tempo_atual - tempo_traducao_inicio
        velocidade = linhas_traduzidas / tempo_total_ate_agora
        linhas_restantes = linhas_nao_vazias - linhas_traduzidas
        tempo_estimado = linhas_restantes / velocidade if velocidade > 0 else 0
        
        # Atualizar barra a cada 10 segundos ou no último lote
        tempo_desde_update = tempo_atual - ultimo_update
        deve_atualizar = (
            tempo_desde_update >= update_interval or
            batch_num == total_batches
        )
        
        if deve_atualizar:
            # Informações de memória GPU (via nvidia-smi = mais preciso)
            memoria_info = ""
            if device == "cuda":
                mem_usada, mem_total = obter_memoria_gpu_real()
                memoria_info = (
                    f"| GPU: {mem_usada:.1f}GB/{mem_total:.1f}GB "
                )
            
            # Timestamp
            agora = datetime.now().strftime('%H:%M:%S')
            
            # Exibir progresso
            info_extra = (
                f"| {batch_num}/{total_batches} "
                f"{memoria_info}"
                f"| {velocidade:.1f} l/s "
                f"| ETA: {tempo_estimado:.0f}s "
                f"| {agora}"
            )
            imprimir_barra_progresso(
                linhas_traduzidas,
                linhas_nao_vazias,
                largura=40,
                info_extra=info_extra
            )
            ultimo_update = tempo_atual
    
    # Nova linha após a barra de progresso
    print("\n")
    
    # Reconstruir arquivo na ordem original
    print("📝 Reconstruindo arquivo traduzido...")
    traducoes_finais = []
    for i, linha in enumerate(linhas):
        if i in resultado_ordenado:
            traducoes_finais.append(resultado_ordenado[i] + '\n')
        else:
            traducoes_finais.append('\n')  # Linha vazia
    
    # Salvar o resultado
    print("💾 Salvando tradução...")
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        f.writelines(traducoes_finais)
    
    # Estatísticas finais
    tempo_total = time.time() - tempo_inicio
    tempo_trad = time.time() - tempo_traducao_inicio
    velocidade_media = linhas_nao_vazias / tempo_trad
    
    print("✅ Tradução concluída!")
    print("\n📊 ESTATÍSTICAS:")
    print(f"   ⏱️  Tempo total: {tempo_total:.2f}s")
    print(f"   📝 Linhas traduzidas: {linhas_nao_vazias}")
    print(f"   ⚡ Velocidade média: {velocidade_media:.2f} linhas/s")
    
    # Estatísticas de GPU se disponível
    if device == "cuda":
        mem_usada, _ = obter_memoria_gpu_real()
        print(f"   🎮 Memória GPU usada: {mem_usada:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    print(f"   💾 Arquivo salvo: {arquivo_saida}")
    print(f"\n📅 Término: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    # Configuração padrão
    arquivo_entrada = "texto.txt"
    arquivo_saida = "texto_traduzido.txt"
    max_tokens = 2048  # Padrão
    
    # Permite passar argumentos pela linha de comando
    if len(sys.argv) > 1:
        arquivo_entrada = sys.argv[1]
    if len(sys.argv) > 2:
        arquivo_saida = sys.argv[2]
    if len(sys.argv) > 3:
        max_tokens = int(sys.argv[3])
    
    # Ajustar max_tokens baseado na GPU disponível
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory > 11:  # GPU com ~12GB (RTX 4070, etc)
            max_tokens = 1024  # Base para cálculo
        elif gpu_memory >= 10:
            max_tokens = 896
        elif gpu_memory >= 8:
            max_tokens = 768
        elif gpu_memory >= 6:
            max_tokens = 640
        else:
            max_tokens = 512

        print(f"🎮 GPU com {gpu_memory:.1f}GB detectada")
        print(f"📦 Configuração otimizada: {max_tokens} tokens/lote\n")
    
    traduzir_arquivo(arquivo_entrada, arquivo_saida, max_tokens=max_tokens)
