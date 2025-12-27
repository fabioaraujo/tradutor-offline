"""
Tradutor de inglês para português usando Transformers
"""
import torch
import time
from datetime import datetime


def imprimir_barra_progresso(atual, total, largura=50, info_extra=""):
    """
    Imprime uma barra de progresso no terminal.
    """
    percentual = (atual / total) * 100
    blocos_preenchidos = int(largura * atual / total)
    barra = "█" * blocos_preenchidos + "░" * (largura - blocos_preenchidos)
    print(f"\r[{barra}] {percentual:.1f}% ({atual}/{total}) {info_extra}", end="", flush=True)


def traduzir_arquivo(arquivo_entrada: str, arquivo_saida: str, batch_size: int = 8):
    """
    Traduz um arquivo de texto de inglês para português.
    
    Args:
        arquivo_entrada: Caminho do arquivo em inglês
        arquivo_saida: Caminho do arquivo de saída em português
        batch_size: Tamanho do lote para processamento
    """
    tempo_inicio = time.time()
    
    print("=" * 70)
    print("🌐 TRADUTOR INGLÊS → PORTUGUÊS")
    print("=" * 70)
    print(f"📅 Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    
    print("📥 Carregando modelo de tradução...")
    # Modelo específico para tradução inglês -> português
    # Usando o modelo correto do Hugging Face
    model_name = "unicamp-dl/translation-en-pt-t5"
    print(f"   Modelo: {model_name}")
    print("   ⚠️  Primeira execução: Baixando modelo (~900MB)...")
    print("   Isso pode levar alguns minutos dependendo da conexão.\n")
    
    tempo_modelo_inicio = time.time()
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        # Usar safetensors para evitar problema de segurança do torch.load
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=True
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
        
        # Otimizações para GPU
        torch.backends.cudnn.benchmark = True
        model = model.to(device)
        
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
    
    # Traduzir em lotes
    print("🔄 Iniciando tradução...")
    print(f"   Tamanho do lote: {batch_size} linhas por vez\n")
    
    traducoes = []
    total_batches = (len(linhas) + batch_size - 1) // batch_size
    linhas_traduzidas = 0
    tempo_traducao_inicio = time.time()
    
    for i in range(0, len(linhas), batch_size):
        batch = linhas[i:i + batch_size]
        batch_num = i // batch_size + 1
        tempo_batch_inicio = time.time()
        
        # Filtrar linhas vazias para tradução
        linhas_para_traduzir = []
        indices_vazios = []
        
        for idx, linha in enumerate(batch):
            linha_limpa = linha.strip()
            if linha_limpa:
                linhas_para_traduzir.append(linha_limpa)
            else:
                indices_vazios.append(idx)
        
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
                    with torch.amp.autocast('cuda'):
                        translated = model.generate(
                            **inputs,
                            max_length=512,
                            num_beams=4,
                            early_stopping=True
                        )
                else:
                    translated = model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True
                    )
            
            traducoes_batch = [
                tokenizer.decode(t, skip_special_tokens=True)
                for t in translated
            ]
            
            # Limpar cache da GPU após cada lote
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            traducoes_batch = []
        
        # Reconstruir o lote com linhas vazias nos lugares corretos
        resultado_batch = []
        traducao_idx = 0
        
        for idx in range(len(batch)):
            if idx in indices_vazios:
                resultado_batch.append('\n')
            else:
                resultado_batch.append(traducoes_batch[traducao_idx] + '\n')
                traducao_idx += 1
        
        traducoes.extend(resultado_batch)
        linhas_traduzidas += len(linhas_para_traduzir)
        
        # Calcular estatísticas
        tempo_batch = time.time() - tempo_batch_inicio
        tempo_total_ate_agora = time.time() - tempo_traducao_inicio
        velocidade = linhas_traduzidas / tempo_total_ate_agora
        linhas_restantes = linhas_nao_vazias - linhas_traduzidas
        tempo_estimado = linhas_restantes / velocidade if velocidade > 0 else 0
        
        # Exibir progresso
        info_extra = (
            f"| Lote {batch_num}/{total_batches} "
            f"| {velocidade:.1f} linhas/s "
            f"| ETA: {tempo_estimado:.0f}s"
        )
        imprimir_barra_progresso(
            linhas_traduzidas,
            linhas_nao_vazias,
            largura=40,
            info_extra=info_extra
        )
    
    # Nova linha após a barra de progresso
    print("\n")
    
    # Salvar o resultado
    print("💾 Salvando tradução...")
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        f.writelines(traducoes)
    
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
        gpu_memory_used = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"   🎮 Memória GPU máxima usada: {gpu_memory_used:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    print(f"   💾 Arquivo salvo: {arquivo_saida}")
    print(f"\n📅 Término: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    # Configuração padrão
    arquivo_entrada = "texto.txt"
    arquivo_saida = "texto_traduzido.txt"
    
    # Permite passar argumentos pela linha de comando
    if len(sys.argv) > 1:
        arquivo_entrada = sys.argv[1]
    if len(sys.argv) > 2:
        arquivo_saida = sys.argv[2]
    
    traduzir_arquivo(arquivo_entrada, arquivo_saida)
