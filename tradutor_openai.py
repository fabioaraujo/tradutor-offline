"""
Tradutor de inglês para português usando API da OpenAI local
"""
import time
import sys
import os
from datetime import datetime
import requests
import json

# Configurar encoding UTF-8 para o terminal Windows
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 > nul')
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def imprimir_barra_progresso(atual, total, largura=50, info_extra=""):
    """
    Imprime uma barra de progresso no terminal com fallback ASCII.
    """
    percentual = (atual / total) * 100
    blocos_preenchidos = int(largura * atual / total)
    blocos_vazios = largura - blocos_preenchidos
    
    barra = "=" * blocos_preenchidos + " " * blocos_vazios
    
    print(
        f"\r[{barra}] {percentual:.1f}% ({atual}/{total}) {info_extra}",
        end="",
        flush=True
    )


def traduzir_com_openai(texto, api_url="http://127.0.0.1:1234/v1", modelo="local-model", is_single_line=False):
    """
    Traduz um texto do inglês para o português usando a API da OpenAI.
    
    Args:
        texto: Texto em inglês para traduzir
        api_url: URL base da API da OpenAI
        modelo: Nome do modelo a ser usado
        is_single_line: Se True, trata como linha única
        
    Returns:
        Texto traduzido em português
    """
    url = f"{api_url}/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if is_single_line:
        system_prompt = "Você é um tradutor profissional. Traduza APENAS o texto fornecido do inglês para o português brasileiro. Retorne SOMENTE a tradução, sem adicionar nenhum texto extra, explicação ou comentário."
        user_prompt = texto
    else:
        system_prompt = "Você é um tradutor profissional. Traduza o texto fornecido do inglês para o português brasileiro, mantendo EXATAMENTE a mesma estrutura de linhas. Cada linha em inglês deve corresponder a UMA linha em português. Não adicione ou remova linhas. Não adicione explicações."
        user_prompt = f"Traduza este texto linha por linha, mantendo a mesma quantidade de linhas:\n\n{texto}"
    
    payload = {
        "model": modelo,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": -1,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        traducao = data['choices'][0]['message']['content'].strip()
        
        return traducao
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Erro na requisição à API: {e}")
        raise
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"\n❌ Erro ao processar resposta da API: {e}")
        raise


def traduzir_arquivo(arquivo_entrada: str, arquivo_saida: str, 
                     api_url: str = "http://127.0.0.1:1234/v1",
                     modelo: str = "local-model",
                     linhas_por_lote: int = 5):
    """
    Traduz um arquivo de texto de inglês para português usando API da OpenAI.
    
    Args:
        arquivo_entrada: Caminho do arquivo em inglês
        arquivo_saida: Caminho do arquivo de saída em português
        api_url: URL base da API da OpenAI
        modelo: Nome do modelo a ser usado
        linhas_por_lote: Número de linhas a traduzir por requisição (padrão: 5 para maior precisão)
    """
    tempo_inicio = time.time()
    
    print("=" * 70)
    print("🌐 TRADUTOR INGLÊS → PORTUGUÊS (OpenAI API)")
    print("=" * 70)
    print(f"📅 Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    
    print(f"🔌 Conectando à API: {api_url}")
    print(f"🤖 Modelo: {modelo}\n")
    
    # Testar conexão com a API
    try:
        print("🔍 Testando conexão com a API...")
        response = requests.get(f"{api_url}/models", timeout=5)
        if response.status_code == 200:
            print("   ✓ API respondendo corretamente\n")
        else:
            print(f"   ⚠️  API retornou status {response.status_code}\n")
    except Exception as e:
        print(f"   ⚠️  Não foi possível verificar a API: {e}")
        print("   Continuando mesmo assim...\n")
    
    # Ler o arquivo
    print(f"📄 Lendo arquivo: {arquivo_entrada}")
    try:
        with open(arquivo_entrada, 'r', encoding='utf-8') as f:
            linhas = f.readlines()
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {arquivo_entrada}")
        return
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return
    
    linhas_nao_vazias = sum(1 for linha in linhas if linha.strip())
    print(f"   Total de linhas: {len(linhas)}")
    print(f"   Linhas com texto: {linhas_nao_vazias}")
    print(f"   Linhas vazias: {len(linhas) - linhas_nao_vazias}\n")
    
    # Traduzir em lotes
    print("🔄 Iniciando tradução...")
    print(f"   Processando em lotes de {linhas_por_lote} linhas\n")
    
    linhas_traduzidas = 0
    tempo_traducao_inicio = time.time()
    
    # Criar lotes
    lotes = []
    lote_atual = []
    lote_indices = []
    
    for i, linha in enumerate(linhas):
        linha_limpa = linha.strip()
        if linha_limpa:
            lote_atual.append(linha_limpa)
            lote_indices.append(i)
            
            # Se o lote atingiu o tamanho máximo, adicionar à lista
            if len(lote_atual) >= linhas_por_lote:
                lotes.append((lote_atual[:], lote_indices[:]))
                lote_atual = []
                lote_indices = []
        else:
            # Linha vazia - adicionar ao lote como marcador
            if lote_atual:
                lotes.append((lote_atual[:], lote_indices[:]))
                lote_atual = []
                lote_indices = []
    
    # Adicionar último lote se houver
    if lote_atual:
        lotes.append((lote_atual, lote_indices))
    
    total_batches = len(lotes)
    print(f"   📦 Total de lotes criados: {total_batches}")
    if total_batches > 0:
        print(f"   📊 Média de linhas por lote: {linhas_nao_vazias / total_batches:.1f}\n")
    
    # Processar cada lote
    resultado_ordenado = {}
    ultimo_update = time.time()
    update_interval = 1  # Atualizar a cada segundo
    
    # Mostrar barra de progresso inicial
    agora_inicial = datetime.now().strftime('%H:%M:%S')
    info_inicial = f"| Iniciando... | {agora_inicial}"
    imprimir_barra_progresso(
        0, linhas_nao_vazias, largura=40, info_extra=info_inicial
    )
    print()
    sys.stdout.flush()
    
    for batch_num, (linhas_para_traduzir, indices) in enumerate(lotes, 1):
        if linhas_para_traduzir:
            # Traduzir linha por linha para garantir precisão
            traducoes_batch = []
            
            for linha in linhas_para_traduzir:
                try:
                    # Traduzir cada linha individualmente
                    texto_traduzido = traduzir_com_openai(linha, api_url, modelo, is_single_line=True)
                    traducoes_batch.append(texto_traduzido)
                except Exception as e:
                    print(f"\n⚠️  Erro ao traduzir linha, mantendo original: {e}")
                    traducoes_batch.append(linha)
            
            # Armazenar traduções com seus índices originais
            for idx_orig, traducao in zip(indices, traducoes_batch):
                resultado_ordenado[idx_orig] = traducao
        
        linhas_traduzidas += len(linhas_para_traduzir)
        
        # Calcular estatísticas
        tempo_atual = time.time()
        tempo_total_ate_agora = tempo_atual - tempo_traducao_inicio
        velocidade = linhas_traduzidas / tempo_total_ate_agora if tempo_total_ate_agora > 0 else 0
        linhas_restantes = linhas_nao_vazias - linhas_traduzidas
        tempo_estimado = linhas_restantes / velocidade if velocidade > 0 else 0
        
        # Atualizar barra
        tempo_desde_update = tempo_atual - ultimo_update
        deve_atualizar = (
            tempo_desde_update >= update_interval or
            batch_num == total_batches
        )
        
        if deve_atualizar:
            agora = datetime.now().strftime('%H:%M:%S')
            info_extra = (
                f"| {batch_num}/{total_batches} "
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
    try:
        with open(arquivo_saida, 'w', encoding='utf-8') as f:
            f.writelines(traducoes_finais)
    except Exception as e:
        print(f"❌ Erro ao salvar arquivo: {e}")
        return
    
    # Estatísticas finais
    tempo_total = time.time() - tempo_inicio
    tempo_trad = time.time() - tempo_traducao_inicio
    velocidade_media = linhas_nao_vazias / tempo_trad if tempo_trad > 0 else 0
    
    print("✅ Tradução concluída!")
    print("\n📊 ESTATÍSTICAS:")
    print(f"   ⏱️  Tempo total: {tempo_total:.2f}s")
    print(f"   📝 Linhas traduzidas: {linhas_nao_vazias}")
    print(f"   ⚡ Velocidade média: {velocidade_media:.2f} linhas/s")
    print(f"   💾 Arquivo salvo: {arquivo_saida}")
    print(f"\n📅 Término: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    # Configuração padrão
    arquivo_entrada = "texto.txt"
    arquivo_saida = "texto_traduzido.txt"
    api_url = "http://127.0.0.1:1234/v1"
    modelo = "local-model"
    linhas_por_lote = 5  # Reduzido para 5 para melhor precisão
    
    # Permite passar argumentos pela linha de comando
    if len(sys.argv) > 1:
        arquivo_entrada = sys.argv[1]
    if len(sys.argv) > 2:
        arquivo_saida = sys.argv[2]
    if len(sys.argv) > 3:
        api_url = sys.argv[3]
    if len(sys.argv) > 4:
        modelo = sys.argv[4]
    if len(sys.argv) > 5:
        linhas_por_lote = int(sys.argv[5])
    
    traduzir_arquivo(
        arquivo_entrada, 
        arquivo_saida, 
        api_url=api_url,
        modelo=modelo,
        linhas_por_lote=linhas_por_lote
    )
