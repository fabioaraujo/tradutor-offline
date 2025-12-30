"""
Script para juntar sentenças quebradas em múltiplas linhas
"""
import sys
import os

# Configurar encoding UTF-8 para o terminal Windows
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 > nul')
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def deve_juntar_com_proxima(linha_atual, proxima_linha):
    """
    Determina se a linha atual deve ser juntada com a próxima.
    
    Args:
        linha_atual: Linha atual (stripped)
        proxima_linha: Próxima linha (stripped)
        
    Returns:
        True se deve juntar, False caso contrário
    """
    if not linha_atual or not proxima_linha:
        return False
    
    # Pontuações que indicam fim de sentença
    fim_sentenca = ('.', '!', '?', ':', '"', '"', '"', ')', ']', '}')
    
    # Se a linha atual termina com pontuação forte, não juntar
    if linha_atual.rstrip().endswith(fim_sentenca):
        return False
    
    # Se a próxima linha começa com letra minúscula, provavelmente é continuação
    if proxima_linha[0].islower():
        return True
    
    # Se a próxima linha começa com "and", "or", "but", "which", etc.
    palavras_continuacao = ['and', 'or', 'but', 'which', 'that', 'who', 'where', 'when', 'how']
    primeira_palavra = proxima_linha.split()[0].lower() if proxima_linha.split() else ""
    if primeira_palavra in palavras_continuacao:
        return True
    
    # Se a linha atual é muito curta (< 40 caracteres) e não termina com pontuação
    if len(linha_atual) < 40 and not linha_atual.endswith(','):
        # Mas a próxima começa com maiúscula, pode ser um nome ou início de frase
        # Não juntar nesses casos
        return False
    
    # Se a linha atual termina com vírgula, provavelmente continua
    if linha_atual.rstrip().endswith(','):
        return True
    
    return False


def juntar_linhas_arquivo(arquivo_entrada, arquivo_saida):
    """
    Junta linhas quebradas em um arquivo de texto.
    
    Args:
        arquivo_entrada: Caminho do arquivo original
        arquivo_saida: Caminho do arquivo processado
    """
    print("=" * 70)
    print("📝 JUNTANDO LINHAS QUEBRADAS")
    print("=" * 70)
    
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
    
    print(f"   Total de linhas: {len(linhas)}\n")
    
    print("🔄 Processando linhas...")
    
    # Processar as linhas
    linhas_processadas = []
    i = 0
    linhas_juntadas = 0
    
    while i < len(linhas):
        linha_atual = linhas[i].rstrip('\n\r')
        
        # Se a linha está vazia, adicionar e continuar
        if not linha_atual.strip():
            linhas_processadas.append(linha_atual + '\n')
            i += 1
            continue
        
        # Juntar com próximas linhas se necessário
        while i + 1 < len(linhas):
            proxima_linha = linhas[i + 1].rstrip('\n\r')
            
            # Se a próxima linha está vazia, não juntar
            if not proxima_linha.strip():
                break
            
            # Verificar se deve juntar
            if deve_juntar_com_proxima(linha_atual.strip(), proxima_linha.strip()):
                # Juntar as linhas com um espaço
                linha_atual = linha_atual + ' ' + proxima_linha.strip()
                i += 1
                linhas_juntadas += 1
            else:
                break
        
        linhas_processadas.append(linha_atual + '\n')
        i += 1
    
    # Salvar o resultado
    print("💾 Salvando arquivo processado...")
    try:
        with open(arquivo_saida, 'w', encoding='utf-8') as f:
            f.writelines(linhas_processadas)
    except Exception as e:
        print(f"❌ Erro ao salvar arquivo: {e}")
        return
    
    print("\n✅ Processamento concluído!")
    print(f"   📊 Linhas originais: {len(linhas)}")
    print(f"   📊 Linhas após junção: {len(linhas_processadas)}")
    print(f"   🔗 Linhas juntadas: {linhas_juntadas}")
    print(f"   💾 Arquivo salvo: {arquivo_saida}")
    print("=" * 70)


if __name__ == "__main__":
    # Configuração padrão
    arquivo_entrada = "texto copy 2.txt"
    arquivo_saida = "texto_juntado.txt"
    
    # Permite passar argumentos pela linha de comando
    if len(sys.argv) > 1:
        arquivo_entrada = sys.argv[1]
    if len(sys.argv) > 2:
        arquivo_saida = sys.argv[2]
    
    juntar_linhas_arquivo(arquivo_entrada, arquivo_saida)
