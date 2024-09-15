import os
import numpy as np
import matplotlib.pyplot as plt
from obspy.io.segy.segy import _read_segy

def list_segy_files_with_sizes(directory):
    """
    Lista todos os arquivos .SEGY no diretório fornecido em ordem alfabética,
    retornando o nome e tamanho em MB.
    """
    segy_files = [
        (f, os.path.getsize(os.path.join(directory, f)) / (1024 * 1024)) 
        for f in os.listdir(directory) if f.endswith('.SEGY') or f.endswith('.SGY')
    ]
    return sorted(segy_files, key=lambda x: x[0])

def get_segy_file_info(file_path):

    """
    Lê o arquivo SEGY e retorna informações básicas como número de traces e amostras.
    """
    
    segy_file = _read_segy(file_path)
    num_traces = len(segy_file.traces)
    num_samples = segy_file.traces[0].header.number_of_samples_in_this_trace if num_traces > 0 else 0

    return num_traces, num_samples

def generate_directory_header(directory):

    """
     HEADER 
     
    """
    
    segy_files = list_segy_files_with_sizes(directory)
    
    print(f"Diretório: {directory}")
    print("Conteúdo:")

    for filename, size in segy_files:
        file_path = os.path.join(directory, filename)
        num_traces, num_samples = get_segy_file_info(file_path)

        print(f"├── {filename} ({size:.2f} MB)")
        print(f"    ├── Número de Traces: {num_traces}")
        print(f"    └── Amostras por Trace: {num_samples}")

def list_segy_files(directory):
    """
    Lista todos os arquivos .SEGY no diretório fornecido em ordem alfabética.
    """
    segy_files = [f for f in os.listdir(directory) if f.endswith('.SEGY') or f.endswith('.SGY')]
    return sorted(segy_files)

def read_segy_file(file_path):

    """
    Lê arquivo SEGY e retorna dados e cabeçalhos das traces.
    """
    segy_file = _read_segy(file_path)
    traces = segy_file.traces
    data = np.array([trace.data for trace in traces])

    return data, segy_file

def plot_seismic_collage_with_spectrogram(data, traces_to_plot=[0, 1, 2], cmap='seismic', fs=24.0, nfft=800, noverlap=700, output_file='seismic_collage.png'):
    """
    Plota um conjunto de gráficos: Wiggle, Intensidade, Linear e Espectrograma, como uma colagem.
    Também salva o gráfico como imagem PNG.
    """
    num_traces, num_samples = data.shape
    t = np.arange(num_samples)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # Gráfico Wiggle Seismic
    max_amplitude = np.max(np.abs(data))
    for i, trace in enumerate(data):
        axs[0, 0].plot(trace / max_amplitude + i, t, color='darkblue', lw=1.5)
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_title("Visualizador da Onda Sísmica", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Trace")
    axs[0, 0].set_ylabel("Samples")
    axs[0, 0].grid(True, linestyle='--', color='gray', alpha=0.5)

    # Gráfico de Intensidade
    im = axs[0, 1].imshow(data.T, cmap=cmap, aspect='auto', interpolation='bilinear')
    fig.colorbar(im, ax=axs[0, 1], label='Amplitude')
    axs[0, 1].set_title("Gráfico de Intensidade da Onda", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Trace")
    axs[0, 1].set_ylabel("Samples")

    # Gráfico de Ondas Sísmicas
    for trace_idx in traces_to_plot:
        if trace_idx < num_traces:
            axs[1, 0].plot(t, data[trace_idx], color='black',label=f'Trace {trace_idx}', lw=0.5)
    axs[1, 0].set_title("Gráfico das Ondas Sísmicas", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Samples (Time)")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].grid(True, linestyle='-', color='black', alpha=0.5)

    # Espectrograma
    if len(traces_to_plot) > 5:  
        trace_idx = traces_to_plot[6]
        if trace_idx < num_traces:
            axs[1, 1].specgram(data[trace_idx], NFFT=nfft, Fs=fs, noverlap=noverlap, cmap=cmap)
            axs[1, 1].set_title(f"Gráfico do Espectrograma", fontsize=12, fontweight='bold')
            axs[1, 1].set_xlabel("Time")
            axs[1, 1].set_ylabel("Frequency")
        else:
            axs[1, 1].set_title("Spectrogram: Trace index out of range", fontsize=12, fontweight='bold')
    else:
        axs[1, 1].set_title("Spectrogram: Not enough traces selected", fontsize=12, fontweight='bold')

    
    plt.savefig(output_file)
    plt.show()
    
 
def choose_segy_file_with_header(directory):
    
    generate_directory_header(directory)
    
    segy_files = list_segy_files_with_sizes(directory)
    if not segy_files:
        raise FileNotFoundError(f"Nenhum arquivo SEGY encontrado no diretório: {directory}")

    print("\nArquivos SEGY encontrados (em ordem alfabética):")
    for idx, (filename, _) in enumerate(segy_files):
        print(f"{idx}: {filename}")

    while True:
        try:
            file_idx = int(input("├── Selecione o número do arquivo SEGY para carregar: "))
            if file_idx < 0 or file_idx >= len(segy_files):
                raise IndexError("Índice de arquivo SEGY inválido.")
            return os.path.join(directory, segy_files[file_idx][0])
        except ValueError:
            print("Por favor, insira um número válido.")
        except IndexError as e:
            print(e)

def ask_plot_parameters():

    """
    Permite ao usuário configurar os parâmetros de plotagem.
    """

    cmap = input("  ├── Digite o cmap para o gráfico. ENTER para continuar com o Padrão: ") or 'inferno'
    try:
        fs = float(input("    ├── Digite o fator de amostragem (Fs). ENTER para continuar com o Padrão: ") or 24.0)
        nfft = int(input("        ├── Digite o tamanho da janela de FFT (NFFT). ENTER para continuar com o Padrão: ") or 800)
        noverlap = int(input("           ├── Digite o valor de sobreposição (noverlap). ENTER para continuar com o Padrão:") or 700)
    except ValueError:
        print("Valores inválidos fornecidos, usando padrões.")
        fs, nfft, noverlap = 24.0, 800, 700

    return cmap, fs, nfft, noverlap



directory_path = '/home/will/Documentos/Sismica/Processamento de Dados/SEIS_NAVAL/Data_Naval/Seismic-Data-Active/'

segy_file_path = choose_segy_file_with_header(directory_path)

cmap, fs, nfft, noverlap = ask_plot_parameters()
 
output_file = input("                 ├── Digite o nome do arquivo de saída (com extensão .png) (padrão: 'SEIS.png'): ") or 'SEIS.png'

data, segy_file = read_segy_file(segy_file_path)
plot_seismic_collage_with_spectrogram(data, traces_to_plot=[0, 1, 2, 3, 4, 5, 6], cmap=cmap, fs=fs, nfft=nfft, noverlap=noverlap, output_file=output_file)
