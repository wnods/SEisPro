import os
import sys
import utm
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy.io.segy.segy import _read_segy
from obspy.signal.invsim import corn_freq_2_paz
from obspy.signal.array_analysis import array_processing
from obspy.imaging.cm import obspy_sequential
from obspy.signal import filter
from obspy import read, Stream
from obspy import UTCDateTime
from scipy import signal
from obspy.signal.filter import bandstop
from matplotlib.colors import Normalize
from obspy.imaging.spectrogram import spectrogram
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# from processing import FkPro
# from routines import util


def list_segy_files_with_sizes(directory):
    segy_files = [
        (f, os.path.getsize(os.path.join(directory, f)) / (1024 * 1024)) 
        for f in os.listdir(directory) if f.endswith('.SEGY') or f.endswith('.SGY')
    ]
    return sorted(segy_files, key=lambda x: x[0])

def get_segy_file_info(file_path):
    segy_file = _read_segy(file_path)
    num_traces = len(segy_file.traces)
    num_samples = segy_file.traces[0].header.number_of_samples_in_this_trace if num_traces > 0 else 0
    return num_traces, num_samples

def generate_directory_header(directory):
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
    segy_files = [f for f in os.listdir(directory) if f.endswith('.SEGY') or f.endswith('.SGY')]
    return sorted(segy_files)

def read_segy_file(file_path):
    segy_file = _read_segy(file_path)
    traces = segy_file.traces
    data = np.array([trace.data for trace in traces])
    return data, segy_file

def plot_seismic_collage_with_spectrogram(data, traces_to_plot=[0, 1, 2], cmap='seismic', fs=24.0, nfft=800, noverlap=700, output_file='seismic_collage.png'):
    num_traces, num_samples = data.shape
    t = np.arange(num_samples)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    max_amplitude = np.max(np.abs(data))
    for i, trace in enumerate(data):
        axs[0, 0].plot(trace / max_amplitude + i, t, color='darkblue', lw=1.5)
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_title("Visualizador da Onda Sísmica", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Traços")
    axs[0, 0].set_ylabel("Tempo [s]")
    axs[0, 0].grid(True, linestyle='--', color='gray', alpha=0.5)

    im = axs[0, 1].imshow(data.T, cmap=cmap, aspect='auto', interpolation='bilinear')
    fig.colorbar(im, ax=axs[0, 1], label='Amplitude')
    axs[0, 1].set_title("Gráfico de Intensidade da Onda", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Traços")
    axs[0, 1].set_ylabel("Tempo [s]")

    for trace_idx in traces_to_plot:
        if trace_idx < num_traces:
            axs[1, 0].plot(t, data[trace_idx], color='black', label=f'Trace {trace_idx}', lw=0.5)
    axs[1, 0].set_title("Gráfico das Ondas Sísmicas", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Tempo [s]")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].grid(True, linestyle='-', color='black', alpha=0.5)

    if len(traces_to_plot) > 5:  
        trace_idx = traces_to_plot[6]
        if trace_idx < num_traces:
            axs[1, 1].specgram(data[trace_idx], NFFT=nfft, Fs=fs, noverlap=noverlap, cmap=cmap)
            axs[1, 1].set_title(f"Gráfico do Espectrograma", fontsize=12, fontweight='bold')
            axs[1, 1].set_xlabel("Tempo [s]")
            axs[1, 1].set_ylabel("Frequência [hz]")
        else:
            axs[1, 1].set_title("Spectrogram: Trace index out of range", fontsize=12, fontweight='bold')

    plt.savefig(output_file)
    plt.show()


    #---------------- Configurações do processamento --------------------#

def plot_filtered_data_with_envelope(data, sample_rate, freqmin=1, freqmax=3, time_window=(80, 90)):
    """
    Plota os dados filtrados com a envoltória de amplitude.
    """
    data_filtered = obspy.signal.filter.bandpass(data, freqmin=freqmin, freqmax=freqmax, df=sample_rate, corners=2, zerophase=True)
    data_envelope = obspy.signal.filter.envelope(data_filtered)

    npts = len(data)
    t = np.arange(0, npts / sample_rate, 1 / sample_rate)

    plt.plot(t, data_filtered, 'k', label='Filtered Data')
    plt.plot(t, data_envelope, 'k:', label='Envelope')
    plt.title('Dados filtrados com Envoltória')
    plt.ylabel('Amplitude')
    plt.xlabel('Tempo [s]')
    plt.xlim(time_window)
    plt.legend()
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

def ask_plot_choice():
    print("Escolha o tipo de gráfico para plotar:")
    print("1: Colagem de gráficos sísmicos")
    print("2: Gráfico de Dados Filtrados com Envoltória")
    
    while True:
        try:
            choice = int(input("Escolha (1 ou 2): "))
            if choice not in [1, 2]:
                raise ValueError("Escolha inválida. Selecione 1 ou 2")
            return choice
        except ValueError as e:
            print(e)

def ask_plot_parameters():
    cmap = input("  ├── Digite o cmap para o gráfico. ENTER para continuar com o Padrão: ") or 'inferno'
    try:
        fs = float(input("    ├── Digite o fator de amostragem (Fs). ENTER para continuar com o Padrão:(rtn= 24)") or 24.0)
        nfft = int(input("        ├── Digite o tamanho da janela de FFT (NFFT). ENTER para continuar com o Padrão:(rtn= 2000) ") or 2000)
        noverlap = int(input("           ├── Digite o valor de sobreposição (noverlap). ENTER para continuar com o Padrão:(rtn= 700)") or 700)
    except ValueError:
        print("Valores inválidos fornecidos, usando padrões.")
        fs, nfft, noverlap = 24.0, 2000, 700

    return cmap, fs, nfft, noverlap

def ask_filter_parameters():
    """
    Permite ao usuário definir as frequências de corte do filtro e o intervalo de tempo do gráfico.
    """
    try:
        freqmin = float(input("├── Digite a frequência mínima para o filtro (rtn= 10 Hz): ") or 10)
        freqmax = float(input("├── Digite a frequência máxima para o filtro (rtn= 30 Hz): ") or 30)
        time_start = float(input("├── Tempo de início da visualização (rtn= 10 s): ") or 10)
        time_end = float(input("├── Tempo de término da visualização (rtn=: 80 s): ") or 80)
    except ValueError:
        print("Valores inválidos fornecidos, usando padrões.")
        freqmin, freqmax = 10, 30
        time_start, time_end = 10, 90

    return freqmin, freqmax, (time_start, time_end)

def main():
    current_directory = os.getcwd()
    print(f"Diretório Atual: {current_directory}\n")
    
    naval_data_directory = os.path.join(current_directory, "Data_Naval")

    options = {
        1: "Sismica Ativa",
        2: "Sismica Passiva"
    }

    print("Selecione o tipo de dados sísmicos:")
    for key, value in options.items():
        print(f"{key}: {value}")

    while True:
        try:
            choice = int(input("Escolha (1 ou 2): "))
            if choice not in options:
                raise ValueError("Escolha inválida. Selecione 1 ou 2.")
            
            selected_directory = os.path.join(naval_data_directory, options[choice])
            break
        except ValueError as e:
            print(e)

    segy_file_path = choose_segy_file_with_header(selected_directory)

    plot_choice = ask_plot_choice()

    if plot_choice == 1:
        cmap, fs, nfft, noverlap = ask_plot_parameters()
        output_file = input("                 ├── Digite o nome do arquivo de saída (com extensão .png) (padrão: 'SEIS.png'): ") or 'SEIS.png'
        data, segy_file = read_segy_file(segy_file_path)
        plot_seismic_collage_with_spectrogram(data, traces_to_plot=[0, 1, 2, 3, 4, 5, 6], cmap=cmap, fs=fs, nfft=nfft, noverlap=noverlap, output_file=output_file)

    elif plot_choice == 2:
        data, segy_file = read_segy_file(segy_file_path)
        sample_rate = 24.0  
        freqmin, freqmax, time_window = ask_filter_parameters()
        plot_filtered_data_with_envelope(data[0], sample_rate, freqmin=freqmin, freqmax=freqmax, time_window=time_window)
        
        

if __name__ == "__main__":
    main()
