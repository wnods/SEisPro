"""

░██████╗███████╗██╗░██████╗░░░░░░██████╗░██████╗░░█████╗░
██╔════╝██╔════╝██║██╔════╝░░░░░░██╔══██╗██╔══██╗██╔══██╗
╚█████╗░█████╗░░██║╚█████╗░█████╗██████╔╝██████╔╝██║░░██║
░╚═══██╗██╔══╝░░██║░╚═══██╗╚════╝██╔═══╝░██╔══██╗██║░░██║
██████╔╝███████╗██║██████╔╝░░░░░░██║░░░░░██║░░██║╚█████╔╝
╚═════╝░╚══════╝╚═╝╚═════╝░░░░░░░╚═╝░░░░░╚═╝░░╚═╝░╚════╝░

#-------------------------------------------------------
# Programa de Análise e Visualização de Dados Sismicos
# Criado por: Wilson Weliton Oliveira de Souza
#-------------------------------------------------------

"""

import os
import sys
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
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from tqdm import tqdm
import time


console = Console()

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
    """
    Mostra a tabela com o conteúdo do diretório SEGY (arquivo, tamanho, número de traces e amostras).
    """
    segy_files = list_segy_files_with_sizes(directory)
    
    console.print(Panel(f"[bold blue]Diretório: {directory}", title="Informação do Diretório", title_align="left"))
    
    table = Table(title="Conteúdo do Diretório SEGY")
    table.add_column("Arquivo", justify="left", style="cyan", no_wrap=True)
    table.add_column("Tamanho (MB)", justify="center", style="green")
    table.add_column("Número de Traces", justify="center", style="magenta")
    table.add_column("Amostras por Trace", justify="center", style="yellow")

    for filename, size in segy_files:
        file_path = os.path.join(directory, filename)
        num_traces, num_samples = get_segy_file_info(file_path)

        table.add_row(
            filename,
            f"{size:.2f}",
            str(num_traces),
            str(num_samples)
        )
    
    console.print(table)

def list_segy_files(directory):
    segy_files = [f for f in os.listdir(directory) if f.endswith('.SEGY') or f.endswith('.SGY')]
    return sorted(segy_files)

def show_segy_files_in_multi_column_table(segy_files, num_columns=4):
    """
    Exibe os arquivos SEGY em uma tabela formatada com múltiplas colunas.
    """
    console.print(Panel("[bold blue]Arquivos SEGY encontrados (em colunas):[/bold blue]"))

    
    table = Table(title="Escolha o Arquivo SEGY", show_header=True, header_style="bold cyan")
    for i in range(num_columns):
        table.add_column(f"Dados SEGY {i + 1}", justify="left", style="blue")

    
    num_files = len(segy_files)
    rows = (num_files + num_columns - 1) // num_columns  

    for row in range(rows):
        row_data = []
        for col in range(num_columns):
            file_idx = row + col * rows
            if file_idx < num_files:
                row_data.append(f"{file_idx}: {segy_files[file_idx]}")
            else:
                row_data.append("") 
        table.add_row(*row_data)

    console.print(table)

def choose_segy_file_with_multi_columns(directory, num_columns=4):
    """
    Exibe a tabela com arquivos SEGY em múltiplas colunas para escolha do usuário.
    """
    segy_files = list_segy_files(directory)
    if not segy_files:
        raise FileNotFoundError(f"Nenhum arquivo SEGY encontrado no diretório: {directory}")

    # Exibe a tabela com arquivos SEGY encontrados, organizados em múltiplas colunas
    show_segy_files_in_multi_column_table(segy_files, num_columns)

    while True:
        try:
            file_idx = int(input("Digite o número do arquivo SEGY para carregar: "))
            if file_idx < 0 or file_idx >= len(segy_files):
                raise IndexError("Índice de arquivo SEGY inválido.")
            return os.path.join(directory, segy_files[file_idx])
        except ValueError:
            print("Por favor, insira um número válido.")
        except IndexError as e:
            print(e)

def read_segy_file(file_path):
    segy_file = _read_segy(file_path)
    traces = segy_file.traces
    data = np.array([trace.data for trace in traces])
    return data, segy_file

def plot_seismic_collage_with_spectrogram(data, traces_to_plot=[0, 1, 2], cmap='seismic', fs=24.0, nfft=800, noverlap=700, output_file='seismic_collage.png', segy_file_name=''):
    num_traces, num_samples = data.shape
    t = np.arange(num_samples)
    
    
    simulate_processing()
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    max_amplitude = np.max(np.abs(data))
    for i, trace in enumerate(data):
        axs[0, 0].plot(trace / max_amplitude + i, t, color='black', lw=1.0)
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

    
    plt.figtext(0.5, 0.01, f"Arquivo utilizado: {segy_file_name}", ha="center", fontsize=10, color="blue")
    
    plt.savefig(output_file)
    plt.show()

def plot_filtered_data_with_envelope(data, sample_rate, freqmin=1, freqmax=3, time_window=(80, 90)):
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
    """
    Exibe a tabela com arquivos SEGY para escolha do usuário.
    """
    segy_files = list_segy_files(directory)
    if not segy_files:
        raise FileNotFoundError(f"Nenhum arquivo SEGY encontrado no diretório: {directory}")

   
    show_segy_files_in_table(segy_files)

    while True:
        try:
            file_idx = int(input("Digite o número do arquivo SEGY para carregar: "))
            if file_idx < 0 or file_idx >= len(segy_files):
                raise IndexError("Índice de arquivo SEGY inválido.")
            return os.path.join(directory, segy_files[file_idx])
        except ValueError:
            print("Por favor, insira um número válido.")
        except IndexError as e:
            print(e)

def ask_plot_choice():
    tree = Tree("[bold blue]Escolha o tipo de gráfico para plotar:[/bold blue]")
    tree.add("[green]1: Colagem de gráficos sísmicos[/green]")
    tree.add("[green]2: Gráfico de Dados Filtrados com Envoltória[/green]")
    console.print(tree)

    while True:
        try:
            choice = int(input("Escolha (1 ou 2): "))
            if choice not in [1, 2]:
                raise ValueError("Escolha inválida. Selecione 1 ou 2")
            return choice
        except ValueError as e:
            print(e)

def ask_plot_parameters():
    tree = Tree("[bold blue]Parâmetros para o gráfico de colagem sísmica[/bold blue]")
    cmap = input("Digite o cmap para o gráfico (ENTER para padrão: gray): ") or 'gray'
    tree.add(f"[green]Cmap: {cmap}[/green]")
    
    try:
        fs = float(input("Digite o fator de amostragem (Fs) (ENTER para padrão: 24.0): ") or 24.0)
        nfft = int(input("Digite o tamanho da janela de FFT (NFFT) (ENTER para padrão: 2000): ") or 2000)
        noverlap = int(input("Digite o valor de sobreposição (noverlap) (ENTER para padrão: 700): ") or 700)
    except ValueError:
        console.print("[bold red]Valores inválidos fornecidos, usando padrões.[/bold red]")
        fs, nfft, noverlap = 24.0, 2000, 700

    tree.add(f"[green]Fs: {fs}[/green]")
    tree.add(f"[green]NFFT: {nfft}[/green]")
    tree.add(f"[green]Noverlap: {noverlap}[/green]")

    console.print(tree)
    return cmap, fs, nfft, noverlap

def ask_filter_parameters():
    tree = Tree("[bold blue]Parâmetros de Filtro[/bold blue]")
    
    try:
        freqmin = float(input("Digite a frequência mínima para o filtro (Hz): ") or 10)
        freqmax = float(input("Digite a frequência máxima para o filtro (Hz): ") or 30)
        time_start = float(input("Tempo de início da visualização (s): ") or 10)
        time_end = float(input("Tempo de término da visualização (s): ") or 80)
    except ValueError:
        console.print("[bold red]Valores inválidos fornecidos, usando padrões.[/bold red]")
        freqmin, freqmax = 10, 30
        time_start, time_end = 10, 90

    tree.add(f"[green]Frequência mínima: {freqmin} Hz[/green]")
    tree.add(f"[green]Frequência máxima: {freqmax} Hz[/green]")
    tree.add(f"[green]Início do tempo: {time_start} s[/green]")
    tree.add(f"[green]Fim do tempo: {time_end} s[/green]")

    console.print(tree)
    return freqmin, freqmax, (time_start, time_end)

def simulate_processing():
    """
    Simula o processamento exibindo uma barra de progresso com tqdm.
    """
    for _ in tqdm(range(100), desc="Processando..."):
        time.sleep(0.05)

def main():
    current_directory = os.getcwd()
    console.print(f"[bold blue]Diretório Atual: {current_directory}[/bold blue]\n")
    
    naval_data_directory = os.path.join(current_directory, "Data")

    options = {
        1: "Sismica Ativa",
        2: "Sismica Passiva"
    }

    console.print("[bold green]Selecione o tipo de dados sísmicos:[/bold green]")
    for key, value in options.items():
        console.print(f"{key}: [cyan]{value}[/cyan]")

    while True:
        try:
            choice = int(input("Escolha (1 ou 2): "))
            if choice not in options:
                raise ValueError("Escolha inválida. Selecione 1 ou 2.")
            
            selected_directory = os.path.join(naval_data_directory, options[choice])
            break
        except ValueError as e:
            print(e)

    
    generate_directory_header(selected_directory)

    
    segy_file_path = choose_segy_file_with_multi_columns(selected_directory, num_columns=4)
    segy_file_name = os.path.basename(segy_file_path)  

    console.print(f"[bold green]Arquivo SEGY selecionado: {segy_file_name}[/bold green]")

    plot_choice = ask_plot_choice()

    if plot_choice == 1:
        cmap, fs, nfft, noverlap = ask_plot_parameters()
        output_file = input("Digite o nome do arquivo de saída (com extensão .png) (padrão: 'SEIS.png'): ") or 'SEIS.png'
        data, segy_file = read_segy_file(segy_file_path)
        plot_seismic_collage_with_spectrogram(data, traces_to_plot=[0, 1, 2, 3, 4, 5, 6], cmap=cmap, fs=fs, nfft=nfft, noverlap=noverlap, output_file=output_file, segy_file_name=segy_file_name)

    elif plot_choice == 2:
        data, segy_file = read_segy_file(segy_file_path)
        sample_rate = 24.0  
        freqmin, freqmax, time_window = ask_filter_parameters()
        plot_filtered_data_with_envelope(data[0], sample_rate, freqmin=freqmin, freqmax=freqmax, time_window=time_window)


if __name__ == "__main__":
    main()
