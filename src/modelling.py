import matplotlib.pyplot as plt
import numpy as np


def model_geophones_and_layers(num_geophones=8, num_layers=3, layer_depths=None):
    """
    Modela geofones e fonte, com camadas horizontais abaixo.
    """
    if layer_depths is None:
        layer_depths = [-100, -200, -300]  

    
    geophone_positions = np.linspace(300, 1500, num_geophones)  
    source_position = 50  

    return geophone_positions, source_position, layer_depths


def plot_geophones_and_layers(geophones, source, layers):
    fig, ax = plt.subplots(figsize=(8, 6))

    
    ax.plot(source, 0, 'ro', markersize=10, label="Fonte (*)")
    ax.text(source - 10, 10, '*', fontsize=15, color='red')

    
    ax.plot(geophones, np.zeros_like(geophones), 'gv', markersize=8, label="Geofones (x)")
    for x in geophones:
        ax.text(x - 5, 10, 'x', fontsize=12, color='green')

    
    for i, depth in enumerate(layers):
        ax.axhline(depth, color='black', linestyle='--')
        ax.text(20, depth + 10, f'Camada {i+1}', fontsize=10, color='black')

    
    ax.set_xlim(0, 1800)
    ax.set_ylim(layers[-1] - 50, 50)  
    ax.set_xlabel("Posição (m)", fontsize=12)
    ax.set_ylabel("Profundidade (m)", fontsize=12)
    ax.set_title("Fonte, Geofones e Camadas", fontsize=14, fontweight='bold')
    ax.legend()

  
    plt.grid(False)
    plt.show()

geophones, source, layers = model_geophones_and_layers()

plot_geophones_and_layers(geophones, source, layers)
