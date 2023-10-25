from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines
import matplotlib.colors as mcolors
from math import sin, cos, radians

def drive_organism(x1, y1, theta, ax):
    # Criação da representação gráfica de um organismo
    circle = Circle([x1, y1], 0.05, edgecolor=mcolors.to_rgba('#73418b', alpha=1), facecolor=mcolors.to_rgba('#B292C1', alpha=0.8), zorder=8)
    ax.add_artist(circle)

    edge = Circle([x1, y1], 0.05, facecolor='None', edgecolor=mcolors.to_rgba('#52294a', alpha=1), zorder=8)
    ax.add_artist(edge)

    tail_len = 0.075 # Cria uma cauda
    
    x2 = cos(radians(theta)) * tail_len + x1
    y2 = sin(radians(theta)) * tail_len + y1

    ax.add_line(lines.Line2D([x1, x2], [y1, y2], color=mcolors.to_rgba('#f65220', alpha=1), linewidth=1, zorder=10))

    pass


def drive_food(x1, y1, ax):
    # Cria a reprsentação gráfica de um alimento
    circle = Circle([x1, y1], 0.03, edgecolor=mcolors.to_rgba('#eebd00', alpha=1), facecolor=mcolors.to_rgba('#E9AA4C', alpha=0.8), zorder=5)
    ax.add_artist(circle)
    
    pass