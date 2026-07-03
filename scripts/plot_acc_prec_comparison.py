from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / 'results'

attacks = ['Face', 'Text', 'Face+Text']
methods = ['FLiD', 'TruFor', 'MMFusion', 'UniVAD']

# (Accuracy, Precision) per method, ordered by `attacks`
ACC = {
    'FLiD':     [75.20, 81.54, 76.80],
    'TruFor':   [64.52, 76.80, 61.40],
    'MMFusion': [41.94, 51.20, 51.72],
    'UniVAD':   [35.48, 49.60, 50.57],
}
PREC = {
    'FLiD':     [61.92, 81.76, 76.80],
    'TruFor':   [50.00, 68.13, 57.91],
    'MMFusion': [37.93, 50.41, 49.45],
    'UniVAD':   [33.45, 49.60, 46.78],
}

COLOURS = {
    'FLiD':     '#1a9850',  # ours — bold green
    'TruFor':   '#4575b4',
    'MMFusion': '#fdae61',
    'UniVAD':   '#bdbdbd',
}


def panel(ax, data, title):
    x = np.arange(len(attacks))
    w = 0.2
    for i, m in enumerate(methods):
        bars = ax.bar(x + (i - 1.5) * w, data[m], w, label=m,
                      color=COLOURS[m], edgecolor='black', linewidth=0.4,
                      zorder=3)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f'{b.get_height():.0f}', ha='center', va='bottom',
                    fontsize=6.5, rotation=0)
    ax.axhline(50, color='grey', ls='--', lw=0.8, zorder=1)  # chance line
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylim(0, 100)
    ax.set_ylabel(title + ' (\\%)' if False else f'{title} (%)')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
panel(axes[0], ACC,  'Accuracy')
panel(axes[1], PREC, 'Precision')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4,
           frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])

for ext in ('png', 'pdf'):
    fig.savefig(RES / f'acc_prec_comparison.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Wrote acc_prec_comparison.png / .pdf')
