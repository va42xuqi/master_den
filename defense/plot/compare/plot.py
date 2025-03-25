import matplotlib.pyplot as plt

# Daten der Modelle für FDE bei 0.4s und 1.0s
models = ['LSTM', 'LMU', 'Transformer', 'BitNet']
fde_0_4s = [16.91, 17.47, 16.66, 18.60]  # FDE bei 0.4s
fde_1s = [64, 64, 67, 72]  # FDE bei 1.0s

# Verschiebung der FDE-Werte, sodass 0.4s immer bei 0 beginnt
fde_0_4s_shifted = [x - fde_0_4s[i] for i, x in enumerate(fde_0_4s)]
fde_1s_shifted = [x - fde_0_4s[i] for i, x in enumerate(fde_1s)]

# Erstelle das Diagramm
fig, ax = plt.subplots(figsize=(10, 8))

# Farben für jedes Modell (Farben passend zu den Modellen)
colors = {'LSTM': 'blue', 'LMU': 'green', 'Transformer': 'red', 'BitNet': 'purple'}

# Plot für jedes Modell, vom gleichen relativen Nullpunkt starten
for i, model in enumerate(models):
    ax.plot([0.4, 1.0], [fde_0_4s_shifted[i], fde_1s_shifted[i]], marker='o', label=model, color=colors[model])
    
    # Dünnerer Pfeil von 0.4s zu 1.0s
    ax.annotate('', xy=(1.0, fde_1s_shifted[i]), xytext=(0.4, fde_0_4s_shifted[i]),
                arrowprops=dict(facecolor=colors[model], edgecolor=colors[model], lw=1.5, alpha=0.7, arrowstyle='->'))

# Achsen- und Titelbezeichnungen
ax.set_title('Modellveränderungen von 0.4s auf 1.0s (relative Verschiebung)', fontsize=16)
ax.set_xlabel('Zeitspanne (s)', fontsize=14)
ax.set_ylabel('Relative Veränderung in FDE (cm)', fontsize=14)
ax.set_xticks([0.4, 1.0])
ax.set_xticklabels(['0.4s', '1.0s'])
ax.legend()

# Layout anpassen und anzeigen
plt.tight_layout()

plt.savefig("model_development.png")
