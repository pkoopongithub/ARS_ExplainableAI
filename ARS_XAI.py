"""
Algorithmisch Rekursive Sequenzanalyse 2.0
Grammatikinduktion aus acht Transkripten
Optimierung durch iterativen Vergleich empirischer und generierter Ketten
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tabulate import tabulate

# ============================================================================
# 1. EMPIRISCHE DATEN: Terminalzeichenketten aus acht Transkripten
# ============================================================================

empirical_chains = [
    # Transkript 1: Metzgerei
    ['KBG', 'VBG', 'KBBd', 'VBBd', 'KBA', 'VBA', 'KBBd', 'VBBd', 'KBA', 'VAA', 'KAA', 'VAV', 'KAV'],
    # Transkript 2: Marktplatz (Kirschen)
    ['VBG', 'KBBd', 'VBBd', 'VAA', 'KAA', 'VBG', 'KBBd', 'VAA', 'KAA'],
    # Transkript 3: Fischstand
    ['KBBd', 'VBBd', 'VAA', 'KAA'],
    # Transkript 4: Gemüsestand (ausfuehrlich)
    ['KBBd', 'VBBd', 'KBA', 'VBA', 'KBBd', 'VBA', 'KAE', 'VAE', 'KAA', 'VAV', 'KAV'],
    # Transkript 5: Gemüsestand (mit KAV zu Beginn)
    ['KAV', 'KBBd', 'VBBd', 'KBBd', 'VAA', 'KAV'],
    # Transkript 6: Käseverkaufsstand
    ['KBG', 'VBG', 'KBBd', 'VBBd', 'KAA'],
    # Transkript 7: Bonbonstand
    ['KBBd', 'VBBd', 'KBA', 'VAA', 'KAA'],
    # Transkript 8: Baeckerei
    ['KBG', 'VBBd', 'KBBd', 'VBA', 'VAA', 'KAA', 'VAV', 'KAV']
]

# ============================================================================
# 2. UeBERGANGSZAEHLUNG UND INITIALE WAHRSCHEINLICHKEITEN
# ============================================================================

def count_transitions(chains):
    """Zaehlt Uebergaenge zwischen Terminalzeichen in allen Ketten"""
    transitions = {}
    for chain in chains:
        for i in range(len(chain) - 1):
            start, end = chain[i], chain[i + 1]
            if start not in transitions:
                transitions[start] = {}
            if end not in transitions[start]:
                transitions[start][end] = 0
            transitions[start][end] += 1
    return transitions

def calculate_probabilities(transitions):
    """Normalisiert Uebergangszaehlungen zu Wahrscheinlichkeiten"""
    probabilities = {}
    for start in transitions:
        total = sum(transitions[start].values())
        probabilities[start] = {end: count / total 
                               for end, count in transitions[start].items()}
    return probabilities

# Initiale Berechnungen
initial_transitions = count_transitions(empirical_chains)
initial_probabilities = calculate_probabilities(initial_transitions)

print("=" * 70)
print("ALGORITHMISCH REKURSIVE SEQUENZANALYSE 2.0")
print("=" * 70)
print("\n1. INITIALE UeBERGANGSWAHRSCHEINLICHKEITEN (AUS EMPIRISCHEN DATEN)")
print("-" * 70)

for start in sorted(initial_probabilities.keys()):
    transitions_str = ", ".join([f"{end}: {prob:.3f}" 
                                 for end, prob in initial_probabilities[start].items()])
    print(f"{start} -> {transitions_str}")

# ============================================================================
# 3. TERMINALZEICHEN UND STARTZEICHEN
# ============================================================================

terminal_symbols = sorted(list(set([item for sublist in empirical_chains 
                                     for item in sublist])))
start_symbol = empirical_chains[0][0]  # KBG als Start (kann angepasst werden)

print(f"\nTerminalzeichen ({len(terminal_symbols)}): {terminal_symbols}")
print(f"Startzeichen: {start_symbol}")

# ============================================================================
# 4. GENERIERUNG KUeNSTLICHER KETTEN
# ============================================================================

def generate_chain(probabilities, start_symbol, max_length=20):
    """Generiert eine Kette basierend auf den Uebergangswahrscheinlichkeiten"""
    chain = [start_symbol]
    current = start_symbol
    
    for _ in range(max_length - 1):
        if current not in probabilities:
            break
        
        next_symbols = list(probabilities[current].keys())
        probs = list(probabilities[current].values())
        
        # Falls keine Folgesymbole vorhanden, abbrechen
        if not next_symbols:
            break
            
        next_symbol = np.random.choice(next_symbols, p=probs)
        chain.append(next_symbol)
        current = next_symbol
        
        # Stopp, wenn wir bei einem Terminal ohne weitere Uebergaenge landen
        if current not in probabilities:
            break
    
    return chain

def generate_multiple_chains(probabilities, start_symbol, n_chains=8, max_length=20):
    """Generiert mehrere Ketten"""
    return [generate_chain(probabilities, start_symbol, max_length) 
            for _ in range(n_chains)]

# ============================================================================
# 5. HAEUFIGKEITSANALYSE
# ============================================================================

def compute_frequencies(chains, terminals):
    """Berechnet relative Haeufigkeiten der Terminalzeichen in Ketten"""
    frequency_array = np.zeros(len(terminals))
    terminal_index = {term: i for i, term in enumerate(terminals)}
    
    for chain in chains:
        for symbol in chain:
            if symbol in terminal_index:
                frequency_array[terminal_index[symbol]] += 1
    
    total = frequency_array.sum()
    if total > 0:
        frequency_array /= total  # Normierung
    
    return frequency_array

# Empirische Haeufigkeiten als Referenz
empirical_frequencies = compute_frequencies(empirical_chains, terminal_symbols)

print("\n2. EMPIRISCHE RELATIVE HAEUFIGKEITEN")
print("-" * 70)
for i, symbol in enumerate(terminal_symbols):
    print(f"{symbol}: {empirical_frequencies[i]:.4f}")

# ============================================================================
# 6. ITERATIVE OPTIMIERUNG DER GRAMMATIK
# ============================================================================

def optimize_grammar(empirical_chains, terminal_symbols, start_symbol,
                     max_iterations=1000, tolerance=0.01, target_correlation=0.9):
    """
    Optimiert die Grammatik durch iterativen Vergleich mit generierten Ketten.
    """
    
    # Initiale Wahrscheinlichkeiten aus empirischen Daten
    transitions = count_transitions(empirical_chains)
    probabilities = calculate_probabilities(transitions)
    
    # Empirische Haeufigkeiten als Zielgroesse
    empirical_freqs = compute_frequencies(empirical_chains, terminal_symbols)
    
    best_correlation = 0
    best_significance = 1
    best_probabilities = None
    history = []
    
    print("\n3. ITERATIVE OPTIMIERUNG")
    print("-" * 70)
    
    for iteration in range(max_iterations):
        # Generiere 8 kuenstliche Ketten
        generated_chains = generate_multiple_chains(probabilities, start_symbol, n_chains=8)
        
        # Berechne Haeufigkeiten der generierten Ketten
        generated_freqs = compute_frequencies(generated_chains, terminal_symbols)
        
        # Korrelationsanalyse
        correlation, p_value = pearsonr(empirical_freqs, generated_freqs)
        history.append((iteration, correlation, p_value))
        
        # Fortschrittsanzeige alle 50 Iterationen
        if iteration % 50 == 0:
            print(f"Iteration {iteration:4d}: Korrelation = {correlation:.4f}, p = {p_value:.4f}")
        
        # Pruefe Abbruchkriterium
        if correlation >= target_correlation and p_value < 0.05:
            best_correlation = correlation
            best_significance = p_value
            best_probabilities = {start: probs.copy() 
                                 for start, probs in probabilities.items()}
            print(f"\nOptimum erreicht bei Iteration {iteration}:")
            print(f"  Korrelation = {correlation:.4f}")
            print(f"  Signifikanz = {p_value:.4f}")
            break
        
        # Anpassung der Wahrscheinlichkeiten
        for start in probabilities:
            for end in probabilities[start]:
                # Fehlerberechnung
                empirical_prob = empirical_freqs[terminal_symbols.index(end)]
                generated_prob = generated_freqs[terminal_symbols.index(end)]
                error = empirical_prob - generated_prob
                
                # Anpassung mit Toleranzfaktor
                probabilities[start][end] += error * tolerance
                
                # Begrenzung auf [0,1]
                probabilities[start][end] = max(0.01, min(0.99, probabilities[start][end]))
        
        # Renormalisierung
        for start in probabilities:
            total = sum(probabilities[start].values())
            if total > 0:
                probabilities[start] = {end: prob / total 
                                       for end, prob in probabilities[start].items()}
    
    # Falls kein Optimum erreicht wurde, nimm die beste Iteration
    if best_probabilities is None:
        # Finde Iteration mit hoechster Korrelation
        best_idx = max(range(len(history)), key=lambda i: history[i][1])
        best_iter, best_correlation, best_significance = history[best_idx]
        best_probabilities = calculate_probabilities(count_transitions(empirical_chains))
        print(f"\nKein Optimum erreicht. Beste Korrelation bei Iteration {best_iter}:")
        print(f"  Korrelation = {best_correlation:.4f}")
        print(f"  Signifikanz = {best_significance:.4f}")
    
    return best_probabilities, best_correlation, best_significance, history

# Optimierung durchfuehren
optimized_probabilities, best_corr, best_sig, history = optimize_grammar(
    empirical_chains, terminal_symbols, start_symbol,
    max_iterations=500, tolerance=0.005, target_correlation=0.9
)

# ============================================================================
# 7. VISUALISIERUNG DER OPTIMIERUNG
# ============================================================================

def plot_optimization_history(history):
    """Visualisiert den Optimierungsverlauf"""
    iterations, correlations, p_values = zip(*history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Korrelationsverlauf
    ax1.plot(iterations, correlations, 'b-', linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Korrelation (Pearson r)')
    ax1.set_title('Optimierungsverlauf: Korrelation zwischen empirischen und generierten Haeufigkeiten')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Zielkorrelation (0.9)')
    ax1.legend()
    
    # p-Wert-Verlauf (logarithmisch)
    p_values = [max(p, 1e-10) for p in p_values]  # Vermeidung von log(0)
    ax2.semilogy(iterations, p_values, 'g-', linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('p-Wert (logarithmisch)')
    ax2.set_title('Signifikanz der Korrelation')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Signifikanzniveau (0.05)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('optimierungsverlauf.png', dpi=150)
    plt.show()

# Optional: Visualisierung (wenn matplotlib verfuegbar)
try:
    plot_optimization_history(history)
    print("\nOptimierungsverlauf wurde als 'optimierungsverlauf.png' gespeichert.")
except:
    print("\n(Hinweis: Fuer Visualisierung ist matplotlib erforderlich)")

# ============================================================================
# 8. AUSGABE DER OPTIMIERTEN GRAMMATIK
# ============================================================================

print("\n" + "=" * 70)
print("4. OPTIMIERTE PROBABILISTISCHE GRAMMATIK")
print("=" * 70)

# Nach Startzeichen sortierte Ausgabe
for start in sorted(optimized_probabilities.keys()):
    transitions = optimized_probabilities[start]
    transitions_str = ", ".join([f"'{end}': {prob:.3f}" 
                                 for end, prob in sorted(transitions.items())])
    print(f"\n{start} -> {transitions_str}")

# ============================================================================
# 9. VALIDIERUNG: VERGLEICH EMPIRISCHER UND GENERIERTER HAEUFIGKEITEN
# ============================================================================

# Generiere neue Ketten mit optimierter Grammatik
validation_chains = generate_multiple_chains(
    optimized_probabilities, start_symbol, n_chains=100, max_length=20
)
validation_frequencies = compute_frequencies(validation_chains, terminal_symbols)

print("\n" + "=" * 70)
print("5. VALIDIERUNG: EMPIRISCHE VS. GENERIERTE HAEUFIGKEITEN")
print("=" * 70)

table_data = []
for i, symbol in enumerate(terminal_symbols):
    table_data.append([
        symbol,
        f"{empirical_frequencies[i]:.4f}",
        f"{validation_frequencies[i]:.4f}",
        f"{abs(empirical_frequencies[i] - validation_frequencies[i]):.4f}"
    ])

print(tabulate(table_data, 
               headers=["Symbol", "Empirisch", "Generiert", "Differenz"],
               tablefmt="grid"))

# Gesamtkorrelation
final_corr, final_p = pearsonr(empirical_frequencies, validation_frequencies)
print(f"\nKorrelation (100 generierte Ketten): r = {final_corr:.4f}, p = {final_p:.4f}")

# ============================================================================
# 10. BEISPIEL-GENERIERTE KETTEN
# ============================================================================

print("\n" + "=" * 70)
print("6. BEISPIEL GENERIERTER TERMINALZEICHENKETTEN")
print("=" * 70)

example_chains = generate_multiple_chains(
    optimized_probabilities, start_symbol, n_chains=5, max_length=15
)

for i, chain in enumerate(example_chains, 1):
    chain_str = " -> ".join(chain)
    print(f"\nKette {i} ({len(chain)} Symbole):")
    print(f"  {chain_str}")

# ============================================================================
# 11. EXPORT DER GRAMMATIK ALS STRUKTUR
# ============================================================================

def export_grammar_as_pcfg(probabilities, filename="optimierte_grammatik.txt"):
    """Exportiert die Grammatik im PCFG-Format"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Optimierte probabilistische kontextfreie Grammatik (PCFG)\n")
        f.write("# Generiert durch Algorithmisch Rekursive Sequenzanalyse 2.0\n\n")
        
        for start in sorted(probabilities.keys()):
            transitions = probabilities[start]
            for end, prob in sorted(transitions.items()):
                f.write(f"{start} -> {end} [{prob:.3f}]\n")
    
    print(f"\nGrammatik wurde als '{filename}' exportiert.")

export_grammar_as_pcfg(optimized_probabilities)

print("\n" + "=" * 70)
print("ALGORITHMISCH REKURSIVE SEQUENZANALYSE ABGESCHLOSSEN")
print("=" * 70)
