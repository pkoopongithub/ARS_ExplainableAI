"""
Algorithmisch Rekursive Sequenzanalyse 3.0
GRAMMATIKINDUKTION DURCH HIERARCHISCHE KOMPRESSION
Bildung von Nonterminalzeichen durch Erkennung von Wiederholungen
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import Counter, defaultdict
import itertools

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
# 2. HIERARCHISCHE GRAMMATIKINDUKTION DURCH KOMPRESSION
# ============================================================================

class GrammarInducer:
    """
    Induziert eine PCFG durch hierarchische Kompression von Wiederholungen.
    Analog zur Bildung neuer Variablen bei Termumformungen.
    """
    
    def __init__(self):
        self.rules = {}  # Nonterminal -> Liste von Produktionen mit Wahrscheinlichkeiten
        self.terminals = set()
        self.nonterminals = set()
        self.start_symbol = None
        self.compression_history = []  # Speichert die Hierarchie der Kompression
        
    def find_best_repetition(self, chains, min_length=2, max_length=5):
        """
        Findet die häufigste wiederholte Sequenz in allen Ketten.
        Berücksichtigt Überlappungen und zählt Vorkommen.
        """
        sequence_counter = Counter()
        
        for chain in chains:
            for length in range(min_length, min(max_length, len(chain) + 1)):
                for i in range(len(chain) - length + 1):
                    seq = tuple(chain[i:i+length])
                    sequence_counter[seq] += 1
        
        # Filtere Sequenzen, die mehrmals vorkommen
        repeated = {seq: count for seq, count in sequence_counter.items() 
                   if count >= 2}
        
        if not repeated:
            return None
        
        # Bewerte Sequenzen: (Häufigkeit * Länge) / (Anzahl einzigartiger Symbole)
        # Bevorzugt längere, häufige Sequenzen mit weniger einzigartigen Symbolen
        best_seq = max(repeated.items(), 
                      key=lambda x: x[1] * len(x[0]) / max(1, len(set(x[0]))))
        
        return best_seq[0]
    
    def compress_sequences(self, chains, sequence, new_nonterminal):
        """
        Ersetzt alle Vorkommen der Sequenz durch das neue Nonterminal.
        """
        compressed_chains = []
        seq_tuple = tuple(sequence)
        seq_len = len(sequence)
        
        for chain in chains:
            new_chain = []
            i = 0
            while i < len(chain):
                if i <= len(chain) - seq_len and tuple(chain[i:i+seq_len]) == seq_tuple:
                    new_chain.append(new_nonterminal)
                    i += seq_len
                else:
                    new_chain.append(chain[i])
                    i += 1
            compressed_chains.append(new_chain)
        
        return compressed_chains
    
    def generate_nonterminal_name(self, base_symbols):
        """
        Generiert einen aussagekräftigen Namen für ein neues Nonterminal.
        """
        # Extrahiere semantische Informationen aus den Symbolen
        prefixes = [s[:2] if isinstance(s, str) else str(s)[:2] for s in base_symbols]
        suffixes = [s[2:] if isinstance(s, str) and len(s) > 2 else '' for s in base_symbols]
        
        # Wenn alle Symbole mit K oder V beginnen, behalte das Muster
        if all(p in ['KB', 'VB', 'KA', 'VA', 'NT'] or p.startswith('NT') for p in prefixes):
            # Behalte das erste und letzte Symbol für die Benennung
            first = base_symbols[0]
            last = base_symbols[-1]
            # Entferne NT_ Präfix für Lesbarkeit, falls vorhanden
            first = first.replace('NT_', '') if isinstance(first, str) else str(first)
            last = last.replace('NT_', '') if isinstance(last, str) else str(last)
            return f"NT_{first}_{last}"
        
        # Generischer Name basierend auf den Symbolen
        symbol_str = "_".join(str(s).replace('NT_', '') for s in base_symbols)
        return f"NT_{symbol_str}"
    
    def induce_grammar(self, chains, max_iterations=20):
        """
        Hauptmethode: Induziert eine PCFG durch iterative Kompression.
        Führt so lange fort, bis keine Wiederholungen mehr gefunden werden.
        """
        current_chains = [list(chain) for chain in chains]
        iteration = 0
        rule_counter = 1
        
        print("\n" + "=" * 70)
        print("HIERARCHISCHE GRAMMATIKINDUKTION")
        print("=" * 70)
        
        while iteration < max_iterations:
            # Finde die beste Wiederholung
            best_seq = self.find_best_repetition(current_chains)
            
            if best_seq is None:
                print(f"\nKeine weiteren Wiederholungen gefunden nach {iteration} Iterationen.")
                break
            
            # Generiere Namen für neues Nonterminal
            new_nonterminal = self.generate_nonterminal_name(best_seq)
            
            # Stelle sicher, dass der Name einzigartig ist
            while new_nonterminal in self.nonterminals:
                new_nonterminal = f"{new_nonterminal}_{rule_counter}"
                rule_counter += 1
            
            print(f"\nIteration {iteration + 1}:")
            print(f"  Gefundene Wiederholung: {' -> '.join(best_seq)}")
            print(f"  Neues Nonterminal: {new_nonterminal}")
            
            # Speichere die Produktionsregel (als Liste, nicht als Tupel)
            self.rules[new_nonterminal] = [(list(best_seq), 1.0)]  # Temporär mit Wahrscheinlichkeit 1
            self.nonterminals.add(new_nonterminal)
            
            # Zähle, wie oft die Regel in den aktuellen Ketten vorkommt
            occurrence_count = 0
            for chain in current_chains:
                for i in range(len(chain) - len(best_seq) + 1):
                    if tuple(chain[i:i+len(best_seq)]) == best_seq:
                        occurrence_count += 1
            
            self.compression_history.append({
                'iteration': iteration,
                'sequence': best_seq,
                'new_symbol': new_nonterminal,
                'occurrences': occurrence_count
            })
            
            # Komprimiere alle Ketten
            current_chains = self.compress_sequences(current_chains, best_seq, new_nonterminal)
            
            # Zeige Beispiel der komprimierten Kette
            if current_chains and len(current_chains[0]) > 0:
                print(f"  Beispiel (komprimiert): {' -> '.join(str(s) for s in current_chains[0])}")
            
            iteration += 1
        
        # Terminale sind die ursprünglichen Symbole, die nie ersetzt wurden
        all_symbols = set()
        for chain in empirical_chains:
            all_symbols.update(chain)
        
        # Symbole, die nie als Nonterminale eingeführt wurden, sind Terminale
        self.terminals = all_symbols - self.nonterminals
        
        # Berechne Wahrscheinlichkeiten für jede Produktion
        self.calculate_probabilities()
        
        # Bestimme das Startsymbol (das oberste Nonterminal)
        if self.nonterminals:
            # Finde Nonterminale, die in keiner anderen Produktion vorkommen
            used_as_child = set()
            for productions in self.rules.values():
                for prod, _ in productions:
                    for sym in prod:
                        if sym in self.nonterminals:
                            used_as_child.add(sym)
            
            possible_starts = self.nonterminals - used_as_child
            if possible_starts:
                self.start_symbol = sorted(possible_starts)[0]  # Nimm das erste
            else:
                # Fallback: nimm das zuletzt erstellte Nonterminal
                self.start_symbol = sorted(self.nonterminals)[-1]
        
        return current_chains
    
    def calculate_probabilities(self):
        """
        Berechnet Wahrscheinlichkeiten für jede Produktionsregel basierend auf
        den Häufigkeiten in den komprimierten Ketten.
        """
        # Rekonstruiere die komprimierten Ketten mit den Nonterminalen
        current_chains = [list(chain) for chain in empirical_chains]
        
        # Wende alle Kompressionen in der richtigen Reihenfolge an
        for hist_entry in self.compression_history:
            sequence = hist_entry['sequence']
            new_symbol = hist_entry['new_symbol']
            current_chains = self.compress_sequences(current_chains, sequence, new_symbol)
        
        # Zähle für jedes Nonterminal, wie oft es vorkommt und welche Expansionen es hat
        expansion_counts = defaultdict(lambda: defaultdict(int))
        
        for chain in current_chains:
            self.count_expansions_in_chain(chain, expansion_counts)
        
        # Konvertiere zu Wahrscheinlichkeiten
        for nonterminal in self.rules:
            if nonterminal in expansion_counts:
                total = sum(expansion_counts[nonterminal].values())
                if total > 0:
                    # Aktualisiere die Produktionen mit Wahrscheinlichkeiten
                    new_productions = []
                    for expansion_tuple, count in expansion_counts[nonterminal].items():
                        new_productions.append((list(expansion_tuple), count / total))
                    self.rules[nonterminal] = new_productions
    
    def count_expansions_in_chain(self, chain, expansion_counts):
        """
        Zählt die Expansionen in einer einzelnen Kette.
        """
        for i, symbol in enumerate(chain):
            if symbol in self.rules:
                # Suche nach der passenden Expansion für dieses Symbol
                # (In den komprimierten Ketten ist jedes Nonterminal genau eine Expansion)
                # Wir müssen die ursprüngliche Sequenz aus der History finden
                for hist_entry in self.compression_history:
                    if hist_entry['new_symbol'] == symbol:
                        expansion = tuple(hist_entry['sequence'])
                        expansion_counts[symbol][expansion] += 1
                        break

# ============================================================================
# 3. GRAMMATIK-INDUKTION DURCHFÜHREN
# ============================================================================

inducer = GrammarInducer()
compressed_chains = inducer.induce_grammar(empirical_chains)

print("\n" + "=" * 70)
print("INDUZIERTE GRAMMATIK")
print("=" * 70)
print(f"\nTerminale ({len(inducer.terminals)}): {sorted(inducer.terminals)}")
print(f"Nonterminale ({len(inducer.nonterminals)}): {sorted(inducer.nonterminals)}")
if inducer.start_symbol:
    print(f"Startsymbol: {inducer.start_symbol}")

print("\nPRODUKTIONSREGELN (mit Wahrscheinlichkeiten):")
for nonterminal in sorted(inducer.rules.keys()):
    productions = inducer.rules[nonterminal]
    if productions:
        prod_strings = []
        for prod, prob in productions:
            prod_str = ' -> '.join(str(s) for s in prod)
            prod_strings.append(f"{prod_str} [{prob:.3f}]")
        print(f"\n{nonterminal} -> {' | '.join(prod_strings)}")

# ============================================================================
# 4. GENERIERUNG MIT DER INDUZIERTEN PCFG
# ============================================================================

class PCFGGenerator:
    """
    Generiert Ketten mit der induzierten PCFG.
    """
    
    def __init__(self, grammar, terminals, start_symbol):
        self.grammar = grammar
        self.terminals = terminals
        self.start_symbol = start_symbol
        
        # Erstelle schnelle Lookup-Tabellen
        self.production_probs = {}
        for nt, prods in grammar.items():
            if prods:  # Nur wenn es Produktionen gibt
                symbols = [prod for prod, _ in prods]
                probs = [prob for _, prob in prods]
                self.production_probs[nt] = (symbols, probs)
    
    def expand(self, symbol, max_depth=10, current_depth=0):
        """
        Expandiert ein Symbol rekursiv.
        """
        if current_depth >= max_depth:
            return [str(symbol)]  # Fallback, wenn zu tief
        
        # Wenn es ein Terminal ist oder keine Produktionen hat
        if symbol in self.terminals or symbol not in self.production_probs:
            return [str(symbol)]
        
        symbols, probs = self.production_probs[symbol]
        if not symbols:  # Falls keine Symbole vorhanden
            return [str(symbol)]
        
        chosen_idx = np.random.choice(len(symbols), p=probs)
        chosen_symbols = symbols[chosen_idx]
        
        result = []
        for sym in chosen_symbols:
            result.extend(self.expand(sym, max_depth, current_depth + 1))
        
        return result
    
    def generate(self, max_depth=10):
        """
        Generiert eine vollständige Kette.
        """
        if not self.start_symbol:
            return []
        
        chain = self.expand(self.start_symbol, max_depth)
        return chain

# Initialisiere Generator
if inducer.start_symbol:
    generator = PCFGGenerator(inducer.rules, inducer.terminals, inducer.start_symbol)
    
    print("\n" + "=" * 70)
    print("GENERIERTE BEISPIELE MIT DER PCFG")
    print("=" * 70)
    
    for i in range(5):
        generated = generator.generate(max_depth=15)
        print(f"\nKette {i+1} ({len(generated)} Symbole):")
        print(f"  {' -> '.join(generated)}")
else:
    print("\nKein Startsymbol gefunden. Generierung übersprungen.")

# ============================================================================
# 5. VALIDIERUNG: VERGLEICH MIT EMPIRISCHEN DATEN
# ============================================================================

def collect_all_symbols(grammar, terminals, start_symbol, num_samples=1000, max_depth=10):
    """
    Sammelt alle möglichen Terminalableitungen aus der Grammatik.
    """
    if not start_symbol:
        return {}
    
    generator = PCFGGenerator(grammar, terminals, start_symbol)
    all_terminals = []
    
    # Generiere viele Ketten
    for _ in range(num_samples):
        chain = generator.generate(max_depth)
        all_terminals.extend(chain)
    
    if not all_terminals:
        return {}
    
    # Berechne Häufigkeiten
    freq = Counter(all_terminals)
    total = len(all_terminals)
    
    return {sym: count/total for sym, count in freq.items()}

# Sammle alle ursprünglichen Terminale
all_original_terminals = []
for chain in empirical_chains:
    all_original_terminals.extend(chain)

original_freq = Counter(all_original_terminals)
total_original = len(all_original_terminals)
if total_original > 0:
    original_dist = {sym: count/total_original for sym, count in original_freq.items()}
else:
    original_dist = {}

# Generierte Verteilung
if inducer.start_symbol:
    generated_dist = collect_all_symbols(inducer.rules, inducer.terminals, inducer.start_symbol)
    
    print("\n" + "=" * 70)
    print("VALIDIERUNG: TERMINAL-VERTEILUNGEN")
    print("=" * 70)
    
    table_data = []
    all_symbols = sorted(set(original_dist.keys()) | set(generated_dist.keys()))
    for sym in all_symbols:
        orig = original_dist.get(sym, 0)
        gen = generated_dist.get(sym, 0)
        table_data.append([sym, f"{orig:.4f}", f"{gen:.4f}", f"{abs(orig-gen):.4f}"])
    
    print(tabulate(table_data, 
                   headers=["Symbol", "Empirisch", "Generiert", "Differenz"],
                   tablefmt="grid"))
    
    # Berechne Korrelation, wenn möglich
    orig_array = [original_dist.get(sym, 0) for sym in all_symbols]
    gen_array = [generated_dist.get(sym, 0) for sym in all_symbols]
    try:
        corr, p_value = pearsonr(orig_array, gen_array)
        print(f"\nKorrelation: r = {corr:.4f}, p = {p_value:.4f}")
    except:
        print("\nKorrelation konnte nicht berechnet werden.")

# ============================================================================
# 6. HIERARCHIE-VISUALISIERUNG
# ============================================================================

def print_grammar_hierarchy(grammar, start_symbol, indent=0, visited=None):
    """
    Gibt die Hierarchie der Grammatik als Baum aus.
    """
    if visited is None:
        visited = set()
    
    if not start_symbol or start_symbol not in grammar:
        return
    
    if start_symbol in visited:
        print("  " * indent + f"├─ {start_symbol} (Zyklus!)")
        return
    
    visited.add(start_symbol)
    
    productions = grammar[start_symbol]
    print("  " * indent + f"├─ {start_symbol}")
    
    for prod, prob in productions:
        print("  " * (indent + 1) + f"├─ [{prob:.3f}] -> ", end="")
        prod_str = []
        for sym in prod:
            if sym in grammar:
                prod_str.append(sym)
            else:
                prod_str.append(f"'{sym}'")
        print(" ".join(prod_str))
        
        # Rekursiv für Nonterminale
        for sym in prod:
            if sym in grammar:
                print_grammar_hierarchy(grammar, sym, indent + 2, visited.copy())

print("\n" + "=" * 70)
print("GRAMMATIK-HIERARCHIE")
print("=" * 70)
if inducer.start_symbol:
    print_grammar_hierarchy(inducer.rules, inducer.start_symbol)
else:
    print("Kein Startsymbol definiert.")

# ============================================================================
# 7. EXPORT DER GRAMMATIK
# ============================================================================

def export_pcfg(grammar, terminals, start_symbol, filename="induzierte_pcfg.txt"):
    """
    Exportiert die PCFG im lesbaren Format.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Induzierte probabilistische kontextfreie Grammatik (PCFG)\n")
        f.write(f"# Startsymbol: {start_symbol}\n")
        f.write(f"# Terminale: {', '.join(sorted(terminals))}\n")
        f.write("# Nonterminale: " + ', '.join(sorted(grammar.keys())) + "\n\n")
        
        for nonterminal in sorted(grammar.keys()):
            productions = grammar[nonterminal]
            for prod, prob in productions:
                prod_str = ' '.join(str(s) for s in prod)
                f.write(f"{nonterminal} -> {prod_str} [{prob:.3f}]\n")
    
    print(f"\nGrammatik wurde als '{filename}' exportiert.")

export_pcfg(inducer.rules, inducer.terminals, inducer.start_symbol)

print("\n" + "=" * 70)
print("GRAMMATIKINDUKTION ABGESCHLOSSEN")
print("=" * 70)
