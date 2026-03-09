"""
ARS GUI - Algorithmic Recursive Sequence Analysis with Graphical User Interface
Erweiterte Version mit Petri-Netzen, Bayesschen Netzen und hybrider Integration

Dieses Programm prüft automatisch die Verfügbarkeit aller benötigten Pakete
und installiert fehlende Pakete bei Bedarf nach.
"""

import sys
import subprocess
import importlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# PAKETVERWALTUNG - ALTERNATIVE ZU PKG_RESOURCES
# ============================================================================

def check_and_install_packages():
    """Prüft und installiert fehlende Python-Pakete (ohne pkg_resources)"""
    
    REQUIRED_PACKAGES = [
        'numpy',
        'scipy',
        'matplotlib',
        'hmmlearn',
        'sklearn-crfsuite',
        'sentence-transformers',
        'networkx',
        'torch',
        'seaborn',
        'tabulate'
    ]
    
    print("=" * 70)
    print("ARS 4.0 - PAKETPRÜFUNG")
    print("=" * 70)
    
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        # Paketnamen für importlib anpassen
        import_name = package.replace('-', '_')
        if package == 'sklearn-crfsuite':
            import_name = 'sklearn_crfsuite'
        elif package == 'sentence-transformers':
            import_name = 'sentence_transformers'
        
        try:
            importlib.import_module(import_name)
            print(f"✓ {package} bereits installiert")
        except ImportError:
            print(f"✗ {package} fehlt")
            missing_packages.append(package)
    
    if missing_packages:
        print("\nInstalliere fehlende Pakete...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"  ✓ {package} erfolgreich installiert")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Fehler bei Installation von {package}: {e}")
                print(f"    Bitte manuell installieren: pip install {package}")
    
    print("\n" + "=" * 70 + "\n")

# Pakete prüfen und installieren
check_and_install_packages()

# ============================================================================
# IMPORTS
# ============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')  # Wichtig für Thread-Sicherheit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter, defaultdict
import threading
import time
import re
import queue

# Optionale Imports mit Fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warnung: networkx nicht verfügbar. Graph-Funktionen deaktiviert.")

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warnung: hmmlearn nicht verfügbar. HMM-Funktionen deaktiviert.")

try:
    from sklearn_crfsuite import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("Warnung: sklearn-crfsuite nicht verfügbar. CRF-Funktionen deaktiviert.")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Warnung: sentence-transformers nicht verfügbar. Embedding-Funktionen deaktiviert.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warnung: torch nicht verfügbar. GNN-Funktionen deaktiviert.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warnung: seaborn nicht verfügbar. Visualisierungsfunktionen eingeschränkt.")


# ============================================================================
# THREAD-SICHERE MATPLOTLIB-FUNKTIONEN
# ============================================================================

class PlotThread:
    """Thread-sichere Plot-Ausführung"""
    
    def __init__(self, root):
        self.root = root
        self.plot_queue = queue.Queue()
        self.start_processor()
    
    def start_processor(self):
        """Startet den Plot-Processor"""
        self.process()
    
    def process(self):
        """Verarbeitet Plot-Aufträge im Hauptthread"""
        try:
            while True:
                func, args, kwargs = self.plot_queue.get_nowait()
                # Im Hauptthread ausführen
                self.root.after(0, lambda: self._execute_plot(func, args, kwargs))
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process)
    
    def _execute_plot(self, func, args, kwargs):
        """Führt Plot-Funktion aus"""
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"Fehler im Plot: {e}")
    
    def plot(self, func, *args, **kwargs):
        """Fügt einen Plot-Auftrag hinzu"""
        self.plot_queue.put((func, args, kwargs))


# ============================================================================
# ARS 2.0 - GRAMMATIK OHNE NONTERMINALE
# ============================================================================

class ARS20:
    """ARS 2.0 - Übergangswahrscheinlichkeiten ohne Nonterminale"""
    
    def __init__(self):
        self.chains = []
        self.terminals = []
        self.start_symbol = None
        self.transitions = {}
        self.probabilities = {}
        self.optimized_probabilities = {}
        self.history = []
        
    def load_chains(self, chains, start_symbol=None):
        """Lädt Terminalzeichenketten"""
        self.chains = chains
        # Alle Terminale aus allen Ketten sammeln
        all_terminals = set()
        for chain in chains:
            for symbol in chain:
                all_terminals.add(symbol)
        self.terminals = sorted(list(all_terminals))
        self.start_symbol = start_symbol if start_symbol else (chains[0][0] if chains else None)
        self.transitions = self.count_transitions(chains)
        self.probabilities = self.calculate_probabilities(self.transitions)
        return True
    
    def count_transitions(self, chains):
        """Zählt Übergänge zwischen Terminalzeichen"""
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
    
    def calculate_probabilities(self, transitions):
        """Normalisiert Übergangszaehlungen zu Wahrscheinlichkeiten"""
        probabilities = {}
        for start in transitions:
            total = sum(transitions[start].values())
            if total > 0:
                probabilities[start] = {end: count / total 
                                       for end, count in transitions[start].items()}
        return probabilities
    
    def print_grammar(self):
        """Gibt die Grammatik aus"""
        lines = []
        lines.append("=" * 70)
        lines.append("ARS 2.0 - ÜBERGANGSWAHRSCHEINLICHKEITEN")
        lines.append("=" * 70)
        lines.append("")
        
        if self.probabilities:
            for start in sorted(self.probabilities.keys()):
                trans = self.probabilities[start]
                trans_str = ", ".join([f"{end}: {prob:.3f}" for end, prob in sorted(trans.items())])
                lines.append(f"{start} -> {trans_str}")
        else:
            lines.append("Keine Übergänge gefunden.")
        
        lines.append(f"\nTerminalzeichen ({len(self.terminals)}): {self.terminals}")
        lines.append(f"Startzeichen: {self.start_symbol}")
        
        return "\n".join(lines)
    
    def generate_chain(self, start_symbol=None, max_length=20):
        """Generiert eine Kette basierend auf Wahrscheinlichkeiten"""
        if not self.optimized_probabilities:
            probs = self.probabilities
        else:
            probs = self.optimized_probabilities
            
        start = start_symbol if start_symbol else self.start_symbol
        if not start or start not in probs:
            return []
            
        chain = [start]
        current = start
        
        for _ in range(max_length - 1):
            if current not in probs:
                break
            
            next_symbols = list(probs[current].keys())
            if not next_symbols:
                break
                
            probs_list = list(probs[current].values())
            if not probs_list:
                break
                
            try:
                next_symbol = np.random.choice(next_symbols, p=probs_list)
                chain.append(next_symbol)
                current = next_symbol
            except:
                break
            
            if current not in probs:
                break
        
        return chain
    
    def compute_frequencies(self, chains):
        """Berechnet relative Häufigkeiten der Terminalzeichen"""
        if not self.terminals:
            return np.array([])
        
        freq_array = np.zeros(len(self.terminals))
        term_index = {term: i for i, term in enumerate(self.terminals)}
        
        for chain in chains:
            for symbol in chain:
                if symbol in term_index:
                    freq_array[term_index[symbol]] += 1
        
        total = freq_array.sum()
        if total > 0:
            freq_array /= total
        
        return freq_array
    
    def optimize(self, max_iterations=500, tolerance=0.005, target_correlation=0.9,
                 progress_callback=None):
        """Optimiert die Grammatik durch iterativen Vergleich"""
        
        # Initiale Wahrscheinlichkeiten
        probs = {}
        for start, p in self.probabilities.items():
            probs[start] = p.copy()
            
        empirical_freqs = self.compute_frequencies(self.chains)
        
        best_correlation = 0
        best_probabilities = None
        history = []
        
        for iteration in range(max_iterations):
            # Generiere Ketten
            generated = [self.generate_chain(max_length=20) for _ in range(8)]
            generated = [g for g in generated if g]  # Entferne leere Ketten
            
            if not generated:
                continue
                
            gen_freqs = self.compute_frequencies(generated)
            
            # Korrelation
            try:
                if len(empirical_freqs) == len(gen_freqs) and len(empirical_freqs) > 1:
                    corr, p_val = pearsonr(empirical_freqs, gen_freqs)
                else:
                    corr, p_val = 0, 1
            except:
                corr, p_val = 0, 1
            
            history.append((iteration, corr, p_val))
            
            # Progress update
            if progress_callback and iteration % 10 == 0:
                progress_callback(iteration, max_iterations, corr, p_val)
            
            # Abbruchkriterium
            if corr >= target_correlation and p_val < 0.05:
                best_correlation = corr
                best_probabilities = {s: p.copy() for s, p in probs.items()}
                break
            
            # Anpassung
            for start in probs:
                for end in list(probs[start].keys()):
                    if end in self.terminals:
                        idx = self.terminals.index(end)
                        if idx < len(empirical_freqs) and idx < len(gen_freqs):
                            emp_prob = empirical_freqs[idx]
                            gen_prob = gen_freqs[idx]
                            error = emp_prob - gen_prob
                            
                            probs[start][end] += error * tolerance
                            probs[start][end] = max(0.01, min(0.99, probs[start][end]))
            
            # Renormalisierung
            for start in probs:
                total = sum(probs[start].values())
                if total > 0:
                    probs[start] = {end: p/total for end, p in probs[start].items()}
        
        if best_probabilities is None and history:
            best_idx = max(range(len(history)), key=lambda i: history[i][1])
            best_correlation = history[best_idx][1]
            best_probabilities = self.probabilities
        
        self.optimized_probabilities = best_probabilities
        self.history = history
        
        return best_probabilities, best_correlation, history


# ============================================================================
# ARS 3.0 - GRAMMATIK MIT NONTERMINALEN (HIERARCHISCHE KOMPRESSION)
# VOLLSTÄNDIG KORRIGIERTE VERSION
# ============================================================================

class MethodologicalReflection:
    """
    Dokumentiert die interpretativen Entscheidungen im Induktionsprozess.
    """
    
    def __init__(self):
        self.interpretation_log = []
        self.sequence_meaning_mapping = {}
    
    def log_interpretation(self, sequence, new_nonterminal, rationale):
        """Dokumentiert eine Interpretationsentscheidung"""
        self.interpretation_log.append({
            'sequence': sequence,
            'new_nonterminal': new_nonterminal,
            'rationale': rationale,
            'timestamp': len(self.interpretation_log)
        })
        
        # Bedeutung der Sequenz explizieren
        aktionen = [self._interpretiere_symbol(s) for s in sequence if isinstance(s, str)]
        self.sequence_meaning_mapping[tuple(sequence)] = {
            'bedeutung': ' → '.join(aktionen),
            'typ': self._klassifiziere_sequenz(sequence)
        }
    
    def _interpretiere_symbol(self, symbol):
        """Gibt die qualitative Bedeutung eines Terminalzeichens zurück"""
        bedeutungen = {
            'KBG': 'Kunden-Gruß',
            'VBG': 'Verkäufer-Gruß',
            'KBBd': 'Kunden-Bedarf (konkret)',
            'VBBd': 'Verkäufer-Nachfrage',
            'KBA': 'Kunden-Antwort',
            'VBA': 'Verkäufer-Reaktion',
            'KAE': 'Kunden-Erkundigung',
            'VAE': 'Verkäufer-Auskunft',
            'KAA': 'Kunden-Abschluss',
            'VAA': 'Verkäufer-Abschluss',
            'KAV': 'Kunden-Verabschiedung',
            'VAV': 'Verkäufer-Verabschiedung',
            'KNG': 'Kunden-Gruß (Variante)',
            'VBG.VBBd': 'Verkäufer-Aktion (kombiniert)'
        }
        return bedeutungen.get(symbol, str(symbol))
    
    def _klassifiziere_sequenz(self, sequence):
        """Klassifiziert den Typ der Interaktionssequenz"""
        seq_str = ' '.join([str(s) for s in sequence])
        if 'KBBd' in seq_str and 'VBBd' in seq_str:
            return 'Bedarfsaushandlung'
        elif 'KAE' in seq_str or 'VAE' in seq_str:
            return 'Informationsaustausch'
        elif 'KAA' in seq_str and 'VAA' in seq_str:
            return 'Transaktionsabschluss'
        else:
            return 'Interaktionssequenz'
    
    def print_summary(self):
        """Gibt eine methodologische Zusammenfassung aus"""
        print("\n" + "=" * 70)
        print("METHODOLOGISCHE REFLEXION")
        print("=" * 70)
        print("\nDokumentierte Interpretationsentscheidungen:")
        
        for log in self.interpretation_log:
            print(f"\n[Interpretation {log['timestamp']+1}]")
            seq_str = ' → '.join([str(s) for s in log['sequence']])
            print(f"  Sequenz: {seq_str}")
            print(f"  → Nonterminal: {log['new_nonterminal']}")
            print(f"  Begründung: {log['rationale']}")
            
            if tuple(log['sequence']) in self.sequence_meaning_mapping:
                mapping = self.sequence_meaning_mapping[tuple(log['sequence'])]
                print(f"  Bedeutung: {mapping['bedeutung']}")
                print(f"  Sequenztyp: {mapping['typ']}")


class GrammarInducer:
    """
    Induziert eine PCFG durch hierarchische Kompression von Wiederholungen.
    Wiederholt den Vorgang, bis nur noch ein Startsymbol übrig bleibt.
    """
    
    def __init__(self):
        self.rules = {}          # Nonterminal -> Produktionen
        self.terminals = set()
        self.nonterminals = set()
        self.start_symbol = None
        self.user_start_symbol = None  # Vom Benutzer definiertes Startzeichen
        self.compression_history = []
        self.reflection = MethodologicalReflection()
        self.chains = []
        self.iteration_count = 0
        self.hierarchy_levels = {}  # Speichert die Hierarchieebene jedes Nonterminals
    
    def load_chains(self, chains, user_start_symbol=None):
        """Lädt Terminalzeichenketten und optional ein benutzerdefiniertes Startzeichen"""
        self.chains = [list(chain) for chain in chains]
        self.user_start_symbol = user_start_symbol
        
        # Alle ursprünglichen Terminale sammeln
        all_symbols = set()
        for chain in chains:
            for symbol in chain:
                all_symbols.add(symbol)
        self.terminals = all_symbols
        return True
    
    def find_best_repetition(self, chains, min_length=2, max_length=5):
        """
        Findet die beste wiederholte Sequenz in allen Ketten.
        Berücksichtigt Häufigkeit, Länge und Komplexität.
        """
        sequence_counter = Counter()
        
        for chain in chains:
            max_len = min(max_length, len(chain))
            for length in range(min_length, max_len + 1):
                for i in range(len(chain) - length + 1):
                    seq = tuple(chain[i:i+length])
                    sequence_counter[seq] += 1
        
        # Nur Sequenzen mit mindestens 2 Vorkommen
        repeated = {seq: count for seq, count in sequence_counter.items() 
                   if count >= 2}
        
        if not repeated:
            return None
        
        # Bewertung: (Häufigkeit * Länge) / Anzahl einzigartiger Symbole
        # Bevorzugt längere, häufigere Muster mit weniger Varianz
        best_seq = max(repeated.items(), 
                      key=lambda x: x[1] * len(x[0]) / max(1, len(set(x[0]))))
        
        return best_seq[0]
    
    def generate_nonterminal_name(self, sequence):
        """
        Generiert einen aussagekräftigen Namen für ein neues Nonterminal.
        """
        if all(isinstance(s, str) and s.startswith(('K', 'V')) for s in sequence):
            # Extrahiere erste und letzte Komponente für die Benennung
            first = sequence[0]
            last = sequence[-1]
            # Bestimme den Typ basierend auf den Symbolen
            seq_str = ' '.join([str(s) for s in sequence])
            if 'KBBd' in seq_str and 'VBBd' in seq_str:
                typ = "BEDARFSKLAERUNG"
            elif ('VAA' in seq_str and 'KAA' in seq_str) or ('VAA' in seq_str and 'KAV' in seq_str):
                typ = "ZAHLUNGSVORGANG"
            elif 'KAE' in seq_str or 'VAE' in seq_str:
                typ = "INFORMATIONSAUSTAUSCH"
            elif 'KBG' in seq_str and 'VBG' in seq_str:
                typ = "BEGRUESSUNG"
            elif 'VAV' in seq_str and 'KAV' in seq_str:
                typ = "VERABSCHIEDUNG"
            else:
                typ = "SEQUENZ"
            return f"NT_{typ}_{first}_{last}"
        else:
            # Für gemischte Sequenzen mit bereits vorhandenen Nonterminalen
            return f"NT_{'_'.join(str(s) for s in sequence)}"
    
    def _describe_sequence(self, sequence):
        """Erzeugt eine semantische Beschreibung der Sequenz"""
        if len(sequence) == 2:
            if all(isinstance(s, str) and len(s) <= 4 for s in sequence):
                return f"{self.reflection._interpretiere_symbol(sequence[0])} → {self.reflection._interpretiere_symbol(sequence[1])}"
            else:
                return f"{sequence[0]} → {sequence[1]}"
        else:
            return f"Sequenz mit {len(sequence)} Schritten"
    
    def compress_sequences(self, chains, sequence, new_nonterminal):
        """
        Ersetzt alle Vorkommen der Sequenz durch das neue Nonterminal.
        """
        compressed = []
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
            compressed.append(new_chain)
        
        return compressed
    
    def all_chains_identical(self, chains):
        """Prüft, ob alle Ketten identisch sind (nur ein Symbol)"""
        if not chains:
            return False
        first = chains[0]
        return all(len(chain) == 1 and chain[0] == first[0] for chain in chains)
    
    def find_top_level_nonterminal(self):
        """
        Findet das oberste Nonterminal in der Hierarchie.
        Das ist dasjenige, das niemals als Teil einer anderen Produktion vorkommt.
        """
        if not self.rules:
            return None
        
        # Sammle alle Symbole, die in Produktionen vorkommen
        symbols_in_productions = set()
        for nt, productions in self.rules.items():
            for prod, _ in productions:
                for sym in prod:
                    symbols_in_productions.add(sym)
        
        # Nonterminale, die niemals als Teil einer anderen Produktion vorkommen
        # sind die obersten in der Hierarchie
        top_level = [nt for nt in self.rules if nt not in symbols_in_productions]
        
        if top_level:
            # Wenn es mehrere gibt, nimm das mit der höchsten Hierarchieebene (späteste Iteration)
            if len(top_level) > 1:
                top_level.sort(key=lambda nt: self.hierarchy_levels.get(nt, 0), reverse=True)
            selected = top_level[0]
            return selected
        
        # Fallback: nimm das Nonterminal mit der höchsten Hierarchieebene
        if self.hierarchy_levels:
            selected = max(self.hierarchy_levels.items(), key=lambda x: x[1])[0]
            return selected
        
        # Letzter Fallback: nimm das erste Nonterminal
        return list(self.rules.keys())[0] if self.rules else None
    
    def induce_grammar(self, max_iterations=50, progress_callback=None):
        """
        Induziert Grammatik durch hierarchische Kompression.
        Wiederholt den Vorgang, bis nur noch ein Startsymbol übrig bleibt
        oder keine weiteren Wiederholungen gefunden werden.
        """
        
        current_chains = [list(chain) for chain in self.chains]
        iteration = 0
        rule_counter = 1
        
        self.rules = {}
        self.nonterminals = set()
        self.compression_history = []
        self.iteration_count = 0
        self.hierarchy_levels = {}
        
        print("\n" + "=" * 70)
        print("HIERARCHISCHE GRAMMATIKINDUKTION")
        print("=" * 70)
        print("\nDer Induktionsprozess wird als EXPLIKATION verstanden:")
        print("- Jedes neue Nonterminal repräsentiert eine INTERPRETATIVE KATEGORIE")
        print("- Die Benennung expliziert die qualitative Bedeutung")
        print("- Der Prozess wird wiederholt, bis nur noch EIN Symbol übrig bleibt")
        print("- Dieses Symbol wird zum STARTSYMBOL der Grammatik\n")
        
        while iteration < max_iterations:
            best_seq = self.find_best_repetition(current_chains)
            
            if best_seq is None:
                print(f"\nKeine weiteren Wiederholungen nach {iteration} Iterationen gefunden.")
                break
            
            # Generiere interpretativen Namen
            new_nonterminal = self.generate_nonterminal_name(best_seq)
            beschreibung = self._describe_sequence(best_seq)
            
            # Stelle Einzigartigkeit sicher
            base_name = new_nonterminal
            while new_nonterminal in self.nonterminals:
                new_nonterminal = f"{base_name}_{rule_counter}"
                rule_counter += 1
            
            # Dokumentiere die interpretative Entscheidung
            rationale = f"Erkanntes Dialogmuster: {beschreibung}"
            self.reflection.log_interpretation(best_seq, new_nonterminal, rationale)
            
            seq_str = ' → '.join([str(s) for s in best_seq])
            print(f"\nIteration {iteration + 1}:")
            print(f"  Erkanntes Muster: {seq_str}")
            print(f"  Interpretation: {beschreibung}")
            print(f"  → Neue Kategorie: {new_nonterminal}")
            
            # Speichere die Regel (vorerst ohne Wahrscheinlichkeit)
            self.rules[new_nonterminal] = [(list(best_seq), 1.0)]
            self.nonterminals.add(new_nonterminal)
            self.hierarchy_levels[new_nonterminal] = iteration  # Speichere die Hierarchieebene
            
            # Vorkommen zählen für die Dokumentation
            occurrences = 0
            for chain in current_chains:
                for i in range(len(chain) - len(best_seq) + 1):
                    if tuple(chain[i:i+len(best_seq)]) == best_seq:
                        occurrences += 1
            
            self.compression_history.append({
                'iteration': iteration,
                'sequence': best_seq,
                'new_symbol': new_nonterminal,
                'occurrences': occurrences
            })
            
            # Komprimiere alle Ketten
            current_chains = self.compress_sequences(current_chains, best_seq, new_nonterminal)
            
            # Zeige ein Beispiel der komprimierten Kette
            if current_chains and current_chains[0]:
                example = ' → '.join([str(s) for s in current_chains[0][:10]])
                print(f"  Beispiel (komprimiert): {example}...")
            
            iteration += 1
            self.iteration_count = iteration
            
            # Prüfe auf vollständige Kompression - alle Ketten sind zu einem Symbol komprimiert
            if self.all_chains_identical(current_chains):
                # Alle Ketten sind zu einem Symbol komprimiert
                if current_chains and current_chains[0]:
                    unique_symbol = current_chains[0][0]
                    
                    # Prüfe, ob das benutzerdefinierte Startzeichen verwendet werden kann
                    if self.user_start_symbol and self.user_start_symbol in self.rules:
                        self.start_symbol = self.user_start_symbol
                        print(f"\nINDUKTION ABGESCHLOSSEN: Benutzerdefiniertes Startsymbol = {self.start_symbol}")
                    elif unique_symbol in self.rules:
                        self.start_symbol = unique_symbol
                        print(f"\nINDUKTION ABGESCHLOSSEN: Komprimiertes Startsymbol = {self.start_symbol}")
                    else:
                        # Finde das oberste Nonterminal
                        self.start_symbol = self.find_top_level_nonterminal()
                        print(f"\nINDUKTION ABGESCHLOSSEN: Oberstes Nonterminal als Startsymbol = {self.start_symbol}")
                    break
        
        # Falls keine vollständige Kompression erreicht wurde, bestimme ein Startsymbol
        if self.start_symbol is None:
            if self.user_start_symbol and self.user_start_symbol in self.rules:
                self.start_symbol = self.user_start_symbol
                print(f"\nKeine vollständige Kompression erreicht. Benutzerdefiniertes Startsymbol: {self.start_symbol}")
            elif self.rules:
                # Finde das oberste Nonterminal in der Hierarchie
                self.start_symbol = self.find_top_level_nonterminal()
                print(f"\nKeine vollständige Kompression erreicht. Oberstes Nonterminal als Startsymbol: {self.start_symbol}")
            else:
                print("\nWARNUNG: Keine Grammatik induziert!")
                return current_chains
        
        # Terminale sind die ursprünglichen Symbole, die nie ersetzt wurden
        all_symbols = set()
        for chain in self.chains:
            for sym in chain:
                all_symbols.add(sym)
        
        # Symbole, die nie als Nonterminale eingeführt wurden, sind Terminale
        self.terminals = all_symbols - self.nonterminals
        
        # Wahrscheinlichkeiten berechnen
        self._calculate_probabilities()
        
        print(f"\nStartsymbol: {self.start_symbol}")
        
        return current_chains
    
    def _calculate_probabilities(self):
        """Berechnet Wahrscheinlichkeiten für jede Produktion basierend auf Häufigkeiten"""
        # Zähle, wie oft jedes Nonterminal expandiert wird
        expansion_counts = defaultdict(Counter)
        
        # Rekonstruiere die Expansionshierarchie aus den Originalketten
        for chain in self.chains:
            self._count_expansions(chain, expansion_counts)
        
        # Konvertiere zu Wahrscheinlichkeiten
        for nonterminal in self.rules:
            if nonterminal in expansion_counts:
                total = sum(expansion_counts[nonterminal].values())
                if total > 0:
                    productions = []
                    for expansion, count in expansion_counts[nonterminal].items():
                        productions.append((list(expansion), count / total))
                    # Sortiere nach Wahrscheinlichkeit (absteigend)
                    productions.sort(key=lambda x: x[1], reverse=True)
                    self.rules[nonterminal] = productions
            # Falls keine Vorkommen gefunden wurden, behalte die initiale Produktion mit Wahrscheinlichkeit 1.0
    
    def _count_expansions(self, sequence, expansion_counts):
        """Rekursive Hilfsfunktion zum Zählen der Expansionen"""
        i = 0
        while i < len(sequence):
            symbol = sequence[i]
            
            # Wenn das Symbol ein Nonterminal ist, zähle seine Expansion
            if symbol in self.rules:
                # Finde die längste passende Expansion
                found = False
                for expansion, _ in self.rules[symbol]:
                    exp_len = len(expansion)
                    if i + exp_len <= len(sequence) and sequence[i:i+exp_len] == expansion:
                        expansion_counts[symbol][tuple(expansion)] += 1
                        self._count_expansions(expansion, expansion_counts)
                        i += exp_len
                        found = True
                        break
                if not found:
                    i += 1
            else:
                i += 1
    
    def print_grammar(self):
        """Gibt die vollständige induzierte Grammatik aus"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("INDUZIERTE GRAMMATIK")
        lines.append("=" * 70)
        lines.append(f"\nTerminale ({len(self.terminals)}): {sorted(self.terminals)}")
        lines.append(f"Nonterminale ({len(self.nonterminals)}): {sorted(self.nonterminals)}")
        lines.append(f"Startsymbol: {self.start_symbol}")
        lines.append(f"Iterationen: {self.iteration_count}")
        
        lines.append("\nPRODUKTIONSREGELN (mit Wahrscheinlichkeiten):")
        for nonterminal in sorted(self.rules.keys()):
            productions = self.rules[nonterminal]
            if productions:
                prod_str = " | ".join([f"{' → '.join(prod)} [{prob:.3f}]" 
                                      for prod, prob in productions])
                lines.append(f"\n{nonterminal} → {prod_str}")
        
        # Zeige die Kompressionshierarchie
        if self.compression_history:
            lines.append("\n\nKOMPRESSIONSHISTORIE:")
            for entry in self.compression_history:
                seq_str = ' → '.join([str(s) for s in entry['sequence']])
                lines.append(f"  Iteration {entry['iteration']+1}: {seq_str} → {entry['new_symbol']} ({entry['occurrences']} Vorkommen)")
        
        return "\n".join(lines)
    
    def generate_chain(self, start_symbol=None, max_depth=20):
        """
        Generiert eine neue Kette mit der induzierten Grammatik.
        Beginnt beim Startsymbol und expandiert rekursiv.
        """
        if not start_symbol:
            start_symbol = self.start_symbol
        
        if not start_symbol:
            return []
        
        # Stelle sicher, dass das Startsymbol in den Regeln existiert
        if start_symbol not in self.rules:
            if self.rules:
                # Versuche das oberste Nonterminal
                start_symbol = self.find_top_level_nonterminal()
            else:
                return []
        
        # Produktionswahrscheinlichkeiten vorbereiten
        prod_probs = {}
        for nt, prods in self.rules.items():
            symbols = [p for p, _ in prods]
            probs = [prob for _, prob in prods]
            if symbols and probs:
                total = sum(probs)
                if total > 0:
                    probs = [p/total for p in probs]
                prod_probs[nt] = (symbols, probs)
        
        def expand(symbol, depth=0):
            """Rekursive Expansion eines Symbols"""
            if depth >= max_depth:
                return [str(symbol)]  # Schutz vor unendlicher Rekursion
            
            # Wenn es ein Terminal ist, gib es zurück
            if symbol in self.terminals:
                return [str(symbol)]
            
            # Wenn es ein Nonterminal mit Produktionen ist
            if symbol in prod_probs:
                symbols, probs = prod_probs[symbol]
                if not symbols:
                    return [str(symbol)]
                
                try:
                    # Wähle eine Produktion basierend auf Wahrscheinlichkeiten
                    chosen_idx = np.random.choice(len(symbols), p=probs)
                    chosen = symbols[chosen_idx]
                except Exception:
                    # Fallback bei Fehlern
                    chosen = symbols[0] if symbols else []
                
                # Expandiere jedes Symbol der gewählten Produktion
                result = []
                for sym in chosen:
                    result.extend(expand(sym, depth + 1))
                return result
            
            # Fallback
            return [str(symbol)]
        
        return expand(start_symbol)
    
    def get_compression_tree(self, symbol=None, depth=0):
        """
        Gibt den Kompressionsbaum als String zurück (für Debugging/Visualisierung).
        """
        if symbol is None:
            symbol = self.start_symbol
        
        if symbol is None:
            return "Kein Startsymbol definiert"
        
        if symbol in self.terminals:
            return "  " * depth + f"└─ {symbol} (Terminal)"
        
        lines = []
        lines.append("  " * depth + f"├─ {symbol}")
        
        if symbol in self.rules:
            productions = self.rules[symbol]
            for i, (prod, prob) in enumerate(productions):
                prefix = "  " * (depth + 1) + "├─ " if i < len(productions) - 1 else "  " * (depth + 1) + "└─ "
                lines.append(prefix + f"[{prob:.3f}] ->")
                for sym in prod:
                    if sym in self.rules:
                        # Rekursiv für Nonterminale
                        subtree = self.get_compression_tree(sym, depth + 2)
                        # Die erste Zeile des Subtree anpassen
                        subtree_lines = subtree.split('\n')
                        for j, line in enumerate(subtree_lines):
                            if j == 0:
                                lines.append(line)
                            else:
                                lines.append(line)
                    else:
                        lines.append("  " * (depth + 2) + f"└─ {sym}")
        
        return "\n".join(lines)


# ============================================================================
# PETRI-NETZE (ARS 4.0 - SZENARIO A)
# ============================================================================

if NETWORKX_AVAILABLE:
    class ARSPetriNet:
        """
        Petri-Netz-Modell für ARS 4.0
        """
        
        def __init__(self, name="ARS_PetriNet"):
            self.name = name
            self.places = {}  # Stellen: name -> Place-Objekt
            self.transitions = {}  # Transitionen: name -> Transition-Objekt
            self.arcs = []  # Kanten: (source, target, weight)
            self.tokens = {}  # Marken: place_name -> Anzahl
            self.hierarchy = {}  # Hierarchie: transition_name -> subnet
            
            # Statistik
            self.firing_history = []
            self.reached_markings = set()
        
        def add_place(self, name, initial_tokens=0, place_type="normal"):
            """
            Fügt eine Stelle hinzu
            place_type: "normal", "resource", "phase", "customer", "seller"
            """
            self.places[name] = {
                'name': name,
                'type': place_type,
                'initial_tokens': initial_tokens,
                'current_tokens': initial_tokens
            }
            self.tokens[name] = initial_tokens
        
        def add_transition(self, name, transition_type="speech_act", 
                           guard=None, subnet=None):
            """
            Fügt eine Transition hinzu
            transition_type: "speech_act", "abstract", "silent"
            guard: Bedingungsfunktion (optional)
            subnet: Subnetz für hierarchische Transitionen
            """
            self.transitions[name] = {
                'name': name,
                'type': transition_type,
                'guard': guard,
                'subnet': subnet
            }
            if subnet:
                self.hierarchy[name] = subnet
        
        def add_arc(self, source, target, weight=1):
            """
            Fügt eine Kante hinzu (source -> target)
            source/target können Stellen oder Transitionen sein
            """
            self.arcs.append({
                'source': source,
                'target': target,
                'weight': weight
            })
        
        def get_preset(self, transition):
            """Gibt die Vorstellen einer Transition zurück"""
            preset = {}
            for arc in self.arcs:
                if arc['target'] == transition and arc['source'] in self.places:
                    preset[arc['source']] = arc['weight']
            return preset
        
        def get_postset(self, transition):
            """Gibt die Nachstellen einer Transition zurück"""
            postset = {}
            for arc in self.arcs:
                if arc['source'] == transition and arc['target'] in self.places:
                    postset[arc['target']] = arc['weight']
            return postset
        
        def is_enabled(self, transition):
            """Prüft, ob eine Transition aktiviert ist"""
            if transition not in self.transitions:
                return False
            
            # Prüfe Vorstellen
            preset = self.get_preset(transition)
            for place, weight in preset.items():
                if self.tokens.get(place, 0) < weight:
                    return False
            
            # Prüfe Guard-Bedingung
            trans_data = self.transitions[transition]
            if trans_data['guard'] and not trans_data['guard'](self):
                return False
            
            return True
        
        def fire(self, transition):
            """Schaltet eine Transition"""
            if not self.is_enabled(transition):
                return False
            
            # Entferne Token von Vorstellen
            preset = self.get_preset(transition)
            for place, weight in preset.items():
                self.tokens[place] -= weight
            
            # Füge Token zu Nachstellen hinzu
            postset = self.get_postset(transition)
            for place, weight in postset.items():
                self.tokens[place] = self.tokens.get(place, 0) + weight
            
            # Protokolliere Schaltvorgang
            self.firing_history.append({
                'transition': transition,
                'marking': self.get_marking_copy()
            })
            
            # Speichere erreichte Markierung
            self.reached_markings.add(self.get_marking_tuple())
            
            return True
        
        def get_marking_copy(self):
            """Gibt eine Kopie der aktuellen Markierung zurück"""
            return self.tokens.copy()
        
        def get_marking_tuple(self):
            """Gibt die Markierung als sortiertes Tupel zurück (für Hash-Set)"""
            return tuple(sorted([(p, self.tokens[p]) for p in self.places]))
        
        def reset(self):
            """Setzt das Netz in den Anfangszustand zurück"""
            for place_name, place_data in self.places.items():
                self.tokens[place_name] = place_data['initial_tokens']
            self.firing_history = []
        
        def simulate(self, transition_sequence):
            """
            Simuliert eine Sequenz von Transitionen
            Gibt Erfolg und letzte Markierung zurück
            """
            self.reset()
            successful = []
            
            for t in transition_sequence:
                if self.is_enabled(t):
                    self.fire(t)
                    successful.append(t)
                else:
                    break
            
            return successful, self.get_marking_copy()

    class PetriNetBuilder:
        """
        Baut Petri-Netze aus ARS-Daten
        """
        
        def __init__(self, terminal_chains, grammar_rules=None):
            self.chains = terminal_chains
            self.grammar = grammar_rules
            self.petri_net = None
            
        def build_basic_net(self):
            """Erstellt ein einfaches Petri-Netz ohne Ressourcen"""
            self.petri_net = ARSPetriNet("ARS_PetriNet_Basic")
            
            # Alle Terminalzeichen als Transitionen
            all_symbols = set()
            for chain in self.chains:
                for sym in chain:
                    all_symbols.add(sym)
            
            # Stellen für Sequenzpositionen
            self.petri_net.add_place("p_start", initial_tokens=1)
            self.petri_net.add_place("p_end", initial_tokens=0)
            
            for i, sym in enumerate(sorted(all_symbols)):
                self.petri_net.add_place(f"p_{sym}_ready", initial_tokens=0)
                self.petri_net.add_transition(f"t_{sym}")
                
                # Verbindungen
                if i == 0:
                    self.petri_net.add_arc("p_start", f"t_{sym}")
                self.petri_net.add_arc(f"t_{sym}", f"p_{sym}_ready")
            
            return self.petri_net
        
        def build_resource_net(self):
            """Erstellt ein Petri-Netz mit Ressourcen"""
            self.petri_net = ARSPetriNet("ARS_PetriNet_Resource")
            
            # Kunde und Verkäufer als Ressourcen
            self.petri_net.add_place("p_customer_present", initial_tokens=1, place_type="customer")
            self.petri_net.add_place("p_customer_ready", initial_tokens=1, place_type="customer")
            self.petri_net.add_place("p_seller_ready", initial_tokens=1, place_type="seller")
            
            # Waren und Geld
            self.petri_net.add_place("p_goods_available", initial_tokens=10, place_type="resource")
            self.petri_net.add_place("p_goods_selected", initial_tokens=0, place_type="resource")
            self.petri_net.add_place("p_money_customer", initial_tokens=20, place_type="resource")
            self.petri_net.add_place("p_money_register", initial_tokens=0, place_type="resource")
            
            # Phasen
            phases = ["Greeting", "Need", "Consult", "Completion", "Farewell"]
            for phase in phases:
                self.petri_net.add_place(f"p_phase_{phase}", initial_tokens=0, place_type="phase")
            self.petri_net.add_place("p_phase_start", initial_tokens=1, place_type="phase")
            
            # Alle Terminalzeichen als Transitionen mit Ressourcen-Anbindung
            all_symbols = set()
            for chain in self.chains:
                for sym in chain:
                    all_symbols.add(sym)
            
            for sym in sorted(all_symbols):
                self.petri_net.add_transition(f"t_{sym}")
                
                # Grundlegende Verbindungen
                if sym.startswith('K'):
                    self.petri_net.add_arc("p_customer_ready", f"t_{sym}")
                    self.petri_net.add_arc(f"t_{sym}", "p_customer_ready")
                else:
                    self.petri_net.add_arc("p_seller_ready", f"t_{sym}")
                    self.petri_net.add_arc(f"t_{sym}", "p_seller_ready")
                
                # Spezielle Verbindungen je nach Symboltyp
                if sym.endswith('A'):  # Abschluss-Symbole
                    self.petri_net.add_arc("p_goods_selected", f"t_{sym}")
                    self.petri_net.add_arc("p_money_customer", f"t_{sym}")
                    self.petri_net.add_arc(f"t_{sym}", "p_goods_available")
                    self.petri_net.add_arc(f"t_{sym}", "p_money_register")
            
            return self.petri_net
        
        def simulate_chain(self, chain):
            """Simuliert eine Kette im Petri-Netz"""
            if not self.petri_net:
                self.build_basic_net()
            
            self.petri_net.reset()
            results = []
            
            for sym in chain:
                trans_name = f"t_{sym}"
                if trans_name in self.petri_net.transitions:
                    enabled = self.petri_net.is_enabled(trans_name)
                    if enabled:
                        self.petri_net.fire(trans_name)
                        results.append((sym, True, "enabled"))
                    else:
                        results.append((sym, False, "not enabled"))
                else:
                    results.append((sym, False, "no transition"))
            
            return results, self.petri_net.get_marking_copy()
else:
    class ARSPetriNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("networkx nicht installiert")
    
    class PetriNetBuilder:
        def __init__(self, *args, **kwargs):
            raise ImportError("networkx nicht installiert")


# ============================================================================
# BAYESSCHE NETZE (ARS 4.0 - SZENARIO B)
# ============================================================================

if HMM_AVAILABLE:
    class ARSHiddenMarkovModel:
        """
        Hidden-Markov-Modell für ARS 4.0
        Korrigierte Version für hmmlearn
        """
        
        def __init__(self, n_states=5):
            self.n_states = n_states
            self.model = None
            self.symbol_to_idx = {}
            self.idx_to_symbol = {}
            self.state_names = {
                0: "Greeting",
                1: "Need Determination",
                2: "Consultation",
                3: "Completion",
                4: "Farewell"
            }
            self.n_features = None
            
        def prepare_data(self, chains):
            """Bereitet Daten für HMM vor"""
            # Symbol-Mapping erstellen
            all_symbols = set()
            for chain in chains:
                for sym in chain:
                    all_symbols.add(sym)
            
            # Stelle sicher, dass alle Symbole Strings sind und keine None-Werte
            all_symbols = {str(s) for s in all_symbols if s is not None}
            
            self.symbol_to_idx = {sym: i for i, sym in enumerate(sorted(all_symbols))}
            self.idx_to_symbol = {i: sym for sym, i in self.symbol_to_idx.items()}
            self.n_features = len(all_symbols)
            
            # Daten in Sequenzen konvertieren
            X = []
            lengths = []
            
            for chain in chains:
                # Stelle sicher, dass jedes Symbol im chain existiert
                seq = []
                for sym in chain:
                    if sym in self.symbol_to_idx:
                        seq.append(self.symbol_to_idx[sym])
                    else:
                        # Fallback: überspringe unbekannte Symbole
                        continue
                
                if seq:  # Nur nicht-leere Sequenzen hinzufügen
                    X.extend(seq)
                    lengths.append(len(seq))
            
            if not X:  # Falls keine Daten vorhanden
                return np.array([]).reshape(-1, 1), np.array([])
            
            return np.array(X).reshape(-1, 1), np.array(lengths)
        
        def initialize_from_ars(self, chains):
            """Initialisiert HMM-Parameter aus ARS-Daten"""
            print("\n=== Initialisiere HMM aus ARS-3.0-Daten ===")
            
            # Zuerst prepare_data aufrufen, um Mapping zu erstellen
            X, lengths = self.prepare_data(chains)
            
            if len(X) == 0:
                print("Warnung: Keine Daten für HMM-Initialisierung")
                return None
            
            # 1. Startwahrscheinlichkeiten
            startprob = np.zeros(self.n_states)
            startprob[0] = 0.7  # Greeting
            startprob[1] = 0.2  # Need Determination
            startprob[4] = 0.1  # Farewell
            
            # 2. Übergangsmatrix
            transmat = np.zeros((self.n_states, self.n_states))
            transmat[0, 1] = 0.8
            transmat[0, 0] = 0.2
            transmat[1, 2] = 0.6
            transmat[1, 3] = 0.3
            transmat[1, 1] = 0.1
            transmat[2, 3] = 0.5
            transmat[2, 2] = 0.4
            transmat[2, 1] = 0.1
            transmat[3, 4] = 0.9
            transmat[3, 3] = 0.1
            transmat[4, 4] = 1.0
            
            # 3. Emissionswahrscheinlichkeiten (gleichverteilt initial)
            emissionprob = np.ones((self.n_states, self.n_features)) / self.n_features
            
            # HMM erstellen
            self.model = hmm.MultinomialHMM(
                n_components=self.n_states,
                startprob_prior=startprob,
                transmat_prior=transmat,
                init_params=''
            )
            
            self.model.startprob_ = startprob
            self.model.transmat_ = transmat
            self.model.emissionprob_ = emissionprob
            
            print(f"HMM initialisiert: {self.n_states} Zustände, {self.n_features} Symbole")
            self.print_parameters()
            
            return self.model
        
        def fit(self, chains, n_iter=100):
            """Trainiert das HMM mit Baum-Welch"""
            X, lengths = self.prepare_data(chains)
            
            if len(X) == 0:
                raise ValueError("Keine Daten zum Trainieren vorhanden")
            
            print(f"\n=== Trainiere HMM mit {len(chains)} Sequenzen ===")
            print(f"Gesamtlänge: {len(X)} Beobachtungen")
            
            if self.model is None:
                self.model = hmm.MultinomialHMM(
                    n_components=self.n_states,
                    n_iter=n_iter,
                    random_state=42
                )
            
            self.model.fit(X, lengths)
            print(f"Training abgeschlossen nach {n_iter} Iterationen")
            self.print_parameters()
            
            return self.model
        
        def print_parameters(self):
            """Gibt die Modellparameter aus"""
            if self.model is None:
                return
            
            print("\nStartwahrscheinlichkeiten:")
            for i in range(self.n_states):
                print(f"  {self.state_names[i]}: {self.model.startprob_[i]:.3f}")
            
            print("\nÜbergangsmatrix:")
            for i in range(self.n_states):
                row = "  " + " ".join([f"{self.model.transmat_[i,j]:.3f}" 
                                       for j in range(self.n_states)])
                print(f"{self.state_names[i]}: {row}")
        
        def decode(self, chain):
            """Viterbi-Dekodierung einer Kette"""
            if self.model is None:
                return None, None
            
            # Konvertiere chain in Indizes
            X_list = []
            for sym in chain:
                if sym in self.symbol_to_idx:
                    X_list.append(self.symbol_to_idx[sym])
                else:
                    # Fallback: überspringe unbekannte Symbole
                    continue
            
            if not X_list:
                return None, None
            
            X = np.array(X_list).reshape(-1, 1)
            
            try:
                logprob, states = self.model.decode(X, algorithm="viterbi")
                return states, np.exp(logprob)
            except:
                return None, None
        
        def get_parameters_string(self):
            """Gibt die HMM-Parameter als String zurück"""
            if self.model is None:
                return "Kein HMM trainiert"
            
            lines = []
            lines.append("Startwahrscheinlichkeiten:")
            for i in range(self.n_states):
                lines.append(f"  {self.state_names[i]}: {self.model.startprob_[i]:.3f}")
            
            lines.append("\nÜbergangsmatrix:")
            for i in range(self.n_states):
                row = "  " + " ".join([f"{self.model.transmat_[i,j]:.3f}" 
                                       for j in range(self.n_states)])
                lines.append(f"{self.state_names[i]}: {row}")
            
            return '\n'.join(lines)
else:
    class ARSHiddenMarkovModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("hmmlearn nicht installiert")


# ============================================================================
# HYBRIDE INTEGRATION (ARS 4.0 - SZENARIO D2)
# ============================================================================

if CRF_AVAILABLE:
    class ARSCRFModel:
        """
        CRF-Modell für sequenzielle Abhängigkeiten
        """
        
        def __init__(self):
            self.crf = CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
            )
            
        def extract_features(self, sequence, i):
            """Extrahiert Features für Position i"""
            features = {
                'bias': 1.0,
                'symbol': sequence[i],
                'prefix_K': sequence[i].startswith('K'),
                'prefix_V': sequence[i].startswith('V'),
                'suffix_A': sequence[i].endswith('A'),
                'suffix_B': sequence[i].endswith('B'),
                'suffix_E': sequence[i].endswith('E'),
                'suffix_G': sequence[i].endswith('G'),
                'suffix_V': sequence[i].endswith('V'),
                'position': i,
                'is_first': i == 0,
                'is_last': i == len(sequence) - 1,
            }
            
            # Kontext-Features
            for offset in [-2, -1, 1, 2]:
                if 0 <= i + offset < len(sequence):
                    sym = sequence[i + offset]
                    features[f'context_{offset:+d}'] = sym
            
            if i > 0:
                features['bigram'] = f"{sequence[i-1]}_{sequence[i]}"
            
            return features
        
        def prepare_data(self, sequences):
            """Bereitet Daten für CRF-Training vor"""
            X = []
            y = []
            
            for seq in sequences:
                X_seq = [self.extract_features(seq, i) for i in range(len(seq))]
                y_seq = [sym for sym in seq]
                X.append(X_seq)
                y.append(y_seq)
            
            return X, y
        
        def fit(self, sequences):
            """Trainiert das CRF-Modell"""
            X, y = self.prepare_data(sequences)
            self.crf.fit(X, y)
            return self
        
        def predict(self, sequence):
            """Sagt Labels für eine Sequenz vorher"""
            X = [self.extract_features(sequence, i) for i in range(len(sequence))]
            return self.crf.predict([X])[0]
        
        def get_top_features(self, n=20):
            """Gibt die wichtigsten Features zurück"""
            if not hasattr(self.crf, 'state_features_'):
                return []
            
            top = sorted(
                self.crf.state_features_.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:n]
            
            return [(attr, label, weight) for (attr, label), weight in top]
else:
    class ARSCRFModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("sklearn-crfsuite nicht installiert")


if TRANSFORMER_AVAILABLE:
    class SemanticValidator:
        """
        Validiert Kategorien mit Transformer-Embeddings
        """
        
        def __init__(self):
            self.model = None
            self.embeddings = {}
            self.symbol_to_texts = self._create_text_mapping()
            
        def _create_text_mapping(self):
            """Erstellt Beispieltexte für Symbole"""
            return {
                'KBG': ['Good day', 'Good morning', 'Hello'],
                'VBG': ['Good day', 'Good morning', 'Hello back'],
                'KBBd': ['One sausage', 'I would like cheese', 'One kilo apples'],
                'VBBd': ['How much?', 'Which kind?', 'Anything else?'],
                'KBA': ['Two hundred grams', 'The white ones', 'Yes please'],
                'VBA': ['All right', 'Coming up', 'Okay'],
                'KAE': ['Can I put in salad?', 'Where from?', 'Is it fresh?'],
                'VAE': ['Better to saute', 'From region', 'Very fresh'],
                'KAA': ['Here you go', 'Thanks', 'Yes thanks'],
                'VAA': ['That will be 8 marks', '3 marks', '14 marks'],
                'KAV': ['Goodbye', 'Bye', 'Have a nice day'],
                'VAV': ['Thank you', 'Have a nice day', 'Goodbye']
            }
        
        def load_model(self):
            """Lädt das Sentence-Transformer-Modell"""
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                return True
            except:
                return False
        
        def compute_embeddings(self):
            """Berechnet Embeddings für alle Symbole"""
            if self.model is None:
                if not self.load_model():
                    return False
            
            for symbol, texts in self.symbol_to_texts.items():
                embeddings = self.model.encode(texts)
                self.embeddings[symbol] = np.mean(embeddings, axis=0)
            
            return True
        
        def similarity_matrix(self):
            """Berechnet Ähnlichkeitsmatrix zwischen Symbolen"""
            if not self.embeddings:
                if not self.compute_embeddings():
                    return None, None
            
            symbols = sorted(self.embeddings.keys())
            n = len(symbols)
            matrix = np.zeros((n, n))
            
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    emb1 = self.embeddings[sym1]
                    emb2 = self.embeddings[sym2]
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    matrix[i, j] = sim
            
            return matrix, symbols
        
        def get_intra_similarities(self):
            """Gibt Intra-Kategorie-Ähnlichkeiten zurück"""
            matrix, symbols = self.similarity_matrix()
            if matrix is None:
                return {}
            
            return {sym: matrix[i, i] for i, sym in enumerate(symbols)}
else:
    class SemanticValidator:
        def __init__(self, *args, **kwargs):
            raise ImportError("sentence-transformers nicht installiert")


if NETWORKX_AVAILABLE:
    class GrammarGraph:
        """
        Repräsentiert Grammatik als Graph
        """
        
        def __init__(self, grammar_rules):
            self.grammar = grammar_rules
            self.graph = nx.DiGraph()
            self.build_graph()
        
        def build_graph(self):
            """Baut Graphen aus Grammatik"""
            for nt, productions in self.grammar.items():
                for prod, prob in productions:
                    for sym in prod:
                        self.graph.add_edge(nt, sym, weight=prob)
        
        def centrality(self):
            """Berechnet Zentralität der Knoten"""
            return nx.degree_centrality(self.graph)
else:
    class GrammarGraph:
        def __init__(self, *args, **kwargs):
            raise ImportError("networkx nicht installiert")


class AttentionVisualizer:
    """
    Visualisiert Attention auf Sequenzen
    """
    
    def __init__(self, chains):
        self.chains = chains
        self.bigram_probs = self._compute_bigram_probs()
    
    def _compute_bigram_probs(self):
        """Berechnet Bigram-Wahrscheinlichkeiten"""
        bigram_counts = defaultdict(int)
        unigram_counts = defaultdict(int)
        
        for chain in self.chains:
            for i in range(len(chain)-1):
                bigram_counts[(chain[i], chain[i+1])] += 1
                unigram_counts[chain[i]] += 1
            if chain:
                unigram_counts[chain[-1]] += 1
        
        probs = {}
        for (prev, next_), count in bigram_counts.items():
            if unigram_counts[prev] > 0:
                probs[(prev, next_)] = count / unigram_counts[prev]
        
        return probs
    
    def attention_weights(self, sequence):
        """Berechnet vereinfachte Attention-Gewichte"""
        n = len(sequence)
        attention = np.zeros((n, n))
        
        for i in range(1, n):
            prev = sequence[i-1]
            current = sequence[i]
            
            if (prev, current) in self.bigram_probs:
                attention[i, i-1] = self.bigram_probs[(prev, current)]
            
            for j in range(i-2, -1, -1):
                attention[i, j] = attention[i, j+1] * 0.5
        
        for i in range(n):
            row_sum = attention[i].sum()
            if row_sum > 0:
                attention[i] /= row_sum
        
        return attention


# ============================================================================
# PLOT-FUNKTIONEN (für Thread-sichere Ausführung)
# ============================================================================

def plot_petri_net(petri_net, filename="petri_net.png"):
    """Plottet ein Petri-Netz"""
    if not NETWORKX_AVAILABLE:
        print("networkx nicht verfügbar")
        return
    
    G = nx.DiGraph()
    
    # Füge Stellen hinzu (Kreise)
    for place in petri_net.places:
        G.add_node(place, type='place', shape='circle')
    
    # Füge Transitionen hinzu (Rechtecke)
    for trans in petri_net.transitions:
        G.add_node(trans, type='transition', shape='box')
    
    # Füge Kanten hinzu
    for arc in petri_net.arcs:
        G.add_edge(arc['source'], arc['target'], weight=arc['weight'])
    
    # Layout
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(15, 10))
    
    # Zeichne Stellen
    place_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'place']
    nx.draw_networkx_nodes(G, pos, nodelist=place_nodes, 
                          node_color='lightblue', node_shape='o', 
                          node_size=1000)
    
    # Zeichne Transitionen
    trans_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'transition']
    nx.draw_networkx_nodes(G, pos, nodelist=trans_nodes, 
                          node_color='lightgreen', node_shape='s', 
                          node_size=800)
    
    # Zeichne Kanten
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
    
    # Zeichne Labels
    labels = {}
    for node in G.nodes():
        if node in petri_net.places:
            labels[node] = f"{node}\n[{petri_net.tokens.get(node, 0)}]"
        else:
            labels[node] = node
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(f"Petri-Netz: {petri_net.name}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_similarity_matrix(matrix, symbols, filename="category_similarity.png"):
    """Plottet Ähnlichkeitsmatrix"""
    plt.figure(figsize=(12, 10))
    if SEABORN_AVAILABLE:
        sns.heatmap(matrix, xticklabels=symbols, yticklabels=symbols,
                   cmap='viridis', annot=True, fmt='.2f')
    else:
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(symbols)), symbols, rotation=90)
        plt.yticks(range(len(symbols)), symbols)
    plt.title('Semantic Similarity Between Categories')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_grammar_graph(graph, filename="grammar_graph.png"):
    """Plottet Grammatik-Graphen"""
    if not NETWORKX_AVAILABLE:
        print("networkx nicht verfügbar")
        return
    
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    node_colors = []
    for node in graph.nodes():
        if node.startswith('NT_'):
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightblue')
    
    nx.draw(graph, pos, node_color=node_colors, with_labels=True,
           node_size=1000, font_size=8, arrows=True, arrowsize=20)
    
    plt.title('Grammar Graph')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_attention(attention, sequence, title="Attention Weights", filename="attention_weights.png"):
    """Plottet Attention-Matrix"""
    plt.figure(figsize=(10, 8))
    if SEABORN_AVAILABLE:
        sns.heatmap(attention, xticklabels=sequence, yticklabels=sequence,
                   cmap='viridis', annot=True, fmt='.2f')
    else:
        plt.imshow(attention, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(sequence)), sequence)
        plt.yticks(range(len(sequence)), sequence)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ============================================================================
# GUI - HAUPTFENSTER
# ============================================================================

class ARSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ARS - Algorithmic Recursive Sequence Analysis 4.0")
        self.root.geometry("1400x900")
        
        # Plot-Thread für sichere Visualisierung
        self.plot_thread = PlotThread(root)
        
        # Queue für GUI-Updates aus Threads
        self.update_queue = queue.Queue()
        self.process_updates()
        
        # Daten
        self.chains = []
        self.terminals = []
        self.delimiter = tk.StringVar(value=",")
        self.start_symbol = tk.StringVar(value="")
        
        # ARS-Objekte
        self.ars20 = ARS20()
        self.ars30 = GrammarInducer()  # ARS 3.0
        self.petri_builder = None
        self.hmm_model = None
        self.crf_model = None
        self.semantic_validator = None
        self.grammar_graph = None
        self.attention_viz = None
        
        # Optimierung
        self.optimization_running = False
        self.opt_progress_var = tk.DoubleVar()
        
        # Verfügbarkeit der optionalen Module
        self.module_status = {
            'networkx': NETWORKX_AVAILABLE,
            'hmmlearn': HMM_AVAILABLE,
            'crf': CRF_AVAILABLE,
            'transformer': TRANSFORMER_AVAILABLE,
            'torch': TORCH_AVAILABLE,
            'seaborn': SEABORN_AVAILABLE
        }
        
        # GUI aufbauen
        self.create_menu()
        self.create_main_panels()
        
        # Status
        self.status_var = tk.StringVar(value="Bereit")
        self.create_statusbar()
        
        # Modulstatus anzeigen
        self.show_module_status()
    
    def process_updates(self):
        """Verarbeitet Updates aus Threads im Hauptthread"""
        try:
            while True:
                update_func = self.update_queue.get_nowait()
                update_func()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_updates)
    
    def safe_gui_update(self, func):
        """Führt eine GUI-Update-Funktion thread-sicher aus"""
        self.update_queue.put(func)
    
    def create_menu(self):
        """Erstellt die Menüleiste"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Datei-Menü
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Datei", menu=file_menu)
        file_menu.add_command(label="Öffnen", command=self.load_file)
        file_menu.add_command(label="Beispiel laden", command=self.load_example)
        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self.root.quit)
        
        # Hilfe-Menü
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Hilfe", menu=help_menu)
        help_menu.add_command(label="Modulstatus", command=self.show_module_status)
        help_menu.add_command(label="Über", command=self.show_about)
    
    def create_main_panels(self):
        """Erstellt die Hauptbereiche"""
        # Hauptframe mit PanedWindow
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Linkes Panel - Eingabe
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        self.create_input_panel(left_frame)
        
        # Rechtes Panel - Notebook mit Tabs
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        self.create_output_panel(right_frame)
    
    def create_input_panel(self, parent):
        """Erstellt das Eingabe-Panel"""
        # Titel
        ttk.Label(parent, text="Eingabe", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=5)
        
        # Delimiter-Auswahl
        delim_frame = ttk.Frame(parent)
        delim_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(delim_frame, text="Trennzeichen:").pack(side=tk.LEFT)
        ttk.Radiobutton(delim_frame, text="Komma (,)", variable=self.delimiter, 
                       value=",").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(delim_frame, text="Semikolon (;)", variable=self.delimiter, 
                       value=";").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(delim_frame, text="Leerzeichen", variable=self.delimiter, 
                       value=" ").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(delim_frame, text="Benutzer", variable=self.delimiter, 
                       value="custom").pack(side=tk.LEFT, padx=2)
        
        self.custom_delimiter = ttk.Entry(delim_frame, width=5)
        self.custom_delimiter.pack(side=tk.LEFT, padx=2)
        self.custom_delimiter.insert(0, "|")
        
        # Text-Eingabe
        ttk.Label(parent, text="Terminalzeichenketten (eine pro Zeile):").pack(anchor=tk.W, pady=5)
        
        self.text_input = scrolledtext.ScrolledText(parent, height=12, font=('Courier', 10))
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Datei laden", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Parsen", command=self.parse_input).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Beispiel", command=self.load_example).pack(side=tk.LEFT, padx=2)
        
        # Startzeichen
        start_frame = ttk.Frame(parent)
        start_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(start_frame, text="Startzeichen:").pack(side=tk.LEFT)
        self.start_entry = ttk.Entry(start_frame, textvariable=self.start_symbol, width=10)
        self.start_entry.pack(side=tk.LEFT, padx=5)
        
        # Info
        self.info_var = tk.StringVar(value="Keine Daten geladen")
        ttk.Label(parent, textvariable=self.info_var, foreground="blue").pack(anchor=tk.W, pady=5)
    
    def create_output_panel(self, parent):
        """Erstellt das Output-Notebook mit Tabs"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: ARS 2.0
        self.tab20 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab20, text="ARS 2.0 (Basis)")
        self.create_ars20_tab()
        
        # Tab 2: ARS 3.0
        self.tab30 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab30, text="ARS 3.0 (Nonterminale)")
        self.create_ars30_tab()
        
        # Tab 3: Petri-Netze
        self.tab_petri = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_petri, text="Petri-Netze")
        self.create_petri_tab()
        
        # Tab 4: Bayessche Netze
        self.tab_bayes = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bayes, text="Bayessche Netze")
        self.create_bayes_tab()
        
        # Tab 5: Hybride Integration
        self.tab_hybrid = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_hybrid, text="Hybrid")
        self.create_hybrid_tab()
        
        # Tab 6: Generierung
        self.tab_gen = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_gen, text="Generierung")
        self.create_generation_tab()
    
    def create_ars20_tab(self):
        """Erstellt ARS 2.0 Tab"""
        # Steuerung
        control = ttk.Frame(self.tab20)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="ARS 2.0 berechnen", 
                  command=self.run_ars20).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Optimierung starten", 
                  command=self.run_optimization).pack(side=tk.LEFT, padx=5)
        
        self.opt_progress = ttk.Progressbar(control, length=200, mode='determinate')
        self.opt_progress.pack(side=tk.LEFT, padx=10)
        
        # Textausgabe
        self.text20 = scrolledtext.ScrolledText(self.tab20, font=('Courier', 10))
        self.text20.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_ars30_tab(self):
        """Erstellt ARS 3.0 Tab"""
        control = ttk.Frame(self.tab30)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Grammatik induzieren", 
                  command=self.run_ars30).pack(side=tk.LEFT, padx=5)
        
        self.ars30_progress = ttk.Progressbar(control, length=200, mode='indeterminate')
        self.ars30_progress.pack(side=tk.LEFT, padx=10)
        
        self.text30 = scrolledtext.ScrolledText(self.tab30, font=('Courier', 10))
        self.text30.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_petri_tab(self):
        """Erstellt Petri-Netz Tab"""
        control = ttk.Frame(self.tab_petri)
        control.pack(fill=tk.X, pady=5)
        
        if self.module_status['networkx']:
            ttk.Button(control, text="Einfaches Netz", 
                      command=self.build_basic_petri).pack(side=tk.LEFT, padx=5)
            ttk.Button(control, text="Netz mit Ressourcen", 
                      command=self.build_resource_petri).pack(side=tk.LEFT, padx=5)
            ttk.Button(control, text="Simuliere Transkript 1", 
                      command=self.simulate_petri).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Label(control, text="networkx nicht verfügbar", 
                     foreground="red").pack(side=tk.LEFT, padx=5)
        
        self.text_petri = scrolledtext.ScrolledText(self.tab_petri, font=('Courier', 10))
        self.text_petri.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_bayes_tab(self):
        """Erstellt Bayessche Netze Tab"""
        control = ttk.Frame(self.tab_bayes)
        control.pack(fill=tk.X, pady=5)
        
        if self.module_status['hmmlearn']:
            ttk.Button(control, text="HMM initialisieren", 
                      command=self.init_hmm).pack(side=tk.LEFT, padx=5)
            ttk.Button(control, text="HMM trainieren", 
                      command=self.train_hmm).pack(side=tk.LEFT, padx=5)
            ttk.Button(control, text="Dekodiere Transkript 1", 
                      command=self.decode_hmm).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Label(control, text="hmmlearn nicht verfügbar", 
                     foreground="red").pack(side=tk.LEFT, padx=5)
        
        self.text_bayes = scrolledtext.ScrolledText(self.tab_bayes, font=('Courier', 10))
        self.text_bayes.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_hybrid_tab(self):
        """Erstellt Hybrid-Tab"""
        control = ttk.Frame(self.tab_hybrid)
        control.pack(fill=tk.X, pady=5)
        
        if self.module_status['crf']:
            ttk.Button(control, text="CRF trainieren", 
                      command=self.train_crf).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Label(control, text="sklearn-crfsuite nicht verfügbar", 
                     foreground="red").pack(side=tk.LEFT, padx=5)
        
        if self.module_status['transformer']:
            ttk.Button(control, text="Semantische Validierung", 
                      command=self.run_semantic).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Label(control, text="sentence-transformers nicht verfügbar", 
                     foreground="red").pack(side=tk.LEFT, padx=5)
        
        if self.module_status['networkx']:
            ttk.Button(control, text="Grammatik-Graph", 
                      command=self.build_grammar_graph).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Label(control, text="networkx nicht verfügbar", 
                     foreground="red").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control, text="Attention visualisieren", 
                  command=self.visualize_attention).pack(side=tk.LEFT, padx=5)
        
        self.text_hybrid = scrolledtext.ScrolledText(self.tab_hybrid, font=('Courier', 10))
        self.text_hybrid.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_generation_tab(self):
        """Erstellt Generierungs-Tab"""
        control = ttk.Frame(self.tab_gen)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Label(control, text="Grammatik:").pack(side=tk.LEFT)
        
        self.gen_source = tk.StringVar(value="ars20")
        ttk.Radiobutton(control, text="ARS 2.0", variable=self.gen_source, 
                       value="ars20").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control, text="ARS 3.0", variable=self.gen_source, 
                       value="ars30").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control, text="Anzahl:").pack(side=tk.LEFT, padx=(20,5))
        self.gen_count = ttk.Spinbox(control, from_=1, to=50, width=5)
        self.gen_count.set(5)
        self.gen_count.pack(side=tk.LEFT)
        
        ttk.Button(control, text="Generieren", 
                  command=self.generate_chains).pack(side=tk.LEFT, padx=20)
        
        self.text_gen = scrolledtext.ScrolledText(self.tab_gen, font=('Courier', 10))
        self.text_gen.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_statusbar(self):
        """Erstellt Statusleiste"""
        status = ttk.Frame(self.root)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(status, length=100, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
    
    def show_module_status(self):
        """Zeigt Status der optionalen Module"""
        status_text = "Modulstatus:\n"
        status_text += f"✓ networkx: {'verfügbar' if self.module_status['networkx'] else 'nicht verfügbar'}\n"
        status_text += f"✓ hmmlearn: {'verfügbar' if self.module_status['hmmlearn'] else 'nicht verfügbar'}\n"
        status_text += f"✓ sklearn-crfsuite: {'verfügbar' if self.module_status['crf'] else 'nicht verfügbar'}\n"
        status_text += f"✓ sentence-transformers: {'verfügbar' if self.module_status['transformer'] else 'nicht verfügbar'}\n"
        status_text += f"✓ torch: {'verfügbar' if self.module_status['torch'] else 'nicht verfügbar'}\n"
        status_text += f"✓ seaborn: {'verfügbar' if self.module_status['seaborn'] else 'nicht verfügbar'}"
        
        messagebox.showinfo("Modulstatus", status_text)
    
    # ========================================================================
    # FUNKTIONEN
    # ========================================================================
    
    def get_actual_delimiter(self):
        """Gibt aktuelles Trennzeichen zurück"""
        delim = self.delimiter.get()
        if delim == "custom":
            return self.custom_delimiter.get()
        return delim
    
    def parse_line(self, line):
        """Parst eine Zeile"""
        line = line.strip()
        if not line:
            return []
        
        delim = self.get_actual_delimiter()
        
        if delim == " ":
            parts = re.split(r'\s+', line)
        else:
            parts = line.split(delim)
        
        return [p.strip() for p in parts if p.strip()]
    
    def parse_input(self):
        """Parst die Eingabe"""
        self.text_input.update()
        text = self.text_input.get("1.0", tk.END)
        lines = text.strip().split('\n')
        
        self.chains = []
        for line in lines:
            chain = self.parse_line(line)
            if chain:
                self.chains.append(chain)
        
        if self.chains:
            # Alle Terminale aus allen Ketten sammeln
            all_symbols = set()
            for chain in self.chains:
                for symbol in chain:
                    all_symbols.add(symbol)
            self.terminals = sorted(all_symbols)
            
            self.info_var.set(f"{len(self.chains)} Ketten, {len(self.terminals)} Terminale")
            self.status_var.set(f"{len(self.chains)} Ketten geladen")
            
            # In ARS-Objekte laden
            self.ars20.load_chains(self.chains, self.start_symbol.get() or None)
            self.ars30.load_chains(self.chains, self.start_symbol.get() or None)
            
            # Petri-Builder initialisieren (falls verfügbar)
            if self.module_status['networkx']:
                self.petri_builder = PetriNetBuilder(self.chains, self.ars30.rules)
            
            # Vorschau anzeigen
            self.show_ars20_preview()
        else:
            messagebox.showwarning("Warnung", "Keine gültigen Ketten gefunden!")
    
    def show_ars20_preview(self):
        """Zeigt ARS 2.0 Vorschau"""
        self.text20.delete("1.0", tk.END)
        self.text20.insert(tk.END, self.ars20.print_grammar())
    
    def run_ars20(self):
        """Führt ARS 2.0 aus"""
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        self.show_ars20_preview()
        self.status_var.set("ARS 2.0 abgeschlossen")
    
    def run_optimization(self):
        """Startet Optimierung"""
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        if self.optimization_running:
            messagebox.showinfo("Info", "Optimierung läuft bereits")
            return
        
        self.optimization_running = True
        self.opt_progress['value'] = 0
        
        def update_progress(iter_num, max_iter, corr, p_val):
            def update():
                self.opt_progress['value'] = iter_num
                self.status_var.set(f"Optimierung: Iteration {iter_num}, r={corr:.4f}")
            self.safe_gui_update(update)
        
        def run():
            try:
                probs, best_corr, history = self.ars20.optimize(progress_callback=update_progress)
                
                def update_display():
                    self.text20.insert(tk.END, "\n" + "="*70 + "\n")
                    self.text20.insert(tk.END, "OPTIMIERTE GRAMMATIK\n")
                    self.text20.insert(tk.END, "="*70 + "\n\n")
                    
                    if probs:
                        for start in sorted(probs.keys()):
                            trans = probs[start]
                            trans_str = ", ".join([f"'{end}': {prob:.3f}" for end, prob in sorted(trans.items())])
                            self.text20.insert(tk.END, f"{start} -> {trans_str}\n")
                    else:
                        self.text20.insert(tk.END, "Keine optimierte Grammatik erhalten.\n")
                    
                    self.text20.insert(tk.END, f"\nBeste Korrelation: {best_corr:.4f}\n")
                    self.status_var.set(f"Optimierung abgeschlossen, r={best_corr:.4f}")
                    self.opt_progress['value'] = 0
                    self.optimization_running = False
                
                self.safe_gui_update(update_display)
            except Exception as e:
                def error_display():
                    messagebox.showerror("Fehler", f"Optimierung fehlgeschlagen:\n{str(e)}")
                    self.optimization_running = False
                self.safe_gui_update(error_display)
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def run_ars30(self):
        """Führt ARS 3.0 aus"""
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        self.ars30_progress.start()
        self.status_var.set("Induziere Grammatik...")
        
        def update_progress(iter_num, max_iter, sequence, new_nt):
            def update():
                self.status_var.set(f"Induktion: {new_nt} gefunden")
            self.safe_gui_update(update)
        
        def run():
            try:
                self.ars30.induce_grammar(progress_callback=update_progress)
                
                def update_display():
                    self.text30.delete("1.0", tk.END)
                    self.text30.insert(tk.END, self.ars30.print_grammar())
                    self.ars30_progress.stop()
                    self.status_var.set("Grammatikinduktion abgeschlossen")
                
                self.safe_gui_update(update_display)
            except Exception as e:
                def error_display():
                    messagebox.showerror("Fehler", f"Grammatikinduktion fehlgeschlagen:\n{str(e)}")
                    self.ars30_progress.stop()
                self.safe_gui_update(error_display)
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def build_basic_petri(self):
        """Erstellt einfaches Petri-Netz"""
        if not self.module_status['networkx']:
            messagebox.showerror("Fehler", "networkx nicht installiert!")
            return
        
        if not self.petri_builder:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        try:
            self.petri_builder.build_basic_net()
            self.text_petri.delete("1.0", tk.END)
            self.text_petri.insert(tk.END, "Einfaches Petri-Netz erstellt:\n")
            self.text_petri.insert(tk.END, f"Stellen: {len(self.petri_builder.petri_net.places)}\n")
            self.text_petri.insert(tk.END, f"Transitionen: {len(self.petri_builder.petri_net.transitions)}\n")
            self.text_petri.insert(tk.END, f"Kanten: {len(self.petri_builder.petri_net.arcs)}\n")
            self.status_var.set("Petri-Netz erstellt")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Erstellen des Petri-Netzes:\n{str(e)}")
    
    def build_resource_petri(self):
        """Erstellt Petri-Netz mit Ressourcen"""
        if not self.module_status['networkx']:
            messagebox.showerror("Fehler", "networkx nicht installiert!")
            return
        
        if not self.petri_builder:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        try:
            self.petri_builder.build_resource_net()
            self.text_petri.delete("1.0", tk.END)
            self.text_petri.insert(tk.END, "Petri-Netz mit Ressourcen erstellt:\n")
            self.text_petri.insert(tk.END, f"Stellen: {len(self.petri_builder.petri_net.places)}\n")
            self.text_petri.insert(tk.END, f"Transitionen: {len(self.petri_builder.petri_net.transitions)}\n")
            self.text_petri.insert(tk.END, f"Kanten: {len(self.petri_builder.petri_net.arcs)}\n")
            self.text_petri.insert(tk.END, "\nRessourcen-Stellen:\n")
            for p, data in self.petri_builder.petri_net.places.items():
                if data['type'] == 'resource':
                    self.text_petri.insert(tk.END, f"  {p}: {data['initial_tokens']} Token\n")
            self.status_var.set("Petri-Netz mit Ressourcen erstellt")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Erstellen des Petri-Netzes:\n{str(e)}")
    
    def simulate_petri(self):
        """Simuliert Transkript 1 im Petri-Netz"""
        if not self.module_status['networkx']:
            messagebox.showerror("Fehler", "networkx nicht installiert!")
            return
        
        if not self.petri_builder or not self.petri_builder.petri_net:
            messagebox.showerror("Fehler", "Kein Petri-Netz vorhanden!")
            return
        
        if not self.chains:
            return
        
        try:
            results, marking = self.petri_builder.simulate_chain(self.chains[0])
            
            self.text_petri.insert(tk.END, "\n" + "="*50 + "\n")
            self.text_petri.insert(tk.END, "Simulation Transkript 1:\n")
            self.text_petri.insert(tk.END, "="*50 + "\n")
            
            for sym, success, reason in results:
                status = "✓" if success else "✗"
                self.text_petri.insert(tk.END, f"{status} {sym}: {reason}\n")
            
            self.text_petri.insert(tk.END, f"\nFinale Markierung:\n")
            for p, tokens in marking.items():
                if tokens > 0:
                    self.text_petri.insert(tk.END, f"  {p}: {tokens}\n")
            
            # Visualisierung
            self.plot_thread.plot(plot_petri_net, self.petri_builder.petri_net, "petri_net.png")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei der Simulation:\n{str(e)}")
    
    def init_hmm(self):
        """Initialisiert HMM"""
        if not self.module_status['hmmlearn']:
            messagebox.showerror("Fehler", "hmmlearn nicht installiert!")
            return
        
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        try:
            self.hmm_model = ARSHiddenMarkovModel(n_states=5)
            result = self.hmm_model.initialize_from_ars(self.chains)
            
            if result is None:
                messagebox.showerror("Fehler", "HMM-Initialisierung fehlgeschlagen - keine Daten?")
                return
            
            self.text_bayes.delete("1.0", tk.END)
            self.text_bayes.insert(tk.END, "HMM initialisiert:\n\n")
            self.text_bayes.insert(tk.END, self.hmm_model.get_parameters_string())
            self.status_var.set("HMM initialisiert")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei HMM-Initialisierung:\n{str(e)}")
    
    def train_hmm(self):
        """Trainiert HMM"""
        if not self.module_status['hmmlearn']:
            messagebox.showerror("Fehler", "hmmlearn nicht installiert!")
            return
        
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        if not self.hmm_model:
            self.hmm_model = ARSHiddenMarkovModel(n_states=5)
            self.hmm_model.initialize_from_ars(self.chains)
        
        self.status_var.set("Trainiere HMM...")
        self.progress_bar.start()
        
        def run():
            try:
                self.hmm_model.fit(self.chains, n_iter=100)
                
                def update_display():
                    self.text_bayes.insert(tk.END, "\n" + "="*50 + "\n")
                    self.text_bayes.insert(tk.END, "Nach Training:\n\n")
                    self.text_bayes.insert(tk.END, self.hmm_model.get_parameters_string())
                    self.status_var.set("HMM-Training abgeschlossen")
                    self.progress_bar.stop()
                
                self.safe_gui_update(update_display)
            except Exception as e:
                def error_display():
                    messagebox.showerror("Fehler", f"HMM-Training fehlgeschlagen:\n{str(e)}")
                    self.progress_bar.stop()
                self.safe_gui_update(error_display)
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def decode_hmm(self):
        """Dekodiert Transkript 1 mit HMM"""
        if not self.module_status['hmmlearn']:
            messagebox.showerror("Fehler", "hmmlearn nicht installiert!")
            return
        
        if not self.hmm_model or not self.hmm_model.model:
            messagebox.showerror("Fehler", "Kein HMM vorhanden!")
            return
        
        if not self.chains:
            return
        
        try:
            states, prob = self.hmm_model.decode(self.chains[0])
            
            if states is None:
                messagebox.showerror("Fehler", "Dekodierung fehlgeschlagen")
                return
            
            self.text_bayes.insert(tk.END, "\n" + "="*50 + "\n")
            self.text_bayes.insert(tk.END, f"Dekodierung Transkript 1 (p={prob:.4f}):\n")
            self.text_bayes.insert(tk.END, "="*50 + "\n")
            
            for i, (sym, state) in enumerate(zip(self.chains[0], states)):
                state_name = self.hmm_model.state_names.get(state, f"State {state}")
                self.text_bayes.insert(tk.END, f"{i+1:2d}: {sym} -> {state_name}\n")
                
        except Exception as e:
            messagebox.showerror("Fehler", f"Dekodierung fehlgeschlagen:\n{str(e)}")
    
    def train_crf(self):
        """Trainiert CRF-Modell"""
        if not self.module_status['crf']:
            messagebox.showerror("Fehler", "sklearn-crfsuite nicht installiert!")
            return
        
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        try:
            self.crf_model = ARSCRFModel()
            self.crf_model.fit(self.chains)
            
            self.text_hybrid.delete("1.0", tk.END)
            self.text_hybrid.insert(tk.END, "CRF trainiert.\n\nTop-Features:\n")
            
            for attr, label, weight in self.crf_model.get_top_features(10):
                self.text_hybrid.insert(tk.END, f"{attr:30s} -> {label:4s} : {weight:+.4f}\n")
            
            # Beispielvorhersage
            if self.chains:
                example = self.chains[0][:5]
                pred = self.crf_model.predict(example)
                self.text_hybrid.insert(tk.END, f"\nBeispiel: {example}\n")
                self.text_hybrid.insert(tk.END, f"Vorhersage: {pred}\n")
            
            self.status_var.set("CRF-Training abgeschlossen")
        except Exception as e:
            messagebox.showerror("Fehler", f"CRF-Training fehlgeschlagen:\n{str(e)}")
    
    def run_semantic(self):
        """Führt semantische Validierung durch"""
        if not self.module_status['transformer']:
            messagebox.showerror("Fehler", "sentence-transformers nicht installiert!")
            return
        
        try:
            self.semantic_validator = SemanticValidator()
            
            self.text_hybrid.insert(tk.END, "\n" + "="*50 + "\n")
            self.text_hybrid.insert(tk.END, "Semantische Validierung:\n")
            self.text_hybrid.insert(tk.END, "="*50 + "\n")
            
            if self.semantic_validator.load_model():
                sims = self.semantic_validator.get_intra_similarities()
                self.text_hybrid.insert(tk.END, "\nIntra-Kategorie-Ähnlichkeiten:\n")
                for sym, sim in sims.items():
                    self.text_hybrid.insert(tk.END, f"  {sym}: {sim:.3f}\n")
                
                # Visualisierung
                matrix, symbols = self.semantic_validator.similarity_matrix()
                if matrix is not None:
                    self.plot_thread.plot(plot_similarity_matrix, matrix, symbols)
                
                self.status_var.set("Semantische Validierung abgeschlossen")
            else:
                self.text_hybrid.insert(tk.END, "Fehler beim Laden des Modells\n")
        except Exception as e:
            messagebox.showerror("Fehler", f"Semantische Validierung fehlgeschlagen:\n{str(e)}")
    
    def build_grammar_graph(self):
        """Erstellt Grammatik-Graph"""
        if not self.module_status['networkx']:
            messagebox.showerror("Fehler", "networkx nicht installiert!")
            return
        
        if not self.ars30.rules:
            messagebox.showerror("Fehler", "Keine ARS-3.0-Grammatik vorhanden!")
            return
        
        try:
            self.grammar_graph = GrammarGraph(self.ars30.rules)
            
            self.text_hybrid.insert(tk.END, "\n" + "="*50 + "\n")
            self.text_hybrid.insert(tk.END, "Grammatik-Graph:\n")
            self.text_hybrid.insert(tk.END, "="*50 + "\n")
            self.text_hybrid.insert(tk.END, f"Knoten: {self.grammar_graph.graph.number_of_nodes()}\n")
            self.text_hybrid.insert(tk.END, f"Kanten: {self.grammar_graph.graph.number_of_edges()}\n")
            
            cent = self.grammar_graph.centrality()
            top = sorted(cent.items(), key=lambda x: x[1], reverse=True)[:5]
            self.text_hybrid.insert(tk.END, "\nZentralste Knoten:\n")
            for node, c in top:
                self.text_hybrid.insert(tk.END, f"  {node}: {c:.3f}\n")
            
            # Visualisierung
            self.plot_thread.plot(plot_grammar_graph, self.grammar_graph.graph)
            
            self.status_var.set("Grammatik-Graph erstellt")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Erstellen des Grammatik-Graphen:\n{str(e)}")
    
    def visualize_attention(self):
        """Visualisiert Attention für Transkript 1"""
        if not self.chains:
            return
        
        try:
            self.attention_viz = AttentionVisualizer(self.chains)
            
            self.text_hybrid.insert(tk.END, "\n" + "="*50 + "\n")
            self.text_hybrid.insert(tk.END, "Attention visualisiert (siehe plot)\n")
            
            # Attention berechnen und plotten
            attention = self.attention_viz.attention_weights(self.chains[0])
            self.plot_thread.plot(plot_attention, attention, self.chains[0])
            
            self.status_var.set("Attention visualisiert")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei Attention-Visualisierung:\n{str(e)}")
    
    def generate_chains(self):
        """Generiert neue Ketten"""
        source = self.gen_source.get()
        count = int(self.gen_count.get())
        
        self.text_gen.delete("1.0", tk.END)
        
        if source == "ars20":
            probs = self.ars20.optimized_probabilities or self.ars20.probabilities
            if not probs:
                self.text_gen.insert(tk.END, "Keine ARS 2.0 Grammatik!\n")
                return
            
            self.text_gen.insert(tk.END, f"ARS 2.0 - {count} generierte Ketten:\n\n")
            for i in range(count):
                chain = self.ars20.generate_chain()
                if chain:
                    self.text_gen.insert(tk.END, f"{i+1}: {' → '.join(chain)}\n")
        
        else:  # ars30
            if not self.ars30.rules:
                self.text_gen.insert(tk.END, "Keine ARS 3.0 Grammatik!\n")
                return
            
            self.text_gen.insert(tk.END, f"ARS 3.0 - {count} generierte Ketten:\n\n")
            for i in range(count):
                chain = self.ars30.generate_chain()
                if chain:
                    self.text_gen.insert(tk.END, f"{i+1}: {' → '.join(chain)}\n")
    
    def load_file(self):
        """Lädt Datei"""
        filename = filedialog.askopenfilename(
            title="Datei auswählen",
            filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.text_input.delete("1.0", tk.END)
                    self.text_input.insert("1.0", content)
                self.status_var.set(f"Geladen: {filename}")
            except Exception as e:
                messagebox.showerror("Fehler", f"Kann Datei nicht laden:\n{e}")
    
    def load_example(self):
        """Lädt Beispieldaten"""
        example = """KBG, VBG, KBBd, VBBd, KBA, VBA, KBBd, VBBd, KBA, VAA, KAA, VAV, KAV
VBG, KBBd, VBBd, VAA, KAA, VBG, KBBd, VAA, KAA
KBBd, VBBd, VAA, KAA
KBBd, VBBd, KBA, VBA, KBBd, VBA, KAE, VAE, KAA, VAV, KAV
KAV, KBBd, VBBd, KBBd, VAA, KAV
KBG, VBG, KBBd, VBBd, KAA
KBBd, VBBd, KBA, VAA, KAA
KBG, VBBd, KBBd, VBA, VAA, KAA, VAV, KAV"""
        
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", example)
        self.parse_input()
    
    def show_about(self):
        """Zeigt Info"""
        about = """ARS 4.0 - Algorithmic Recursive Sequence Analysis

Funktionen:
- ARS 2.0: Basis-Grammatik mit Optimierung
- ARS 3.0: Hierarchische Grammatik mit Nonterminalen
- Petri-Netze: Nebenläufigkeit und Ressourcen
- Bayessche Netze: HMM für latente Zustände
- Hybride Integration: CRF, Embeddings, Attention

Das Programm prüft automatisch die Verfügbarkeit aller benötigten
Pakete und installiert fehlende Pakete bei Bedarf nach.

© 2026 Paul Koop"""
        
        messagebox.showinfo("Über ARS", about)


# ============================================================================
# HAUPTFUNKTION
# ============================================================================

def main():
    root = tk.Tk()
    app = ARSGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
