"""
ARSXAI9.py - Algorithmic Recursive Sequence Analysis with Explainable AI
========================================================================
Universelle Analyseplattform für beliebige Terminalzeichenketten mit
PCFG-basierter Mustererkennung und natürlichen Erklärungen.

Version: 9.0 (PCFG-basiert - KEINE 5-Bit-Kodierung mehr!)

Kernkonzepte:
- Hierarchische Grammatikinduktion (ARS 3.0) als zentrale Wissensbasis
- Natürlichsprachliche Erklärungen aus gelernten Strukturen
- Wiederholungsmuster werden zu Nonterminalen abstrahiert
- Keine willkürlichen Kodierungen oder Annahmen mehr
"""

# ============================================================================
# UMWELTVARIABLEN FÜR WARNUNGEN
# ============================================================================

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# ============================================================================
# STANDARD BIBLIOTHEKEN
# ============================================================================

import sys
import queue
import threading
import re
import json
import subprocess
import importlib
import warnings
import logging
import shutil
import glob
from datetime import datetime
from collections import defaultdict, Counter

# ============================================================================
# WARNUNGEN VOLLSTÄNDIG UNTERDRÜCKEN
# ============================================================================

# Alle Warnungen konfigurieren
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="hmmlearn")
warnings.filterwarnings("ignore", message="MultinomialHMM has undergone major changes")

# hmmlearn Logger unterdrücken
logging.getLogger('hmmlearn').setLevel(logging.ERROR)

# ============================================================================
# HMMLEARN MONKEY-PATCH
# ============================================================================

try:
    import hmmlearn
    if hasattr(hmmlearn, 'hmm'):
        original_init = hmmlearn.hmm.MultinomialHMM.__init__
        def patched_init(self, *args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                original_init(self, *args, **kwargs)
        hmmlearn.hmm.MultinomialHMM.__init__ = patched_init
        print("✓ hmmlearn Monkey-Patch angewendet")
except ImportError:
    pass

# ============================================================================
# PAKETVERWALTUNG
# ============================================================================

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
    'tabulate',
    'graphviz'
]

def check_and_install_packages():
    """Prüft und installiert fehlende Python-Pakete"""
    print("=" * 70)
    print("ARSXAI9 - PAKETPRÜFUNG")
    print("=" * 70)
    
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        import_name = package.replace('-', '_')
        special_imports = {
            'sklearn-crfsuite': 'sklearn_crfsuite',
            'sentence-transformers': 'sentence_transformers'
        }
        import_name = special_imports.get(package, import_name)
        
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
    
    print("\n" + "=" * 70 + "\n")

check_and_install_packages()

# ============================================================================
# GRAPHVIZ KONFIGURATION
# ============================================================================

def setup_graphviz():
    """Konfiguriert Graphviz und prüft Installation"""
    GRAPHVIZ_AVAILABLE = False
    
    if sys.platform == 'win32':
        possible_paths = [
            r'C:\Program Files\Graphviz\bin',
            r'C:\Program Files (x86)\Graphviz\bin',
            r'C:\Graphviz\bin',
        ]
        
        for base_path in [r'C:\Program Files\Graphviz*', r'C:\Program Files (x86)\Graphviz*']:
            for path in glob.glob(base_path):
                bin_path = os.path.join(path, 'bin')
                if os.path.exists(bin_path):
                    possible_paths.append(bin_path)
        
        dot_path = shutil.which('dot')
        if dot_path:
            print(f"✓ Graphviz (dot) gefunden in: {dot_path}")
            GRAPHVIZ_AVAILABLE = True
        else:
            for path in possible_paths:
                if os.path.exists(path):
                    dot_exe = os.path.join(path, 'dot.exe')
                    if os.path.exists(dot_exe):
                        os.environ['PATH'] += os.pathsep + path
                        print(f"✓ Graphviz gefunden in: {path}")
                        GRAPHVIZ_AVAILABLE = True
                        break
    else:
        if shutil.which('dot'):
            print("✓ Graphviz (dot) gefunden")
            GRAPHVIZ_AVAILABLE = True
    
    return GRAPHVIZ_AVAILABLE

GRAPHVIZ_AVAILABLE = setup_graphviz()

# ============================================================================
# TKINTER IMPORTS
# ============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ============================================================================
# WISSENSCHAFTLICHE BIBLIOTHEKEN
# ============================================================================

import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ============================================================================
# OPTIONALE IMPORTS MIT STATUS-ERFASSUNG
# ============================================================================

MODULE_STATUS = {
    'networkx': False,
    'hmmlearn': False,
    'crf': False,
    'transformer': False,
    'seaborn': False,
    'graphviz': GRAPHVIZ_AVAILABLE
}

try:
    import networkx as nx
    MODULE_STATUS['networkx'] = True
except ImportError:
    pass

try:
    from hmmlearn import hmm
    MODULE_STATUS['hmmlearn'] = True
except ImportError:
    pass

try:
    from sklearn_crfsuite import CRF
    MODULE_STATUS['crf'] = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    MODULE_STATUS['transformer'] = True
except ImportError:
    pass

try:
    import seaborn as sns
    MODULE_STATUS['seaborn'] = True
except ImportError:
    pass

try:
    import graphviz
    MODULE_STATUS['graphviz'] = GRAPHVIZ_AVAILABLE
except ImportError:
    MODULE_STATUS['graphviz'] = False


# ============================================================================
# THREAD-SICHERE PLOT-FUNKTIONEN
# ============================================================================

class PlotThread:
    """Thread-sichere Plot-Ausführung"""
    
    def __init__(self, root):
        self.root = root
        self.plot_queue = queue.Queue()
        self.process()
    
    def process(self):
        try:
            while True:
                func, args, kwargs = self.plot_queue.get_nowait()
                self.root.after(0, lambda: func(*args, **kwargs))
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process)
    
    def plot(self, func, *args, **kwargs):
        self.plot_queue.put((func, args, kwargs))


# ============================================================================
# XAI BASISKLASSE FÜR ALLE MODELLE
# ============================================================================

class XAIModel:
    """Einheitliches XAI-Interface für alle Modelle"""
    
    def __init__(self, name):
        self.name = name
        self.description = ""
        self.confidence = 0.5
        self.trained = False
    
    def train(self, chains):
        """Trainiert das Modell auf den Daten"""
        raise NotImplementedError
    
    def explain(self, data, detail_level='normal'):
        """Liefert XAI-konforme Erklärung"""
        raise NotImplementedError
    
    def get_confidence(self):
        return self.confidence
    
    def get_info(self):
        return {
            'name': self.name,
            'description': self.description,
            'confidence': self.confidence,
            'trained': self.trained
        }


# ============================================================================
# ARS 2.0 - BASIS-GRAMMATIK (für Vergleich, optional)
# ============================================================================

class ARS20(XAIModel):
    """ARS 2.0 - Übergangswahrscheinlichkeiten ohne Nonterminale"""
    
    def __init__(self):
        super().__init__("ARS 2.0 - Basis-Grammatik")
        self.description = "Einfache Bigramm-Übergangswahrscheinlichkeiten"
        self.chains = []
        self.terminals = []
        self.start_symbol = None
        self.transitions = {}
        self.probabilities = {}
    
    def train(self, chains, start_symbol=None):
        self.chains = chains
        all_terminals = set()
        for chain in chains:
            for symbol in chain:
                all_terminals.add(symbol)
        self.terminals = sorted(list(all_terminals))
        self.start_symbol = start_symbol if start_symbol else (chains[0][0] if chains else None)
        self.transitions = self._count_transitions(chains)
        self.probabilities = self._calculate_probabilities(self.transitions)
        self.trained = True
        self.confidence = self._calculate_confidence()
        return True
    
    def _count_transitions(self, chains):
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
    
    def _calculate_probabilities(self, transitions):
        probabilities = {}
        for start in transitions:
            total = sum(transitions[start].values())
            if total > 0:
                probabilities[start] = {end: count / total for end, count in transitions[start].items()}
        return probabilities
    
    def _calculate_confidence(self):
        if not self.chains:
            return 0.0
        total_transitions = sum(len(chain)-1 for chain in self.chains)
        confidence = min(1.0, np.log10(total_transitions + 1) / 2)
        return round(confidence, 3)
    
    def explain(self, data, detail_level='normal'):
        explanation = {
            'model': self.name,
            'confidence': self.confidence,
            'type': None,
            'content': []
        }
        
        if isinstance(data, tuple) and len(data) == 2:
            start, end = data
            explanation['type'] = 'transition'
            if start in self.probabilities and end in self.probabilities[start]:
                prob = self.probabilities[start][end]
                count = self.transitions[start][end]
                total = sum(self.transitions[start].values())
                explanation['content'] = [
                    f"Übergang {start} → {end}:",
                    f"  Wahrscheinlichkeit: {prob:.1%}",
                    f"  Beobachtet: {count} von {total} Fällen"
                ]
            else:
                explanation['content'] = [f"Übergang {start} → {end} nicht beobachtet"]
        
        elif isinstance(data, str):
            explanation['type'] = 'symbol'
            symbol = data
            if symbol in self.probabilities:
                explanation['content'] = [f"Mögliche Folgesymbole für {symbol}:"]
                for end, prob in sorted(self.probabilities[symbol].items(), key=lambda x: -x[1]):
                    count = self.transitions[symbol][end]
                    explanation['content'].append(f"  {end}: {prob:.1%} ({count}x)")
            else:
                explanation['content'] = [f"Keine Folgesymbole für {symbol} bekannt"]
        
        return explanation
    
    def generate_chain(self, start_symbol=None, max_length=20):
        if not self.trained:
            return []
        probs = self.probabilities
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
            try:
                next_symbol = np.random.choice(next_symbols, p=probs_list)
                chain.append(next_symbol)
                current = next_symbol
            except:
                break
        return chain
    
    def get_grammar_string(self):
        lines = ["=" * 60, f"{self.name}", "=" * 60, ""]
        if self.probabilities:
            for start in sorted(self.probabilities.keys()):
                trans = self.probabilities[start]
                trans_str = ", ".join([f"{end}: {prob:.3f}" for end, prob in sorted(trans.items())])
                lines.append(f"{start} -> {trans_str}")
        else:
            lines.append("Keine Übergänge gefunden.")
        lines.append(f"\nTerminalzeichen ({len(self.terminals)}): {self.terminals}")
        lines.append(f"Startzeichen: {self.start_symbol}")
        lines.append(f"Konfidenz: {self.confidence:.0%}")
        return "\n".join(lines)


# ============================================================================
# ARS 3.0 - HIERARCHISCHE GRAMMATIK (ZENTRALE WISSENSBASIS)
# ============================================================================

class MethodologicalReflection:
    """Methodologische Reflexion für ARS 3.0"""
    
    def __init__(self):
        self.interpretation_log = []
    
    def log_interpretation(self, sequence, new_nonterminal, rationale):
        self.interpretation_log.append({
            'sequence': sequence,
            'new_nonterminal': new_nonterminal,
            'rationale': rationale,
            'timestamp': len(self.interpretation_log)
        })
    
    def print_summary(self):
        lines = ["\n" + "=" * 60, "ERKANNTE MUSTER", "=" * 60]
        for log in self.interpretation_log:
            lines.append(f"\n[{log['timestamp']+1}] {log['new_nonterminal']}")
            lines.append(f"  Sequenz: {' → '.join(log['sequence'])}")
            lines.append(f"  Begründung: {log['rationale']}")
        return "\n".join(lines)


class GrammarInducer(XAIModel):
    """
    ARS 3.0 - Hierarchische Grammatikinduktion
    ZENTRALE WISSENSBASIS für alle XAI-Erklärungen
    """
    
    def __init__(self):
        super().__init__("ARS 3.0 - Hierarchische Grammatik")
        self.description = "Induziert hierarchische Nonterminale aus wiederholten Sequenzen"
        self.rules = {}                 # PCFG-Regeln: nonterminal -> [(production, probability)]
        self.terminals = set()           # Grundlegende Symbole
        self.nonterminals = set()        # Abstrakte Kategorien
        self.symbol_to_nonterminals = defaultdict(set)  # Welche Nonterminale enthalten ein Symbol?
        self.start_symbol = None
        self.compression_history = []
        self.reflection = MethodologicalReflection()
        self.chains = []
        self.iteration_count = 0
        self.hierarchy_levels = {}
        self.induction_done = False
    
    def train(self, chains, max_iterations=20):
        """Induziert Grammatik aus Ketten"""
        self.chains = [list(chain) for chain in chains]
        
        # Sammle alle Terminale
        all_symbols = set()
        for chain in chains:
            for symbol in chain:
                all_symbols.add(symbol)
        self.terminals = all_symbols
        
        current_chains = [list(chain) for chain in chains]
        iteration = 0
        rule_counter = 1
        
        self.rules = {}
        self.nonterminals = set()
        self.symbol_to_nonterminals = defaultdict(set)
        self.compression_history = []
        self.hierarchy_levels = {}
        
        while iteration < max_iterations:
            best_seq = self._find_best_repetition(current_chains)
            
            if best_seq is None:
                break
            
            new_nonterminal = self._generate_nonterminal_name(best_seq)
            base_name = new_nonterminal
            while new_nonterminal in self.nonterminals:
                new_nonterminal = f"{base_name}_{rule_counter}"
                rule_counter += 1
            
            rationale = self._generate_rationale(best_seq)
            self.reflection.log_interpretation(best_seq, new_nonterminal, rationale)
            
            self.rules[new_nonterminal] = [(list(best_seq), 1.0)]
            self.nonterminals.add(new_nonterminal)
            self.hierarchy_levels[new_nonterminal] = iteration
            
            # Aktualisiere symbol_to_nonterminals
            for symbol in best_seq:
                self.symbol_to_nonterminals[symbol].add(new_nonterminal)
            
            # Zähle Vorkommen
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
            
            current_chains = self._compress_sequences(current_chains, best_seq, new_nonterminal)
            iteration += 1
            self.iteration_count = iteration
            
            if self._all_chains_identical(current_chains):
                if current_chains and current_chains[0]:
                    unique_symbol = current_chains[0][0]
                    if unique_symbol in self.rules:
                        self.start_symbol = unique_symbol
                    else:
                        self.start_symbol = self._find_top_level_nonterminal()
                    break
        
        if self.start_symbol is None:
            self.start_symbol = self._find_top_level_nonterminal()
        
        # Terminale aktualisieren (alles, was nicht Nonterminal ist)
        all_symbols = set()
        for chain in self.chains:
            for sym in chain:
                all_symbols.add(sym)
        self.terminals = all_symbols - self.nonterminals
        
        # Berechne Wahrscheinlichkeiten
        self._calculate_probabilities()
        
        self.induction_done = True
        self.trained = True
        self.confidence = self._calculate_confidence()
        
        return current_chains
    
    def _find_best_repetition(self, chains, min_length=2, max_length=5):
        """Findet die beste Wiederholung in den Ketten"""
        sequence_counter = Counter()
        
        for chain in chains:
            max_len = min(max_length, len(chain))
            for length in range(min_length, max_len + 1):
                for i in range(len(chain) - length + 1):
                    seq = tuple(chain[i:i+length])
                    sequence_counter[seq] += 1
        
        repeated = {seq: count for seq, count in sequence_counter.items() if count >= 2}
        if not repeated:
            return None
        
        # Bewerte nach: Vorkommen * Länge / (1 + Anzahl verschiedener Symbole)
        best_seq = max(repeated.items(), 
                      key=lambda x: x[1] * len(x[0]) / max(1, len(set(x[0]))))
        return best_seq[0]
    
    def _generate_nonterminal_name(self, sequence):
        """Generiert einen neutralen Namen für ein Nonterminal"""
        first = sequence[0] if sequence else "X"
        last = sequence[-1] if sequence else "X"
        return f"P_{first}_{last}_{len(sequence)}"
    
    def _generate_rationale(self, sequence):
        """Generiert eine Erklärung, warum dieses Muster erkannt wurde"""
        occurrences = sum(1 for chain in self.chains if self._sequence_in_chain(sequence, chain))
        total = len(self.chains)
        percentage = (occurrences / total) * 100
        
        if len(sequence) == 2:
            return f"Die Zweierfolge {' → '.join(sequence)} kommt in {percentage:.0f}% aller Ketten vor"
        else:
            return f"Die {len(sequence)}-teilige Sequenz {' → '.join(sequence)} kommt in {percentage:.0f}% aller Ketten vor"
    
    def _sequence_in_chain(self, sequence, chain):
        """Prüft, ob eine Sequenz in einer Kette vorkommt"""
        seq_len = len(sequence)
        for i in range(len(chain) - seq_len + 1):
            if tuple(chain[i:i+seq_len]) == sequence:
                return True
        return False
    
    def _compress_sequences(self, chains, sequence, new_nonterminal):
        """Komprimiert Ketten durch Einführung eines Nonterminals"""
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
    
    def _all_chains_identical(self, chains):
        if not chains:
            return False
        first = chains[0]
        return all(len(chain) == 1 and chain[0] == first[0] for chain in chains)
    
    def _find_top_level_nonterminal(self):
        """Findet das oberste Nonterminal in der Hierarchie"""
        if not self.rules:
            return None
        
        symbols_in_productions = set()
        for nt, productions in self.rules.items():
            for prod, _ in productions:
                for sym in prod:
                    symbols_in_productions.add(sym)
        
        top_level = [nt for nt in self.rules if nt not in symbols_in_productions]
        
        if top_level:
            return top_level[0]
        
        if self.hierarchy_levels:
            return max(self.hierarchy_levels.items(), key=lambda x: x[1])[0]
        
        return list(self.rules.keys())[0] if self.rules else None
    
    def _calculate_probabilities(self):
        """Berechnet Wahrscheinlichkeiten für Produktionen"""
        expansion_counts = defaultdict(Counter)
        
        for chain in self.chains:
            self._count_expansions(chain, expansion_counts)
        
        for nonterminal in self.rules:
            if nonterminal in expansion_counts:
                total = sum(expansion_counts[nonterminal].values())
                if total > 0:
                    productions = []
                    for expansion, count in expansion_counts[nonterminal].items():
                        productions.append((list(expansion), count / total))
                    productions.sort(key=lambda x: x[1], reverse=True)
                    self.rules[nonterminal] = productions
    
    def _count_expansions(self, sequence, expansion_counts):
        i = 0
        while i < len(sequence):
            symbol = sequence[i]
            if symbol in self.rules:
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
    
    def _calculate_confidence(self):
        """Berechnet Konfidenz basierend auf Kompressionsrate"""
        if not self.chains or not self.compression_history:
            return 0.0
        
        total_original = sum(len(chain) for chain in self.chains)
        total_compressed = total_original
        
        for hist in self.compression_history:
            savings = (len(hist['sequence']) - 1) * hist['occurrences']
            total_compressed -= savings
        
        compression_ratio = 1 - (total_compressed / total_original) if total_original > 0 else 0
        confidence = min(1.0, compression_ratio * 1.5)
        
        return round(confidence, 3)
    
    def explain(self, data, detail_level='normal'):
        """
        Erklärt ein Symbol basierend auf seiner Rolle in der Grammatik
        
        Args:
            data: Symbol (String) oder Sequenz (Liste)
            detail_level: 'simple', 'normal', oder 'detailed'
        
        Returns:
            Dictionary mit Erklärung
        """
        explanation = {
            'model': self.name,
            'confidence': self.confidence,
            'type': None,
            'content': []
        }
        
        if isinstance(data, list):
            # Erkläre eine Sequenz
            explanation['type'] = 'sequence'
            seq = tuple(data)
            
            # Suche nach direktem Match
            for hist in self.compression_history:
                if tuple(hist['sequence']) == seq:
                    percentage = (hist['occurrences'] / len(self.chains)) * 100
                    explanation['content'].append(
                        f"✅ Diese Sequenz wurde als **{hist['new_symbol']}** abstrahiert "
                        f"und kommt in {percentage:.0f}% der Ketten vor."
                    )
                    
                    if detail_level == 'detailed':
                        explanation['content'].append(
                            f"   Grund: {self.reflection.interpretation_log[hist['iteration']]['rationale']}"
                        )
                    break
            else:
                explanation['content'].append(
                    "Diese Sequenz bildet kein eigenständiges wiederkehrendes Muster."
                )
        
        elif isinstance(data, str):
            # Erkläre ein Symbol
            explanation['type'] = 'symbol'
            symbol = data
            
            if symbol in self.nonterminals:
                explanation['content'].append(f"📦 **{symbol}** ist ein **abstraktes Muster**.")
                if symbol in self.rules:
                    for prod, prob in self.rules[symbol]:
                        prod_str = ' → '.join(prod)
                        explanation['content'].append(f"  • {symbol} → {prod_str} (p={prob:.2f})")
            
            elif symbol in self.terminals:
                explanation['content'].append(f"🔤 **{symbol}** ist ein **grundlegendes Symbol**.")
                
                # In welchen Mustern kommt es vor?
                patterns = self.symbol_to_nonterminals.get(symbol, set())
                if patterns:
                    explanation['content'].append(f"\n📊 Es kommt in folgenden wiederkehrenden Mustern vor:")
                    for nt in sorted(patterns):
                        # Finde die Produktion, in der es vorkommt
                        for prod, prob in self.rules.get(nt, []):
                            if symbol in prod:
                                idx = prod.index(symbol)
                                context = []
                                if idx > 0:
                                    context.append(f"nach {prod[idx-1]}")
                                if idx < len(prod) - 1:
                                    context.append(f"vor {prod[idx+1]}")
                                
                                # Finde die Häufigkeit dieses Musters
                                for hist in self.compression_history:
                                    if hist['new_symbol'] == nt:
                                        percentage = (hist['occurrences'] / len(self.chains)) * 100
                                        break
                                else:
                                    percentage = 0
                                
                                explanation['content'].append(
                                    f"\n  • **{nt}** ({percentage:.0f}% der Ketten):"
                                )
                                explanation['content'].append(
                                    f"    {nt} → {' → '.join(prod)}"
                                )
                                if context:
                                    explanation['content'].append(
                                        f"    Position: {', '.join(context)}"
                                    )
                
                # Hierarchische Einbettung
                hierarchy = self._get_hierarchy(symbol)
                if hierarchy:
                    explanation['content'].append(f"\n🏗️ **Hierarchische Einbettung:**")
                    for level, (nt, prod) in enumerate(hierarchy):
                        indent = "  " * level
                        explanation['content'].append(
                            f"{indent}└─ in {nt} → {' → '.join(prod)}"
                        )
            
            else:
                explanation['content'].append(f"❓ Symbol {symbol} ist unbekannt.")
        
        return explanation
    
    def _get_hierarchy(self, symbol, max_depth=5):
        """Ermittelt die hierarchische Einbettung eines Symbols"""
        hierarchy = []
        current = symbol
        depth = 0
        
        while depth < max_depth:
            parents = list(self.symbol_to_nonterminals.get(current, set()))
            if not parents:
                break
            
            # Wähle den ersten Parent (oder den mit der höchsten Hierarchie)
            parent = parents[0]
            for prod, prob in self.rules.get(parent, []):
                if current in prod:
                    hierarchy.append((parent, prod))
                    current = parent
                    depth += 1
                    break
            else:
                break
        
        return hierarchy
    
    def get_pattern_summary(self):
        """Gibt eine Zusammenfassung aller erkannten Muster"""
        lines = []
        lines.append("=" * 70)
        lines.append("ERKANNTE WIEDERKEHRENDE MUSTER")
        lines.append("=" * 70)
        lines.append("")
        
        for hist in self.compression_history:
            percentage = (hist['occurrences'] / len(self.chains)) * 100
            lines.append(f"\n📌 **{hist['new_symbol']}** ({percentage:.0f}% der Ketten):")
            lines.append(f"   {' → '.join(hist['sequence'])}")
            lines.append(f"   Grund: {self.reflection.interpretation_log[hist['iteration']]['rationale']}")
        
        return "\n".join(lines)
    
    def generate_chain(self, start_symbol=None, max_depth=20):
        """Generiert eine neue Kette basierend auf der Grammatik"""
        if not self.trained:
            return []
        
        if not start_symbol:
            start_symbol = self.start_symbol
        
        if not start_symbol or start_symbol not in self.rules:
            if self.rules:
                start_symbol = self._find_top_level_nonterminal()
            else:
                return []
        
        # Bereite Produktionen mit Wahrscheinlichkeiten vor
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
            if depth >= max_depth:
                return [str(symbol)]
            
            if symbol in self.terminals:
                return [str(symbol)]
            
            if symbol not in prod_probs:
                return [str(symbol)]
            
            symbols, probs = prod_probs[symbol]
            if not symbols:
                return [str(symbol)]
            
            try:
                chosen_idx = np.random.choice(len(symbols), p=probs)
                chosen = symbols[chosen_idx]
            except Exception:
                chosen = symbols[0] if symbols else []
            
            result = []
            for sym in chosen:
                result.extend(expand(sym, depth + 1))
            return result
        
        return expand(start_symbol)
    
    def get_grammar_string(self):
        """Gibt die vollständige Grammatik als String zurück"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"{self.name}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"📝 **Terminale** ({len(self.terminals)}): {sorted(self.terminals)}")
        lines.append(f"📦 **Nonterminale** ({len(self.nonterminals)}): {sorted(self.nonterminals)}")
        lines.append(f"🎯 **Startsymbol**: {self.start_symbol}")
        lines.append(f"📊 **Konfidenz**: {self.confidence:.0%}")
        lines.append(f"🔄 **Iterationen**: {self.iteration_count}")
        lines.append("")
        lines.append("**PRODUKTIONSREGELN:**")
        
        for nonterminal in sorted(self.rules.keys()):
            productions = self.rules[nonterminal]
            if productions:
                prod_str = " | ".join([f"{' → '.join(prod)} [{prob:.3f}]" 
                                      for prod, prob in productions])
                lines.append(f"\n{nonterminal} → {prod_str}")
        
        lines.append("")
        lines.append(self.reflection.print_summary())
        
        return "\n".join(lines)


# ============================================================================
# NATÜRLICHSPRACHLICHER ERKLÄRER
# ============================================================================

class NaturalLanguageExplainer:
    """
    Generiert menschenlesbare Erklärungen aus der Grammatik
    """
    
    def __init__(self, grammar_inducer):
        self.grammar = grammar_inducer
        self.pattern_names = self._generate_pattern_names()
    
    def _generate_pattern_names(self):
        """Generiert verständliche Namen für wiederkehrende Muster"""
        names = {}
        for nt in self.grammar.nonterminals:
            productions = self.grammar.rules.get(nt, [])
            if not productions:
                continue
            
            seq = productions[0][0]
            
            # Versuche, einen semantischen Namen zu generieren
            if len(seq) == 2:
                names[nt] = f"Zweierschritt {seq[0]}→{seq[1]}"
            elif len(seq) == 3:
                names[nt] = f"Dreierschritt {seq[0]}…{seq[2]}"
            else:
                names[nt] = f"{len(seq)}-teiliges Muster"
        
        return names
    
    def explain_symbol(self, symbol):
        """Erklärt ein Symbol in natürlicher Sprache"""
        exp = self.grammar.explain(symbol, 'detailed')
        
        lines = []
        lines.append(f"🔍 **Erklärung für Symbol '{symbol}'**")
        lines.append("=" * 70)
        
        for line in exp['content']:
            lines.append(line)
        
        # Füge Konfidenz hinzu
        lines.append("")
        lines.append(f"✅ **Konfidenz dieser Analyse**: {self.grammar.confidence:.0%}")
        
        return "\n".join(lines)
    
    def explain_sequence(self, sequence):
        """Erklärt eine ganze Sequenz"""
        seq_str = ' → '.join(sequence)
        lines = []
        lines.append(f"🔍 **Erklärung für Sequenz:** {seq_str}")
        lines.append("=" * 70)
        lines.append("")
        
        # Finde die hierarchische Struktur
        current = sequence
        level = 0
        found_patterns = []
        
        while current and level < 5:
            # Suche nach Nonterminal, das diese Sequenz repräsentiert
            matched = False
            for nt in self.grammar.nonterminals:
                for prod, prob in self.grammar.rules.get(nt, []):
                    if prod == current:
                        for hist in self.grammar.compression_history:
                            if hist['new_symbol'] == nt:
                                percentage = (hist['occurrences'] / len(self.grammar.chains)) * 100
                                break
                        else:
                            percentage = 0
                        
                        found_patterns.append({
                            'level': level,
                            'nonterminal': nt,
                            'sequence': current,
                            'probability': prob,
                            'frequency': percentage
                        })
                        
                        # Ersetze durch Nonterminal für nächste Ebene
                        current = [nt]
                        matched = True
                        break
                if matched:
                    break
            
            if not matched:
                break
            level += 1
        
        if found_patterns:
            lines.append("**Hierarchische Struktur:**")
            for pattern in found_patterns:
                indent = "  " * pattern['level']
                seq_str = ' → '.join(pattern['sequence'])
                lines.append(
                    f"{indent}└─ **{pattern['nonterminal']}** = {seq_str} "
                    f"(in {pattern['frequency']:.0f}% der Ketten)"
                )
        else:
            lines.append("Diese Sequenz bildet kein eigenständiges Muster.")
        
        # Zeige, wo diese Sequenz vorkommt
        occurrences = 0
        for chain in self.grammar.chains:
            if self._sequence_in_chain(sequence, chain):
                occurrences += 1
        
        if occurrences > 0:
            percentage = (occurrences / len(self.grammar.chains)) * 100
            lines.append("")
            lines.append(f"📊 **Vorkommen**: {percentage:.0f}% der Ketten ({occurrences} von {len(self.grammar.chains)})")
        
        return "\n".join(lines)
    
    def _sequence_in_chain(self, sequence, chain):
        """Prüft, ob eine Sequenz in einer Kette vorkommt"""
        seq_len = len(sequence)
        for i in range(len(chain) - seq_len + 1):
            if chain[i:i+seq_len] == sequence:
                return True
        return False
    
    def get_summary(self):
        """Gibt eine Zusammenfassung aller Erkenntnisse"""
        lines = []
        lines.append("=" * 70)
        lines.append("📊 **ZUSAMMENFASSUNG DER ANALYSE**")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append(f"**Datenbasis:** {len(self.grammar.chains)} Ketten")
        lines.append(f"**Terminale Symbole:** {len(self.grammar.terminals)}")
        lines.append(f"**Erkannte Muster:** {len(self.grammar.nonterminals)}")
        lines.append(f"**Kompressionsrate:** {self.grammar.confidence:.0%}")
        lines.append("")
        
        lines.append("**Häufigste Muster:**")
        for hist in sorted(self.grammar.compression_history, key=lambda x: -x['occurrences'])[:5]:
            percentage = (hist['occurrences'] / len(self.grammar.chains)) * 100
            name = self.pattern_names.get(hist['new_symbol'], hist['new_symbol'])
            lines.append(f"  • {name}: {percentage:.0f}% der Ketten")
        
        return "\n".join(lines)


# ============================================================================
# HMM - HIDDEN MARKOV MODELS (angepasst)
# ============================================================================

class ARSHiddenMarkovModel(XAIModel):
    """Bayessche Netze - Hidden Markov Models für latente Phasen"""
    
    def __init__(self, n_states=5):
        super().__init__("HMM - Bayessches Netz")
        self.description = "Modelliert latente Phasen in den Sequenzen"
        self.n_states = n_states
        self.model = None
        self.code_to_idx = {}
        self.idx_to_code = {}
        self.state_names = {i: f"Phase {i}" for i in range(n_states)}
        self.n_features = None
        self.trained = False
    
    def train(self, chains, n_iter=100):
        if not MODULE_STATUS['hmmlearn']:
            raise ImportError("hmmlearn nicht installiert")
        
        X, lengths = self._prepare_data(chains)
        
        if len(X) == 0:
            raise ValueError("Keine gültigen Daten zum Trainieren")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Versuche verschiedene Konfigurationen
            success = False
            last_error = None
            
            try:
                self.model = hmm.MultinomialHMM(
                    n_components=self.n_states,
                    n_iter=n_iter,
                    random_state=42,
                    tol=0.01,
                    init_params="ste",
                    params="ste"
                )
                self.model.fit(X, lengths)
                success = True
            except Exception as e:
                last_error = e
                
                try:
                    if X.ndim == 2 and X.shape[1] > 1:
                        X_int = np.argmax(X, axis=1).reshape(-1, 1).astype(np.int32)
                    else:
                        X_int = X.astype(np.int32)
                    
                    self.model = hmm.MultinomialHMM(
                        n_components=self.n_states,
                        n_iter=n_iter,
                        random_state=42
                    )
                    self.model.fit(X_int, lengths)
                    success = True
                except Exception as e2:
                    last_error = e2
        
        if not success:
            raise ValueError(f"HMM-Training fehlgeschlagen: {last_error}")
        
        self.trained = True
        self.confidence = self._calculate_confidence()
        
        return self.model
    
    def _prepare_data(self, chains):
        all_symbols = set()
        for chain in chains:
            for sym in chain:
                all_symbols.add(sym)
        
        if not all_symbols:
            return np.array([]).reshape(-1, 1), np.array([])
        
        self.code_to_idx = {sym: i for i, sym in enumerate(sorted(all_symbols))}
        self.idx_to_code = {i: sym for sym, i in self.code_to_idx.items()}
        self.n_features = len(all_symbols)
        
        X_list = []
        lengths = []
        
        for chain in chains:
            if chain:
                seq_length = len(chain)
                one_hot_seq = np.zeros((seq_length, self.n_features), dtype=np.int32)
                valid = True
                for i, sym in enumerate(chain):
                    if sym in self.code_to_idx:
                        one_hot_seq[i, self.code_to_idx[sym]] = 1
                    else:
                        valid = False
                        break
                if valid:
                    X_list.append(one_hot_seq)
                    lengths.append(seq_length)
        
        if not X_list:
            return np.array([]), np.array([])
        
        X = np.vstack(X_list)
        return X, np.array(lengths, dtype=np.int32)
    
    def _calculate_confidence(self):
        if not self.model:
            return 0.5
        base_conf = 0.7
        if hasattr(self.model, 'monitor_'):
            if hasattr(self.model.monitor_, 'converged') and self.model.monitor_.converged:
                base_conf += 0.2
        return min(1.0, base_conf)
    
    def explain(self, data, detail_level='normal'):
        explanation = {
            'model': self.name,
            'confidence': self.confidence,
            'type': None,
            'content': []
        }
        
        if not self.trained:
            explanation['content'] = ["Modell nicht trainiert"]
            return explanation
        
        if isinstance(data, list):
            explanation['type'] = 'sequence'
            states, prob = self.decode_chain(data)
            if states is not None:
                explanation['content'].append(f"Dekodierte Phasen (p={prob:.4f}):")
                for i, (sym, state) in enumerate(zip(data, states)):
                    explanation['content'].append(f"  {i+1}. {sym} → {self.state_names[state]}")
            else:
                explanation['content'] = ["Dekodierung fehlgeschlagen"]
        
        elif isinstance(data, str):
            explanation['type'] = 'symbol'
            symbol = data
            if symbol in self.code_to_idx:
                sym_idx = self.code_to_idx[symbol]
                explanation['content'].append(f"Wahrscheinlichste Phasen für {symbol}:")
                probs = [(state, self.model.emissionprob_[state, sym_idx]) 
                        for state in range(self.n_states)]
                probs.sort(key=lambda x: -x[1])
                for state, prob in probs[:3]:
                    explanation['content'].append(f"  {self.state_names[state]}: {prob:.3f}")
            else:
                explanation['content'] = [f"Symbol {symbol} nicht im Modell"]
        
        return explanation
    
    def decode_chain(self, chain):
        if not self.trained or self.model is None:
            return None, None
        
        X, _ = self._prepare_data([chain])
        if len(X) == 0:
            return None, None
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logprob, states = self.model.decode(X, algorithm="viterbi")
                return states, np.exp(logprob)
        except:
            return None, None
    
    def generate_chain(self, max_length=20):
        if not self.trained or self.model is None:
            return []
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, states = self.model.sample(max_length)
                chain = []
                if X.ndim == 2 and X.shape[1] > 1:
                    indices = np.argmax(X, axis=1)
                else:
                    indices = X.flatten()
                for idx in indices:
                    idx_int = int(round(idx))
                    if idx_int in self.idx_to_code:
                        chain.append(self.idx_to_code[idx_int])
                return chain
        except Exception as e:
            print(f"Fehler bei HMM-Generierung: {e}")
            return []
    
    def get_parameters_string(self):
        if not self.trained or self.model is None:
            return "Modell nicht trainiert"
        
        lines = ["=" * 60, f"{self.name}", "=" * 60, ""]
        lines.append("Startwahrscheinlichkeiten:")
        for i in range(self.n_states):
            lines.append(f"  {self.state_names[i]}: {self.model.startprob_[i]:.3f}")
        
        lines.append("\nÜbergangsmatrix:")
        for i in range(self.n_states):
            row = "  " + " ".join([f"{self.model.transmat_[i,j]:.3f}" for j in range(self.n_states)])
            lines.append(f"{self.state_names[i]}: {row}")
        
        lines.append(f"\nKonfidenz: {self.confidence:.0%}")
        return "\n".join(lines)


# ============================================================================
# CRF - CONDITIONAL RANDOM FIELDS (angepasst)
# ============================================================================

class ARSCRFModel(XAIModel):
    """CRF für kontext-sensitive Sequenzanalyse"""
    
    def __init__(self):
        super().__init__("CRF - Conditional Random Fields")
        self.description = "Kontext-sensitive Vorhersage mit Feature-Gewichten"
        self.crf = None
        self.feature_importances = {}
    
    def train(self, chains, max_iterations=100):
        if not MODULE_STATUS['crf']:
            raise ImportError("sklearn-crfsuite nicht installiert")
        
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=max_iterations,
            all_possible_transitions=True
        )
        
        X, y = self._prepare_data(chains)
        self.crf.fit(X, y)
        self.trained = True
        self.confidence = self._calculate_confidence()
        self._extract_feature_importances()
        
        return self.crf
    
    def _prepare_data(self, sequences):
        X = []
        y = []
        for seq in sequences:
            X_seq = [self._extract_features(seq, i) for i in range(len(seq))]
            y_seq = [sym for sym in seq]
            X.append(X_seq)
            y.append(y_seq)
        return X, y
    
    def _extract_features(self, sequence, i):
        features = {
            'bias': 1.0,
            'symbol': sequence[i],
            'position': i,
            'is_first': i == 0,
            'is_last': i == len(sequence) - 1,
        }
        
        for offset in [-2, -1, 1, 2]:
            if 0 <= i + offset < len(sequence):
                features[f'context_{offset:+d}'] = sequence[i + offset]
        
        if i > 0:
            features['bigram'] = f"{sequence[i-1]}_{sequence[i]}"
        
        return features
    
    def _calculate_confidence(self):
        if not hasattr(self.crf, 'state_features_'):
            return 0.5
        n_features = len(self.crf.state_features_)
        return round(min(1.0, n_features / 100), 3)
    
    def _extract_feature_importances(self):
        if not hasattr(self.crf, 'state_features_'):
            return
        for (attr, label), weight in self.crf.state_features_.items():
            if attr not in self.feature_importances:
                self.feature_importances[attr] = []
            self.feature_importances[attr].append((label, weight))
    
    def explain(self, data, detail_level='normal'):
        explanation = {
            'model': self.name,
            'confidence': self.confidence,
            'type': None,
            'content': []
        }
        
        if not self.trained:
            explanation['content'] = ["Modell nicht trainiert"]
            return explanation
        
        if isinstance(data, list):
            explanation['type'] = 'sequence'
            try:
                X = [self._extract_features(data, i) for i in range(len(data))]
                pred = self.crf.predict([X])[0]
                
                explanation['content'].append("Vorhergesagte Sequenz:")
                for i, (sym, pred_sym) in enumerate(zip(data, pred)):
                    match = "✓" if sym == pred_sym else "✗"
                    explanation['content'].append(f"  {i+1}. {sym} → {pred_sym} {match}")
                
                if detail_level == 'detailed':
                    explanation['content'].append("\nWichtigste Merkmale:")
                    if hasattr(self.crf, 'state_features_'):
                        top_features = sorted(
                            self.crf.state_features_.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:5]
                        for (attr, label), weight in top_features:
                            explanation['content'].append(f"  {attr} → {label}: {weight:+.3f}")
            except Exception as e:
                explanation['content'] = [f"Fehler bei Vorhersage: {e}"]
        
        return explanation
    
    def get_feature_string(self, n=20):
        if not hasattr(self.crf, 'state_features_'):
            return "Keine Feature-Informationen"
        
        lines = ["=" * 60, f"{self.name} - Wichtigste Merkmale", "=" * 60, ""]
        top = sorted(self.crf.state_features_.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
        for (attr, label), weight in top:
            lines.append(f"{attr:30s} → {label:4s} : {weight:+.4f}")
        lines.append(f"\nKonfidenz: {self.confidence:.0%}")
        return "\n".join(lines)


# ============================================================================
# GENERIERUNGSKOMPONENTE
# ============================================================================

class ChainGenerator(XAIModel):
    """Generiert neue Ketten basierend auf trainierten Modellen"""
    
    def __init__(self):
        super().__init__("Generator - Synthetische Ketten")
        self.description = "Generiert neue Sequenzen aus gelernten Modellen"
        self.source_model = None
    
    def train(self, chains):
        self.trained = True
        self.confidence = 0.8
        return True
    
    def set_source_model(self, model):
        self.source_model = model
    
    def generate(self, count=5, max_length=20):
        if not self.source_model:
            return []
        
        chains = []
        for i in range(count):
            chain = None
            if hasattr(self.source_model, 'generate_chain'):
                try:
                    chain = self.source_model.generate_chain(max_length=max_length)
                except TypeError:
                    try:
                        chain = self.source_model.generate_chain()
                        if chain and len(chain) > max_length:
                            chain = chain[:max_length]
                    except:
                        pass
            elif hasattr(self.source_model, 'model') and hasattr(self.source_model.model, 'sample'):
                try:
                    X, states = self.source_model.model.sample(max_length)
                    chain = []
                    for idx in X.flatten():
                        if int(idx) in self.source_model.idx_to_code:
                            chain.append(self.source_model.idx_to_code[int(idx)])
                except:
                    pass
            
            if chain and len(chain) > 0:
                chains.append(chain)
        
        return chains
    
    def explain(self, data, detail_level='normal'):
        explanation = {
            'model': self.name,
            'confidence': self.confidence,
            'type': 'generation',
            'content': []
        }
        
        if isinstance(data, int):
            n = data
            chains = self.generate(n)
            explanation['content'].append(f"{n} generierte Ketten:")
            for i, chain in enumerate(chains, 1):
                explanation['content'].append(f"  {i}. {' → '.join(chain)}")
            
            if detail_level == 'detailed' and self.source_model:
                explanation['content'].append(f"\nBasiert auf Modell: {self.source_model.name}")
                explanation['content'].append(f"Modell-Konfidenz: {self.source_model.get_confidence():.0%}")
        
        return explanation


# ============================================================================
# PETRI-NETZE (angepasst)
# ============================================================================

if MODULE_STATUS['networkx']:
    class ARSPetriNet(XAIModel):
        """Petri-Netze für ressourcenbasierte Prozessmodellierung"""
        
        def __init__(self, name="ARS_PetriNet"):
            super().__init__(f"Petri-Netz - {name}")
            self.description = "Ressourcen-basierte Prozessmodellierung mit Token"
            self.name = name
            self.places = {}
            self.transitions = {}
            self.arcs = []
            self.tokens = {}
        
        def train(self, chains):
            all_symbols = set()
            for chain in chains:
                for sym in chain:
                    all_symbols.add(sym)
            
            self.add_place("p_start", initial_tokens=1)
            self.add_place("p_end", initial_tokens=0)
            
            for i, sym in enumerate(sorted(all_symbols)):
                self.add_place(f"p_{sym}_ready", initial_tokens=0)
                self.add_transition(f"t_{sym}")
                if i == 0:
                    self.add_arc("p_start", f"t_{sym}")
                self.add_arc(f"t_{sym}", f"p_{sym}_ready")
            
            self.trained = True
            self.confidence = self._calculate_confidence(chains)
            return True
        
        def add_place(self, name, initial_tokens=0, place_type="normal"):
            self.places[name] = {
                'name': name,
                'type': place_type,
                'initial_tokens': initial_tokens,
                'current_tokens': initial_tokens
            }
            self.tokens[name] = initial_tokens
        
        def add_transition(self, name, transition_type="speech_act", guard=None, subnet=None):
            self.transitions[name] = {
                'name': name,
                'type': transition_type,
                'guard': guard,
                'subnet': subnet
            }
        
        def add_arc(self, source, target, weight=1):
            self.arcs.append({'source': source, 'target': target, 'weight': weight})
        
        def _calculate_confidence(self, chains):
            if not chains:
                return 0.0
            total_symbols = sum(len(chain) for chain in chains)
            covered = 0
            for chain in chains:
                self.reset()
                for sym in chain:
                    trans_name = f"t_{sym}"
                    if trans_name in self.transitions and self.is_enabled(trans_name):
                        covered += 1
                        self.fire(trans_name)
            coverage = covered / total_symbols if total_symbols > 0 else 0
            return round(coverage, 3)
        
        def reset(self):
            for place_name, place_data in self.places.items():
                self.tokens[place_name] = place_data['initial_tokens']
        
        def is_enabled(self, transition):
            if transition not in self.transitions:
                return False
            for arc in self.arcs:
                if arc['target'] == transition and arc['source'] in self.places:
                    if self.tokens.get(arc['source'], 0) < arc['weight']:
                        return False
            return True
        
        def fire(self, transition):
            if not self.is_enabled(transition):
                return False
            for arc in self.arcs:
                if arc['target'] == transition and arc['source'] in self.places:
                    self.tokens[arc['source']] -= arc['weight']
            for arc in self.arcs:
                if arc['source'] == transition and arc['target'] in self.places:
                    self.tokens[arc['target']] = self.tokens.get(arc['target'], 0) + arc['weight']
            return True
        
        def explain(self, data, detail_level='normal'):
            explanation = {
                'model': self.name,
                'confidence': self.confidence,
                'type': None,
                'content': []
            }
            
            if isinstance(data, list):
                explanation['type'] = 'simulation'
                self.reset()
                explanation['content'].append(f"Simulation der Kette:")
                for i, sym in enumerate(data):
                    trans_name = f"t_{sym}"
                    enabled = self.is_enabled(trans_name)
                    if enabled:
                        self.fire(trans_name)
                        status = "✓ aktiviert"
                    else:
                        status = "✗ nicht aktiviert"
                    explanation['content'].append(f"  {i+1}. {sym}: {status}")
            
            elif isinstance(data, str):
                explanation['type'] = 'transition'
                trans_name = f"t_{data}"
                if trans_name in self.transitions:
                    enabled = self.is_enabled(trans_name)
                    explanation['content'].append(f"Transition {trans_name}:")
                    explanation['content'].append(f"  Aktiviert: {'Ja' if enabled else 'Nein'}")
                else:
                    explanation['content'] = [f"Transition {trans_name} nicht gefunden"]
            
            return explanation
        
        def get_net_string(self):
            lines = ["=" * 60, f"{self.name}", "=" * 60, ""]
            lines.append(f"Stellen ({len(self.places)}):")
            for place, data in self.places.items():
                lines.append(f"  {place}: {data['initial_tokens']} Token ({data['type']})")
            lines.append(f"\nTransitionen ({len(self.transitions)}):")
            for trans in self.transitions:
                lines.append(f"  {trans}")
            lines.append(f"\nKanten ({len(self.arcs)}):")
            for arc in self.arcs:
                lines.append(f"  {arc['source']} → {arc['target']} [{arc['weight']}]")
            lines.append(f"\nKonfidenz: {self.confidence:.0%}")
            return "\n".join(lines)


# ============================================================================
# XAI MODELLVERWALTER
# ============================================================================

class XAIModelManager:
    """Verwaltet alle XAI-Modelle und ermöglicht Vergleich"""
    
    def __init__(self):
        self.models = {}
        self.active_models = set()
    
    def register_model(self, name, model):
        self.models[name] = model
        self.active_models.add(name)
    
    def activate_model(self, name):
        if name in self.models:
            self.active_models.add(name)
    
    def deactivate_model(self, name):
        self.active_models.discard(name)
    
    def train_all(self, chains):
        results = {}
        for name in self.active_models:
            if name in self.models:
                try:
                    self.models[name].train(chains)
                    results[name] = {
                        'success': True,
                        'confidence': self.models[name].get_confidence()
                    }
                except Exception as e:
                    results[name] = {
                        'success': False,
                        'error': str(e)
                    }
        return results
    
    def explain_all(self, data, detail_level='normal'):
        explanations = {}
        for name in self.active_models:
            if name in self.models:
                try:
                    explanations[name] = self.models[name].explain(data, detail_level)
                except Exception as e:
                    explanations[name] = {
                        'model': name,
                        'error': str(e),
                        'content': [f"Fehler: {e}"]
                    }
        return explanations
    
    def get_model_info(self):
        info = {}
        for name, model in self.models.items():
            info[name] = {
                'name': model.name,
                'description': model.description,
                'confidence': model.get_confidence(),
                'trained': model.trained,
                'active': name in self.active_models
            }
        return info


# ============================================================================
# DATENVALIDIERUNG
# ============================================================================

class DataValidator:
    """Prüft die geladenen Transkripte auf Qualität und Konsistenz"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
    
    def validate_chains(self, chains):
        self.issues = []
        self.warnings = []
        
        if not chains:
            self.issues.append(("error", "Keine Ketten gefunden"))
            return self.issues, self.warnings
        
        empty_chains = [i for i, chain in enumerate(chains) if not chain]
        if empty_chains:
            self.warnings.append(("warning", f"Leere Ketten an Positionen: {empty_chains}"))
        
        short_chains = [i for i, chain in enumerate(chains) if len(chain) < 2]
        if short_chains:
            self.warnings.append(("warning", f"Sehr kurze Ketten (<2 Symbole): {short_chains}"))
        
        all_symbols = set()
        for chain in chains:
            for symbol in chain:
                all_symbols.add(symbol)
        
        symbol_groups = self._group_similar_symbols(all_symbols)
        for base, similar in symbol_groups.items():
            if len(similar) > 1:
                self.warnings.append(("info", f"Ähnliche Symbole gefunden: {', '.join(similar)}"))
        
        return self.issues, self.warnings
    
    def _group_similar_symbols(self, symbols):
        groups = defaultdict(list)
        for sym in symbols:
            normalized = re.sub(r'[^A-Za-z]', '', sym.upper())
            groups[normalized].append(sym)
        return {k: v for k, v in groups.items() if len(v) > 1}
    
    def suggest_corrections(self):
        suggestions = []
        for severity, msg in self.warnings:
            if "leer" in msg:
                suggestions.append("Leere Zeilen sollten entfernt werden")
            elif "kurz" in msg:
                suggestions.append("Sehr kurze Ketten könnten Rauschen sein")
            elif "Ähnlich" in msg:
                suggestions.append("Prüfen Sie auf Tippfehler in den Symbolen")
        return suggestions


# ============================================================================
# VISUALISIERUNGSKOMPONENTE
# ============================================================================

class DerivationVisualizer:
    """Visualisiert die gelernte Grammatik"""
    
    def __init__(self, root, plot_thread):
        self.root = root
        self.plot_thread = plot_thread
    
    def plot_grammar_hierarchy(self, grammar):
        """Zeigt die hierarchische Struktur der Grammatik"""
        if not MODULE_STATUS['networkx']:
            return
        
        G = nx.DiGraph()
        
        # Füge Knoten hinzu
        for nt in grammar.nonterminals:
            G.add_node(nt, type='nonterminal', size=500)
        for t in grammar.terminals:
            G.add_node(t, type='terminal', size=300)
        
        # Füge Kanten hinzu
        for nt, productions in grammar.rules.items():
            for prod, prob in productions:
                for sym in prod:
                    G.add_edge(nt, sym, weight=prob, label=f"{prob:.2f}")
        
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Zeichne Nonterminale
        nonterm_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'nonterminal']
        nx.draw_networkx_nodes(G, pos, nodelist=nonterm_nodes, 
                              node_color='lightgreen', node_size=500)
        
        # Zeichne Terminale
        term_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'terminal']
        nx.draw_networkx_nodes(G, pos, nodelist=term_nodes, 
                              node_color='lightblue', node_size=300)
        
        # Zeichne Kanten
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
        
        # Beschriftungen
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title("Hierarchische Struktur der Grammatik")
        plt.axis('off')
        self.plot_thread.plot(lambda: plt.show())
    
    def plot_pattern_frequency(self, grammar):
        """Zeigt die Häufigkeit der erkannten Muster"""
        if not grammar.compression_history:
            return
        
        patterns = []
        frequencies = []
        colors = []
        
        for hist in grammar.compression_history[:10]:  # Top 10
            patterns.append(hist['new_symbol'])
            freq = (hist['occurrences'] / len(grammar.chains)) * 100
            frequencies.append(freq)
            colors.append(plt.cm.viridis(freq / 100))
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(patterns)), frequencies, color=colors)
        plt.xticks(range(len(patterns)), patterns, rotation=45, ha='right')
        plt.ylabel('Vorkommen in % der Ketten')
        plt.title('Häufigkeit der erkannten Muster')
        
        # Werte auf Balken
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{freq:.0f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        self.plot_thread.plot(lambda: plt.show())


# ============================================================================
# MULTI-FORMAT EXPORTER (angepasst)
# ============================================================================

class MultiFormatExporter:
    """Exportiert Ergebnisse in verschiedene Formate"""
    
    def __init__(self):
        self.export_path = "exports"
        os.makedirs(self.export_path, exist_ok=True)
    
    def to_json(self, data, filename=None):
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.export_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return filepath
    
    def to_html(self, data, filename=None):
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.export_path, filename)
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head><title>ARSXAI9 Analysebericht</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        html.append("h1 { color: #2c3e50; }")
        html.append("h2 { color: #34495e; border-bottom: 2px solid #3498db; }")
        html.append(".pattern-box { background-color: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }")
        html.append(".confidence-high { color: green; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        html.append(f"<h1>ARSXAI9 Analysebericht</h1>")
        html.append(f"<p>Generiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        if 'grammar' in data:
            html.append("<h2>Erkannte Muster</h2>")
            for pattern in data['grammar'].get('patterns', []):
                html.append(f"<div class='pattern-box'>")
                html.append(f"<h3>{pattern['name']}</h3>")
                html.append(f"<p><strong>Sequenz:</strong> {' → '.join(pattern['sequence'])}</p>")
                html.append(f"<p><strong>Vorkommen:</strong> {pattern['frequency']:.0f}% der Ketten</p>")
                html.append(f"<p><strong>Begründung:</strong> {pattern['rationale']}</p>")
                html.append(f"</div>")
        
        html.append("</body></html>")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(html))
        
        return filepath
    
    def to_latex(self, data, filename=None):
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        filepath = os.path.join(self.export_path, filename)
        
        latex = []
        latex.append("\\documentclass{article}")
        latex.append("\\usepackage[utf8]{inputenc}")
        latex.append("\\usepackage{booktabs}")
        latex.append("\\begin{document}")
        latex.append("\\title{ARSXAI9 Analyseergebnisse}")
        latex.append(f"\\date{{{datetime.now().strftime('%Y-%m-%d')}}}")
        latex.append("\\maketitle")
        
        if 'grammar' in data:
            latex.append("\\section{Erkannte Muster}")
            latex.append("\\begin{tabular}{lll}")
            latex.append("\\toprule")
            latex.append("Muster & Sequenz & Häufigkeit \\\\")
            latex.append("\\midrule")
            for pattern in data['grammar'].get('patterns', []):
                seq = ' $\\rightarrow$ '.join(pattern['sequence'])
                latex.append(f"{pattern['name']} & {seq} & {pattern['frequency']:.0f}\\% \\\\")
            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
        
        latex.append("\\end{document}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(latex))
        
        return filepath


# ============================================================================
# GUI - HAUPTFENSTER (angepasst für PCFG-basierte XAI)
# ============================================================================

class ARSXAI9GUI:
    """Haupt-GUI für ARSXAI9"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ARSXAI9 - PCFG-basierte Musteranalyse mit XAI")
        self.root.geometry("1600x1000")
        
        self.plot_thread = PlotThread(root)
        self.update_queue = queue.Queue()
        self.process_updates()
        
        # Daten
        self.chains = []
        self.terminals = set()
        self.comments = []
        self.delimiter = tk.StringVar(value=",")
        
        # Komponenten
        self.validator = DataValidator()
        self.grammar = GrammarInducer()  # ZENTRALE WISSENSBASIS
        self.explainer = None
        self.model_manager = XAIModelManager()
        self.visualizer = DerivationVisualizer(root, self.plot_thread)
        self.exporter = MultiFormatExporter()
        
        # Weitere Modelle registrieren (optional)
        self._register_models()
        
        # GUI erstellen
        self.create_menu()
        self.create_main_panels()
        self.status_var = tk.StringVar(value="Bereit")
        self.create_statusbar()
        
        self.show_module_status()
    
    def _register_models(self):
        """Registriert alle XAI-Modelle"""
        # ARS 3.0 ist das Hauptmodell
        self.model_manager.register_model('ARS30', self.grammar)
        
        # Optionale Zusatzmodelle
        ars20 = ARS20()
        self.model_manager.register_model('ARS20', ars20)
        
        if MODULE_STATUS['hmmlearn']:
            hmm_model = ARSHiddenMarkovModel(n_states=5)
            self.model_manager.register_model('HMM', hmm_model)
        
        if MODULE_STATUS['crf']:
            crf_model = ARSCRFModel()
            self.model_manager.register_model('CRF', crf_model)
        
        if MODULE_STATUS['networkx']:
            petri_model = ARSPetriNet("ARS_PetriNet")
            self.model_manager.register_model('Petri', petri_model)
        
        generator = ChainGenerator()
        self.model_manager.register_model('Generator', generator)
        
        # Alle aktivieren
        for name in self.model_manager.models:
            self.model_manager.activate_model(name)
    
    def process_updates(self):
        try:
            while True:
                update_func = self.update_queue.get_nowait()
                update_func()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_updates)
    
    def safe_gui_update(self, func):
        self.update_queue.put(func)
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Datei", menu=file_menu)
        file_menu.add_command(label="Transkripte laden", command=self.load_transcripts)
        file_menu.add_command(label="Beispiel laden", command=self.load_example)
        file_menu.add_separator()
        file_menu.add_command(label="Exportieren", command=self.show_export_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self.root.quit)
        
        analyze_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analyse", menu=analyze_menu)
        analyze_menu.add_command(label="Grammatik induzieren", command=self.run_grammar_induction)
        analyze_menu.add_command(label="Alle Modelle trainieren", command=self.train_all_models)
        analyze_menu.add_command(label="Validierung durchführen", command=self.run_validation)
        
        xai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="XAI", menu=xai_menu)
        xai_menu.add_command(label="Erklärung für Symbol", command=self.ask_explanation)
        xai_menu.add_command(label="Erklärung für Sequenz", command=self.explain_sequence)
        xai_menu.add_command(label="Musterübersicht", command=self.show_patterns)
        xai_menu.add_command(label="Modellvergleich", command=self.compare_models)
        
        vis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualisierung", menu=vis_menu)
        vis_menu.add_command(label="Grammatik-Hierarchie", command=self.plot_grammar)
        vis_menu.add_command(label="Muster-Häufigkeiten", command=self.plot_patterns)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Hilfe", menu=help_menu)
        help_menu.add_command(label="Modulstatus", command=self.show_module_status)
        help_menu.add_command(label="Über", command=self.show_about)
    
    def create_main_panels(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        self.create_input_panel(left_frame)
        
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        self.create_output_panel(right_frame)
    
    def create_input_panel(self, parent):
        ttk.Label(parent, text="Eingabe", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=5)
        
        delim_frame = ttk.Frame(parent)
        delim_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(delim_frame, text="Trennzeichen:").pack(side=tk.LEFT)
        ttk.Radiobutton(delim_frame, text="Komma (,)", variable=self.delimiter, 
                       value=",").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(delim_frame, text="Semikolon (;)", variable=self.delimiter, 
                       value=";").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(delim_frame, text="Leerzeichen", variable=self.delimiter, 
                       value=" ").pack(side=tk.LEFT, padx=2)
        
        self.custom_delimiter = ttk.Entry(delim_frame, width=5)
        self.custom_delimiter.pack(side=tk.LEFT, padx=2)
        self.custom_delimiter.insert(0, "|")
        
        ttk.Label(parent, text="Transkripte (eine pro Zeile, # für Kommentare):").pack(anchor=tk.W, pady=5)
        
        self.text_input = scrolledtext.ScrolledText(parent, height=15, font=('Courier', 10))
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=5)
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Datei laden", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Parsen", command=self.parse_input).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Beispiel", command=self.load_example).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Grammatik induzieren", command=self.run_grammar_induction).pack(side=tk.LEFT, padx=2)
        
        self.info_var = tk.StringVar(value="Keine Daten geladen")
        ttk.Label(parent, textvariable=self.info_var, foreground="blue").pack(anchor=tk.W, pady=5)
        
        self.warning_text = scrolledtext.ScrolledText(parent, height=5, font=('Courier', 9), 
                                                      foreground="orange")
        self.warning_text.pack(fill=tk.X, pady=5)
    
    def create_output_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Grammatik (Haupttab)
        self.tab_grammar = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_grammar, text="Grammatik")
        self.create_grammar_tab()
        
        # Tab 2: Muster
        self.tab_patterns = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_patterns, text="Erkannte Muster")
        self.create_patterns_tab()
        
        # Tab 3: XAI
        self.tab_xai = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_xai, text="XAI-Erklärungen")
        self.create_xai_tab()
        
        # Tab 4: Modelle
        self.tab_models = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_models, text="Weitere Modelle")
        self.create_models_tab()
        
        # Tab 5: Statistiken
        self.tab_stats = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_stats, text="Statistiken")
        self.create_statistics_tab()
    
    def create_grammar_tab(self):
        control = ttk.Frame(self.tab_grammar)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Grammatik anzeigen", 
                  command=self.show_grammar).pack(side=tk.LEFT, padx=5)
        
        self.text_grammar = scrolledtext.ScrolledText(self.tab_grammar, font=('Courier', 10))
        self.text_grammar.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_patterns_tab(self):
        control = ttk.Frame(self.tab_patterns)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Musterübersicht", 
                  command=self.show_patterns).pack(side=tk.LEFT, padx=5)
        
        self.text_patterns = scrolledtext.ScrolledText(self.tab_patterns, font=('Courier', 10))
        self.text_patterns.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_xai_tab(self):
        question_frame = ttk.Frame(self.tab_xai)
        question_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(question_frame, text="Symbol:").pack(side=tk.LEFT)
        self.symbol_entry = ttk.Entry(question_frame, width=10)
        self.symbol_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(question_frame, text="Symbol erklären", 
                  command=self.ask_explanation).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(question_frame, text="Sequenz:").pack(side=tk.LEFT, padx=(20,5))
        self.sequence_entry = ttk.Entry(question_frame, width=30)
        self.sequence_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(question_frame, text="Sequenz erklären", 
                  command=self.explain_sequence).pack(side=tk.LEFT, padx=5)
        
        self.text_xai = scrolledtext.ScrolledText(self.tab_xai, font=('Courier', 10))
        self.text_xai.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_models_tab(self):
        control = ttk.Frame(self.tab_models)
        control.pack(fill=tk.X, pady=5)
        
        self.model_vars = {}
        for name in self.model_manager.models:
            if name != 'ARS30':  # ARS 3.0 ist immer aktiv
                var = tk.BooleanVar(value=True)
                self.model_vars[name] = var
                ttk.Checkbutton(control, text=name, variable=var,
                              command=lambda n=name: self.toggle_model(n)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control, text="Alle trainieren", 
                  command=self.train_all_models).pack(side=tk.LEFT, padx=20)
        
        self.text_models = scrolledtext.ScrolledText(self.tab_models, font=('Courier', 10))
        self.text_models.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_statistics_tab(self):
        control = ttk.Frame(self.tab_stats)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Statistiken berechnen", 
                  command=self.calculate_statistics).pack(side=tk.LEFT, padx=5)
        
        self.text_stats = scrolledtext.ScrolledText(self.tab_stats, font=('Courier', 10))
        self.text_stats.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_statusbar(self):
        status = ttk.Frame(self.root)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(status, length=100, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
    
    def toggle_model(self, name):
        if self.model_vars[name].get():
            self.model_manager.activate_model(name)
        else:
            self.model_manager.deactivate_model(name)
    
    def get_actual_delimiter(self):
        delim = self.delimiter.get()
        if delim == "custom":
            return self.custom_delimiter.get()
        return delim
    
    def parse_line(self, line):
        line = line.strip()
        if not line or line.startswith('#'):
            return []
        delim = self.get_actual_delimiter()
        if delim == " ":
            parts = re.split(r'\s+', line)
        else:
            parts = line.split(delim)
        return [p.strip() for p in parts if p.strip()]
    
    def parse_input(self):
        text = self.text_input.get("1.0", tk.END)
        lines = text.strip().split('\n')
        
        self.chains = []
        self.comments = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                self.comments.append(line)
                continue
            chain = self.parse_line(line)
            if chain:
                self.chains.append(chain)
        
        if self.chains:
            self.terminals = set()
            for chain in self.chains:
                for symbol in chain:
                    self.terminals.add(symbol)
            
            self.info_var.set(f"{len(self.chains)} Ketten, {len(self.terminals)} Terminale")
            self.status_var.set(f"{len(self.chains)} Ketten geladen")
            
            self.run_validation()
            self.run_grammar_induction()
        else:
            messagebox.showwarning("Warnung", "Keine gültigen Ketten gefunden!")
    
    def run_validation(self):
        if not self.chains:
            return
        
        issues, warnings = self.validator.validate_chains(self.chains)
        
        self.warning_text.delete("1.0", tk.END)
        if warnings:
            self.warning_text.insert(tk.END, "VALIDIERUNGSWARNUNGEN:\n")
            for severity, msg in warnings:
                self.warning_text.insert(tk.END, f"  {msg}\n")
            
            suggestions = self.validator.suggest_corrections()
            if suggestions:
                self.warning_text.insert(tk.END, "\nKORREKTURVORSCHLÄGE:\n")
                for s in suggestions:
                    self.warning_text.insert(tk.END, f"  • {s}\n")
        else:
            self.warning_text.insert(tk.END, "✓ Keine Validierungsprobleme")
    
    def run_grammar_induction(self):
        """Führt die Grammatikinduktion durch (ZENTRAL)"""
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        self.status_var.set("Induziere Grammatik...")
        self.progress_bar.start()
        
        def run():
            try:
                self.grammar.train(self.chains)
                self.explainer = NaturalLanguageExplainer(self.grammar)
                
                def update():
                    self.show_grammar()
                    self.show_patterns()
                    self.status_var.set(f"Grammatik induziert: {len(self.grammar.nonterminals)} Muster gefunden")
                    self.progress_bar.stop()
                
                self.safe_gui_update(update)
            except Exception as e:
                def error():
                    messagebox.showerror("Fehler", f"Grammatikinduktion fehlgeschlagen:\n{str(e)}")
                    self.progress_bar.stop()
                
                self.safe_gui_update(error)
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def show_grammar(self):
        """Zeigt die vollständige Grammatik"""
        self.text_grammar.delete("1.0", tk.END)
        self.text_grammar.insert(tk.END, self.grammar.get_grammar_string())
    
    def show_patterns(self):
        """Zeigt die erkannten Muster"""
        self.text_patterns.delete("1.0", tk.END)
        self.text_patterns.insert(tk.END, self.grammar.get_pattern_summary())
    
    def ask_explanation(self):
        """Erklärt ein Symbol"""
        if not self.explainer:
            messagebox.showerror("Fehler", "Keine Grammatik vorhanden!")
            return
        
        symbol = self.symbol_entry.get().strip()
        if not symbol:
            messagebox.showwarning("Warnung", "Bitte ein Symbol eingeben!")
            return
        
        explanation = self.explainer.explain_symbol(symbol)
        
        self.text_xai.delete("1.0", tk.END)
        self.text_xai.insert(tk.END, explanation)
    
    def explain_sequence(self):
        """Erklärt eine Sequenz"""
        if not self.explainer:
            messagebox.showerror("Fehler", "Keine Grammatik vorhanden!")
            return
        
        seq_text = self.sequence_entry.get().strip()
        if not seq_text:
            messagebox.showwarning("Warnung", "Bitte eine Sequenz eingeben!")
            return
        
        # Parse die Sequenz (gleiches Trennzeichen wie Haupteingabe)
        delim = self.get_actual_delimiter()
        if delim == " ":
            sequence = seq_text.split()
        else:
            sequence = [s.strip() for s in seq_text.split(delim) if s.strip()]
        
        explanation = self.explainer.explain_sequence(sequence)
        
        self.text_xai.delete("1.0", tk.END)
        self.text_xai.insert(tk.END, explanation)
    
    def train_all_models(self):
        """Trainiert alle aktiven Zusatzmodelle"""
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        self.status_var.set("Trainiere Modelle...")
        self.progress_bar.start()
        
        def run():
            try:
                results = self.model_manager.train_all(self.chains)
                
                def update():
                    self.text_models.delete("1.0", tk.END)
                    self.text_models.insert(tk.END, "MODELL-TRAINING ERGEBNISSE\n")
                    self.text_models.insert(tk.END, "=" * 70 + "\n\n")
                    
                    for name, result in results.items():
                        if name == 'ARS30':
                            continue  # Wurde schon trainiert
                        
                        model = self.model_manager.models.get(name)
                        info = model.get_info() if model else {}
                        
                        if result.get('success'):
                            self.text_models.insert(tk.END, f"✓ {name}\n")
                            self.text_models.insert(tk.END, f"  Beschreibung: {info.get('description', '')}\n")
                            self.text_models.insert(tk.END, f"  Konfidenz: {info.get('confidence', 0):.0%}\n\n")
                        else:
                            self.text_models.insert(tk.END, f"✗ {name}: {result.get('error', 'Unbekannter Fehler')}\n\n")
                    
                    # Generator konfigurieren
                    if 'Generator' in self.model_manager.models:
                        self.model_manager.models['Generator'].set_source_model(self.grammar)
                    
                    self.status_var.set("Modell-Training abgeschlossen")
                    self.progress_bar.stop()
                
                self.safe_gui_update(update)
            except Exception as e:
                def error():
                    messagebox.showerror("Fehler", f"Training fehlgeschlagen:\n{str(e)}")
                    self.progress_bar.stop()
                
                self.safe_gui_update(error)
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def compare_models(self):
        """Vergleicht die Erklärungen verschiedener Modelle für ein Symbol"""
        if not self.explainer:
            messagebox.showerror("Fehler", "Keine Analyse vorhanden!")
            return
        
        symbol = self.symbol_entry.get().strip()
        if not symbol:
            messagebox.showwarning("Warnung", "Bitte ein Symbol eingeben!")
            return
        
        explanations = self.model_manager.explain_all(symbol)
        
        self.text_xai.delete("1.0", tk.END)
        self.text_xai.insert(tk.END, f"🔍 **MODELLVERGLEICH für '{symbol}'**\n")
        self.text_xai.insert(tk.END, "=" * 70 + "\n\n")
        
        for model_name, exp in explanations.items():
            self.text_xai.insert(tk.END, f"\n📌 **{model_name}**\n")
            self.text_xai.insert(tk.END, "-" * 40 + "\n")
            if 'content' in exp:
                for line in exp['content']:
                    self.text_xai.insert(tk.END, f"{line}\n")
    
    def plot_grammar(self):
        """Visualisiert die Grammatik-Hierarchie"""
        if not MODULE_STATUS['networkx']:
            messagebox.showerror("Fehler", "networkx nicht verfügbar!")
            return
        
        self.visualizer.plot_grammar_hierarchy(self.grammar)
    
    def plot_patterns(self):
        """Visualisiert die Muster-Häufigkeiten"""
        self.visualizer.plot_pattern_frequency(self.grammar)
    
    def calculate_statistics(self):
        """Berechnet grundlegende Statistiken"""
        if not self.chains:
            return
        
        stats = []
        stats.append("STATISTISCHE KENNZAHLEN")
        stats.append("=" * 60)
        
        chain_lengths = [len(chain) for chain in self.chains]
        stats.append(f"\nAnzahl Ketten: {len(self.chains)}")
        stats.append(f"Anzahl Terminale: {len(self.terminals)}")
        stats.append(f"Durchschnittliche Länge: {np.mean(chain_lengths):.1f}")
        stats.append(f"Minimale Länge: {min(chain_lengths)}")
        stats.append(f"Maximale Länge: {max(chain_lengths)}")
        
        stats.append(f"\nErkannte Muster: {len(self.grammar.nonterminals)}")
        stats.append(f"Kompressionsrate: {self.grammar.confidence:.0%}")
        
        symbol_counts = Counter()
        for chain in self.chains:
            symbol_counts.update(chain)
        
        stats.append("\nHäufigste Symbole:")
        for sym, count in symbol_counts.most_common(10):
            stats.append(f"  {sym}: {count}x")
        
        self.text_stats.delete("1.0", tk.END)
        self.text_stats.insert(tk.END, "\n".join(stats))
    
    def show_export_dialog(self):
        """Zeigt Export-Dialog"""
        if not self.grammar.trained:
            messagebox.showerror("Fehler", "Keine Grammatik vorhanden!")
            return
        
        # Bereite Export-Daten vor
        patterns = []
        for hist in self.grammar.compression_history:
            patterns.append({
                'name': hist['new_symbol'],
                'sequence': hist['sequence'],
                'occurrences': hist['occurrences'],
                'frequency': (hist['occurrences'] / len(self.chains)) * 100,
                'rationale': self.grammar.reflection.interpretation_log[hist['iteration']]['rationale']
            })
        
        export_data = {
            'grammar': {
                'patterns': patterns,
                'terminals': list(self.grammar.terminals),
                'nonterminals': list(self.grammar.nonterminals),
                'confidence': self.grammar.confidence
            },
            'statistics': {
                'n_chains': len(self.chains),
                'n_terminals': len(self.terminals),
                'avg_length': np.mean([len(c) for c in self.chains]) if self.chains else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Exportieren")
        dialog.geometry("300x250")
        
        ttk.Label(dialog, text="Export-Format:").pack(pady=10)
        
        def export_json():
            filepath = self.exporter.to_json(export_data)
            messagebox.showinfo("Export erfolgreich", f"Gespeichert als:\n{filepath}")
            dialog.destroy()
        
        def export_html():
            filepath = self.exporter.to_html(export_data)
            messagebox.showinfo("Export erfolgreich", f"Gespeichert als:\n{filepath}")
            dialog.destroy()
        
        def export_latex():
            filepath = self.exporter.to_latex(export_data)
            messagebox.showinfo("Export erfolgreich", f"Gespeichert als:\n{filepath}")
            dialog.destroy()
        
        ttk.Button(dialog, text="JSON", command=export_json).pack(pady=5)
        ttk.Button(dialog, text="HTML (Bericht)", command=export_html).pack(pady=5)
        ttk.Button(dialog, text="LaTeX", command=export_latex).pack(pady=5)
    
    def load_file(self):
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
                self.parse_input()
            except Exception as e:
                messagebox.showerror("Fehler", f"Kann Datei nicht laden:\n{e}")
    
    def load_transcripts(self):
        self.load_file()
    
    def load_example(self):
        """Lädt ein Beispiel mit C-Symbolen"""
        example = """# Beispieltranskripte mit C-Symbolen (analog zu Verkaufsgesprächen)
# Jede Zeile enthält eine Sequenz von Terminalzeichen

# Transkript 1: Standard
CBG, BBG, CBBd, BBBd, CBA, BBA, CBBd, BBBd, CBA, BAA, CAA, BAB, CAB

# Transkript 2: Mit Wiederholungen
CBG, BBG, CBBd, BBBd, CBBd, BBBd, CBA, BBA, BAA, CAA, BAB, CAB

# Transkript 3: Kurz
CBG, BBG, CBBd, BBBd, CBA, BBA, BAA, CAA, BAB, CAB

# Transkript 4: Mit Beratung
CBG, BBG, CBBd, BBBd, CBA, BBA, CAE, BAE, CBA, BBA, BAA, CAA, BAB, CAB

# Transkript 5: Viele Wiederholungen
CBG, BBG, CBBd, BBBd, CBBd, BBBd, CBBd, BBBd, CBA, BBA, BAA, CAA, BAB, CAB"""
        
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", example)
        self.parse_input()
    
    def show_module_status(self):
        status_text = "MODULSTATUS:\n"
        status_text += "=" * 40 + "\n"
        for module, available in MODULE_STATUS.items():
            status = "✓ verfügbar" if available else "✗ nicht verfügbar"
            status_text += f"{module:15s}: {status}\n"
        
        status_text += "\nREGISTRIERTE MODELLE:\n"
        for name, info in self.model_manager.get_model_info().items():
            status_text += f"  {name}: {info['description'][:30]}...\n"
        
        messagebox.showinfo("Modulstatus", status_text)
    
    def show_about(self):
        about = """ARSXAI9 - Algorithmic Recursive Sequence Analysis with Explainable AI

Version 9.0 (PCFG-basiert - KEINE 5-Bit-Kodierung mehr!)

Zentrale Neuerungen:
• Hierarchische Grammatik als einzige Wissensbasis
• Natürlichsprachliche Erklärungen aus gelernten Mustern
• Keine willkürlichen Kodierungen oder Annahmen
• Muster werden zu Nonterminalen abstrahiert

Integrierte Modelle (optional):
• ARS 2.0 - Basis-Grammatik
• HMM - Bayessche Netze
• CRF - Conditional Random Fields
• Petri-Netze - Ressourcenmodellierung

XAI-Features:
• Erklärungen in natürlicher Sprache
• Hierarchische Musterübersicht
• Modellvergleich
• Umfangreiche Exportformate

© 2024 - Explainable AI Research"""
        
        messagebox.showinfo("Über ARSXAI9", about)


# ============================================================================
# HAUPTFUNKTION
# ============================================================================

def main():
    """Hauptfunktion"""
    root = tk.Tk()
    app = ARSXAI9GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
