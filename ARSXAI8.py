"""
ARSXAI8.py - Algorithmic Recursive Sequence Analysis with Explainable AI
========================================================================
Universelle Analyseplattform für beliebige Terminalzeichenketten mit
automatischer Strukturableitung und XAI-Komponenten.

Version: 8.0 (Vollständige Integration aller ARS-Komponenten mit Generierungs-Korrekturen)

Integrierte Module:
- ARS 2.0: Basis-Grammatik mit Bigramm-Wahrscheinlichkeiten
- ARS 3.0: Hierarchische Grammatikinduktion mit Nonterminalen
- HMM: Bayessche Netze für latente Phasen
- CRF: Conditional Random Fields für kontext-sensitive Analyse
- Petri-Netze: Ressourcen-basierte Prozessmodellierung
- Generierung: Synthetische Erzeugung neuer Sequenzen

XAI-Features:
- Einheitliches Erklärungs-Interface für alle Modelle
- Modellvergleich mit Konsensanalyse
- Konfidenzmetriken für alle Ableitungen
- Interaktive "Warum?"-Erklärungen
- Export in verschiedene Formate
"""

import sys
import subprocess
import importlib
import warnings
import traceback
import os
import json
from datetime import datetime
from collections import Counter, defaultdict
import threading
import queue
import re

# ============================================================================
# WARNUNGEN UNTERDRÜCKEN
# ============================================================================

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

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
    print("ARSXAI8 - PAKETPRÜFUNG")
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
# GRAPHVIZ KONFIGURATION FÜR WINDOWS
# ============================================================================

import os
import sys
import shutil

def setup_graphviz():
    """Konfiguriert Graphviz für Windows und prüft Installation"""
    GRAPHVIZ_AVAILABLE = False
    
    if sys.platform == 'win32':
        # Mögliche Installationspfade
        possible_paths = [
            r'C:\Program Files\Graphviz\bin',
            r'C:\Program Files (x86)\Graphviz\bin',
            r'C:\Graphviz\bin',
        ]
        
        # Prüfe zuerst mit shutil.which()
        if shutil.which('dot'):
            print("✓ Graphviz (dot) im PATH gefunden")
            GRAPHVIZ_AVAILABLE = True
        else:
            # Versuche Pfade manuell hinzuzufügen
            for path in possible_paths:
                dot_exe = os.path.join(path, 'dot.exe')
                if os.path.exists(dot_exe):
                    os.environ['PATH'] += os.pathsep + path
                    print(f"✓ Graphviz gefunden in: {path}")
                    GRAPHVIZ_AVAILABLE = True
                    break
            
            if not GRAPHVIZ_AVAILABLE:
                print("\n" + "="*70)
                print("⚠️  GRAPHVIX NICHT GEFUNDEN")
                print("="*70)
                print("\nFür Automaten-Visualisierung wird Graphviz benötigt.")
                print("\nInstallation:")
                print("  1. Laden Sie Graphviz herunter von:")
                print("     https://graphviz.org/download/")
                print("  2. Wählen Sie: graphviz-12.2.1 (64-bit) EXE installer")
                print("  3. Bei Installation HAKEN SETZEN bei:")
                print("     'Add Graphviz to the system PATH'")
                print("  4. Programm neu starten")
                print("\nOder mit Chocolatey (als Administrator):")
                print("  choco install graphviz")
                print("="*70 + "\n")
    else:
        # Linux/Mac: Prüfe mit which
        if shutil.which('dot'):
            print("✓ Graphviz (dot) gefunden")
            GRAPHVIZ_AVAILABLE = True
        else:
            print("\n⚠️  Graphviz nicht gefunden. Installieren mit:")
            print("  Ubuntu/Debian: sudo apt-get install graphviz")
            print("  Mac: brew install graphviz")
    
    return GRAPHVIZ_AVAILABLE

# Graphviz konfigurieren
GRAPHVIZ_AVAILABLE = setup_graphviz()

# ============================================================================
# IMPORTS
# ============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Optionale Imports mit Status-Erfassung
MODULE_STATUS = {}

try:
    import networkx as nx
    MODULE_STATUS['networkx'] = True
except ImportError:
    MODULE_STATUS['networkx'] = False

try:
    from hmmlearn import hmm
    MODULE_STATUS['hmmlearn'] = True
except ImportError:
    MODULE_STATUS['hmmlearn'] = False

try:
    from sklearn_crfsuite import CRF
    MODULE_STATUS['crf'] = True
except ImportError:
    MODULE_STATUS['crf'] = False

try:
    from sentence_transformers import SentenceTransformer
    MODULE_STATUS['transformer'] = True
except ImportError:
    MODULE_STATUS['transformer'] = False

try:
    import seaborn as sns
    MODULE_STATUS['seaborn'] = True
except ImportError:
    MODULE_STATUS['seaborn'] = False

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
    """
    Einheitliches XAI-Interface für alle Modelle
    """
    
    def __init__(self, name):
        self.name = name
        self.description = ""
        self.confidence = 0.5
        self.trained = False
    
    def train(self, chains):
        """Trainiert das Modell auf den Daten"""
        raise NotImplementedError
    
    def explain(self, data, detail_level='normal'):
        """
        Liefert XAI-konforme Erklärung
        
        Args:
            data: Zu erklärende Daten (Kette, Symbol, Übergang)
            detail_level: 'simple', 'normal', oder 'detailed'
        
        Returns:
            Dictionary mit Erklärung
        """
        raise NotImplementedError
    
    def get_confidence(self):
        """Liefert Konfidenz für das Modell (0-1)"""
        return self.confidence
    
    def get_info(self):
        """Liefert Metainformationen über das Modell"""
        return {
            'name': self.name,
            'description': self.description,
            'confidence': self.confidence,
            'trained': self.trained
        }
    
    def visualize(self):
        """Optionale Visualisierung"""
        return None


# ============================================================================
# ARS 2.0 - BASIS-GRAMMATIK
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
        self.optimized_probabilities = {}
        self.history = []
    
    def train(self, chains, start_symbol=None):
        """Trainiert ARS 2.0 auf den Ketten"""
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
        """Zählt Übergänge zwischen Symbolen"""
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
        """Berechnet Wahrscheinlichkeiten aus Übergangszählungen"""
        probabilities = {}
        for start in transitions:
            total = sum(transitions[start].values())
            if total > 0:
                probabilities[start] = {end: count / total 
                                       for end, count in transitions[start].items()}
        return probabilities
    
    def _calculate_confidence(self):
        """Berechnet Konfidenz basierend auf Datenmenge"""
        if not self.chains:
            return 0.0
        
        total_transitions = sum(len(chain)-1 for chain in self.chains)
        # Je mehr Transitionen, desto höher die Konfidenz (logarithmisch)
        confidence = min(1.0, np.log10(total_transitions + 1) / 2)
        return round(confidence, 3)
    
    def explain(self, data, detail_level='normal'):
        """
        Erklärt ARS 2.0-Vorhersagen
        
        data kann sein:
        - Ein Symbol (erkläre mögliche Folgesymbole)
        - Ein Tupel (start, end) (erkläre spezifischen Übergang)
        """
        explanation = {
            'model': self.name,
            'confidence': self.confidence,
            'type': None,
            'content': []
        }
        
        if isinstance(data, tuple) and len(data) == 2:
            # Spezifischer Übergang
            start, end = data
            explanation['type'] = 'transition'
            
            if start in self.probabilities and end in self.probabilities[start]:
                prob = self.probabilities[start][end]
                count = self.transitions[start][end]
                total = sum(self.transitions[start].values())
                
                explanation['content'] = [
                    f"Übergang {start} → {end}:",
                    f"  Wahrscheinlichkeit: {prob:.1%}",
                    f"  Beobachtet: {count} von {total} Fällen",
                    f"  Konfidenz: {self.confidence:.0%}"
                ]
                
                if detail_level == 'detailed':
                    # Zeige alternative Übergänge
                    alternatives = [(e, p) for e, p in self.probabilities[start].items() if e != end]
                    alternatives.sort(key=lambda x: -x[1])
                    if alternatives:
                        explanation['content'].append("\n  Alternative Übergänge:")
                        for alt, p in alternatives[:3]:
                            explanation['content'].append(f"    {alt}: {p:.1%}")
            else:
                explanation['content'] = [f"Übergang {start} → {end} nicht beobachtet"]
        
        elif isinstance(data, str):
            # Symbol - zeige mögliche Folgesymbole
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
    
    def generate_chain(self, start_symbol=None, max_length=20, max_depth=None):
        """
        Generiert eine neue Kette basierend auf gelernten Wahrscheinlichkeiten
        
        Args:
            start_symbol: Optionales Startsymbol
            max_length: Maximale Länge
            max_depth: Alias für max_length (für Kompatibilität)
        """
        if max_depth is not None:
            max_length = max_depth
            
        if not self.trained:
            return []
        
        probs = self.optimized_probabilities if self.optimized_probabilities else self.probabilities
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
    
    def get_grammar_string(self):
        """Gibt die Grammatik als String zurück"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"{self.name}")
        lines.append("=" * 60)
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
        lines.append(f"Konfidenz: {self.confidence:.0%}")
        
        return "\n".join(lines)


# ============================================================================
# ARS 3.0 - HIERARCHISCHE GRAMMATIK
# ============================================================================

class MethodologicalReflection:
    """Methodologische Reflexion für ARS 3.0"""
    
    def __init__(self):
        self.interpretation_log = []
        self.sequence_meaning_mapping = {}
    
    def log_interpretation(self, sequence, new_nonterminal, rationale):
        self.interpretation_log.append({
            'sequence': sequence,
            'new_nonterminal': new_nonterminal,
            'rationale': rationale,
            'timestamp': len(self.interpretation_log)
        })
    
    def print_summary(self):
        lines = ["\n" + "=" * 60, "METHODOLOGISCHE REFLEXION", "=" * 60]
        for log in self.interpretation_log:
            lines.append(f"\n[{log['timestamp']+1}] {log['new_nonterminal']}")
            lines.append(f"  Sequenz: {' → '.join(log['sequence'])}")
            lines.append(f"  Begründung: {log['rationale']}")
        return "\n".join(lines)


class GrammarInducer(XAIModel):
    """ARS 3.0 - Hierarchische Grammatikinduktion"""
    
    def __init__(self):
        super().__init__("ARS 3.0 - Hierarchische Grammatik")
        self.description = "Induziert hierarchische Nonterminale aus wiederholten Sequenzen"
        self.rules = {}
        self.terminals = set()
        self.nonterminals = set()
        self.start_symbol = None
        self.user_start_symbol = None
        self.compression_history = []
        self.reflection = MethodologicalReflection()
        self.chains = []
        self.iteration_count = 0
        self.hierarchy_levels = {}
        self.induction_done = False
    
    def train(self, chains, user_start_symbol=None, max_iterations=20):
        """Induziert Grammatik aus Ketten"""
        self.chains = [list(chain) for chain in chains]
        self.user_start_symbol = user_start_symbol
        
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
            
            rationale = f"Erkanntes wiederholtes Muster: {' → '.join(best_seq)}"
            self.reflection.log_interpretation(best_seq, new_nonterminal, rationale)
            
            self.rules[new_nonterminal] = [(list(best_seq), 1.0)]
            self.nonterminals.add(new_nonterminal)
            self.hierarchy_levels[new_nonterminal] = iteration
            
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
                    if self.user_start_symbol and self.user_start_symbol in self.rules:
                        self.start_symbol = self.user_start_symbol
                    elif unique_symbol in self.rules:
                        self.start_symbol = unique_symbol
                    else:
                        self.start_symbol = self._find_top_level_nonterminal()
                    break
        
        if self.start_symbol is None:
            if self.user_start_symbol and self.user_start_symbol in self.rules:
                self.start_symbol = self.user_start_symbol
            elif self.rules:
                self.start_symbol = self._find_top_level_nonterminal()
        
        # Terminale aktualisieren
        all_symbols = set()
        for chain in self.chains:
            for sym in chain:
                all_symbols.add(sym)
        self.terminals = all_symbols - self.nonterminals
        
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
        """Generiert einen Namen für ein neues Nonterminal"""
        if all(isinstance(s, str) for s in sequence):
            # Versuche, einen semantischen Namen zu generieren
            seq_str = ' '.join(sequence)
            if any('B' in s for s in sequence) and any('d' in s for s in sequence):
                typ = "BEDARF"
            elif any('A' in s for s in sequence) and any('E' in s for s in sequence):
                typ = "BERATUNG"
            elif any('A' in s for s in sequence) and any('A' in s for s in sequence):
                typ = "ABSCHLUSS"
            elif any('G' in s for s in sequence):
                typ = "BEGRUESSUNG"
            elif any('V' in s for s in sequence):
                typ = "VERABSCHIEDUNG"
            else:
                typ = "SEQUENZ"
            return f"NT_{typ}_{len(sequence)}"
        else:
            return f"NT_SEQ_{len(sequence)}"
    
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
        """Prüft, ob alle Ketten identisch sind"""
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
    
    def _calculate_confidence(self):
        """Berechnet Konfidenz basierend auf Kompressionsrate"""
        if not self.chains or not self.compression_history:
            return 0.0
        
        # Berechne durchschnittliche Kompression
        total_original = sum(len(chain) for chain in self.chains)
        total_compressed = total_original
        
        for hist in self.compression_history:
            # Jede Kompression spart (Länge-1) Symbole pro Vorkommen
            savings = (len(hist['sequence']) - 1) * hist['occurrences']
            total_compressed -= savings
        
        compression_ratio = 1 - (total_compressed / total_original) if total_original > 0 else 0
        confidence = min(1.0, compression_ratio * 1.5)  # Skalieren, aber max 1.0
        
        return round(confidence, 3)
    
    def explain(self, data, detail_level='normal'):
        """
        Erklärt ARS 3.0-Strukturen
        
        data kann sein:
        - Ein Symbol (erkläre, ob Terminal oder Nonterminal)
        - Eine Sequenz (erkläre, ob sie ein Pattern bildet)
        """
        explanation = {
            'model': self.name,
            'confidence': self.confidence,
            'type': None,
            'content': []
        }
        
        if isinstance(data, tuple) or isinstance(data, list):
            # Sequenz - prüfe auf Pattern
            seq = tuple(data) if isinstance(data, list) else data
            explanation['type'] = 'sequence'
            
            # Suche nach Matching in der History
            matches = []
            for hist in self.compression_history:
                if tuple(hist['sequence']) == seq:
                    matches.append(hist)
            
            if matches:
                for match in matches:
                    explanation['content'].append(
                        f"Sequenz {' → '.join(seq)} wurde als {match['new_symbol']} "
                        f"komprimiert ({match['occurrences']} Vorkommen)"
                    )
                    
                    if detail_level == 'detailed':
                        # Zeige, wo die Sequenz vorkommt
                        explanation['content'].append(
                            f"  Grund: Wiederholtes Muster in Iteration {match['iteration']+1}"
                        )
            else:
                # Prüfe, ob Teil eines größeren Patterns
                for hist in self.compression_history:
                    pattern = hist['sequence']
                    if all(s in pattern for s in seq):
                        explanation['content'].append(
                            f"Sequenz ist Teil von Pattern: {' → '.join(pattern)}"
                        )
                        break
                else:
                    explanation['content'].append("Sequenz bildet kein wiederkehrendes Muster")
        
        elif isinstance(data, str):
            # Symbol - erkläre, ob Terminal oder Nonterminal
            explanation['type'] = 'symbol'
            symbol = data
            
            if symbol in self.nonterminals:
                explanation['content'].append(f"{symbol} ist ein Nonterminal")
                if symbol in self.rules:
                    for prod, prob in self.rules[symbol]:
                        explanation['content'].append(
                            f"  → {' → '.join(prod)} (p={prob:.2f})"
                        )
            elif symbol in self.terminals:
                explanation['content'].append(f"{symbol} ist ein Terminal")
                
                # Zeige, in welchen Produktionen es vorkommt
                appearing_in = []
                for nt, prods in self.rules.items():
                    for prod, _ in prods:
                        if symbol in prod:
                            appearing_in.append(nt)
                if appearing_in:
                    explanation['content'].append(f"  Kommt vor in: {', '.join(set(appearing_in))}")
            else:
                explanation['content'].append(f"{symbol} ist unbekannt")
        
        return explanation
    
    def generate_chain(self, start_symbol=None, max_depth=20, max_length=None):
        """
        Generiert eine neue Kette basierend auf der Grammatik
        
        Args:
            start_symbol: Optionales Startsymbol
            max_depth: Maximale Rekursionstiefe
            max_length: Alias für max_depth (für Kompatibilität)
        """
        if max_length is not None:
            max_depth = max_length
            
        if not self.trained:
            return []
        
        if not start_symbol:
            start_symbol = self.start_symbol
        
        if not start_symbol:
            return []
        
        if start_symbol not in self.rules:
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
        """Gibt die Grammatik als String zurück"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"{self.name}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Terminale ({len(self.terminals)}): {sorted(self.terminals)}")
        lines.append(f"Nonterminale ({len(self.nonterminals)}): {sorted(self.nonterminals)}")
        lines.append(f"Startsymbol: {self.start_symbol}")
        lines.append(f"Iterationen: {self.iteration_count}")
        lines.append(f"Konfidenz: {self.confidence:.0%}")
        lines.append("")
        lines.append("PRODUKTIONSREGELN:")
        
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
# HMM - HIDDEN MARKOV MODELS
# ============================================================================

class ARSHiddenMarkovModel(XAIModel):
    """Bayessche Netze - Hidden Markov Models für latente Phasen"""
    
    def __init__(self, n_states=5):
        super().__init__("HMM - Bayessches Netz")
        self.description = "Modelliert latente Gesprächsphasen"
        self.n_states = n_states
        self.model = None
        self.code_to_idx = {}
        self.idx_to_code = {}
        self.state_names = {
            0: "Phase 0 (Beginn)",
            1: "Phase 1 (Früh)",
            2: "Phase 2 (Mitte)",
            3: "Phase 3 (Spät)",
            4: "Phase 4 (Ende)"
        }
        self.n_features = None
        self.trained = False
    
    def train(self, chains, n_iter=100):
        """Trainiert HMM mit Baum-Welch"""
        if not MODULE_STATUS['hmmlearn']:
            raise ImportError("hmmlearn nicht installiert")
        
        # Bereite Daten vor (One-Hot-Encoding)
        X, lengths = self._prepare_data(chains)
        
        if len(X) == 0:
            raise ValueError("Keine gültigen Daten zum Trainieren")
        
        # Erstelle und trainiere Modell
        self.model = hmm.MultinomialHMM(
            n_components=self.n_states,
            n_iter=n_iter,
            random_state=42
        )
        
        self.model.fit(X, lengths)
        self.trained = True
        self.confidence = self._calculate_confidence()
        
        return self.model
    
    def _prepare_data(self, chains):
        """Bereitet Daten für HMM vor (One-Hot-Encoding)"""
        # Sammle alle Symbole
        all_symbols = set()
        for chain in chains:
            for sym in chain:
                all_symbols.add(sym)
        
        if not all_symbols:
            return np.array([]).reshape(-1, 1), np.array([])
        
        # Mapping für Symbole
        self.code_to_idx = {sym: i for i, sym in enumerate(sorted(all_symbols))}
        self.idx_to_code = {i: sym for sym, i in self.code_to_idx.items()}
        self.n_features = len(all_symbols)
        
        # Konvertiere zu One-Hot-Encoding
        X_list = []
        lengths = []
        
        for chain in chains:
            if chain:
                seq_length = len(chain)
                one_hot_seq = np.zeros((seq_length, self.n_features))
                for i, sym in enumerate(chain):
                    if sym in self.code_to_idx:
                        one_hot_seq[i, self.code_to_idx[sym]] = 1
                X_list.append(one_hot_seq)
                lengths.append(seq_length)
        
        if not X_list:
            return np.array([]), np.array([])
        
        X = np.vstack(X_list)
        return X, np.array(lengths)
    
    def _calculate_confidence(self):
        """Berechnet Konfidenz basierend auf Modell-Konvergenz"""
        if not self.model or not hasattr(self.model, 'monitor_'):
            return 0.5
        
        # Nutze Konvergenz-Informationen
        if hasattr(self.model.monitor_, 'converged'):
            return 0.9 if self.model.monitor_.converged else 0.6
        return 0.7
    
    def explain(self, data, detail_level='normal'):
        """
        Erklärt HMM-Vorhersagen
        
        data kann sein:
        - Eine Kette (erkläre Zustandssequenz)
        - Ein Symbol (erkläre wahrscheinlichste Zustände)
        """
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
            # Kette - dekodiere Zustände
            explanation['type'] = 'sequence'
            
            try:
                # Bereite Daten vor
                X, _ = self._prepare_data([data])
                if len(X) > 0:
                    logprob, states = self.model.decode(X, algorithm="viterbi")
                    
                    explanation['content'].append(f"Dekodierte Zustandssequenz (p={np.exp(logprob):.4f}):")
                    for i, (sym, state) in enumerate(zip(data, states)):
                        state_name = self.state_names.get(state, f"State {state}")
                        explanation['content'].append(f"  {i+1}. {sym} → {state_name}")
                    
                    if detail_level == 'detailed':
                        # Zeige Übergangswahrscheinlichkeiten
                        explanation['content'].append("\nÜbergangswahrscheinlichkeiten:")
                        for i in range(len(states)-1):
                            from_state = states[i]
                            to_state = states[i+1]
                            prob = self.model.transmat_[from_state, to_state]
                            explanation['content'].append(f"  {from_state}→{to_state}: {prob:.3f}")
            except Exception as e:
                explanation['content'] = [f"Fehler bei Dekodierung: {e}"]
        
        elif isinstance(data, str):
            # Symbol - zeige Emissionswahrscheinlichkeiten
            explanation['type'] = 'symbol'
            symbol = data
            
            if symbol in self.code_to_idx:
                sym_idx = self.code_to_idx[symbol]
                explanation['content'].append(f"Emissionswahrscheinlichkeiten für {symbol}:")
                
                probs = [(state, self.model.emissionprob_[state, sym_idx]) 
                        for state in range(self.n_states)]
                probs.sort(key=lambda x: -x[1])
                
                for state, prob in probs:
                    state_name = self.state_names.get(state, f"State {state}")
                    explanation['content'].append(f"  {state_name}: {prob:.3f}")
            else:
                explanation['content'] = [f"Symbol {symbol} nicht im Modell"]
        
        return explanation
    
    def decode_chain(self, chain):
        """Dekodiert eine Kette mit Viterbi"""
        if not self.trained:
            return None, None
        
        X, _ = self._prepare_data([chain])
        if len(X) == 0:
            return None, None
        
        try:
            logprob, states = self.model.decode(X, algorithm="viterbi")
            return states, np.exp(logprob)
        except:
            return None, None
    
    def generate_chain(self, max_length=20, start_state=None):
        """
        Generiert eine neue Kette aus dem HMM
        
        Args:
            max_length: Maximale Länge
            start_state: Optionaler Startzustand
        """
        if not self.trained or self.model is None:
            return []
        
        try:
            # Generiere Sequenz aus HMM
            X, states = self.model.sample(max_length)
            
            # Konvertiere zu Symbolen
            chain = []
            for idx in X.flatten():
                if int(idx) in self.idx_to_code:
                    chain.append(self.idx_to_code[int(idx)])
            
            return chain
        except Exception as e:
            print(f"Fehler bei HMM-Generierung: {e}")
            return []
    
    def get_parameters_string(self):
        """Gibt Modellparameter als String zurück"""
        if not self.trained:
            return "Modell nicht trainiert"
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"{self.name}")
        lines.append("=" * 60)
        lines.append("")
        lines.append("Startwahrscheinlichkeiten:")
        for i in range(self.n_states):
            lines.append(f"  {self.state_names[i]}: {self.model.startprob_[i]:.3f}")
        
        lines.append("\nÜbergangsmatrix:")
        for i in range(self.n_states):
            row = "  " + " ".join([f"{self.model.transmat_[i,j]:.3f}" 
                                   for j in range(self.n_states)])
            lines.append(f"{self.state_names[i]}: {row}")
        
        lines.append(f"\nKonfidenz: {self.confidence:.0%}")
        
        return "\n".join(lines)


# ============================================================================
# CRF - CONDITIONAL RANDOM FIELDS
# ============================================================================

class ARSCRFModel(XAIModel):
    """CRF für kontext-sensitive Sequenzanalyse"""
    
    def __init__(self):
        super().__init__("CRF - Conditional Random Fields")
        self.description = "Kontext-sensitive Vorhersage mit Feature-Gewichten"
        self.crf = None
        self.feature_importances = {}
    
    def train(self, chains, max_iterations=100):
        """Trainiert CRF auf den Ketten"""
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
        """Bereitet Daten für CRF vor"""
        X = []
        y = []
        for seq in sequences:
            X_seq = [self._extract_features(seq, i) for i in range(len(seq))]
            y_seq = [sym for sym in seq]
            X.append(X_seq)
            y.append(y_seq)
        return X, y
    
    def _extract_features(self, sequence, i):
        """Extrahiert Features für Position i"""
        features = {
            'bias': 1.0,
            'symbol': sequence[i],
            'position': i,
            'is_first': i == 0,
            'is_last': i == len(sequence) - 1,
        }
        
        # Kontext-Features
        for offset in [-2, -1, 1, 2]:
            if 0 <= i + offset < len(sequence):
                features[f'context_{offset:+d}'] = sequence[i + offset]
        
        # Bigramm
        if i > 0:
            features['bigram'] = f"{sequence[i-1]}_{sequence[i]}"
        
        return features
    
    def _calculate_confidence(self):
        """Berechnet Konfidenz basierend auf Feature-Anzahl"""
        if not hasattr(self.crf, 'state_features_'):
            return 0.5
        
        n_features = len(self.crf.state_features_)
        confidence = min(1.0, n_features / 100)  # Skaliere mit Feature-Anzahl
        return round(confidence, 3)
    
    def _extract_feature_importances(self):
        """Extrahiert die wichtigsten Features"""
        if not hasattr(self.crf, 'state_features_'):
            return
        
        # Sammle Feature-Gewichte
        for (attr, label), weight in self.crf.state_features_.items():
            if attr not in self.feature_importances:
                self.feature_importances[attr] = []
            self.feature_importances[attr].append((label, weight))
    
    def explain(self, data, detail_level='normal'):
        """
        Erklärt CRF-Vorhersagen
        
        data kann sein:
        - Eine Kette (erkläre Vorhersage)
        - Ein Feature-Name (erkläre Feature-Wichtigkeit)
        """
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
            # Kette - erkläre Vorhersage
            explanation['type'] = 'sequence'
            
            try:
                X = [self._extract_features(data, i) for i in range(len(data))]
                pred = self.crf.predict([X])[0]
                
                explanation['content'].append("Vorhergesagte Sequenz:")
                for i, (sym, pred_sym) in enumerate(zip(data, pred)):
                    match = "✓" if sym == pred_sym else "✗"
                    explanation['content'].append(f"  {i+1}. {sym} → {pred_sym} {match}")
                
                if detail_level == 'detailed':
                    # Zeige wichtige Features für diese Vorhersage
                    explanation['content'].append("\nWichtige Features:")
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
        
        elif isinstance(data, str):
            # Feature-Name - erkläre Feature
            explanation['type'] = 'feature'
            feature = data
            
            if feature in self.feature_importances:
                explanation['content'].append(f"Feature '{feature}':")
                for label, weight in sorted(self.feature_importances[feature], 
                                           key=lambda x: -abs(x[1]))[:5]:
                    direction = "fördert" if weight > 0 else "hemmt"
                    explanation['content'].append(f"  {direction} {label}: {weight:+.3f}")
            else:
                explanation['content'] = [f"Feature '{feature}' nicht gefunden"]
        
        return explanation
    
    def get_feature_string(self, n=20):
        """Gibt die wichtigsten Features als String zurück"""
        if not hasattr(self.crf, 'state_features_'):
            return "Keine Feature-Informationen"
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"{self.name} - Wichtigste Features")
        lines.append("=" * 60)
        lines.append("")
        
        top = sorted(
            self.crf.state_features_.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:n]
        
        for (attr, label), weight in top:
            lines.append(f"{attr:30s} → {label:4s} : {weight:+.4f}")
        
        lines.append(f"\nKonfidenz: {self.confidence:.0%}")
        
        return "\n".join(lines)


# ============================================================================
# PETRI-NETZE
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
            self.hierarchy = {}
            self.firing_history = []
            self.reached_markings = set()
        
        def train(self, chains):
            """Baut Petri-Netz aus Ketten (angepasst für XAI)"""
            # Sammle alle Symbole
            all_symbols = set()
            for chain in chains:
                for sym in chain:
                    all_symbols.add(sym)
            
            # Basis-Places
            self.add_place("p_start", initial_tokens=1)
            self.add_place("p_end", initial_tokens=0)
            
            # Places und Transitions für jedes Symbol
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
            """Fügt eine Stelle hinzu"""
            self.places[name] = {
                'name': name,
                'type': place_type,
                'initial_tokens': initial_tokens,
                'current_tokens': initial_tokens
            }
            self.tokens[name] = initial_tokens
        
        def add_transition(self, name, transition_type="speech_act", guard=None, subnet=None):
            """Fügt eine Transition hinzu"""
            self.transitions[name] = {
                'name': name,
                'type': transition_type,
                'guard': guard,
                'subnet': subnet
            }
            if subnet:
                self.hierarchy[name] = subnet
        
        def add_arc(self, source, target, weight=1):
            """Fügt eine Kante hinzu"""
            self.arcs.append({'source': source, 'target': target, 'weight': weight})
        
        def _calculate_confidence(self, chains):
            """Berechnet Konfidenz basierend auf Abdeckung"""
            if not chains:
                return 0.0
            
            # Wie viele der Symbole können feuern?
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
            """Setzt das Netz zurück"""
            for place_name, place_data in self.places.items():
                self.tokens[place_name] = place_data['initial_tokens']
            self.firing_history = []
        
        def is_enabled(self, transition):
            """Prüft, ob eine Transition aktiviert ist"""
            if transition not in self.transitions:
                return False
            
            # Prüfe Vorbereich
            for arc in self.arcs:
                if arc['target'] == transition and arc['source'] in self.places:
                    if self.tokens.get(arc['source'], 0) < arc['weight']:
                        return False
            
            # Prüfe Guard
            trans_data = self.transitions[transition]
            if trans_data['guard'] and not trans_data['guard'](self):
                return False
            
            return True
        
        def fire(self, transition):
            """Feuert eine Transition"""
            if not self.is_enabled(transition):
                return False
            
            # Entferne Token aus Vorbereich
            for arc in self.arcs:
                if arc['target'] == transition and arc['source'] in self.places:
                    self.tokens[arc['source']] -= arc['weight']
            
            # Füge Token zu Nachbereich hinzu
            for arc in self.arcs:
                if arc['source'] == transition and arc['target'] in self.places:
                    self.tokens[arc['target']] = self.tokens.get(arc['target'], 0) + arc['weight']
            
            self.firing_history.append({'transition': transition, 'tokens': self.tokens.copy()})
            return True
        
        def explain(self, data, detail_level='normal'):
            """
            Erklärt Petri-Netz-Zustände
            
            data kann sein:
            - Ein Symbol (erkläre, ob Transition aktiviert)
            - Eine Kette (simuliere und erkläre)
            """
            explanation = {
                'model': self.name,
                'confidence': self.confidence,
                'type': None,
                'content': []
            }
            
            if isinstance(data, list):
                # Kette - simuliere
                explanation['type'] = 'simulation'
                self.reset()
                
                explanation['content'].append(f"Simulation der Kette:")
                for i, sym in enumerate(data):
                    trans_name = f"t_{sym}"
                    enabled = self.is_enabled(trans_name)
                    
                    if enabled:
                        self.fire(trans_name)
                        status = "✓ aktiviert und gefeuert"
                    else:
                        status = "✗ nicht aktiviert"
                    
                    explanation['content'].append(f"  {i+1}. {sym}: {status}")
                    
                    if detail_level == 'detailed' and not enabled:
                        # Zeige, warum nicht aktiviert
                        reasons = []
                        for arc in self.arcs:
                            if arc['target'] == trans_name and arc['source'] in self.places:
                                if self.tokens.get(arc['source'], 0) < arc['weight']:
                                    reasons.append(f"{arc['source']}: {self.tokens.get(arc['source'],0)}/{arc['weight']}")
                        if reasons:
                            explanation['content'].append(f"     Fehlende Token: {', '.join(reasons)}")
                
                explanation['content'].append(f"\nEndmarkierung:")
                for place, tokens in self.tokens.items():
                    if tokens > 0:
                        explanation['content'].append(f"  {place}: {tokens}")
            
            elif isinstance(data, str):
                # Symbol - prüfe Transition
                explanation['type'] = 'transition'
                trans_name = f"t_{data}"
                
                if trans_name in self.transitions:
                    enabled = self.is_enabled(trans_name)
                    explanation['content'].append(f"Transition {trans_name}:")
                    explanation['content'].append(f"  Aktiviert: {'Ja' if enabled else 'Nein'}")
                    
                    if detail_level == 'detailed':
                        # Zeige Vor- und Nachbereich
                        pre = [arc for arc in self.arcs if arc['target'] == trans_name]
                        post = [arc for arc in self.arcs if arc['source'] == trans_name]
                        
                        if pre:
                            explanation['content'].append("  Benötigt:")
                            for arc in pre:
                                explanation['content'].append(f"    {arc['source']}: {arc['weight']}")
                        
                        if post:
                            explanation['content'].append("  Produziert:")
                            for arc in post:
                                explanation['content'].append(f"    {arc['target']}: {arc['weight']}")
                else:
                    explanation['content'] = [f"Transition {trans_name} nicht gefunden"]
            
            return explanation
        
        def get_net_string(self):
            """Gibt Netz-Beschreibung als String zurück"""
            lines = []
            lines.append("=" * 60)
            lines.append(f"{self.name}")
            lines.append("=" * 60)
            lines.append("")
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
# GENERIERUNGSKOMPONENTE (KORRIGIERT)
# ============================================================================

class ChainGenerator(XAIModel):
    """Generiert neue Ketten basierend auf trainierten Modellen"""
    
    def __init__(self):
        super().__init__("Generator - Synthetische Ketten")
        self.description = "Generiert neue Sequenzen aus gelernten Modellen"
        self.source_model = None
    
    def train(self, chains):
        """Trainiert den Generator (nutzt vorhandene Modelle)"""
        self.trained = True
        self.confidence = 0.8
        return True
    
    def set_source_model(self, model):
        """Setzt das Quellmodell für die Generierung"""
        self.source_model = model
    
    def generate(self, count=5, max_length=20):
        """
        Generiert neue Ketten mit robuster Fehlerbehandlung für verschiedene Modell-Typen
        
        Args:
            count: Anzahl der zu generierenden Ketten
            max_length: Maximale Länge pro Kette
        """
        if not self.source_model:
            return []
        
        chains = []
        for i in range(count):
            chain = None
            
            # Verschiedene Methoden-Signaturen ausprobieren
            if hasattr(self.source_model, 'generate_chain'):
                # Versuche mit max_length
                try:
                    chain = self.source_model.generate_chain(max_length=max_length)
                except TypeError:
                    try:
                        # Versuche ohne Parameter
                        chain = self.source_model.generate_chain()
                        if chain and len(chain) > max_length:
                            chain = chain[:max_length]
                    except Exception as e1:
                        try:
                            # Versuche mit max_depth
                            chain = self.source_model.generate_chain(max_depth=max_length)
                        except:
                            pass
            
            # HMM-Fall (falls keine generate_chain Methode)
            elif hasattr(self.source_model, 'model') and hasattr(self.source_model.model, 'sample'):
                try:
                    X, states = self.source_model.model.sample(max_length)
                    chain = []
                    for idx in X.flatten():
                        if int(idx) in self.source_model.idx_to_code:
                            chain.append(self.source_model.idx_to_code[int(idx)])
                except Exception as e:
                    print(f"HMM-Generierung fehlgeschlagen: {e}")
            
            if chain and len(chain) > 0:
                chains.append(chain)
        
        return chains
    
    def explain(self, data, detail_level='normal'):
        """Erklärt den Generierungsprozess"""
        explanation = {
            'model': self.name,
            'confidence': self.confidence,
            'type': 'generation',
            'content': []
        }
        
        if isinstance(data, int):
            # Generiere Erklärung für n Ketten
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
# XAI MODELLVERWALTER
# ============================================================================

class XAIModelManager:
    """
    Verwaltet alle XAI-Modelle und ermöglicht Vergleich
    """
    
    def __init__(self):
        self.models = {}
        self.active_models = set()
        self.comparison_cache = {}
    
    def register_model(self, name, model):
        """Registriert ein Modell"""
        self.models[name] = model
        self.active_models.add(name)
    
    def activate_model(self, name):
        """Aktiviert ein Modell für Vergleiche"""
        if name in self.models:
            self.active_models.add(name)
    
    def deactivate_model(self, name):
        """Deaktiviert ein Modell"""
        self.active_models.discard(name)
    
    def train_all(self, chains):
        """Trainiert alle aktiven Modelle"""
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
        """
        Sammelt Erklärungen aller aktiven Modelle
        
        Returns:
            Dictionary mit Modell-Erklärungen und Konsens
        """
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
        
        # Berechne Konsens
        consensus = self._calculate_consensus(explanations)
        
        return {
            'explanations': explanations,
            'consensus': consensus,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_consensus(self, explanations):
        """Berechnet Konsens zwischen Modellen"""
        consensus = {
            'agreement': 0.0,
            'disagreements': [],
            'confidence': 0.0
        }
        
        # Einfache Metrik: Übereinstimmung bei Klassifikation
        # Kann je nach Datentyp erweitert werden
        if len(explanations) < 2:
            return consensus
        
        # Sammle alle Konfidenzen
        confidences = [e.get('confidence', 0) for e in explanations.values() if 'confidence' in e]
        if confidences:
            consensus['confidence'] = np.mean(confidences)
        
        return consensus
    
    def compare_models(self, data):
        """
        Vergleicht Modellvorhersagen für gegebene Daten
        
        Returns:
            Dictionary mit Vergleichsergebnissen
        """
        comparison = {
            'data': data,
            'predictions': {},
            'agreement_matrix': None
        }
        
        # Sammle Vorhersagen (vereinfacht - je nach Modelltyp)
        for name in self.active_models:
            model = self.models.get(name)
            if model and hasattr(model, 'explain'):
                try:
                    exp = model.explain(data, 'simple')
                    comparison['predictions'][name] = {
                        'confidence': model.get_confidence(),
                        'summary': exp.get('content', [''])[0] if exp.get('content') else ''
                    }
                except:
                    pass
        
        return comparison
    
    def get_model_info(self):
        """Gibt Informationen über alle Modelle zurück"""
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
        """Validiert alle Ketten und sammelt Probleme"""
        self.issues = []
        self.warnings = []
        
        if not chains:
            self.issues.append(("error", "Keine Ketten gefunden"))
            return self.issues, self.warnings
        
        # Prüfe auf leere Ketten
        empty_chains = [i for i, chain in enumerate(chains) if not chain]
        if empty_chains:
            self.warnings.append(("warning", f"Leere Ketten an Positionen: {empty_chains}"))
        
        # Prüfe auf Mindestlänge
        short_chains = [i for i, chain in enumerate(chains) if len(chain) < 2]
        if short_chains:
            self.warnings.append(("warning", f"Sehr kurze Ketten (<2 Symbole): {short_chains}"))
        
        # Prüfe auf konsistente Symbolschreibweise
        all_symbols = set()
        for chain in chains:
            for symbol in chain:
                all_symbols.add(symbol)
        
        # Prüfe auf mögliche Tippfehler (Ähnlichkeit)
        symbol_groups = self._group_similar_symbols(all_symbols)
        for base, similar in symbol_groups.items():
            if len(similar) > 1:
                self.warnings.append(("info", f"Ähnliche Symbole gefunden: {', '.join(similar)}"))
        
        return self.issues, self.warnings
    
    def _group_similar_symbols(self, symbols):
        """Gruppiert ähnliche Symbole (für Tippfehlererkennung)"""
        groups = defaultdict(list)
        for sym in symbols:
            # Normalisiere: Großbuchstaben, entferne Sonderzeichen
            normalized = re.sub(r'[^A-Za-z]', '', sym.upper())
            groups[normalized].append(sym)
        return {k: v for k, v in groups.items() if len(v) > 1}
    
    def suggest_corrections(self):
        """Generiert Korrekturvorschläge basierend auf gefundenen Problemen"""
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
# ABLEITUNGSSTRATEGIEN FÜR KODIERUNG
# ============================================================================

class CodingStrategy:
    """Basisklasse für Kodierungsstrategien"""
    
    def __init__(self, name):
        self.name = name
        self.confidence = 0.0
    
    def derive(self, chains):
        """Leitet Kodierung aus Ketten ab"""
        raise NotImplementedError
    
    def explain(self, symbol, code):
        """Erklärt, warum ein Symbol so kodiert wurde"""
        raise NotImplementedError


class PositionBasedCoding(CodingStrategy):
    """Strategie 1: Kodierung basierend auf Position in der Sequenz"""
    
    def __init__(self):
        super().__init__("Positionsbasierte Kodierung")
    
    def derive(self, chains):
        coding = {}
        
        # Sammle Positionsstatistiken
        positions = defaultdict(list)
        for chain in chains:
            for i, sym in enumerate(chain):
                positions[sym].append(i)
        
        for symbol in positions:
            # Bit 1: Sprecher (vereinfacht: K=0, V=1)
            speaker = "0" if symbol.startswith('K') else "1"
            
            # Bits 2-3: Phase aus durchschnittlicher Position
            avg_pos = np.mean(positions[symbol])
            max_pos = max([max(pos) for pos in positions.values() if pos])
            phase_norm = avg_pos / max_pos if max_pos > 0 else 0
            
            if phase_norm < 0.25:
                phase = "00"
            elif phase_norm < 0.5:
                phase = "01"
            elif phase_norm < 0.75:
                phase = "10"
            else:
                phase = "11"
            
            # Bits 4-5: Subphase (vereinfacht: 00)
            subphase = "00"
            
            code = f"{speaker}{phase}{subphase}"
            coding[symbol] = {
                'code': code,
                'evidence': {
                    'avg_position': avg_pos,
                    'phase_norm': phase_norm
                }
            }
        
        self.confidence = 0.7
        return coding
    
    def explain(self, symbol, code_data):
        code = code_data['code']
        evidence = code_data['evidence']
        
        bits = list(code)
        phase_map = {"00": "Begrüßung", "01": "Bedarf", "10": "Abschluss", "11": "Verabschiedung"}
        phase = phase_map.get(''.join(bits[1:3]), "Unbekannt")
        
        return (f"Sprecher: {'Kunde' if bits[0]=='0' else 'Verkäufer'} | "
                f"Phase: {phase} (Pos.{evidence['avg_position']:.1f}) | "
                f"Subphase: Basis")


class PatternBasedCoding(CodingStrategy):
    """Strategie 2: Kodierung basierend auf wiederkehrenden Mustern"""
    
    def __init__(self):
        super().__init__("Musterbasierte Kodierung")
    
    def derive(self, chains):
        coding = {}
        
        # Analysiere Nachbarschaftsbeziehungen
        neighbors = defaultdict(Counter)
        for chain in chains:
            for i, sym in enumerate(chain):
                if i > 0:
                    neighbors[sym][chain[i-1]] += 1
                if i < len(chain)-1:
                    neighbors[sym][chain[i+1]] += 1
        
        for symbol in neighbors:
            speaker = "0" if symbol.startswith('K') else "1"
            
            # Phase aus typischen Nachbarn
            common_neighbors = neighbors[symbol].most_common(3)
            if any(n[0].endswith('G') for n in common_neighbors):
                phase = "00"
            elif any('B' in n[0] for n in common_neighbors):
                phase = "01"
            elif any('E' in n[0] for n in common_neighbors):
                phase = "10"
            elif any('V' in n[0] for n in common_neighbors):
                phase = "11"
            else:
                phase = "01"
            
            code = f"{speaker}{phase}00"
            coding[symbol] = {
                'code': code,
                'evidence': {
                    'common_neighbors': [(n, c) for n, c in common_neighbors[:3]]
                }
            }
        
        self.confidence = 0.75
        return coding
    
    def explain(self, symbol, code_data):
        code = code_data['code']
        evidence = code_data['evidence']
        
        bits = list(code)
        phase_map = {"00": "Begrüßung", "01": "Bedarf", "10": "Abschluss", "11": "Verabschiedung"}
        phase = phase_map.get(''.join(bits[1:3]), "Unbekannt")
        
        neighbor_info = ", ".join([n for n, _ in evidence['common_neighbors'][:2]])
        return (f"Sprecher: {'Kunde' if bits[0]=='0' else 'Verkäufer'} | "
                f"Phase: {phase} (Nachbarn: {neighbor_info}) | "
                f"Subphase: Basis")


class StatisticalBasedCoding(CodingStrategy):
    """Strategie 3: Kodierung basierend auf statistischen Verteilungen"""
    
    def __init__(self):
        super().__init__("Statistisch basierte Kodierung")
    
    def derive(self, chains):
        coding = {}
        
        # Berechne Häufigkeiten
        frequencies = defaultdict(int)
        total_symbols = 0
        for chain in chains:
            for sym in chain:
                frequencies[sym] += 1
                total_symbols += 1
        
        # Berechne Positionen
        first_positions = {}
        last_positions = {}
        for chain in chains:
            if chain:
                first_positions[chain[0]] = first_positions.get(chain[0], 0) + 1
                last_positions[chain[-1]] = last_positions.get(chain[-1], 0) + 1
        
        for symbol in frequencies:
            speaker = "0" if symbol.startswith('K') else "1"
            
            # Phase basierend auf Position und Häufigkeit
            rel_freq = frequencies[symbol] / total_symbols
            first_prob = first_positions.get(symbol, 0) / len(chains)
            last_prob = last_positions.get(symbol, 0) / len(chains)
            
            if first_prob > 0.5:
                phase = "00"
            elif last_prob > 0.5:
                phase = "11"
            elif rel_freq > 0.15:
                phase = "01"
            else:
                phase = "10"
            
            code = f"{speaker}{phase}00"
            coding[symbol] = {
                'code': code,
                'evidence': {
                    'frequency': rel_freq,
                    'first_prob': first_prob,
                    'last_prob': last_prob
                }
            }
        
        self.confidence = 0.8
        return coding
    
    def explain(self, symbol, code_data):
        code = code_data['code']
        evidence = code_data['evidence']
        
        bits = list(code)
        phase_map = {"00": "Begrüßung", "01": "Bedarf", "10": "Abschluss", "11": "Verabschiedung"}
        phase = phase_map.get(''.join(bits[1:3]), "Unbekannt")
        
        return (f"Sprecher: {'Kunde' if bits[0]=='0' else 'Verkäufer'} | "
                f"Phase: {phase} (Häufigkeit: {evidence['frequency']:.0%}) | "
                f"Subphase: Basis")


# ============================================================================
# ABLEITUNGSMANAGER
# ============================================================================

class DerivationManager:
    """Verwaltet verschiedene Ableitungsstrategien"""
    
    def __init__(self):
        self.strategies = [
            PositionBasedCoding(),
            PatternBasedCoding(),
            StatisticalBasedCoding()
        ]
        self.results = {}
        self.consensus_coding = {}
    
    def derive_all(self, chains):
        """Wendet alle Strategien an und berechnet Konsens"""
        self.results = {}
        
        for strategy in self.strategies:
            try:
                coding = strategy.derive(chains)
                self.results[strategy.name] = {
                    'coding': coding,
                    'confidence': strategy.confidence
                }
            except Exception as e:
                self.results[strategy.name] = {
                    'error': str(e),
                    'coding': {}
                }
        
        # Berechne Konsens-Kodierung
        self.consensus_coding = self._calculate_consensus()
        
        return self.results
    
    def _calculate_consensus(self):
        """Berechnet Konsens-Kodierung über alle Strategien"""
        if not self.results:
            return {}
        
        # Sammle alle Symbole
        all_symbols = set()
        for result in self.results.values():
            if 'coding' in result:
                all_symbols.update(result['coding'].keys())
        
        consensus = {}
        for symbol in all_symbols:
            votes = defaultdict(int)
            evidences = []
            
            for strategy_name, result in self.results.items():
                if 'coding' in result and symbol in result['coding']:
                    code_data = result['coding'][symbol]
                    votes[code_data['code']] += 1
                    evidences.append({
                        'strategy': strategy_name,
                        'code': code_data['code'],
                        'evidence': code_data.get('evidence', {})
                    })
            
            if votes:
                most_common = max(votes.items(), key=lambda x: x[1])
                consensus[symbol] = {
                    'code': most_common[0],
                    'agreement': most_common[1] / len(self.results),
                    'alternatives': dict(votes),
                    'evidence': evidences,
                    'confidence': most_common[1] / len(self.results)
                }
        
        return consensus
    
    def get_explanation(self, symbol):
        """Generiert Erklärung für ein Symbol"""
        if symbol not in self.consensus_coding:
            return f"Keine Daten für Symbol {symbol}"
        
        data = self.consensus_coding[symbol]
        explanation = []
        
        explanation.append(f"Symbol: {symbol}")
        explanation.append(f"Konsens-Kodierung: {data['code']}")
        explanation.append(f"Übereinstimmung: {data['agreement']:.0%}")
        explanation.append("")
        explanation.append("Erklärungen der einzelnen Strategien:")
        
        for evidence in data['evidence']:
            strategy = next((s for s in self.strategies if s.name == evidence['strategy']), None)
            if strategy:
                detailed = strategy.explain(symbol, evidence)
                explanation.append(f"\n  {evidence['strategy']}:")
                explanation.append(f"    {detailed}")
        
        return "\n".join(explanation)


# ============================================================================
# VISUALISIERUNGSKOMPONENTE
# ============================================================================

class DerivationVisualizer:
    """Visualisiert Ableitungsprozesse und Modelle"""
    
    def __init__(self, root, plot_thread):
        self.root = root
        self.plot_thread = plot_thread
    
    def plot_coding_comparison(self, coding_results, symbols):
        """Vergleicht Kodierungsergebnisse verschiedener Strategien"""
        if not coding_results or not symbols:
            return
        
        fig, axes = plt.subplots(len(coding_results), 1, figsize=(12, 4*len(coding_results)))
        if len(coding_results) == 1:
            axes = [axes]
        
        for idx, (strategy_name, result) in enumerate(coding_results.items()):
            ax = axes[idx]
            
            display_symbols = symbols[:10]
            codes = []
            for sym in display_symbols:
                if sym in result.get('coding', {}):
                    code = result['coding'][sym].get('code', '?????')
                    # Konvertiere zu numerischer Darstellung
                    code_nums = [int(b) for b in code if b in '01']
                    if len(code_nums) < 5:
                        code_nums = [0] * 5
                    codes.append(code_nums[:5])
                else:
                    codes.append([0, 0, 0, 0, 0])
            
            im = ax.imshow(codes, cmap='Blues', aspect='auto', interpolation='nearest')
            ax.set_xticks(range(5))
            ax.set_xticklabels(['Bit1', 'Bit2', 'Bit3', 'Bit4', 'Bit5'])
            ax.set_yticks(range(len(display_symbols)))
            ax.set_yticklabels(display_symbols)
            ax.set_title(f"{strategy_name} (Konfidenz: {result.get('confidence', 0):.0%})")
        
        plt.tight_layout()
        self.plot_thread.plot(lambda: plt.show())
    
    def plot_confidence_comparison(self, model_manager):
        """Vergleicht Konfidenzen aller Modelle"""
        info = model_manager.get_model_info()
        
        names = []
        confidences = []
        colors = []
        
        for name, model_info in info.items():
            if model_info['trained']:
                names.append(model_info['name'])
                confidences.append(model_info['confidence'])
                colors.append('green' if model_info['active'] else 'gray')
        
        if not names:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(names)), confidences, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Konfidenz')
        ax.set_title('Modell-Konfidenzen im Vergleich')
        ax.set_ylim(0, 1)
        
        # Werte auf Balken
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{conf:.0%}', ha='center', va='bottom')
        
        plt.tight_layout()
        self.plot_thread.plot(lambda: plt.show())


# ============================================================================
# INTERAKTIVE ERKLÄRUNGSKOMPONENTE
# ============================================================================

class InteractiveExplainer:
    """
    Ermöglicht dem Nutzer, nach Gründen für Ableitungen zu fragen
    """
    
    def __init__(self, model_manager, derivation_manager):
        self.model_manager = model_manager
        self.derivation_manager = derivation_manager
        self.question_history = []
    
    def why_symbol(self, symbol):
        """Warum wurde Symbol so kodiert?"""
        explanation = []
        explanation.append(f"🔍 Erklärung für Symbol '{symbol}':")
        explanation.append("=" * 60)
        
        # Kodierungserklärung
        if symbol in self.derivation_manager.consensus_coding:
            coding_exp = self.derivation_manager.get_explanation(symbol)
            explanation.append(coding_exp)
        else:
            explanation.append(f"Symbol '{symbol}' nicht in Kodierung gefunden")
        
        # Modellerklärungen
        explanation.append("\n" + "=" * 60)
        explanation.append("Modell-spezifische Erklärungen:")
        
        model_exps = self.model_manager.explain_all(symbol)
        for model_name, exp in model_exps['explanations'].items():
            if 'content' in exp and exp['content']:
                explanation.append(f"\n{model_name}:")
                for line in exp['content'][:3]:  # Erste 3 Zeilen
                    explanation.append(f"  {line}")
        
        self.question_history.append(('symbol', symbol))
        return "\n".join(explanation)
    
    def why_transition(self, from_state, symbol, to_state=None):
        """Warum wurde dieser Übergang gelernt?"""
        explanation = []
        explanation.append(f"🔍 Erklärung für Übergang:")
        explanation.append(f"   {from_state} --({symbol})--> {to_state if to_state else '?'}")
        explanation.append("=" * 60)
        
        # Sammle Erklärungen von Modellen, die Übergänge unterstützen
        for name in self.model_manager.active_models:
            model = self.model_manager.models.get(name)
            if hasattr(model, 'explain'):
                try:
                    exp = model.explain((from_state, symbol, to_state))
                    if exp and 'content' in exp:
                        explanation.append(f"\n{name}:")
                        explanation.extend(exp['content'])
                except:
                    pass
        
        self.question_history.append(('transition', f"{from_state}--{symbol}"))
        return "\n".join(explanation)
    
    def compare_models(self, data):
        """Vergleicht, wie verschiedene Modelle die Daten erklären"""
        comparison = self.model_manager.compare_models(data)
        
        explanation = []
        explanation.append(f"🔍 MODELLVERGLEICH für: {data}")
        explanation.append("=" * 60)
        
        for name, pred in comparison.get('predictions', {}).items():
            explanation.append(f"\n{name}:")
            explanation.append(f"  Konfidenz: {pred.get('confidence', 0):.0%}")
            explanation.append(f"  {pred.get('summary', 'Keine Aussage')}")
        
        return "\n".join(explanation)
    
    def get_history_string(self):
        """Gibt den Frageverlauf aus"""
        lines = ["📋 Frageverlauf (letzte 10):"]
        for i, (qtype, subject) in enumerate(self.question_history[-10:], 1):
            if qtype == 'symbol':
                lines.append(f"  {i}. Symbol: '{subject}'")
            else:
                lines.append(f"  {i}. Übergang: {subject}")
        return "\n".join(lines)


# ============================================================================
# GENERISCHER ENTSCHEIDUNGSAUTOMAT
# ============================================================================

class GenericDialogueAutomaton:
    """Generischer Automat, der Regeln aus Daten lernt"""
    
    def __init__(self):
        self.states = set()
        self.transitions = {}  # (state, symbol) -> new_state
        self.accepting_states = set()
        self.current_state = None
        self.state_counter = 0
        self.confidence_metrics = {}  # (state, symbol) -> confidence
        self.state_assignments_cache = {}
    
    def learn_from_chains(self, chains):
        """Lernt Automaten-Regeln aus beobachteten Ketten"""
        if not chains:
            return
        
        self.create_states_from_positions(chains)
        self.learn_transitions(chains)
        self.determine_accepting_states(chains)
        self._calculate_confidences(chains)
    
    def create_states_from_positions(self, chains):
        """Erstellt Zustände basierend auf Positionen"""
        for i in range(5):
            self.states.add(f"q_phase_{i}")
        self.states.add("q_start")
        self.states.add("q_error")
        self.current_state = "q_start"
    
    def assign_states(self, chain):
        """Weist jeder Position einen Zustand zu"""
        cache_key = tuple(chain)
        if cache_key in self.state_assignments_cache:
            return self.state_assignments_cache[cache_key]
        
        states = []
        for i, _ in enumerate(chain):
            progress = i / max(1, len(chain) - 1)
            if progress < 0.2:
                phase = 0
            elif progress < 0.4:
                phase = 1
            elif progress < 0.6:
                phase = 2
            elif progress < 0.8:
                phase = 3
            else:
                phase = 4
            states.append(f"q_phase_{phase}")
        
        self.state_assignments_cache[cache_key] = states
        return states
    
    def learn_transitions(self, chains):
        """Lernt Übergänge aus den Daten"""
        transition_counts = defaultdict(Counter)
        
        for chain in chains:
            states = self.assign_states(chain)
            for i in range(len(chain)-1):
                curr_state = states[i]
                next_state = states[i+1]
                symbol = chain[i+1]
                transition_counts[(curr_state, symbol)][next_state] += 1
        
        self.transitions = {}
        for (state, symbol), targets in transition_counts.items():
            if targets:
                most_common = max(targets.items(), key=lambda x: x[1])
                self.transitions[(state, symbol)] = most_common[0]
    
    def determine_accepting_states(self, chains):
        """Bestimmt akzeptierende Zustände"""
        end_states = Counter()
        
        for chain in chains:
            if chain:
                states = self.assign_states(chain)
                if states:
                    end_states[states[-1]] += 1
        
        total = len(chains)
        for state, count in end_states.items():
            if count / total > 0.2:
                self.accepting_states.add(state)
        
        if not self.accepting_states and "q_phase_4" in self.states:
            self.accepting_states.add("q_phase_4")
    
    def _calculate_confidences(self, chains):
        """Berechnet Konfidenzwerte für alle gelernten Übergänge"""
        transition_occurrences = defaultdict(list)
        
        for chain in chains:
            states = self.assign_states(chain)
            for i in range(len(chain)-1):
                curr_state = states[i]
                symbol = chain[i+1]
                next_state = states[i+1]
                
                key = (curr_state, symbol)
                predicted = self.transitions.get(key)
                transition_occurrences[key].append(next_state == predicted)
        
        self.confidence_metrics = {}
        for (state, symbol), occurrences in transition_occurrences.items():
            if (state, symbol) in self.transitions:
                correct = sum(occurrences)
                total = len(occurrences)
                confidence = correct / total if total > 0 else 0
                
                # Beobachtungsfaktor
                obs_factor = min(1.0, total / 10)
                confidence = confidence * 0.7 + obs_factor * 0.3
                
                self.confidence_metrics[(state, symbol)] = round(confidence, 3)
    
    def transition(self, symbol):
        """Führt einen Übergang aus"""
        if self.current_state is None:
            self.current_state = "q_start"
        
        key = (self.current_state, symbol)
        if key in self.transitions:
            self.current_state = self.transitions[key]
            return self.current_state, True, self.confidence_metrics.get(key, 0.5)
        else:
            self.current_state = "q_error"
            return "q_error", False, 0.0
    
    def validate_chain(self, chain):
        """Validiert eine ganze Kette"""
        self.current_state = "q_start"
        protocol = []
        first_error = None
        
        for i, symbol in enumerate(chain):
            new_state, success, confidence = self.transition(symbol)
            protocol.append({
                'position': i + 1,
                'symbol': symbol,
                'state': new_state,
                'success': success,
                'confidence': confidence
            })
            
            if new_state == "q_error" and first_error is None:
                possible = []
                for (state, sym), next_state in self.transitions.items():
                    if state == self.current_state:
                        possible.append(f"{sym}→{next_state}")
                
                explanation = f"Kein gültiger Übergang von {self.current_state}"
                if possible:
                    explanation += f"\nMöglich: {', '.join(possible[:3])}"
                
                first_error = {
                    'position': i + 1,
                    'symbol': symbol,
                    'explanation': explanation
                }
        
        valid = self.current_state in self.accepting_states and first_error is None
        return valid, self.current_state, protocol, first_error
    
    def get_rules_string(self):
        """Gibt die gelernten Regeln zurück"""
        lines = []
        lines.append("GELERNTE AUTOMATEN-REGELN:")
        lines.append("=" * 60)
        
        if not self.transitions:
            lines.append("  Keine Regeln gelernt.")
            return "\n".join(lines)
        
        rules_by_state = defaultdict(list)
        for (state, symbol), next_state in self.transitions.items():
            rules_by_state[state].append((symbol, next_state))
        
        for state in sorted(rules_by_state.keys()):
            lines.append(f"\n{state}:")
            for symbol, next_state in sorted(rules_by_state[state]):
                conf = self.confidence_metrics.get((state, symbol), 0)
                stars = "★" * int(conf * 5) + "☆" * (5 - int(conf * 5))
                lines.append(f"  {symbol} → {next_state}  {stars} ({conf:.0%})")
        
        lines.append(f"\nAkzeptierend: {', '.join(sorted(self.accepting_states))}")
        return "\n".join(lines)


# ============================================================================
# MULTI-FORMAT EXPORTER
# ============================================================================

class MultiFormatExporter:
    """Exportiert Ergebnisse in verschiedene Formate"""
    
    def __init__(self):
        self.export_path = "exports"
        os.makedirs(self.export_path, exist_ok=True)
    
    def to_json(self, data, filename=None):
        """Exportiert als JSON"""
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.export_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return filepath
    
    def to_csv(self, data, filename=None):
        """Exportiert als CSV"""
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.export_path, filename)
        
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if 'coding' in data:
                writer.writerow(['Symbol', 'Code', 'Konfidenz', 'Strategien'])
                for symbol, info in data['coding'].items():
                    strategies = len(info.get('evidence', []))
                    writer.writerow([
                        symbol, 
                        info.get('code', ''), 
                        f"{info.get('confidence', 0):.0%}",
                        strategies
                    ])
            
            if 'model_comparison' in data:
                writer.writerow([])
                writer.writerow(['Modell', 'Konfidenz', 'Trainiert', 'Aktiv'])
                for name, model_info in data['model_comparison'].items():
                    writer.writerow([
                        model_info['name'],
                        f"{model_info['confidence']:.0%}",
                        model_info['trained'],
                        model_info['active']
                    ])
        
        return filepath
    
    def to_html(self, data, filename=None):
        """Exportiert als interaktiven HTML-Bericht"""
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        filepath = os.path.join(self.export_path, filename)
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head><title>ARSXAI8 Analysebericht</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        html.append("h1 { color: #2c3e50; }")
        html.append("h2 { color: #34495e; border-bottom: 2px solid #3498db; }")
        html.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #3498db; color: white; }")
        html.append("tr:nth-child(even) { background-color: #f2f2f2; }")
        html.append(".confidence-high { color: green; font-weight: bold; }")
        html.append(".confidence-medium { color: orange; }")
        html.append(".confidence-low { color: red; }")
        html.append(".model-box { background-color: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        html.append(f"<h1>ARSXAI8 Analysebericht</h1>")
        html.append(f"<p>Generiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Kodierungsergebnisse
        if 'coding' in data:
            html.append("<h2>Kodierungsergebnisse</h2>")
            html.append("<table>")
            html.append("<tr><th>Symbol</th><th>Code</th><th>Konfidenz</th><th>Übereinstimmung</th></tr>")
            
            for symbol, info in sorted(data['coding'].items()):
                conf = info.get('confidence', 0)
                conf_class = "confidence-high" if conf > 0.7 else "confidence-medium" if conf > 0.4 else "confidence-low"
                agreement = info.get('agreement', 0)
                html.append(f"<tr>")
                html.append(f"<td>{symbol}</td>")
                html.append(f"<td><code>{info.get('code', '')}</code></td>")
                html.append(f"<td class='{conf_class}'>{conf:.0%}</td>")
                html.append(f"<td>{agreement:.0%}</td>")
                html.append(f"</tr>")
            
            html.append("</table>")
        
        # Modell-Vergleich
        if 'model_comparison' in data:
            html.append("<h2>Modell-Vergleich</h2>")
            for name, model_info in data['model_comparison'].items():
                status = "✅" if model_info['active'] else "⭕"
                trained = "✓" if model_info['trained'] else "✗"
                conf_class = "confidence-high" if model_info['confidence'] > 0.7 else "confidence-medium" if model_info['confidence'] > 0.4 else "confidence-low"
                
                html.append(f"<div class='model-box'>")
                html.append(f"<h3>{status} {model_info['name']}</h3>")
                html.append(f"<p><strong>Beschreibung:</strong> {model_info['description']}</p>")
                html.append(f"<p><strong>Konfidenz:</strong> <span class='{conf_class}'>{model_info['confidence']:.0%}</span></p>")
                html.append(f"<p><strong>Trainiert:</strong> {trained}</p>")
                html.append(f"</div>")
        
        # Statistiken
        if 'statistics' in data:
            html.append("<h2>Statistiken</h2>")
            html.append("<table>")
            html.append("<tr><th>Metrik</th><th>Wert</th></tr>")
            for key, value in data['statistics'].items():
                html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            html.append("</table>")
        
        html.append("</body>")
        html.append("</html>")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(html))
        
        return filepath
    
    def to_latex(self, data, filename=None):
        """Exportiert als LaTeX für wissenschaftliche Publikationen"""
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        
        filepath = os.path.join(self.export_path, filename)
        
        latex = []
        latex.append("\\documentclass{article}")
        latex.append("\\usepackage[utf8]{inputenc}")
        latex.append("\\usepackage{booktabs}")
        latex.append("\\usepackage{graphicx}")
        latex.append("\\begin{document}")
        
        latex.append("\\title{ARSXAI8 Analyseergebnisse}")
        latex.append(f"\\date{{{datetime.now().strftime('%Y-%m-%d')}}}")
        latex.append("\\maketitle")
        
        if 'coding' in data:
            latex.append("\\section{Kodierung der Terminalzeichen}")
            latex.append("\\begin{tabular}{lll}")
            latex.append("\\toprule")
            latex.append("Symbol & Code & Konfidenz \\\\")
            latex.append("\\midrule")
            
            for symbol, info in sorted(data['coding'].items()):
                latex.append(f"{symbol} & {info.get('code', '')} & {info.get('confidence', 0):.0%} \\\\")
            
            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
        
        if 'model_comparison' in data:
            latex.append("\\section{Modell-Vergleich}")
            latex.append("\\begin{tabular}{lll}")
            latex.append("\\toprule")
            latex.append("Modell & Konfidenz & Status \\\\")
            latex.append("\\midrule")
            
            for name, model_info in data['model_comparison'].items():
                status = "aktiv" if model_info['active'] else "inaktiv"
                latex.append(f"{model_info['name']} & {model_info['confidence']:.0%} & {status} \\\\")
            
            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
        
        latex.append("\\end{document}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(latex))
        
        return filepath


# ============================================================================
# GUI - HAUPTFENSTER
# ============================================================================

class ARSXAI8GUI:
    """Haupt-GUI für ARSXAI8"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ARSXAI8 - Algorithmic Recursive Sequence Analysis with Explainable AI")
        self.root.geometry("1600x1000")
        
        # Threading
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
        self.derivation_manager = DerivationManager()
        self.automaton = GenericDialogueAutomaton()
        self.model_manager = XAIModelManager()
        self.generator = ChainGenerator()
        self.visualizer = DerivationVisualizer(root, self.plot_thread)
        self.exporter = MultiFormatExporter()
        
        # Modelle registrieren
        self._register_models()
        
        # GUI erstellen
        self.create_menu()
        self.create_main_panels()
        self.status_var = tk.StringVar(value="Bereit")
        self.create_statusbar()
        
        # Erklärer (wird nach Analyse initialisiert)
        self.explainer = None
        
        self.show_module_status()
    
    def _register_models(self):
        """Registriert alle XAI-Modelle"""
        # ARS 2.0
        ars20 = ARS20()
        self.model_manager.register_model('ARS20', ars20)
        
        # ARS 3.0
        ars30 = GrammarInducer()
        self.model_manager.register_model('ARS30', ars30)
        
        # HMM
        if MODULE_STATUS['hmmlearn']:
            hmm_model = ARSHiddenMarkovModel(n_states=5)
            self.model_manager.register_model('HMM', hmm_model)
        
        # CRF
        if MODULE_STATUS['crf']:
            crf_model = ARSCRFModel()
            self.model_manager.register_model('CRF', crf_model)
        
        # Petri-Netz
        if MODULE_STATUS['networkx']:
            petri_model = ARSPetriNet("ARS_PetriNet")
            self.model_manager.register_model('Petri', petri_model)
        
        # Generator
        self.model_manager.register_model('Generator', self.generator)
        
        # Alle Modelle aktivieren
        for name in self.model_manager.models:
            self.model_manager.activate_model(name)
    
    def process_updates(self):
        """Verarbeitet asynchrone GUI-Updates"""
        try:
            while True:
                update_func = self.update_queue.get_nowait()
                update_func()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_updates)
    
    def safe_gui_update(self, func):
        """Führt Funktion im Hauptthread aus"""
        self.update_queue.put(func)
    
    def create_menu(self):
        """Erstellt die Menüleiste"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Datei-Menü
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Datei", menu=file_menu)
        file_menu.add_command(label="Transkripte laden", command=self.load_transcripts)
        file_menu.add_command(label="Beispiel laden", command=self.load_example)
        file_menu.add_separator()
        file_menu.add_command(label="Exportieren", command=self.show_export_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self.root.quit)
        
        # Analyse-Menü
        analyze_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analyse", menu=analyze_menu)
        analyze_menu.add_command(label="Alle Strategien anwenden", command=self.run_all_strategies)
        analyze_menu.add_command(label="Alle Modelle trainieren", command=self.train_all_models)
        analyze_menu.add_command(label="Automaten lernen", command=self.learn_automaton)
        analyze_menu.add_command(label="Validierung durchführen", command=self.run_validation)
        
        # XAI-Menü
        xai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="XAI", menu=xai_menu)
        xai_menu.add_command(label="Erklärung für Symbol", command=self.ask_explanation)
        xai_menu.add_command(label="Modellvergleich", command=self.compare_models)
        xai_menu.add_command(label="Konfidenzen vergleichen", command=self.plot_model_confidences)
        xai_menu.add_command(label="Was-wäre-wenn", command=self.what_if_dialog)
        
        # Generierung-Menü
        gen_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Generierung", menu=gen_menu)
        gen_menu.add_command(label="Mit ARS 2.0 generieren", command=lambda: self.generate_with('ARS20'))
        gen_menu.add_command(label="Mit ARS 3.0 generieren", command=lambda: self.generate_with('ARS30'))
        gen_menu.add_command(label="Mit HMM generieren", command=lambda: self.generate_with('HMM'))
        
        # Visualisierung-Menü
        vis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualisierung", menu=vis_menu)
        vis_menu.add_command(label="Kodierungsvergleich", command=self.plot_coding_comparison)
        vis_menu.add_command(label="Modell-Konfidenzen", command=self.plot_model_confidences)
        vis_menu.add_command(label="Automaten-Graph", command=self.plot_automaton)
        
        # Hilfe-Menü
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Hilfe", menu=help_menu)
        help_menu.add_command(label="Modulstatus", command=self.show_module_status)
        help_menu.add_command(label="Über", command=self.show_about)
    
    def create_main_panels(self):
        """Erstellt die Haupt-Panels"""
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Linkes Panel - Eingabe
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        self.create_input_panel(left_frame)
        
        # Rechtes Panel - Ausgabe mit Notebook
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        self.create_output_panel(right_frame)
    
    def create_input_panel(self, parent):
        """Erstellt das Eingabe-Panel"""
        ttk.Label(parent, text="Eingabe", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=5)
        
        # Trennzeichen
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
        
        # Text-Eingabe
        ttk.Label(parent, text="Transkripte (eine pro Zeile, # für Kommentare):").pack(anchor=tk.W, pady=5)
        
        self.text_input = scrolledtext.ScrolledText(parent, height=15, font=('Courier', 10))
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Datei laden", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Parsen", command=self.parse_input).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Beispiel", command=self.load_example).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Alle trainieren", command=self.train_all_models).pack(side=tk.LEFT, padx=2)
        
        # Info
        self.info_var = tk.StringVar(value="Keine Daten geladen")
        ttk.Label(parent, textvariable=self.info_var, foreground="blue").pack(anchor=tk.W, pady=5)
        
        # Warnungen
        self.warning_text = scrolledtext.ScrolledText(parent, height=5, font=('Courier', 9), 
                                                      foreground="orange")
        self.warning_text.pack(fill=tk.X, pady=5)
    
    def create_output_panel(self, parent):
        """Erstellt das Ausgabe-Panel mit Notebook-Tabs"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Kodierung
        self.tab_coding = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_coding, text="Kodierung")
        self.create_coding_tab()
        
        # Tab 2: Modelle
        self.tab_models = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_models, text="Modelle")
        self.create_models_tab()
        
        # Tab 3: Automat
        self.tab_automaton = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_automaton, text="Automat")
        self.create_automaton_tab()
        
        # Tab 4: XAI
        self.tab_xai = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_xai, text="XAI")
        self.create_xai_tab()
        
        # Tab 5: Generierung
        self.tab_generation = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_generation, text="Generierung")
        self.create_generation_tab()
        
        # Tab 6: Statistiken
        self.tab_stats = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_stats, text="Statistiken")
        self.create_statistics_tab()
    
    def create_coding_tab(self):
        """Erstellt den Kodierungs-Tab"""
        control = ttk.Frame(self.tab_coding)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Alle Strategien", 
                  command=self.run_all_strategies).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Konsens anzeigen", 
                  command=self.show_consensus).pack(side=tk.LEFT, padx=5)
        
        self.text_coding = scrolledtext.ScrolledText(self.tab_coding, font=('Courier', 10))
        self.text_coding.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_models_tab(self):
        """Erstellt den Modelle-Tab"""
        # Modell-Auswahl
        control = ttk.Frame(self.tab_models)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Label(control, text="Aktive Modelle:").pack(side=tk.LEFT)
        
        self.model_vars = {}
        for name in self.model_manager.models:
            var = tk.BooleanVar(value=True)
            self.model_vars[name] = var
            ttk.Checkbutton(control, text=name, variable=var,
                          command=lambda n=name: self.toggle_model(n)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control, text="Alle trainieren", 
                  command=self.train_all_models).pack(side=tk.LEFT, padx=20)
        
        # Modell-Ausgabe
        self.text_models = scrolledtext.ScrolledText(self.tab_models, font=('Courier', 10))
        self.text_models.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_automaton_tab(self):
        """Erstellt den Automaten-Tab"""
        control = ttk.Frame(self.tab_automaton)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Automaten lernen", 
                  command=self.learn_automaton).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Kette validieren", 
                  command=self.validate_chain).pack(side=tk.LEFT, padx=5)
        
        self.text_automaton = scrolledtext.ScrolledText(self.tab_automaton, font=('Courier', 10))
        self.text_automaton.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_xai_tab(self):
        """Erstellt den XAI-Tab"""
        # Frage-Eingabe
        question_frame = ttk.Frame(self.tab_xai)
        question_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(question_frame, text="Symbol:").pack(side=tk.LEFT)
        self.symbol_entry = ttk.Entry(question_frame, width=10)
        self.symbol_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(question_frame, text="Warum?", 
                  command=self.ask_explanation).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(question_frame, text="Modelle vergleichen", 
                  command=self.compare_models).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(question_frame, text="Was-wäre-wenn", 
                  command=self.what_if_dialog).pack(side=tk.LEFT, padx=5)
        
        # Ausgabe
        self.text_xai = scrolledtext.ScrolledText(self.tab_xai, font=('Courier', 10))
        self.text_xai.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_generation_tab(self):
        """Erstellt den Generierung-Tab"""
        control = ttk.Frame(self.tab_generation)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Label(control, text="Quellmodell:").pack(side=tk.LEFT)
        self.gen_source = tk.StringVar(value="ARS20")
        ttk.Radiobutton(control, text="ARS 2.0", variable=self.gen_source, 
                       value="ARS20").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control, text="ARS 3.0", variable=self.gen_source, 
                       value="ARS30").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control, text="HMM", variable=self.gen_source, 
                       value="HMM").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control, text="Anzahl:").pack(side=tk.LEFT, padx=(20,5))
        self.gen_count = ttk.Spinbox(control, from_=1, to=50, width=5)
        self.gen_count.set(5)
        self.gen_count.pack(side=tk.LEFT)
        
        ttk.Button(control, text="Generieren", 
                  command=self.generate_chains).pack(side=tk.LEFT, padx=20)
        
        self.text_generation = scrolledtext.ScrolledText(self.tab_generation, font=('Courier', 10))
        self.text_generation.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_statistics_tab(self):
        """Erstellt den Statistik-Tab"""
        control = ttk.Frame(self.tab_stats)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Statistiken berechnen", 
                  command=self.calculate_statistics).pack(side=tk.LEFT, padx=5)
        
        self.text_stats = scrolledtext.ScrolledText(self.tab_stats, font=('Courier', 10))
        self.text_stats.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_statusbar(self):
        """Erstellt die Statusleiste"""
        status = ttk.Frame(self.root)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(status, length=100, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
    
    def toggle_model(self, name):
        """Aktiviert/deaktiviert ein Modell"""
        if self.model_vars[name].get():
            self.model_manager.activate_model(name)
        else:
            self.model_manager.deactivate_model(name)
    
    def get_actual_delimiter(self):
        """Gibt das tatsächliche Trennzeichen zurück"""
        delim = self.delimiter.get()
        if delim == "custom":
            return self.custom_delimiter.get()
        return delim
    
    def parse_line(self, line):
        """Parst eine einzelne Zeile"""
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
        """Parst die Eingabe"""
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
            self.run_all_strategies()
            self.train_all_models()
        else:
            messagebox.showwarning("Warnung", "Keine gültigen Ketten gefunden!")
    
    def run_validation(self):
        """Führt Validierung durch"""
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
    
    def run_all_strategies(self):
        """Wendet alle Kodierungsstrategien an"""
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        self.status_var.set("Wende Kodierungsstrategien an...")
        self.progress_bar.start()
        
        def run():
            try:
                results = self.derivation_manager.derive_all(self.chains)
                
                def update():
                    self.text_coding.delete("1.0", tk.END)
                    self.text_coding.insert(tk.END, "ERGEBNISSE DER KODIERUNGSSTRATEGIEN\n")
                    self.text_coding.insert(tk.END, "=" * 70 + "\n\n")
                    
                    for strategy_name, result in results.items():
                        self.text_coding.insert(tk.END, f"\n{strategy_name}:\n")
                        self.text_coding.insert(tk.END, "-" * 40 + "\n")
                        
                        if 'error' in result:
                            self.text_coding.insert(tk.END, f"Fehler: {result['error']}\n")
                        else:
                            conf = result.get('confidence', 0)
                            self.text_coding.insert(tk.END, f"Konfidenz: {conf:.0%}\n\n")
                            
                            for symbol, code_data in list(result['coding'].items())[:15]:
                                code = code_data.get('code', '?????')
                                self.text_coding.insert(tk.END, f"  {symbol}: {code}\n")
                    
                    self.show_consensus()
                    self.status_var.set("Kodierung abgeschlossen")
                    self.progress_bar.stop()
                
                self.safe_gui_update(update)
            except Exception as e:
                def error():
                    messagebox.showerror("Fehler", f"Analyse fehlgeschlagen:\n{str(e)}")
                    self.progress_bar.stop()
                
                self.safe_gui_update(error)
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def show_consensus(self):
        """Zeigt den Konsens aller Strategien"""
        if not self.derivation_manager.consensus_coding:
            return
        
        self.text_coding.insert(tk.END, "\n" + "=" * 70 + "\n")
        self.text_coding.insert(tk.END, "KONSENS-KODIERUNG (Mehrheitsentscheidung)\n")
        self.text_coding.insert(tk.END, "=" * 70 + "\n\n")
        
        for symbol, data in sorted(self.derivation_manager.consensus_coding.items()):
            agreement = data.get('agreement', 0)
            conf_color = "✓" if agreement > 0.66 else "⚠️" if agreement > 0.33 else "❌"
            self.text_coding.insert(tk.END, 
                f"{conf_color} {symbol}: {data['code']} (Übereinstimmung: {agreement:.0%})\n")
    
    def train_all_models(self):
        """Trainiert alle aktiven Modelle"""
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
                        model = self.model_manager.models.get(name)
                        info = model.get_info() if model else {}
                        
                        if result.get('success'):
                            self.text_models.insert(tk.END, f"✓ {name}\n")
                            self.text_models.insert(tk.END, f"  Beschreibung: {info.get('description', '')}\n")
                            self.text_models.insert(tk.END, f"  Konfidenz: {info.get('confidence', 0):.0%}\n\n")
                        else:
                            self.text_models.insert(tk.END, f"✗ {name}: {result.get('error', 'Unbekannter Fehler')}\n\n")
                    
                    # Generator konfigurieren
                    if 'ARS20' in self.model_manager.models:
                        self.generator.set_source_model(self.model_manager.models['ARS20'])
                    
                    # Erklärer initialisieren
                    self.explainer = InteractiveExplainer(self.model_manager, self.derivation_manager)
                    
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
    
    def learn_automaton(self):
        """Lernt Automaten-Regeln"""
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        self.status_var.set("Lerne Automaten-Regeln...")
        
        try:
            self.automaton.learn_from_chains(self.chains)
            
            self.text_automaton.delete("1.0", tk.END)
            self.text_automaton.insert(tk.END, self.automaton.get_rules_string())
            
            self.status_var.set("Automaten gelernt")
        except Exception as e:
            messagebox.showerror("Fehler", f"Automaten-Lernen fehlgeschlagen:\n{str(e)}")
    
    def validate_chain(self):
        """Validiert eine ausgewählte Kette"""
        if not self.chains or not self.automaton.transitions:
            messagebox.showerror("Fehler", "Keine Daten oder kein Automat!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Kette auswählen")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Verfügbare Ketten:").pack(pady=5)
        
        listbox = tk.Listbox(dialog)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for i, chain in enumerate(self.chains[:10]):
            listbox.insert(tk.END, f"{i+1}: {' → '.join(chain[:10])}...")
        
        def validate_selected():
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                chain = self.chains[idx]
                
                valid, state, protocol, error = self.automaton.validate_chain(chain)
                
                result_text = []
                result_text.append(f"VALIDIERUNG KETTE {idx+1}\n")
                result_text.append("=" * 50 + "\n")
                result_text.append(f"Ergebnis: {'✓ GÜLTIG' if valid else '✗ UNGÜLTIG'}\n")
                result_text.append(f"Endzustand: {state}\n\n")
                
                if error:
                    result_text.append(f"❌ Fehler bei Position {error['position']}: {error['symbol']}\n")
                    result_text.append(f"   {error['explanation']}\n\n")
                
                result_text.append("Entscheidungspfad:\n")
                for step in protocol:
                    result_text.append(
                        f"  {step['position']}: {step['symbol']} → {step['state']} "
                        f"(Konf: {step['confidence']:.0%})\n"
                    )
                
                self.text_automaton.insert(tk.END, "\n" + "\n".join(result_text))
                dialog.destroy()
        
        ttk.Button(dialog, text="Validieren", command=validate_selected).pack(pady=5)
    
    def ask_explanation(self):
        """Fragt nach Erklärung für ein Symbol"""
        if not self.explainer:
            messagebox.showerror("Fehler", "Keine Analyse vorhanden!")
            return
        
        symbol = self.symbol_entry.get().strip()
        if not symbol:
            messagebox.showwarning("Warnung", "Bitte ein Symbol eingeben!")
            return
        
        explanation = self.explainer.why_symbol(symbol)
        
        self.text_xai.delete("1.0", tk.END)
        self.text_xai.insert(tk.END, explanation)
        self.text_xai.insert(tk.END, "\n\n" + self.explainer.get_history_string())
    
    def compare_models(self):
        """Vergleicht Modelle für ein Symbol"""
        if not self.explainer:
            messagebox.showerror("Fehler", "Keine Analyse vorhanden!")
            return
        
        symbol = self.symbol_entry.get().strip()
        if not symbol:
            messagebox.showwarning("Warnung", "Bitte ein Symbol eingeben!")
            return
        
        comparison = self.explainer.compare_models(symbol)
        
        self.text_xai.delete("1.0", tk.END)
        self.text_xai.insert(tk.END, comparison)
    
    def what_if_dialog(self):
        """Öffnet Was-wäre-wenn Dialog"""
        if not self.explainer:
            messagebox.showerror("Fehler", "Keine Analyse vorhanden!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Was-wäre-wenn Simulation")
        dialog.geometry("400x250")
        
        ttk.Label(dialog, text="Symbol:").grid(row=0, column=0, padx=5, pady=5)
        symbol_entry = ttk.Entry(dialog)
        symbol_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Alternativer Code (5 Bit):").grid(row=1, column=0, padx=5, pady=5)
        code_entry = ttk.Entry(dialog)
        code_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Modell:").grid(row=2, column=0, padx=5, pady=5)
        model_var = tk.StringVar(value="ARS20")
        model_combo = ttk.Combobox(dialog, textvariable=model_var, 
                                   values=list(self.model_manager.active_models))
        model_combo.grid(row=2, column=1, padx=5, pady=5)
        
        def simulate():
            symbol = symbol_entry.get().strip()
            code = code_entry.get().strip()
            model_name = model_var.get()
            
            if symbol and code and model_name in self.model_manager.models:
                model = self.model_manager.models[model_name]
                # Hier könnte die Was-wäre-wenn-Logik implementiert werden
                explanation = f"Simulation für {symbol} als {code} mit {model_name}\n"
                explanation += "Diese Funktion ist in Entwicklung."
                
                self.text_xai.delete("1.0", tk.END)
                self.text_xai.insert(tk.END, explanation)
                dialog.destroy()
        
        ttk.Button(dialog, text="Simulieren", command=simulate).grid(row=3, column=0, columnspan=2, pady=20)
    
    def generate_with(self, model_name):
        """Generiert Ketten mit einem bestimmten Modell"""
        if model_name not in self.model_manager.models:
            messagebox.showerror("Fehler", f"Modell {model_name} nicht verfügbar!")
            return
        
        model = self.model_manager.models[model_name]
        if not model.trained:
            messagebox.showerror("Fehler", f"Modell {model_name} nicht trainiert!")
            return
        
        self.gen_source.set(model_name)
        self.generate_chains()
    
    def generate_chains(self):
        """Generiert neue Ketten"""
        source = self.gen_source.get()
        count = int(self.gen_count.get())
        
        if source not in self.model_manager.models:
            messagebox.showerror("Fehler", f"Modell {source} nicht verfügbar!")
            return
        
        model = self.model_manager.models[source]
        self.generator.set_source_model(model)
        
        chains = self.generator.generate(count)
        
        self.text_generation.delete("1.0", tk.END)
        self.text_generation.insert(tk.END, f"GENERIERTE KETTEN mit {model.name}\n")
        self.text_generation.insert(tk.END, "=" * 60 + "\n\n")
        
        for i, chain in enumerate(chains, 1):
            self.text_generation.insert(tk.END, f"{i}: {' → '.join(chain)}\n")
        
        # Erklärung anzeigen
        exp = self.generator.explain(count, 'detailed')
        if 'content' in exp:
            self.text_generation.insert(tk.END, "\n" + "=" * 60 + "\n")
            self.text_generation.insert(tk.END, "ERKLÄRUNG:\n")
            for line in exp['content']:
                self.text_generation.insert(tk.END, line + "\n")
    
    def plot_coding_comparison(self):
        """Visualisiert Kodierungsvergleich"""
        if not self.derivation_manager.results:
            messagebox.showerror("Fehler", "Keine Kodierungsergebnisse!")
            return
        
        symbols = list(self.terminals)[:10]
        self.visualizer.plot_coding_comparison(self.derivation_manager.results, symbols)
    
    def plot_model_confidences(self):
        """Visualisiert Modell-Konfidenzen"""
        self.visualizer.plot_confidence_comparison(self.model_manager)
    
    def plot_automaton(self):
        """Visualisiert Automaten"""
        if not self.automaton or not self.automaton.transitions:
            messagebox.showerror("Fehler", "Kein Automat vorhanden!")
            return
        
        if MODULE_STATUS['graphviz']:
            # Hier müsste die Graphviz-Visualisierung implementiert werden
            self.text_automaton.insert(tk.END, "\n" + self.automaton.get_rules_string())
        else:
            self.text_automaton.insert(tk.END, "\n" + self.automaton.get_rules_string())
    
    def calculate_statistics(self):
        """Berechnet Statistiken"""
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
        
        symbol_counts = Counter()
        for chain in self.chains:
            symbol_counts.update(chain)
        
        stats.append("\nHäufigste Symbole:")
        for sym, count in symbol_counts.most_common(10):
            stats.append(f"  {sym}: {count}x")
        
        if self.comments:
            stats.append(f"\nKommentare: {len(self.comments)}")
        
        self.text_stats.delete("1.0", tk.END)
        self.text_stats.insert(tk.END, "\n".join(stats))
    
    def show_export_dialog(self):
        """Zeigt Export-Dialog"""
        if not self.derivation_manager.consensus_coding:
            messagebox.showerror("Fehler", "Keine Daten zum Exportieren!")
            return
        
        export_data = {
            'coding': self.derivation_manager.consensus_coding,
            'terminals': list(self.terminals),
            'chains': self.chains,
            'comments': self.comments,
            'model_comparison': self.model_manager.get_model_info(),
            'statistics': {
                'n_chains': len(self.chains),
                'n_terminals': len(self.terminals),
                'avg_length': np.mean([len(c) for c in self.chains]) if self.chains else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Exportieren")
        dialog.geometry("300x300")
        
        ttk.Label(dialog, text="Export-Format:").pack(pady=10)
        
        def export_json():
            filepath = self.exporter.to_json(export_data)
            messagebox.showinfo("Export erfolgreich", f"Gespeichert als:\n{filepath}")
            dialog.destroy()
        
        def export_csv():
            filepath = self.exporter.to_csv(export_data)
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
        ttk.Button(dialog, text="CSV", command=export_csv).pack(pady=5)
        ttk.Button(dialog, text="HTML (Bericht)", command=export_html).pack(pady=5)
        ttk.Button(dialog, text="LaTeX", command=export_latex).pack(pady=5)
    
    def load_file(self):
        """Lädt eine Datei"""
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
        """Alias für load_file"""
        self.load_file()
    
    def load_example(self):
        """Lädt ein Beispiel"""
        example = """# Beispieltranskripte für Verkaufsgespräche
# Jede Zeile enthält eine Sequenz von Terminalzeichen

# Transkript 1: Standard-Verkauf
KBG, VBG, KBBd, VBBd, KBA, VBA, KBBd, VBBd, KBA, VAA, KAA, VAV, KAV

# Transkript 2: Mit Wiederholungen in der Bedarfsphase
KBG, VBG, KBBd, VBBd, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV

# Transkript 3: Kurzer Verkauf
KBG, VBG, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV

# Transkript 4: Mit Beratungsphase
KBG, VBG, KBBd, VBBd, KBA, VBA, KAE, VAE, KBA, VBA, VAA, KAA, VAV, KAV

# Transkript 5: Mit vielen Bedarfswiederholungen
KBG, VBG, KBBd, VBBd, KBBd, VBBd, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV"""
        
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", example)
        self.parse_input()
    
    def show_module_status(self):
        """Zeigt Modulstatus"""
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
        """Zeigt Über-Informationen"""
        about = """ARSXAI8 - Algorithmic Recursive Sequence Analysis with Explainable AI

Version 8.0 (Vollständige Integration mit Generierungs-Korrekturen)

Integrierte Modelle:
• ARS 2.0 - Basis-Grammatik
• ARS 3.0 - Hierarchische Grammatik
• HMM - Bayessche Netze
• CRF - Conditional Random Fields
• Petri-Netze - Ressourcenmodellierung
• Generator - Synthetische Ketten

XAI-Features:
• Mehrere Ableitungsstrategien
• Modellvergleich mit Konsensanalyse
• Interaktive Erklärungen
• Konfidenzmetriken
• Umfangreiche Exportformate

© 2024 - Explainable AI Research"""
        
        messagebox.showinfo("Über ARSXAI8", about)


# ============================================================================
# HAUPTFUNKTION
# ============================================================================

def main():
    """Hauptfunktion"""
    root = tk.Tk()
    app = ARSXAI8GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
