"""
ARSXAI7.py - Algorithmic Recursive Sequence Analysis with Explainable AI
========================================================================
Universelle Analyseplattform für beliebige Terminalzeichenketten mit
automatischer Strukturableitung und XAI-Komponenten.

Kernkonzepte:
- Laden beliebiger Transkriptdateien mit Kommentaren
- Automatische Extraktion der Terminal-Menge
- Mehrere Strategien zur Ableitung von Kodierungen und Regeln
- Konfidenzmetriken für alle Ableitungen
- Interaktive Erklärungskomponente
- Visualisierung der Ableitungsprozesse
- Progressive Learning aus neuen Daten
- Export in verschiedene Formate

Version: 7.0 (Vollständige XAI-Integration)
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
# PAKETVERWALTUNG
# ============================================================================

warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    'graphviz'  # Neu für Automaten-Visualisierung
]

def check_and_install_packages():
    """Prüft und installiert fehlende Python-Pakete"""
    print("=" * 70)
    print("ARSXAI7 - PAKETPRÜFUNG")
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
    MODULE_STATUS['graphviz'] = True
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
# DATENVALIDIERUNG
# ============================================================================

class DataValidator:
    """
    Prüft die geladenen Transkripte auf Qualität und Konsistenz
    """
    
    def __init__(self):
        self.issues = []
        self.warnings = []
    
    def validate_chains(self, chains):
        """Validiert alle Ketten und sammelt Probleme"""
        self.issues = []
        self.warnings = []
        
        if not chains:
            self.issues.append(("error", "Keine Ketten gefunden"))
            return self.issues
        
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
        symbol_groups = self.group_similar_symbols(all_symbols)
        for base, similar in symbol_groups.items():
            if len(similar) > 1:
                self.warnings.append(("info", f"Ähnliche Symbole gefunden: {', '.join(similar)}"))
        
        return self.issues, self.warnings
    
    def group_similar_symbols(self, symbols):
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
        """Leitet Kodierung aus Ketten ab - muss in Subklassen implementiert werden"""
        raise NotImplementedError
    
    def explain(self, symbol, code):
        """Erklärt, warum ein Symbol so kodiert wurde"""
        raise NotImplementedError


class PositionBasedCoding(CodingStrategy):
    """
    Strategie 1: Kodierung basierend auf Position in der Sequenz
    - Bit 1: Sprecher (K/V aus Präfix)
    - Bits 2-3: Phase aus durchschnittlicher Position
    - Bits 4-5: Subphase aus Wiederholungsmuster
    """
    
    def __init__(self):
        super().__init__("Positionsbasierte Kodierung")
    
    def derive(self, chains):
        """Leitet Kodierung aus Positionen ab"""
        coding = {}
        
        # Sammle Positionsstatistiken
        positions = defaultdict(list)
        for chain in chains:
            for i, sym in enumerate(chain):
                positions[sym].append(i)
        
        # Sammle Übergangsmuster
        transitions = defaultdict(Counter)
        for chain in chains:
            for i in range(len(chain)-1):
                transitions[chain[i]][chain[i+1]] += 1
        
        for symbol in positions:
            # Bit 1: Sprecher (K=0, V=1)
            speaker = "0" if symbol.startswith('K') else "1"
            
            # Bits 2-3: Phase aus durchschnittlicher Position
            avg_pos = np.mean(positions[symbol])
            max_pos = max([max(pos) for pos in positions.values() if pos])
            phase_norm = avg_pos / max_pos if max_pos > 0 else 0
            
            if phase_norm < 0.25:
                phase = "00"  # Begrüßung
            elif phase_norm < 0.5:
                phase = "01"  # Bedarf
            elif phase_norm < 0.75:
                phase = "10"  # Abschluss
            else:
                phase = "11"  # Verabschiedung
            
            # Bits 4-5: Subphase aus Wiederholungsmuster
            # Wie oft wird das Symbol wiederholt?
            repeat_count = sum(1 for pos_list in positions.values() 
                             if len(pos_list) > 1 for pos in pos_list)
            if repeat_count > len(chains) * 0.3:
                subphase = "01"  # Folge (bei häufigen Wiederholungen)
            else:
                subphase = "00"  # Basis
            
            code = f"{speaker}{phase}{subphase}"
            coding[symbol] = {
                'code': code,
                'evidence': {
                    'avg_position': avg_pos,
                    'phase_norm': phase_norm,
                    'repeat_count': repeat_count
                }
            }
        
        self.confidence = self.calculate_confidence(coding, chains)
        return coding
    
    def calculate_confidence(self, coding, chains):
        """Berechnet Konfidenz der Ableitung"""
        # Je konsistenter die Positionen, desto höher die Konfidenz
        confidence = 0.7  # Basis-Konfidenz
        return confidence
    
    def explain(self, symbol, code_data):
        """Erklärt die positionsbasierte Kodierung"""
        code = code_data['code']
        evidence = code_data['evidence']
        
        bits = list(code)
        explanation = []
        
        # Sprecher-Erklärung
        speaker = "Kunde" if bits[0] == "0" else "Verkäufer"
        explanation.append(f"Sprecher: {speaker} (aus Präfix '{symbol[0]}')")
        
        # Phasen-Erklärung
        phase_map = {"00": "Begrüßung", "01": "Bedarf", "10": "Abschluss", "11": "Verabschiedung"}
        phase = phase_map.get(''.join(bits[1:3]), "Unbekannt")
        explanation.append(f"Phase: {phase} (durchschnittliche Position {evidence['avg_position']:.1f})")
        
        # Subphasen-Erklärung
        subphase = "Folge" if bits[3:] == "01" else "Basis"
        if bits[3:] == "01":
            explanation.append(f"Subphase: {subphase} (häufige Wiederholungen)")
        else:
            explanation.append(f"Subphase: {subphase} (seltene Wiederholungen)")
        
        return " | ".join(explanation)


class PatternBasedCoding(CodingStrategy):
    """
    Strategie 2: Kodierung basierend auf wiederkehrenden Mustern
    - Bit 1: Sprecher (K/V)
    - Bits 2-3: Phase aus Nachbarschaftsbeziehungen
    - Bits 4-5: Subphase aus Pattern-Position
    """
    
    def __init__(self):
        super().__init__("Musterbasierte Kodierung")
    
    def derive(self, chains):
        """Leitet Kodierung aus Mustern ab"""
        coding = {}
        
        # Finde häufige Patterns
        patterns = self.find_patterns(chains)
        
        # Analysiere Nachbarschaftsbeziehungen
        neighbors = defaultdict(Counter)
        for chain in chains:
            for i, sym in enumerate(chain):
                if i > 0:
                    neighbors[sym][chain[i-1]] += 1
                if i < len(chain)-1:
                    neighbors[sym][chain[i+1]] += 1
        
        for symbol in neighbors:
            # Bit 1: Sprecher
            speaker = "0" if symbol.startswith('K') else "1"
            
            # Bits 2-3: Phase aus typischen Nachbarn
            common_neighbors = neighbors[symbol].most_common(3)
            if any(n[0].endswith('G') for n in common_neighbors):
                phase = "00"  # Begrüßung
            elif any(n[0].endswith('d') for n in common_neighbors):
                phase = "01"  # Bedarf
            elif any(n[0].endswith('E') for n in common_neighbors):
                phase = "10"  # Abschluss (Beratung)
            elif any(n[0].endswith('V') for n in common_neighbors):
                phase = "11"  # Verabschiedung
            else:
                phase = "01"  # Default: Bedarf
            
            # Bits 4-5: Subphase aus Pattern-Teilnahme
            in_patterns = any(symbol in p for p in patterns)
            subphase = "01" if in_patterns else "00"
            
            code = f"{speaker}{phase}{subphase}"
            coding[symbol] = {
                'code': code,
                'evidence': {
                    'common_neighbors': [(n, c) for n, c in common_neighbors],
                    'in_patterns': in_patterns,
                    'patterns_found': len(patterns)
                }
            }
        
        self.confidence = 0.75
        return coding
    
    def find_patterns(self, chains, min_length=2, min_occurrences=2):
        """Findet wiederkehrende Pattern in den Ketten"""
        patterns = []
        pattern_counter = Counter()
        
        for chain in chains:
            for length in range(min_length, min(5, len(chain))):
                for i in range(len(chain) - length + 1):
                    pattern = tuple(chain[i:i+length])
                    pattern_counter[pattern] += 1
        
        return [p for p, count in pattern_counter.items() if count >= min_occurrences]
    
    def explain(self, symbol, code_data):
        """Erklärt die musterbasierte Kodierung"""
        code = code_data['code']
        evidence = code_data['evidence']
        
        bits = list(code)
        explanation = []
        
        explanation.append(f"Sprecher: {'Kunde' if bits[0]=='0' else 'Verkäufer'}")
        
        phase_map = {"00": "Begrüßung", "01": "Bedarf", "10": "Abschluss", "11": "Verabschiedung"}
        phase = phase_map.get(''.join(bits[1:3]), "Unbekannt")
        
        neighbor_info = ", ".join([f"{n}({c}x)" for n, c in evidence['common_neighbors'][:2]])
        explanation.append(f"Phase: {phase} (häufige Nachbarn: {neighbor_info})")
        
        if evidence['in_patterns']:
            explanation.append(f"Subphase: Folge (Teil von {evidence['patterns_found']} wiederkehrenden Mustern)")
        else:
            explanation.append("Subphase: Basis (kein Teil wiederkehrender Muster)")
        
        return " | ".join(explanation)


class StatisticalBasedCoding(CodingStrategy):
    """
    Strategie 3: Kodierung basierend auf statistischen Verteilungen
    - Verwendet Korrelationsanalysen und Häufigkeitsverteilungen
    """
    
    def __init__(self):
        super().__init__("Statistisch basierte Kodierung")
    
    def derive(self, chains):
        """Leitet Kodierung aus Statistiken ab"""
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
        
        # Berechne Übergangsmatrix
        transitions = defaultdict(Counter)
        for chain in chains:
            for i in range(len(chain)-1):
                transitions[chain[i]][chain[i+1]] += 1
        
        for symbol in frequencies:
            # Bit 1: Sprecher
            speaker = "0" if symbol.startswith('K') else "1"
            
            # Bits 2-3: Phase basierend auf Position und Häufigkeit
            rel_freq = frequencies[symbol] / total_symbols
            first_prob = first_positions.get(symbol, 0) / len(chains)
            last_prob = last_positions.get(symbol, 0) / len(chains)
            
            if first_prob > 0.5:
                phase = "00"  # Meist am Anfang = Begrüßung
            elif last_prob > 0.5:
                phase = "11"  # Meist am Ende = Verabschiedung
            elif rel_freq > 0.15:
                phase = "01"  # Häufig in der Mitte = Bedarf
            else:
                phase = "10"  # Seltener = Abschluss/Beratung
            
            # Bits 4-5: Subphase basierend auf Transitionsvielfalt
            transition_count = len(transitions[symbol])
            if transition_count > 3:
                subphase = "01"  # Viele verschiedene Folge-Symbole = Folge
            else:
                subphase = "00"  # Wenige Folge-Symbole = Basis
            
            code = f"{speaker}{phase}{subphase}"
            coding[symbol] = {
                'code': code,
                'evidence': {
                    'frequency': rel_freq,
                    'first_prob': first_prob,
                    'last_prob': last_prob,
                    'transition_count': transition_count
                }
            }
        
        self.confidence = 0.8
        return coding
    
    def explain(self, symbol, code_data):
        """Erklärt die statistisch basierte Kodierung"""
        code = code_data['code']
        evidence = code_data['evidence']
        
        bits = list(code)
        explanation = []
        
        explanation.append(f"Sprecher: {'Kunde' if bits[0]=='0' else 'Verkäufer'}")
        
        phase_map = {"00": "Begrüßung", "01": "Bedarf", "10": "Abschluss", "11": "Verabschiedung"}
        phase = phase_map.get(''.join(bits[1:3]), "Unbekannt")
        
        phase_reason = []
        if evidence['first_prob'] > 0.5:
            phase_reason.append(f"beginnt oft ({evidence['first_prob']:.0%})")
        if evidence['last_prob'] > 0.5:
            phase_reason.append(f"endet oft ({evidence['last_prob']:.0%})")
        if evidence['frequency'] > 0.15:
            phase_reason.append(f"häufig ({evidence['frequency']:.0%})")
        
        explanation.append(f"Phase: {phase} (" + ", ".join(phase_reason) + ")")
        
        if evidence['transition_count'] > 3:
            explanation.append(f"Subphase: Folge ({evidence['transition_count']} verschiedene Folgesymbole)")
        else:
            explanation.append(f"Subphase: Basis ({evidence['transition_count']} Folgesymbole)")
        
        return " | ".join(explanation)


# ============================================================================
# KONFIDENZMETRIKEN
# ============================================================================

class ConfidenceMetrics:
    """
    Berechnet Konfidenzwerte für alle abgeleiteten Strukturen
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_rule_confidence(self, rule, observations):
        """
        Berechnet Konfidenz für eine Regel basierend auf:
        - Anzahl der Beobachtungen
        - Konsistenz der Beobachtungen
        - Stabilität über verschiedene Ketten
        """
        if not observations:
            return 0.0
        
        n_observations = len(observations)
        n_unique = len(set(observations))
        
        # Je mehr Beobachtungen, desto höher die Konfidenz (logarithmisch)
        confidence = min(1.0, np.log10(n_observations + 1) / 2)
        
        # Je konsistenter, desto höher die Konfidenz
        if n_unique == 1:
            consistency = 1.0
        else:
            # Berechne Entropie der Verteilung
            counts = Counter(observations)
            probs = [c/n_observations for c in counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs)
            max_entropy = np.log2(len(counts))
            consistency = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        
        confidence = (confidence + consistency) / 2
        
        return round(confidence, 3)
    
    def calculate_coding_confidence(self, coding_results):
        """
        Vergleicht Ergebnisse verschiedener Kodierungsstrategien
        """
        if len(coding_results) < 2:
            return 0.5
        
        # Berechne Übereinstimmung zwischen Strategien
        agreements = []
        symbols = set()
        for result in coding_results:
            symbols.update(result.keys())
        
        for symbol in symbols:
            codes = [r.get(symbol, {}).get('code', None) for r in coding_results]
            codes = [c for c in codes if c is not None]
            if len(codes) > 1:
                # Wie viele stimmen überein?
                most_common = Counter(codes).most_common(1)[0]
                agreement = most_common[1] / len(codes)
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.5
    
    def highlight_uncertain_rules(self, rules, threshold=0.7):
        """Markiert Regeln mit niedriger Konfidenz"""
        uncertain = []
        for rule, confidence in rules.items():
            if confidence < threshold:
                uncertain.append({
                    'rule': rule,
                    'confidence': confidence,
                    'suggestion': self.suggest_review(rule, confidence)
                })
        return uncertain
    
    def suggest_review(self, rule, confidence):
        """Generiert Vorschlag zur manuellen Überprüfung"""
        if confidence < 0.3:
            return "⚠️ Sehr unsicher - Datenbasis prüfen"
        elif confidence < 0.5:
            return "⚠️ Unsicher - Manuelle Validierung empfohlen"
        elif confidence < 0.7:
            return "⚠️ Eingeschränkt sicher - Bei Bedarf prüfen"
        return "✓ Ausreichend sicher"


# ============================================================================
# ABLEITUNGSMANAGER
# ============================================================================

class DerivationManager:
    """
    Verwaltet verschiedene Ableitungsstrategien und deren Ergebnisse
    """
    
    def __init__(self):
        self.strategies = [
            PositionBasedCoding(),
            PatternBasedCoding(),
            StatisticalBasedCoding()
        ]
        self.results = {}
        self.confidence_metrics = ConfidenceMetrics()
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
        self.consensus_coding = self.calculate_consensus()
        
        return self.results
    
    def calculate_consensus(self):
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
                # Mehrheitsentscheidung
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
        """Generiert Erklärung für ein Symbol basierend auf allen Strategien"""
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
            explanation.append(f"\n  {evidence['strategy']}:")
            explanation.append(f"    Code: {evidence['code']}")
            if evidence.get('evidence'):
                for key, value in evidence['evidence'].items():
                    explanation.append(f"    {key}: {value}")
        
        return "\n".join(explanation)


# ============================================================================
# GENERISCHER ENTSCHEIDUNGSAUTOMAT (KORRIGIERT)
# ============================================================================

class GenericDialogueAutomaton:
    """
    Generischer Automat, der Regeln aus Daten lernt
    """
    
    def __init__(self):
        self.states = set()
        self.transitions = {}  # (state, symbol) -> new_state
        self.accepting_states = set()
        self.current_state = None
        self.state_counter = 0
        self.confidence_metrics = {}  # (state, symbol) -> confidence
        self.state_assignments_cache = {}  # Cache für Zustandszuweisungen
    
    def learn_from_chains(self, chains):
        """
        Lernt Automaten-Regeln aus beobachteten Ketten
        """
        if not chains:
            return
        
        # Zustände basierend auf Positionen erstellen
        self.create_states_from_positions(chains)
        
        # Übergänge aus beobachteten Transitionen lernen
        self.learn_transitions(chains)
        
        # Akzeptierende Zustände bestimmen
        self.determine_accepting_states(chains)
        
        # Konfidenzen berechnen (JETZT KORREKT IMPLEMENTIERT)
        self._calculate_confidences(chains)
    
    def create_states_from_positions(self, chains):
        """
        Erstellt Zustände basierend auf Positionen in der Sequenz
        """
        # Zustände basierend auf Phasen (0-4)
        for i in range(5):
            state_name = f"q_phase_{i}"
            self.states.add(state_name)
        
        self.states.add("q_start")
        self.states.add("q_error")
        self.current_state = "q_start"
    
    def assign_states(self, chain):
        """
        Weist jeder Position in einer Kette einen Zustand zu
        """
        cache_key = tuple(chain)
        if cache_key in self.state_assignments_cache:
            return self.state_assignments_cache[cache_key]
        
        states = []
        for i, symbol in enumerate(chain):
            # Einfache positionsbasierte Zustandszuweisung
            # Phase basierend auf Fortschritt in der Sequenz
            progress = i / max(1, len(chain) - 1)
            if progress < 0.2:
                phase = 0  # Beginn
            elif progress < 0.4:
                phase = 1  # Frühe Phase
            elif progress < 0.6:
                phase = 2  # Mittlere Phase
            elif progress < 0.8:
                phase = 3  # Späte Phase
            else:
                phase = 4  # Ende
            
            states.append(f"q_phase_{phase}")
        
        self.state_assignments_cache[cache_key] = states
        return states
    
    def learn_transitions(self, chains):
        """
        Lernt Übergangswahrscheinlichkeiten aus den Daten
        """
        # Zähle Übergänge
        transition_counts = defaultdict(Counter)
        
        for chain in chains:
            states = self.assign_states(chain)
            for i in range(len(chain)-1):
                curr_state = states[i]
                next_state = states[i+1]
                symbol = chain[i+1]  # Symbol, das den Übergang auslöst
                transition_counts[(curr_state, symbol)][next_state] += 1
        
        # Bestimme wahrscheinlichste Übergänge
        self.transitions = {}
        for (state, symbol), targets in transition_counts.items():
            if targets:
                most_common = max(targets.items(), key=lambda x: x[1])
                self.transitions[(state, symbol)] = most_common[0]
    
    def determine_accepting_states(self, chains):
        """
        Bestimmt akzeptierende Zustände (wo Ketten enden)
        """
        end_states = Counter()
        
        for chain in chains:
            if chain:
                states = self.assign_states(chain)
                if states:
                    end_states[states[-1]] += 1
        
        # Zustände, in denen mindestens 20% der Ketten enden
        total = len(chains)
        for state, count in end_states.items():
            if count / total > 0.2:
                self.accepting_states.add(state)
        
        # Falls keine Zustände gefunden, nimm Phase 4 als Standard
        if not self.accepting_states and "q_phase_4" in self.states:
            self.accepting_states.add("q_phase_4")
    
    def _calculate_confidences(self, chains):
        """
        Berechnet Konfidenzwerte für alle gelernten Übergänge
        """
        # Sammle alle Vorkommen für jeden Übergang
        transition_occurrences = defaultdict(list)
        
        for chain in chains:
            states = self.assign_states(chain)
            for i in range(len(chain)-1):
                curr_state = states[i]
                symbol = chain[i+1]
                next_state = states[i+1]
                
                key = (curr_state, symbol)
                transition_occurrences[key].append(next_state == self.transitions.get(key))
        
        # Berechne Konfidenz für jeden Übergang
        self.confidence_metrics = {}
        for (state, symbol), occurrences in transition_occurrences.items():
            if (state, symbol) in self.transitions:
                # Konfidenz = Anteil der korrekten Vorhersagen
                correct = sum(occurrences)
                total = len(occurrences)
                confidence = correct / total if total > 0 else 0
                
                # Zusätzlich: Berücksichtige Anzahl der Beobachtungen
                # Je mehr Beobachtungen, desto höher die Konfidenz
                observation_factor = min(1.0, total / 10)  # Bei 10+ Beobachtungen maximal
                confidence = confidence * 0.7 + observation_factor * 0.3
                
                self.confidence_metrics[(state, symbol)] = round(confidence, 3)
    
    def transition(self, symbol):
        """
        Führt einen Übergang basierend auf dem Symbol durch
        """
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
        """
        Validiert eine ganze Kette
        """
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
                # Finde mögliche gültige Übergänge für Erklärung
                possible_transitions = []
                for (state, sym), next_state in self.transitions.items():
                    if state == self.current_state:
                        possible_transitions.append(f"{sym}→{next_state}")
                
                explanation = f"Kein gültiger Übergang von {self.current_state}"
                if possible_transitions:
                    explanation += f"\nMögliche Übergänge: {', '.join(possible_transitions[:3])}"
                
                first_error = {
                    'position': i + 1,
                    'symbol': symbol,
                    'explanation': explanation
                }
        
        valid = self.current_state in self.accepting_states and first_error is None
        return valid, self.current_state, protocol, first_error
    
    def get_rules_string(self):
        """
        Gibt die gelernten Regeln als String zurück
        """
        lines = []
        lines.append("GELERNTE AUTOMATEN-REGELN:")
        lines.append("=" * 60)
        
        if not self.transitions:
            lines.append("  Keine Regeln gelernt.")
            return "\n".join(lines)
        
        # Gruppiere nach Ausgangszustand
        rules_by_state = defaultdict(list)
        for (state, symbol), next_state in self.transitions.items():
            rules_by_state[state].append((symbol, next_state))
        
        for state in sorted(rules_by_state.keys()):
            lines.append(f"\n{state}:")
            for symbol, next_state in sorted(rules_by_state[state]):
                conf = self.confidence_metrics.get((state, symbol), 0)
                conf_stars = "★" * int(conf * 5) + "☆" * (5 - int(conf * 5))
                lines.append(f"  {symbol} → {next_state}  {conf_stars} ({conf:.0%})")
        
        lines.append(f"\nAkzeptierende Zustände: {', '.join(sorted(self.accepting_states))}")
        
        return "\n".join(lines)

# ============================================================================
# INTERAKTIVE ERKLÄRUNGSKOMPONENTE
# ============================================================================

class InteractiveExplainer:
    """
    Ermöglicht dem Nutzer, nach Gründen für Ableitungen zu fragen
    """
    
    def __init__(self, derivation_manager, automaton):
        self.derivation_manager = derivation_manager
        self.automaton = automaton
        self.question_history = []
    
    def why_this_coding(self, symbol):
        """Warum wurde Symbol genau so kodiert?"""
        if symbol not in self.derivation_manager.consensus_coding:
            return f"Symbol '{symbol}' nicht gefunden"
        
        data = self.derivation_manager.consensus_coding[symbol]
        explanation = []
        
        explanation.append(f"🔍 Erklärung für Kodierung von '{symbol}':")
        explanation.append("=" * 60)
        explanation.append(f"Konsens-Kodierung: {data['code']}")
        explanation.append(f"Übereinstimmung: {data['agreement']:.0%}")
        explanation.append("")
        
        # Zeige, wie jede Strategie entschieden hat
        explanation.append("Einzelne Strategien:")
        for evidence in data['evidence']:
            explanation.append(f"\n  📊 {evidence['strategy']}:")
            explanation.append(f"    Code: {evidence['code']}")
            
            # Detailerklärung der Strategie
            strategy = next((s for s in self.derivation_manager.strategies 
                           if s.name == evidence['strategy']), None)
            if strategy:
                detailed = strategy.explain(symbol, evidence)
                explanation.append(f"    → {detailed}")
        
        self.question_history.append(('coding', symbol))
        return "\n".join(explanation)
    
    def why_this_transition(self, from_state, symbol, to_state):
        """Warum wurde dieser Übergang gelernt?"""
        explanation = []
        explanation.append(f"🔍 Erklärung für Übergang:")
        explanation.append(f"   {from_state} --({symbol})--> {to_state}")
        explanation.append("=" * 60)
        
        # Suche nach Belegen in den Daten
        confidence = self.automaton.confidence_metrics.get((from_state, symbol), 0)
        explanation.append(f"Konfidenz: {confidence:.0%}")
        
        if confidence > 0.7:
            explanation.append("✓ Dieser Übergang wurde häufig und konsistent beobachtet")
        elif confidence > 0.4:
            explanation.append("⚠️ Dieser Übergang wurde mehrfach, aber nicht immer konsistent beobachtet")
        else:
            explanation.append("⚠️ Dieser Übergang wurde selten beobachtet")
        
        self.question_history.append(('transition', f"{from_state}--{symbol}"))
        return "\n".join(explanation)
    
    def what_if(self, symbol, alternative_code):
        """
        Was wäre, wenn man anders kodiert hätte?
        Simuliert die Auswirkungen einer alternativen Kodierung
        """
        explanation = []
        explanation.append(f"🔮 Simulation: '{symbol}' als {alternative_code} kodieren")
        explanation.append("=" * 60)
        
        # Vergleiche mit aktueller Kodierung
        current = self.derivation_manager.consensus_coding.get(symbol, {})
        if current:
            explanation.append(f"Aktuelle Kodierung: {current.get('code', '?')}")
            explanation.append(f"Abweichung: {self.hamming_distance(current.get('code', ''), alternative_code)} Bit(s)")
        
        # Zeige mögliche Konsequenzen
        explanation.append("\nMögliche Auswirkungen:")
        explanation.append("  • Automaten-Regeln müssten neu gelernt werden")
        explanation.append("  • Phasenzuordnung würde sich ändern")
        explanation.append("  • Statistiken würden sich verschieben")
        
        return "\n".join(explanation)
    
    def hamming_distance(self, code1, code2):
        """Berechnet Hamming-Distanz zwischen zwei Codes"""
        if len(code1) != len(code2):
            return -1
        return sum(c1 != c2 for c1, c2 in zip(code1, code2))
    
    def get_history_string(self):
        """Gibt den Frageverlauf aus"""
        lines = ["📋 Frageverlauf:"]
        for i, (qtype, subject) in enumerate(self.question_history[-10:], 1):
            if qtype == 'coding':
                lines.append(f"  {i}. Kodierung von '{subject}'")
            else:
                lines.append(f"  {i}. Übergang {subject}")
        return "\n".join(lines)


# ============================================================================
# PROGRESSIVE LEARNING
# ============================================================================

class ProgressiveLearner:
    """
    Lernt kontinuierlich aus neuen Daten und passt Regeln an
    """
    
    def __init__(self):
        self.knowledge_base = {
            'versions': [],
            'current_version': 0
        }
        self.learning_rate = 0.1
    
    def incorporate_new_data(self, chains, current_coding, current_rules):
        """
        Integriert neue Daten und aktualisiert Regeln
        """
        version = {
            'timestamp': datetime.now().isoformat(),
            'n_chains': len(chains),
            'coding': current_coding.copy() if current_coding else {},
            'rules': current_rules.copy() if current_rules else {},
            'changes': []
        }
        
        # Vergleiche mit vorheriger Version
        if self.knowledge_base['versions']:
            prev = self.knowledge_base['versions'][-1]
            changes = self.detect_changes(prev, version)
            version['changes'] = changes
        
        self.knowledge_base['versions'].append(version)
        self.knowledge_base['current_version'] = len(self.knowledge_base['versions']) - 1
        
        return version
    
    def detect_changes(self, prev, current):
        """Erkennt Änderungen zwischen zwei Versionen"""
        changes = []
        
        # Kodierungs-Änderungen
        prev_coding = prev.get('coding', {})
        curr_coding = current.get('coding', {})
        
        for symbol, code_data in curr_coding.items():
            prev_code = prev_coding.get(symbol, {}).get('code')
            curr_code = code_data.get('code')
            
            if prev_code and curr_code and prev_code != curr_code:
                changes.append({
                    'type': 'coding_change',
                    'symbol': symbol,
                    'old': prev_code,
                    'new': curr_code,
                    'reason': 'Angepasst an neue Daten'
                })
        
        return changes
    
    def show_evolution(self):
        """Zeigt, wie sich Regeln über Zeit verändert haben"""
        if not self.knowledge_base['versions']:
            return "Keine Versionshistorie verfügbar"
        
        lines = []
        lines.append("📈 EVOLUTION DER ABGELEITETEN STRUKTUREN")
        lines.append("=" * 60)
        
        for i, version in enumerate(self.knowledge_base['versions']):
            lines.append(f"\nVersion {i+1} - {version['timestamp']}")
            lines.append(f"  {version['n_chains']} Ketten analysiert")
            
            if version['changes']:
                lines.append("  Änderungen:")
                for change in version['changes']:
                    lines.append(f"    • {change['symbol']}: {change['old']} → {change['new']}")
        
        return "\n".join(lines)


# ============================================================================
# VISUALISIERUNGSKOMPONENTEN
# ============================================================================

class DerivationVisualizer:
    """
    Visualisiert den Ableitungsprozess
    """
    
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
            
            # Extrahiere Codes für die ersten 10 Symbole
            display_symbols = symbols[:10]
            codes = []
            for sym in display_symbols:
                if sym in result.get('coding', {}):
                    code = result['coding'][sym].get('code', '?????')
                    codes.append(code)
                else:
                    codes.append('?????')
            
            # Erstelle Heatmap-ähnliche Darstellung
            code_matrix = [[int(bit) for bit in code] for code in codes]
            
            im = ax.imshow(code_matrix, cmap='Blues', aspect='auto', interpolation='nearest')
            ax.set_xticks(range(5))
            ax.set_xticklabels(['Bit1\nSprecher', 'Bit2\nPhase', 'Bit3\nPhase', 'Bit4\nSub', 'Bit5\nSub'])
            ax.set_yticks(range(len(display_symbols)))
            ax.set_yticklabels(display_symbols)
            ax.set_title(f"{strategy_name} (Konfidenz: {result.get('confidence', 0):.0%})")
            
            # Werte in Zellen schreiben
            for i in range(len(display_symbols)):
                for j in range(5):
                    text = ax.text(j, i, code_matrix[i][j], ha='center', va='center')
        
        plt.tight_layout()
        self.plot_thread.plot(lambda: plt.show())
    
    def plot_confidence_heatmap(self, automaton):
        """Zeigt Konfidenzen der Regeln als Heatmap"""
        if not automaton or not automaton.transitions:
            return
        
        # Sammle alle Zustände und Symbole
        states = sorted(set(s for (s, _) in automaton.transitions.keys()))
        symbols = sorted(set(sym for (_, sym) in automaton.transitions.keys()))
        
        # Erstelle Konfidenz-Matrix
        confidence_matrix = np.zeros((len(states), len(symbols)))
        
        for i, state in enumerate(states):
            for j, symbol in enumerate(symbols):
                confidence_matrix[i, j] = automaton.confidence_metrics.get((state, symbol), 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(confidence_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(symbols)))
        ax.set_xticklabels(symbols, rotation=45)
        ax.set_yticks(range(len(states)))
        ax.set_yticklabels(states)
        ax.set_title("Konfidenz der Automaten-Regeln")
        
        plt.colorbar(im, ax=ax, label='Konfidenz')
        plt.tight_layout()
        self.plot_thread.plot(lambda: plt.show())
    
    def plot_automaton_graph(self, automaton):
        """Visualisiert den Automaten als Graph"""
        if not MODULE_STATUS['graphviz']:
            return
        
        dot = graphviz.Digraph(comment='Gelernter Automat')
        
        # Zustände hinzufügen
        for state in automaton.states:
            if state in automaton.accepting_states:
                dot.node(state, state, shape='doublecircle')
            elif state == 'q_error':
                dot.node(state, state, shape='box', color='red')
            else:
                dot.node(state, state, shape='circle')
        
        # Übergänge hinzufügen
        for (state, symbol), next_state in automaton.transitions.items():
            conf = automaton.confidence_metrics.get((state, symbol), 0)
            label = f"{symbol}\n({conf:.0%})"
            dot.edge(state, next_state, label=label)
        
        self.plot_thread.plot(lambda: dot.render('automaton_graph', view=True))


# ============================================================================
# MULTI-FORMAT EXPORTER
# ============================================================================

class MultiFormatExporter:
    """
    Exportiert Ergebnisse in verschiedene Formate
    """
    
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
                writer.writerow(['Symbol', 'Code', 'Konfidenz'])
                for symbol, info in data['coding'].items():
                    writer.writerow([symbol, info.get('code', ''), info.get('confidence', '')])
        
        return filepath
    
    def to_html(self, data, filename=None):
        """Exportiert als interaktiven HTML-Bericht"""
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        filepath = os.path.join(self.export_path, filename)
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head><title>ARSXAI7 Analysebericht</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #4CAF50; color: white; }")
        html.append("tr:nth-child(even) { background-color: #f2f2f2; }")
        html.append(".confidence-high { color: green; }")
        html.append(".confidence-medium { color: orange; }")
        html.append(".confidence-low { color: red; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        html.append(f"<h1>ARSXAI7 Analysebericht</h1>")
        html.append(f"<p>Generiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        if 'coding' in data:
            html.append("<h2>Kodierungsergebnisse</h2>")
            html.append("<table>")
            html.append("<tr><th>Symbol</th><th>Code</th><th>Konfidenz</th></tr>")
            
            for symbol, info in data['coding'].items():
                conf = info.get('confidence', 0)
                conf_class = "confidence-high" if conf > 0.7 else "confidence-medium" if conf > 0.4 else "confidence-low"
                html.append(f"<tr>")
                html.append(f"<td>{symbol}</td>")
                html.append(f"<td>{info.get('code', '')}</td>")
                html.append(f"<td class='{conf_class}'>{conf:.0%}</td>")
                html.append(f"</tr>")
            
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
        latex.append("\\begin{document}")
        
        latex.append("\\section{ARSXAI7 Analyseergebnisse}")
        
        if 'coding' in data:
            latex.append("\\subsection{Kodierung der Terminalzeichen}")
            latex.append("\\begin{tabular}{lll}")
            latex.append("\\toprule")
            latex.append("Symbol & Code & Konfidenz \\\\")
            latex.append("\\midrule")
            
            for symbol, info in data['coding'].items():
                latex.append(f"{symbol} & {info.get('code', '')} & {info.get('confidence', 0):.0%} \\\\")
            
            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
        
        latex.append("\\end{document}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(latex))
        
        return filepath


# ============================================================================
# GUI - HAUPTFENSTER
# ============================================================================

class ARSXAI7GUI:
    """Haupt-GUI für ARSXAI7"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ARSXAI7 - Algorithmic Recursive Sequence Analysis with Explainable AI")
        self.root.geometry("1600x1000")
        
        # Threading und Updates
        self.plot_thread = PlotThread(root)
        self.update_queue = queue.Queue()
        self.process_updates()
        
        # Datenstrukturen
        self.chains = []
        self.terminals = set()
        self.comments = []
        self.delimiter = tk.StringVar(value=",")
        
        # Analysekomponenten
        self.validator = DataValidator()
        self.derivation_manager = DerivationManager()
        self.automaton = GenericDialogueAutomaton()
        self.explainer = None
        self.learner = ProgressiveLearner()
        self.visualizer = DerivationVisualizer(root, self.plot_thread)
        self.exporter = MultiFormatExporter()
        
        # GUI-Elemente
        self.create_menu()
        self.create_main_panels()
        self.status_var = tk.StringVar(value="Bereit")
        self.create_statusbar()
        
        # Modulstatus anzeigen
        self.show_module_status()
    
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
        analyze_menu.add_command(label="Automaten lernen", command=self.learn_automaton)
        analyze_menu.add_command(label="Validierung durchführen", command=self.run_validation)
        
        # XAI-Menü
        xai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="XAI", menu=xai_menu)
        xai_menu.add_command(label="Erklärung für Symbol", command=self.ask_explanation)
        xai_menu.add_command(label="Regeln erklären", command=self.explain_rules)
        xai_menu.add_command(label="Was-wäre-wenn Simulation", command=self.what_if_dialog)
        
        # Visualisierung-Menü
        vis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualisierung", menu=vis_menu)
        vis_menu.add_command(label="Kodierungsvergleich", command=self.plot_coding_comparison)
        vis_menu.add_command(label="Konfidenz-Heatmap", command=self.plot_confidence)
        vis_menu.add_command(label="Automaten-Graph", command=self.plot_automaton)
        
        # Hilfe-Menü
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Hilfe", menu=help_menu)
        help_menu.add_command(label="Modulstatus", command=self.show_module_status)
        help_menu.add_command(label="Evolution anzeigen", command=self.show_evolution)
        help_menu.add_command(label="Über", command=self.show_about)
    
    def create_main_panels(self):
        """Erstellt die Haupt-Panels"""
        # Haupt-PanedWindow
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
        # Titel
        ttk.Label(parent, text="Eingabe", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=5)
        
        # Trennzeichen-Auswahl
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
        
        # Info-Label
        self.info_var = tk.StringVar(value="Keine Daten geladen")
        ttk.Label(parent, textvariable=self.info_var, foreground="blue").pack(anchor=tk.W, pady=5)
        
        # Validierungswarnungen
        self.warning_text = scrolledtext.ScrolledText(parent, height=5, font=('Courier', 9), 
                                                      foreground="orange")
        self.warning_text.pack(fill=tk.X, pady=5)
    
    def create_output_panel(self, parent):
        """Erstellt das Ausgabe-Panel mit Notebook-Tabs"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Kodierungsergebnisse
        self.tab_coding = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_coding, text="Kodierung")
        self.create_coding_tab()
        
        # Tab 2: Automat
        self.tab_automaton = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_automaton, text="Automat")
        self.create_automaton_tab()
        
        # Tab 3: Erklärungen (XAI)
        self.tab_xai = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_xai, text="XAI - Erklärungen")
        self.create_xai_tab()
        
        # Tab 4: Statistiken
        self.tab_stats = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_stats, text="Statistiken")
        self.create_statistics_tab()
        
        # Tab 5: Evolution
        self.tab_evolution = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_evolution, text="Evolution")
        self.create_evolution_tab()
    
    def create_coding_tab(self):
        """Erstellt den Kodierungs-Tab"""
        # Steuerung
        control = ttk.Frame(self.tab_coding)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Alle Strategien", 
                  command=self.run_all_strategies).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Konsens berechnen", 
                  command=self.show_consensus).pack(side=tk.LEFT, padx=5)
        
        # Text-Ausgabe
        self.text_coding = scrolledtext.ScrolledText(self.tab_coding, font=('Courier', 10))
        self.text_coding.pack(fill=tk.BOTH, expand=True, pady=5)
    
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
        # Eingabe für Fragen
        question_frame = ttk.Frame(self.tab_xai)
        question_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(question_frame, text="Symbol:").pack(side=tk.LEFT)
        self.symbol_entry = ttk.Entry(question_frame, width=10)
        self.symbol_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(question_frame, text="Warum diese Kodierung?", 
                  command=self.ask_explanation).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(question_frame, text="Was-wäre-wenn", 
                  command=self.what_if_dialog).pack(side=tk.LEFT, padx=5)
        
        # Text-Ausgabe für Erklärungen
        self.text_xai = scrolledtext.ScrolledText(self.tab_xai, font=('Courier', 10))
        self.text_xai.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_statistics_tab(self):
        """Erstellt den Statistik-Tab"""
        control = ttk.Frame(self.tab_stats)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Statistiken berechnen", 
                  command=self.calculate_statistics).pack(side=tk.LEFT, padx=5)
        
        self.text_stats = scrolledtext.ScrolledText(self.tab_stats, font=('Courier', 10))
        self.text_stats.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_evolution_tab(self):
        """Erstellt den Evolution-Tab"""
        control = ttk.Frame(self.tab_evolution)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Evolution anzeigen", 
                  command=self.show_evolution).pack(side=tk.LEFT, padx=5)
        
        self.text_evolution = scrolledtext.ScrolledText(self.tab_evolution, font=('Courier', 10))
        self.text_evolution.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_statusbar(self):
        """Erstellt die Statusleiste"""
        status = ttk.Frame(self.root)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(status, length=100, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
    
    def get_actual_delimiter(self):
        """Gibt das tatsächliche Trennzeichen zurück"""
        delim = self.delimiter.get()
        if delim == "custom":
            return self.custom_delimiter.get()
        return delim
    
    def parse_line(self, line):
        """Parst eine einzelne Zeile in Symbole"""
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
        """Parst die Eingabe und extrahiert Ketten und Kommentare"""
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
            # Extrahiere Terminal-Menge
            self.terminals = set()
            for chain in self.chains:
                for symbol in chain:
                    self.terminals.add(symbol)
            
            self.info_var.set(f"{len(self.chains)} Ketten, {len(self.terminals)} Terminale")
            self.status_var.set(f"{len(self.chains)} Ketten geladen")
            
            # Validierung durchführen
            self.run_validation()
            
            # Automatische Analyse starten
            self.run_all_strategies()
        else:
            messagebox.showwarning("Warnung", "Keine gültigen Ketten gefunden!")
    
    def run_validation(self):
        """Führt die Datenvalidierung durch"""
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
            self.warning_text.insert(tk.END, "✓ Keine Validierungsprobleme gefunden")
    
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
                    
                    # Konsens anzeigen
                    self.show_consensus()
                    
                    # Erklärer initialisieren
                    self.explainer = InteractiveExplainer(self.derivation_manager, self.automaton)
                    
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
    
    def learn_automaton(self):
        """Lernt Automaten-Regeln aus den Daten"""
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        self.status_var.set("Lerne Automaten-Regeln...")
        
        try:
            self.automaton.learn_from_chains(self.chains)
            
            self.text_automaton.delete("1.0", tk.END)
            self.text_automaton.insert(tk.END, self.automaton.get_rules_string())
            
            # Progressive Learning
            version = self.learner.incorporate_new_data(
                self.chains, 
                self.derivation_manager.consensus_coding,
                self.automaton.transitions
            )
            
            self.status_var.set(f"Automaten gelernt (Version {self.learner.knowledge_base['current_version']+1})")
        except Exception as e:
            messagebox.showerror("Fehler", f"Automaten-Lernen fehlgeschlagen:\n{str(e)}")
    
    def validate_chain(self):
        """Validiert eine ausgewählte Kette"""
        if not self.chains or not self.automaton.transitions:
            messagebox.showerror("Fehler", "Keine Daten oder kein Automat!")
            return
        
        # Einfachen Dialog für Kettenauswahl
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
        
        explanation = self.explainer.why_this_coding(symbol)
        
        self.text_xai.delete("1.0", tk.END)
        self.text_xai.insert(tk.END, explanation)
        self.text_xai.insert(tk.END, "\n\n" + self.explainer.get_history_string())
    
    def what_if_dialog(self):
        """Öffnet Dialog für Was-wäre-wenn Simulation"""
        if not self.explainer:
            messagebox.showerror("Fehler", "Keine Analyse vorhanden!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Was-wäre-wenn Simulation")
        dialog.geometry("400x200")
        
        ttk.Label(dialog, text="Symbol:").grid(row=0, column=0, padx=5, pady=5)
        symbol_entry = ttk.Entry(dialog)
        symbol_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Alternativer Code (5 Bit):").grid(row=1, column=0, padx=5, pady=5)
        code_entry = ttk.Entry(dialog)
        code_entry.grid(row=1, column=1, padx=5, pady=5)
        
        def simulate():
            symbol = symbol_entry.get().strip()
            code = code_entry.get().strip()
            
            if symbol and code:
                explanation = self.explainer.what_if(symbol, code)
                
                self.text_xai.delete("1.0", tk.END)
                self.text_xai.insert(tk.END, explanation)
                dialog.destroy()
        
        ttk.Button(dialog, text="Simulieren", command=simulate).grid(row=2, column=0, columnspan=2, pady=20)
    
    def explain_rules(self):
        """Erklärt die gelernten Automaten-Regeln"""
        if not self.automaton or not self.automaton.transitions:
            messagebox.showerror("Fehler", "Kein Automat vorhanden!")
            return
        
        explanation = []
        explanation.append("ERKLÄRUNG DER AUTOMATEN-REGELN")
        explanation.append("=" * 60)
        
        for (state, symbol), next_state in self.automaton.transitions.items():
            conf = self.automaton.confidence_metrics.get((state, symbol), 0)
            explanation.append(f"\nRegel: {state} --({symbol})--> {next_state}")
            explanation.append(f"  Konfidenz: {conf:.0%}")
            
            if conf > 0.7:
                explanation.append("  ✓ Diese Regel wurde in vielen Ketten konsistent beobachtet")
            elif conf > 0.4:
                explanation.append("  ⚠️ Diese Regel wurde mehrfach, aber nicht immer konsistent beobachtet")
            else:
                explanation.append("  ❌ Diese Regel basiert auf wenigen Beobachtungen")
        
        self.text_xai.delete("1.0", tk.END)
        self.text_xai.insert(tk.END, "\n".join(explanation))
    
    def plot_coding_comparison(self):
        """Visualisiert den Kodierungsvergleich"""
        if not self.derivation_manager.results:
            messagebox.showerror("Fehler", "Keine Kodierungsergebnisse vorhanden!")
            return
        
        symbols = list(self.terminals)[:10]  # Erste 10 Symbole
        self.visualizer.plot_coding_comparison(self.derivation_manager.results, symbols)
    
    def plot_confidence(self):
        """Visualisiert Konfidenzen"""
        if not self.automaton or not self.automaton.transitions:
            messagebox.showerror("Fehler", "Kein Automat vorhanden!")
            return
        
        self.visualizer.plot_confidence_heatmap(self.automaton)
    
    def plot_automaton(self):
        """Visualisiert den Automaten"""
        if not self.automaton or not self.automaton.transitions:
            messagebox.showerror("Fehler", "Kein Automat vorhanden!")
            return
        
        self.visualizer.plot_automaton_graph(self.automaton)
    
    def calculate_statistics(self):
        """Berechnet und zeigt Statistiken an"""
        if not self.chains:
            return
        
        stats = []
        stats.append("STATISTISCHE KENNZAHLEN")
        stats.append("=" * 60)
        
        # Grundstatistiken
        chain_lengths = [len(chain) for chain in self.chains]
        stats.append(f"\nAnzahl Ketten: {len(self.chains)}")
        stats.append(f"Anzahl Terminale: {len(self.terminals)}")
        stats.append(f"Durchschnittliche Länge: {np.mean(chain_lengths):.1f}")
        stats.append(f"Minimale Länge: {min(chain_lengths)}")
        stats.append(f"Maximale Länge: {max(chain_lengths)}")
        
        # Häufigste Symbole
        symbol_counts = Counter()
        for chain in self.chains:
            symbol_counts.update(chain)
        
        stats.append("\nHäufigste Symbole:")
        for sym, count in symbol_counts.most_common(10):
            stats.append(f"  {sym}: {count}x")
        
        # Kommentare
        if self.comments:
            stats.append(f"\nKommentare: {len(self.comments)}")
        
        self.text_stats.delete("1.0", tk.END)
        self.text_stats.insert(tk.END, "\n".join(stats))
    
    def show_evolution(self):
        """Zeigt die Evolution der gelernten Strukturen"""
        evolution = self.learner.show_evolution()
        
        self.text_evolution.delete("1.0", tk.END)
        self.text_evolution.insert(tk.END, evolution)
    
    def show_export_dialog(self):
        """Zeigt Dialog für Export-Optionen"""
        if not self.derivation_manager.consensus_coding:
            messagebox.showerror("Fehler", "Keine Daten zum Exportieren!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Exportieren")
        dialog.geometry("300x250")
        
        export_data = {
            'coding': self.derivation_manager.consensus_coding,
            'terminals': list(self.terminals),
            'chains': self.chains,
            'comments': self.comments,
            'timestamp': datetime.now().isoformat()
        }
        
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
        """Zeigt den Status der optionalen Module"""
        status_text = "MODULSTATUS:\n"
        status_text += "=" * 40 + "\n"
        for module, available in MODULE_STATUS.items():
            status = "✓ verfügbar" if available else "✗ nicht verfügbar"
            status_text += f"{module:15s}: {status}\n"
        
        messagebox.showinfo("Modulstatus", status_text)
    
    def show_about(self):
        """Zeigt Über-Informationen"""
        about = """ARSXAI7 - Algorithmic Recursive Sequence Analysis with Explainable AI

Version 7.0 (Vollständige XAI-Integration)

Kernfunktionen:
• Universelle Analyse beliebiger Terminalzeichenketten
• Mehrere Strategien zur automatischen Strukturableitung
• Konfidenzmetriken für alle Ableitungen
• Interaktive Erklärungskomponente ("Warum?")
• Progressive Learning aus neuen Daten
• Umfangreiche Visualisierungen
• Export in verschiedene Formate

© 2024 - Explainable AI Research"""
        
        messagebox.showinfo("Über ARSXAI7", about)


# ============================================================================
# HAUPTFUNKTION
# ============================================================================

def main():
    """Hauptfunktion"""
    root = tk.Tk()
    app = ARSXAI7GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
