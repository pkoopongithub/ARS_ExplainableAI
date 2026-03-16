"""
ARSXAI10.py - Algorithmic Recursive Sequence Analysis mit Explainable AI
========================================================================
ERWEITERUNG von ARSXAI9.py um:
- Depth-Bounded PCFG (Tiefenbeschränkung)
- MDL-Optimierung (Kompression als Gütekriterium)
- PrefixSpan für große Korpora (optional)
- SemInfo-Maximierung für semantische Namen (optional)
- Automatische Paketinstallation

Version: 10.0 (Evolutionäre Erweiterung mit automatischer PrefixSpan-Installation)
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

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="hmmlearn")
warnings.filterwarnings("ignore", message="MultinomialHMM has undergone major changes")
logging.getLogger('hmmlearn').setLevel(logging.ERROR)

# ============================================================================
# PAKETVERWALTUNG (erweitert um prefixspan)
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

OPTIONAL_PACKAGES = [
    'prefixspan'  # Für große Datenmengen
]

def check_and_install_packages():
    """Prüft und installiert fehlende Python-Pakete"""
    print("=" * 70)
    print("ARSXAI10 - PAKETPRÜFUNG")
    print("=" * 70)
    
    # Pflichtpakete
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
        print("\nInstalliere fehlende Pflichtpakete...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"  ✓ {package} erfolgreich installiert")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Fehler bei Installation von {package}: {e}")
    
    # Optionale Pakete (nur Hinweis, keine automatische Installation)
    print("\n" + "-" * 70)
    print("OPTIONALE PAKETE (für erweiterte Funktionen):")
    print("-" * 70)
    
    for package in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"✓ {package} verfügbar")
        except ImportError:
            print(f"ℹ️  {package} nicht installiert (optional)")
            if package == 'prefixspan':
                print("   Für große Datenmengen (>1000 Ketten) empfohlen.")
                print("   Installation: pip install prefixspan")
    
    print("\n" + "=" * 70 + "\n")

# Pakete prüfen
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
    
    if not GRAPHVIZ_AVAILABLE:
        print("\n⚠️  Graphviz nicht gefunden. Installieren für Automaten-Visualisierung:")
        print("   Windows: https://graphviz.org/download/")
        print("   Linux: sudo apt-get install graphviz")
        print("   Mac: brew install graphviz")
    
    return GRAPHVIZ_AVAILABLE

GRAPHVIZ_AVAILABLE = setup_graphviz()

# ============================================================================
# IMPORTS AUS ARSXAI9 (wiederverwendung)
# ============================================================================

# Hinweis: Für die Ausführung muss ARSXAI9.py im gleichen Verzeichnis sein!
try:
    from ARSXAI9 import (
        PlotThread, XAIModel, ARS20, GrammarInducer, 
        MethodologicalReflection, ARSHiddenMarkovModel, ARSCRFModel,
        ARSPetriNet, ChainGenerator, XAIModelManager, DataValidator,
        DerivationVisualizer, NaturalLanguageExplainer, MultiFormatExporter,
        ARSXAI9GUI, MODULE_STATUS, GRAPHVIZ_AVAILABLE as BASE_GRAPHVIZ
    )
    print("✓ ARSXAI9.py erfolgreich importiert")
except ImportError as e:
    print(f"✗ Fehler beim Import von ARSXAI9.py: {e}")
    print("   Stellen Sie sicher, dass ARSXAI9.py im gleichen Verzeichnis ist!")
    sys.exit(1)

# ============================================================================
# GLOBALE VARIABLEN FÜR OPTIONALE PAKETE
# ============================================================================

# Diese müssen VOR ihrer ersten Verwendung deklariert werden!
PREFIXSPAN_AVAILABLE = False
SEMINFO_AVAILABLE = False

# ============================================================================
# NEUE IMPORTS FÜR ARSXAI10 (mit erweiterter Installation)
# ============================================================================

# PrefixSpan für große Daten (optional, mit Installationshilfe)
try:
    from prefixspan import PrefixSpan
    PREFIXSPAN_AVAILABLE = True
    print("✓ PrefixSpan verfügbar - für große Datenmengen optimiert")
except ImportError:
    PREFIXSPAN_AVAILABLE = False
    print("ℹ️  PrefixSpan nicht installiert (optional) - nutze Standard-Mustererkennung")
    print("   Für große Datenmengen (>1000 Ketten) Installation empfohlen:")
    print("   pip install prefixspan")
    
    # Biete automatische Installation an (optional)
    try:
        response = input("   PrefixSpan jetzt installieren? (j/n): ").lower()
        if response == 'j' or response == 'ja' or response == 'y' or response == 'yes':
            print("   Installiere PrefixSpan...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "prefixspan"])
            from prefixspan import PrefixSpan
            PREFIXSPAN_AVAILABLE = True
            print("   ✓ PrefixSpan erfolgreich installiert!")
    except:
        print("   Installation übersprungen.")

# Sentence-Transformers für semantische Namen (optional)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMINFO_AVAILABLE = True
    print("✓ Sentence-Transformers verfügbar - semantische Namen aktivierbar")
except ImportError:
    SEMINFO_AVAILABLE = False
    print("ℹ️  Sentence-Transformers nicht installiert (optional)")
    print("   Für semantische Namen: pip install sentence-transformers")

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
# MDL-OPTIMIZER (Hilfsklasse für Minimum Description Length)
# ============================================================================

class MDLOptimizer:
    """
    Implementiert Minimum Description Length Prinzip.
    Bewertet Grammatiken nach Kompressionsrate.
    """
    
    def __init__(self):
        self.compression_history = []
    
    def calculate_compression_ratio(self, original_chains, grammar):
        """
        Berechnet Kompressionsrate: 1 - (komprimierte_Länge / originale_Länge)
        
        Args:
            original_chains: Liste der ursprünglichen Ketten
            grammar: GrammarInducer-Objekt mit Regeln
        
        Returns:
            float: Kompressionsrate (0-1), höher ist besser
        """
        if not grammar.rules:
            return 0.0
        
        # Originale Länge (Anzahl der Terminalsymbole)
        original_length = sum(len(chain) for chain in original_chains)
        
        # Komprimierte Länge (Nonterminale zählen als 1)
        compressed_length = 0
        for chain in original_chains:
            compressed = self._compress_chain(chain, grammar)
            compressed_length += len(compressed)
        
        ratio = 1 - (compressed_length / original_length) if original_length > 0 else 0
        self.compression_history.append(ratio)
        return round(ratio, 3)
    
    def _compress_chain(self, chain, grammar, max_iter=10):
        """Wendet Grammatikregeln iterativ an, um Kette zu komprimieren"""
        current = list(chain)
        for _ in range(max_iter):
            changed = False
            # Suche nach anwendbaren Regeln (rückwärts, längste zuerst)
            for nt, productions in sorted(grammar.rules.items(), 
                                         key=lambda x: -len(x[1][0][0] if x[1] else 0)):
                for prod, _ in productions:
                    prod_len = len(prod)
                    i = 0
                    while i <= len(current) - prod_len:
                        if current[i:i+prod_len] == prod:
                            current[i:i+prod_len] = [nt]
                            changed = True
                            break
                        i += 1
            if not changed:
                break
        return current
    
    def compare_grammars(self, grammar1, grammar2, chains):
        """
        Vergleicht zwei Grammatiken nach MDL-Prinzip.
        
        Returns:
            dict: Vergleichsergebnisse mit Scores
        """
        ratio1 = self.calculate_compression_ratio(chains, grammar1)
        ratio2 = self.calculate_compression_ratio(chains, grammar2)
        
        # Grammatikkomplexität (Anzahl der Regeln)
        complexity1 = len(grammar1.rules) if hasattr(grammar1, 'rules') else 0
        complexity2 = len(grammar2.rules) if hasattr(grammar2, 'rules') else 0
        
        # MDL-Score: Kompression - Komplexitätsstrafe
        mdl1 = ratio1 - (complexity1 * 0.01)  # Einfache Strafe
        mdl2 = ratio2 - (complexity2 * 0.01)
        
        return {
            'grammar1': {
                'compression_ratio': ratio1,
                'complexity': complexity1,
                'mdl_score': mdl1
            },
            'grammar2': {
                'compression_ratio': ratio2,
                'complexity': complexity2,
                'mdl_score': mdl2
            },
            'better': 'grammar1' if mdl1 > mdl2 else 'grammar2' if mdl2 > mdl1 else 'equal'
        }
    
    def optimal_cutoff(self, compression_gains):
        """
        Findet natürliche Grenze für Iterationen (Elbow-Methode).
        
        Args:
            compression_gains: Liste der Kompressionsgewinne pro Iteration
        
        Returns:
            int: Optimale Anzahl von Iterationen
        """
        if len(compression_gains) < 3:
            return len(compression_gains)
        
        # Einfache Elbow-Erkennung: Wo ist der Knick?
        gains = np.array(compression_gains)
        if len(gains) < 2:
            return len(gains)
        
        # Berechne Differenzen
        diffs = np.diff(gains)
        if len(diffs) < 1:
            return len(gains)
        
        # Finde ersten Punkt mit stark abnehmendem Gewinn
        threshold = np.mean(diffs) * 0.5
        for i, diff in enumerate(diffs):
            if diff < threshold:
                return i + 1
        
        return len(gains)
    
    def get_statistics_string(self):
        """Gibt Statistik der Kompression als String zurück"""
        if not self.compression_history:
            return "Keine Kompressionsdaten vorhanden."
        
        lines = []
        lines.append("📊 **MDL-Kompressionsstatistik**")
        lines.append("=" * 50)
        
        for i, ratio in enumerate(self.compression_history):
            lines.append(f"Iteration {i+1}: {ratio:.1%} Kompression")
        
        optimal = self.optimal_cutoff(self.compression_history)
        lines.append(f"\n✅ Optimale Iterationen: {optimal}")
        
        return "\n".join(lines)


# ============================================================================
# SEMINFO-MAXIMIZER (optional, für semantische Namen)
# ============================================================================

class SemInfoMaximizer:
    """
    Maximiert semantische Information der Nonterminale.
    Benötigt sentence-transformers.
    """
    
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.model = None
        self.embeddings = {}
        self.semantic_cache = {}
        
        if SEMINFO_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"✓ SemInfo-Modell geladen: {model_name}")
            except Exception as e:
                print(f"✗ SemInfo-Modell konnte nicht geladen werden: {e}")
    
    def compute_embeddings(self, symbols):
        """
        Erstellt Embeddings für eine Liste von Symbolen.
        
        Args:
            symbols: Liste von Strings (Symbole)
        
        Returns:
            dict: Symbol -> Embedding-Vektor
        """
        if self.model is None:
            return {}
        
        # Filtere Symbole, die noch nicht im Cache sind
        to_compute = [s for s in symbols if s not in self.embeddings]
        
        if to_compute:
            try:
                # Generiere Embeddings für alle neuen Symbole
                new_embeddings = self.model.encode(to_compute)
                for sym, emb in zip(to_compute, new_embeddings):
                    self.embeddings[sym] = emb
            except Exception as e:
                print(f"Fehler bei Embedding-Berechnung: {e}")
        
        return self.embeddings
    
    def semantic_coherence(self, sequence):
        """
        Misst semantische Kohärenz einer Sequenz.
        
        Args:
            sequence: Liste von Symbolen
        
        Returns:
            float: Kohärenz-Score (0-1), höher = zusammenhängender
        """
        if self.model is None or len(sequence) < 2:
            return 0.5
        
        # Stelle sicher, dass Embeddings vorhanden sind
        self.compute_embeddings(sequence)
        
        # Berechne paarweise Ähnlichkeiten
        similarities = []
        for i in range(len(sequence) - 1):
            sym1 = sequence[i]
            sym2 = sequence[i + 1]
            
            if sym1 in self.embeddings and sym2 in self.embeddings:
                emb1 = self.embeddings[sym1]
                emb2 = self.embeddings[sym2]
                
                # Kosinus-Ähnlichkeit
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                similarities.append(max(0, sim))  # Auf 0-1 beschränken
        
        if not similarities:
            return 0.5
        
        return float(np.mean(similarities))
    
    def suggest_name(self, sequence):
        """
        Generiert einen semantischen Namen für eine Sequenz.
        
        Args:
            sequence: Liste von Symbolen
        
        Returns:
            str: Vorgeschlagener Name oder None
        """
        if self.model is None or len(sequence) < 2:
            return None
        
        # Prüfe Cache
        cache_key = tuple(sequence)
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
        
        # Berechne durchschnittliches Embedding
        self.compute_embeddings(sequence)
        valid_embs = [self.embeddings[s] for s in sequence if s in self.embeddings]
        
        if not valid_embs:
            return None
        
        mean_emb = np.mean(valid_embs, axis=0)
        
        # Hier könnte man eine Suche nach ähnlichen Konzepten implementieren
        # Für jetzt: Generische Namen basierend auf Kohärenz
        coherence = self.semantic_coherence(sequence)
        
        if coherence > 0.8:
            name = "KOHÄRENT"
        elif coherence > 0.6:
            name = "ZUSAMMENHÄNGEND"
        elif coherence > 0.4:
            name = "GEMISCHT"
        else:
            name = "DIVERGENT"
        
        result = f"{name}_{len(sequence)}"
        self.semantic_cache[cache_key] = result
        return result
    
    def get_status_string(self):
        """Gibt Status des SemInfo-Maximizers zurück"""
        lines = []
        lines.append("🧠 **SemInfo-Maximizer Status**")
        lines.append("=" * 50)
        lines.append(f"Modell: {self.model_name}")
        lines.append(f"Verfügbar: {'✓' if self.model else '✗'}")
        lines.append(f"Gecachte Embeddings: {len(self.embeddings)}")
        lines.append(f"Semantische Namen im Cache: {len(self.semantic_cache)}")
        return "\n".join(lines)


# ============================================================================
# DEPTH-BOUNDED GRAMMAR INDUCER (KERN DER ERWEITERUNG)
# ============================================================================

class DepthBoundedGrammarInducer(GrammarInducer):
    """
    Erweitert GrammarInducer um Tiefenbeschränkung und MDL-Optimierung.
    
    Attribute:
        max_depth: Maximale Hierarchietiefe (default 5)
        use_mdl: MDL-Optimierung aktivieren (default True)
        use_prefixspan: PrefixSpan für große Daten (default False)
        depth_map: nonterminal -> Tiefe
        mdl_optimizer: MDLOptimizer-Instanz
        seminfo: SemInfoMaximizer-Instanz (optional)
    """
    
    def __init__(self, max_depth=5, use_mdl=True, use_prefixspan=False, use_seminfo=False):
        super().__init__()
        self.max_depth = max_depth
        self.use_mdl = use_mdl
        self.use_prefixspan = use_prefixspan and PREFIXSPAN_AVAILABLE
        self.use_seminfo = use_seminfo and SEMINFO_AVAILABLE
        
        self.depth_map = {}  # nonterminal -> Tiefe
        self.mdl_scores = {}  # für Optimierung
        self.mdl_optimizer = MDLOptimizer()
        self.seminfo = SemInfoMaximizer() if use_seminfo and SEMINFO_AVAILABLE else None
        
        self.compression_gains = []  # Für Cutoff-Erkennung
    
    def train(self, chains, max_iterations=20):
        """Überschriebene Train-Methode mit Tiefenbeschränkung"""
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
        self.depth_map = {}
        self.mdl_scores = {}
        self.compression_gains = []
        
        while iteration < max_iterations:
            best_seq = self._find_best_repetition(current_chains)
            
            if best_seq is None:
                break
            
            # Prüfe Tiefenbeschränkung
            depth = self._estimate_depth(best_seq)
            if depth > self.max_depth:
                # Überspringe zu tiefe Muster
                self._mark_as_skipped(best_seq)
                continue
            
            new_nonterminal = self._generate_nonterminal_name(best_seq, depth)
            base_name = new_nonterminal
            while new_nonterminal in self.nonterminals:
                new_nonterminal = f"{base_name}_{rule_counter}"
                rule_counter += 1
            
            rationale = self._generate_rationale(best_seq)
            self.reflection.log_interpretation(best_seq, new_nonterminal, rationale)
            
            self.rules[new_nonterminal] = [(list(best_seq), 1.0)]
            self.nonterminals.add(new_nonterminal)
            self.hierarchy_levels[new_nonterminal] = iteration
            self.depth_map[new_nonterminal] = depth
            
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
                'occurrences': occurrences,
                'depth': depth
            })
            
            # MDL-Gain berechnen
            if self.use_mdl:
                gain = self._mdl_gain(best_seq, occurrences)
                self.mdl_scores[new_nonterminal] = gain
                self.compression_gains.append(gain)
            
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
        
        # Terminale aktualisieren
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
    
    def _find_best_repetition(self, chains):
        """MDL-optimierte Musterauswahl"""
        
        # PrefixSpan für große Daten
        if self.use_prefixspan and len(chains) > 1000:
            return self._find_with_prefixspan(chains)
        
        # Standard-Mustererkennung (von GrammarInducer)
        sequence_counter = Counter()
        
        for chain in chains:
            max_len = min(5, len(chain))  # Begrenze Länge für Performance
            for length in range(2, max_len + 1):
                for i in range(len(chain) - length + 1):
                    seq = tuple(chain[i:i+length])
                    sequence_counter[seq] += 1
        
        repeated = {seq: count for seq, count in sequence_counter.items() if count >= 2}
        if not repeated:
            return None
        
        if self.use_mdl:
            # MDL-basierte Bewertung
            best_score = -float('inf')
            best_seq = None
            
            for seq, count in repeated.items():
                gain = self._mdl_gain(seq, count)
                if gain > best_score:
                    best_score = gain
                    best_seq = seq
            
            return best_seq
        else:
            # Standard-Bewertung (wie GrammarInducer)
            return max(repeated.items(), 
                      key=lambda x: x[1] * len(x[0]) / max(1, len(set(x[0]))))[0]
    
    def _mdl_gain(self, sequence, count):
        """
        Berechnet MDL-Ersparnis (Kompression).
        
        Args:
            sequence: Tuple von Symbolen
            count: Anzahl der Vorkommen
        
        Returns:
            float: Ersparnis (positiv = lohnend)
        """
        # Kosten ohne Kompression: jedes Vorkommen zählt als Länge
        original_cost = len(sequence) * count
        
        # Kosten mit Kompression: 1 (Nonterminal) pro Vorkommen + Definition
        compressed_cost = count + len(sequence)
        
        gain = original_cost - compressed_cost
        return gain / (original_cost + 1)  # Normalisiert
    
    def _find_with_prefixspan(self, chains, min_support=2):
        """Findet Muster mit PrefixSpan (für große Daten)"""
        if not PREFIXSPAN_AVAILABLE:
            return None
        
        try:
            ps = PrefixSpan(chains)
            patterns = ps.frequent(min_support)
            
            # Filtere Muster mit Länge >= 2
            valid_patterns = [(seq, support) for seq, support in patterns if len(seq) >= 2]
            
            if not valid_patterns:
                return None
            
            # Bestes Muster nach MDL oder Länge
            if self.use_mdl:
                best = max(valid_patterns, 
                          key=lambda x: self._mdl_gain(tuple(x[0]), x[1]))
                return tuple(best[0])
            else:
                best = max(valid_patterns, 
                          key=lambda x: len(x[0]) * x[1])
                return tuple(best[0])
                
        except Exception as e:
            print(f"PrefixSpan-Fehler: {e}")
            return None
    
    def _generate_nonterminal_name(self, sequence, depth):
        """Generiert Namen mit Tiefeninformation und optional semantisch"""
        
        first = sequence[0] if sequence else "X"
        last = sequence[-1] if sequence else "X"
        
        # Semantischer Name (optional)
        if self.use_seminfo and self.seminfo:
            semantic = self.seminfo.suggest_name(sequence)
            if semantic:
                return f"{semantic}_d{depth}"
        
        return f"P_{first}_{last}_{len(sequence)}_d{depth}"
    
    def _estimate_depth(self, sequence):
        """
        Schätzt benötigte Tiefe für eine Sequenz.
        Berücksichtigt, ob Symbole bereits Nonterminale sind.
        """
        max_depth = 0
        for sym in sequence:
            if sym in self.depth_map:
                max_depth = max(max_depth, self.depth_map[sym] + 1)
        return max_depth
    
    def _mark_as_skipped(self, sequence):
        """Markiert ein Muster als übersprungen (zu tief)"""
        if not hasattr(self, 'skipped_patterns'):
            self.skipped_patterns = []
        self.skipped_patterns.append({
            'sequence': sequence,
            'depth': self._estimate_depth(sequence)
        })
    
    def get_depth_statistics(self):
        """Gibt Tiefenstatistik aus"""
        lines = []
        lines.append("📊 **TIEFENSTATISTIK**")
        lines.append("=" * 60)
        
        if not self.depth_map:
            lines.append("Keine Tiefeninformationen verfügbar.")
            return "\n".join(lines)
        
        # Verteilung der Tiefen
        depth_counts = Counter(self.depth_map.values())
        lines.append("\nTiefenverteilung:")
        for depth in sorted(depth_counts.keys()):
            count = depth_counts[depth]
            percentage = (count / len(self.depth_map)) * 100
            lines.append(f"  Tiefe {depth}: {count} Nonterminale ({percentage:.1f}%)")
        
        # Nonterminale nach Tiefe
        lines.append("\nNonterminale nach Tiefe:")
        for depth in sorted(set(self.depth_map.values())):
            nts = [nt for nt, d in self.depth_map.items() if d == depth]
            lines.append(f"  Tiefe {depth}: {', '.join(nts[:5])}" + 
                        (f" ... und {len(nts)-5} weitere" if len(nts) > 5 else ""))
        
        # Übersprungene Muster
        if hasattr(self, 'skipped_patterns') and self.skipped_patterns:
            lines.append(f"\n⚠️ Übersprungene Muster (Tiefe > {self.max_depth}):")
            for pattern in self.skipped_patterns[:5]:
                seq_str = ' → '.join(pattern['sequence'])
                lines.append(f"  • {seq_str} (Tiefe {pattern['depth']})")
        
        return "\n".join(lines)
    
    def get_mdl_statistics(self):
        """Gibt MDL-Statistik aus"""
        return self.mdl_optimizer.get_statistics_string()
    
    def compare_with_standard(self, standard_grammar, chains):
        """Vergleicht diese Grammatik mit der Standard-Grammatik"""
        return self.mdl_optimizer.compare_grammars(self, standard_grammar, chains)


# ============================================================================
# ERWEITERTE GUI (erbt von ARSXAI9GUI)
# ============================================================================

class ARSXAI10GUI(ARSXAI9GUI):
    """
    Erweiterte GUI mit neuen Optionen für Depth-Bounded PCFG und MDL.
    Alle bestehenden Tabs und Funktionen bleiben erhalten!
    """
    
    def __init__(self, root):
        # Initialisiere erweiterte Parameter
        self.use_depth_bounded = tk.BooleanVar(value=True)
        self.max_depth = tk.IntVar(value=5)
        self.use_mdl = tk.BooleanVar(value=True)
        self.use_prefixspan = tk.BooleanVar(value=False)
        self.use_seminfo = tk.BooleanVar(value=False)
        
        # Rufe Elternkonstruktor auf
        super().__init__(root)
        
        # Füge erweiterte Tabs und Parameter hinzu
        self._add_advanced_induction_tab()
    
    def _register_models(self):
        """Erweiterte Modell-Registrierung"""
        super()._register_models()  # Behalte alle alten Modelle
        
        # Füge Depth-Bounded Modell hinzu
        depth_model = DepthBoundedGrammarInducer(
            max_depth=self.max_depth.get(),
            use_mdl=self.use_mdl.get(),
            use_prefixspan=self.use_prefixspan.get() and PREFIXSPAN_AVAILABLE,
            use_seminfo=self.use_seminfo.get()
        )
        self.model_manager.register_model('DepthBoundedPCFG', depth_model)
        self.model_manager.activate_model('DepthBoundedPCFG')
    
    def _add_advanced_induction_tab(self):
        """Fügt neuen Tab für erweiterte Induktion hinzu"""
        self.tab_advanced = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_advanced, text="Erweiterte Induktion")
        
        # Parameter-Frame
        param_frame = ttk.LabelFrame(self.tab_advanced, text="Parameter")
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Tiefen-Regler
        depth_frame = ttk.Frame(param_frame)
        depth_frame.pack(fill=tk.X, pady=5)
        ttk.Label(depth_frame, text="Maximale Tiefe:").pack(side=tk.LEFT)
        self.depth_slider = ttk.Scale(depth_frame, from_=1, to=10, 
                                       orient=tk.HORIZONTAL,
                                       variable=self.max_depth)
        self.depth_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        ttk.Label(depth_frame, textvariable=self.max_depth).pack(side=tk.LEFT, padx=5)
        
        # Checkboxen
        ttk.Checkbutton(param_frame, text="MDL-Optimierung", 
                       variable=self.use_mdl).pack(anchor=tk.W, pady=2)
        
        prefix_frame = ttk.Frame(param_frame)
        prefix_frame.pack(fill=tk.X, pady=2)
        self.prefix_check = ttk.Checkbutton(prefix_frame, text="PrefixSpan verwenden",
                                           variable=self.use_prefixspan)
        self.prefix_check.pack(side=tk.LEFT)
        if not PREFIXSPAN_AVAILABLE:
            ttk.Label(prefix_frame, text="(nicht verfügbar - pip install prefixspan)", 
                     foreground="orange").pack(side=tk.LEFT, padx=5)
            self.use_prefixspan.set(False)
            self.prefix_check.config(state='disabled')
        
        seminfo_frame = ttk.Frame(param_frame)
        seminfo_frame.pack(fill=tk.X, pady=2)
        self.seminfo_check = ttk.Checkbutton(seminfo_frame, text="Semantische Namen (SemInfo)",
                                            variable=self.use_seminfo)
        self.seminfo_check.pack(side=tk.LEFT)
        if not SEMINFO_AVAILABLE:
            ttk.Label(seminfo_frame, text="(sentence-transformers fehlt)", 
                     foreground="orange").pack(side=tk.LEFT, padx=5)
            self.use_seminfo.set(False)
            self.seminfo_check.config(state='disabled')
        
        # Aktions-Buttons
        action_frame = ttk.Frame(self.tab_advanced)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(action_frame, text="Tiefenstatistik anzeigen",
                  command=self.show_depth_statistics).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="MDL-Statistik anzeigen",
                  command=self.show_mdl_statistics).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="Mit Standard vergleichen",
                  command=self.compare_with_standard).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="PrefixSpan installieren",
                  command=self.install_prefixspan).pack(side=tk.LEFT, padx=5)
        
        # Ausgabebereich
        output_frame = ttk.LabelFrame(self.tab_advanced, text="Ausgabe")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.text_advanced = scrolledtext.ScrolledText(output_frame, font=('Courier', 10))
        self.text_advanced.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def install_prefixspan(self):
        """Installiert PrefixSpan on-demand"""
        # Hier muss PREFIXSPAN_AVAILABLE als global deklariert werden, 
        # weil wir es ändern wollen!
        global PREFIXSPAN_AVAILABLE
        
        if PREFIXSPAN_AVAILABLE:
            messagebox.showinfo("Info", "PrefixSpan ist bereits installiert!")
            return
        
        result = messagebox.askyesno("Installation", 
                                     "PrefixSpan wird für große Datenmengen empfohlen.\n\n"
                                     "Jetzt installieren?")
        if result:
            try:
                self.text_advanced.insert(tk.END, "Installiere PrefixSpan...\n")
                self.root.update()
                
                subprocess.check_call([sys.executable, "-m", "pip", "install", "prefixspan"])
                
                # Versuche erneut zu importieren
                from prefixspan import PrefixSpan
                PREFIXSPAN_AVAILABLE = True
                
                self.text_advanced.insert(tk.END, "✓ PrefixSpan erfolgreich installiert!\n")
                self.prefix_check.config(state='normal')
                
                messagebox.showinfo("Erfolg", "PrefixSpan wurde erfolgreich installiert!")
                
            except Exception as e:
                self.text_advanced.insert(tk.END, f"✗ Fehler bei Installation: {e}\n")
                messagebox.showerror("Fehler", f"Installation fehlgeschlagen:\n{e}")
    
    def run_grammar_induction(self):
        """Führt Grammatikinduktion durch (mit neuen Optionen)"""
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        self.status_var.set("Induziere Grammatik...")
        self.progress_bar.start()
        
        def run():
            try:
                # Wähle Inducer basierend auf Einstellungen
                if self.use_depth_bounded.get():
                    inducer = DepthBoundedGrammarInducer(
                        max_depth=self.max_depth.get(),
                        use_mdl=self.use_mdl.get(),
                        use_prefixspan=self.use_prefixspan.get() and len(self.chains) > 1000,
                        use_seminfo=self.use_seminfo.get()
                    )
                else:
                    inducer = GrammarInducer()
                
                inducer.train(self.chains)
                self.grammar = inducer
                self.explainer = NaturalLanguageExplainer(self.grammar)
                
                def update():
                    self.show_grammar()
                    self.show_patterns()
                    self.status_var.set(f"Grammatik induziert: {len(self.grammar.nonterminals)} Muster gefunden")
                    self.progress_bar.stop()
                    
                    # Zeige Erfolgsmeldung im Advanced-Tab
                    if hasattr(self, 'text_advanced'):
                        self.text_advanced.delete("1.0", tk.END)
                        self.text_advanced.insert(tk.END, "✅ Grammatikinduktion abgeschlossen.\n\n")
                        if isinstance(inducer, DepthBoundedGrammarInducer):
                            self.text_advanced.insert(tk.END, f"Max. Tiefe: {self.max_depth.get()}\n")
                            self.text_advanced.insert(tk.END, f"MDL-Optimierung: {'aktiv' if self.use_mdl.get() else 'inaktiv'}\n")
                            self.text_advanced.insert(tk.END, f"PrefixSpan: {'aktiv' if self.use_prefixspan.get() and PREFIXSPAN_AVAILABLE else 'inaktiv'}\n")
                            if self.use_seminfo.get() and SEMINFO_AVAILABLE:
                                self.text_advanced.insert(tk.END, inducer.seminfo.get_status_string() + "\n")
                
                self.safe_gui_update(update)
                
            except Exception as e:
                def error():
                    messagebox.showerror("Fehler", f"Grammatikinduktion fehlgeschlagen:\n{str(e)}")
                    self.progress_bar.stop()
                
                self.safe_gui_update(error)
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def show_depth_statistics(self):
        """Zeigt Tiefenstatistik im Advanced-Tab an"""
        if not hasattr(self, 'grammar') or not isinstance(self.grammar, DepthBoundedGrammarInducer):
            messagebox.showerror("Fehler", "Keine Depth-Bounded Grammatik vorhanden!")
            return
        
        stats = self.grammar.get_depth_statistics()
        self.text_advanced.delete("1.0", tk.END)
        self.text_advanced.insert(tk.END, stats)
    
    def show_mdl_statistics(self):
        """Zeigt MDL-Statistik im Advanced-Tab an"""
        if not hasattr(self, 'grammar') or not isinstance(self.grammar, DepthBoundedGrammarInducer):
            messagebox.showerror("Fehler", "Keine Depth-Bounded Grammatik vorhanden!")
            return
        
        stats = self.grammar.get_mdl_statistics()
        self.text_advanced.delete("1.0", tk.END)
        self.text_advanced.insert(tk.END, stats)
    
    def compare_with_standard(self):
        """Vergleicht Depth-Bounded mit Standard-Grammatik"""
        if not hasattr(self, 'grammar') or not isinstance(self.grammar, DepthBoundedGrammarInducer):
            messagebox.showerror("Fehler", "Keine Depth-Bounded Grammatik vorhanden!")
            return
        
        # Erstelle Standard-Grammatik zum Vergleich
        standard = GrammarInducer()
        standard.train(self.chains)
        
        comparison = self.grammar.compare_with_standard(standard, self.chains)
        
        self.text_advanced.delete("1.0", tk.END)
        self.text_advanced.insert(tk.END, "📊 **MDL-VERGLEICH**\n")
        self.text_advanced.insert(tk.END, "=" * 60 + "\n\n")
        
        self.text_advanced.insert(tk.END, "**Depth-Bounded Grammatik:**\n")
        self.text_advanced.insert(tk.END, f"  Kompression: {comparison['grammar1']['compression_ratio']:.1%}\n")
        self.text_advanced.insert(tk.END, f"  Komplexität: {comparison['grammar1']['complexity']} Regeln\n")
        self.text_advanced.insert(tk.END, f"  MDL-Score: {comparison['grammar1']['mdl_score']:.3f}\n\n")
        
        self.text_advanced.insert(tk.END, "**Standard Grammatik:**\n")
        self.text_advanced.insert(tk.END, f"  Kompression: {comparison['grammar2']['compression_ratio']:.1%}\n")
        self.text_advanced.insert(tk.END, f"  Komplexität: {comparison['grammar2']['complexity']} Regeln\n")
        self.text_advanced.insert(tk.END, f"  MDL-Score: {comparison['grammar2']['mdl_score']:.3f}\n\n")
        
        winner = "Depth-Bounded" if comparison['better'] == 'grammar1' else "Standard" if comparison['better'] == 'grammar2' else "beide gleich"
        self.text_advanced.insert(tk.END, f"🏆 **Besser: {winner}**\n")


# ============================================================================
# EXPORT-ERWEITERUNGEN
# ============================================================================

class ExtendedExporter(MultiFormatExporter):
    """Erweiterter Exporter mit Tiefenstatistik und MDL-Scores"""
    
    def to_json(self, data, filename=None):
        """Erweiterter JSON-Export mit Tiefeninfo"""
        if 'grammar' in data and hasattr(data['grammar'], 'get'):
            # Füge Tiefenstatistik hinzu, wenn vorhanden
            if 'depth_bounded' in data and data['depth_bounded']:
                data['depth_statistics'] = {
                    'max_depth': data.get('max_depth', 5),
                    'depth_distribution': dict(Counter(data['depth_map'].values())) if 'depth_map' in data else {}
                }
        
        return super().to_json(data, filename)
    
    def to_html(self, data, filename=None):
        """Erweiterter HTML-Export mit MDL-Scores"""
        filepath = super().to_html(data, filename)
        
        # Hier könnte man nachträglich MDL-Info einfügen
        # Für jetzt: Standard-HTML reicht
        
        return filepath


# ============================================================================
# MODULSTATUS AKTUALISIEREN
# ============================================================================

MODULE_STATUS['prefixspan'] = PREFIXSPAN_AVAILABLE
MODULE_STATUS['seminfo'] = SEMINFO_AVAILABLE

# ============================================================================
# HAUPTFUNKTION
# ============================================================================

def main():
    """Hauptfunktion"""
    print("\n" + "=" * 70)
    print("ARSXAI10 - ERWEITERTE GRAMMATIKINDUKTION GESTARTET")
    print("=" * 70)
    print(f"PrefixSpan verfügbar: {'✓' if PREFIXSPAN_AVAILABLE else '✗'}")
    print(f"SemInfo verfügbar: {'✓' if SEMINFO_AVAILABLE else '✗'}")
    print("=" * 70 + "\n")
    
    root = tk.Tk()
    app = ARSXAI10GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
