"""
ARS GUI - Algorithmic Recursive Sequence Analysis with Graphical User Interface
Erweiterte Version mit formalem Entscheidungsautomaten und 5-Bit-Kodierung

Neue Funktionen:
- 5-Bit-Kodierung der Terminalzeichen
- Deterministischer endlicher Automat zur Wohlgeformtheitsprüfung
- Explizite, rekonstruierbare Entscheidungspfade
- Trennung von struktureller Validierung und statistischer Analyse
"""

import sys
import subprocess
import importlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# PAKETVERWALTUNG
# ============================================================================

def check_and_install_packages():
    """Prüft und installiert fehlende Python-Pakete"""
    
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
from collections import Counter, defaultdict
import threading
import re
import queue

# Optionale Imports mit Fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from sklearn_crfsuite import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


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
# 5-BIT-KODIERUNG DER TERMINALZEICHEN
# ============================================================================

class TerminalCoding:
    """
    5-Bit-Kodierung der Terminalzeichen nach dem Schema:
    [S][P1P2][U1U2]
    
    S: 0 = Kunde, 1 = Verkäufer
    P1P2: 00 = BG, 01 = B, 10 = A, 11 = AV
    U1U2: 00 = Basis, 01 = Folge
    """
    
    # Mapping von Symbolen auf 5-Bit-Codes
    SYMBOL_TO_CODE = {
        'KBG': '00000',
        'VBG': '10000',
        'KBBd': '00100',
        'VBBd': '10100',
        'KBA': '00101',
        'VBA': '10101',
        'KAE': '01000',
        'VAE': '11000',
        'KAA': '01001',
        'VAA': '11001',
        'KAV': '01100',
        'VAV': '11100'
    }
    
    # Rückwärts-Mapping für Anzeige
    CODE_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_CODE.items()}
    
    @classmethod
    def encode(cls, symbol):
        """Wandelt ein Symbol in seinen 5-Bit-Code um"""
        return cls.SYMBOL_TO_CODE.get(symbol, None)
    
    @classmethod
    def decode(cls, code):
        """Wandelt einen 5-Bit-Code zurück in das Symbol"""
        return cls.CODE_TO_SYMBOL.get(code, code)
    
    @classmethod
    def encode_chain(cls, chain):
        """Wandelt eine ganze Kette in 5-Bit-Codes um"""
        encoded = []
        for sym in chain:
            code = cls.encode(sym)
            if code:
                encoded.append(code)
            else:
                encoded.append(sym)  # Fallback für unbekannte Symbole
        return encoded
    
    @classmethod
    def decode_chain(cls, coded_chain):
        """Wandelt eine kodierte Kette zurück in Symbole"""
        decoded = []
        for code in coded_chain:
            if len(code) == 5 and all(c in '01' for c in code):
                sym = cls.decode(code)
                decoded.append(sym)
            else:
                decoded.append(code)
        return decoded


# ============================================================================
# FORMALER ENTSCHEIDUNGSAUTOMAT
# ============================================================================

class DialogueAutomaton:
    """
    Deterministischer endlicher Automat zur Prüfung der Wohlgeformtheit
    von Dialogsequenzen basierend auf der 5-Bit-Kodierung.
    """
    
    # Zustände
    Q0 = 'q0'        # Start
    Q_BG = 'q_BG'    # Begrüßung
    Q_B = 'q_B'      # Bedarf
    Q_A = 'q_A'      # Abschluss
    Q_AV = 'q_AV'    # Verabschiedung
    Q_ERR = 'q_err'  # Fehler
    
    # Akzeptierende Zustände
    ACCEPTING = {Q_AV}
    
    # Zustandsnamen für Ausgabe
    STATE_NAMES = {
        Q0: 'Start',
        Q_BG: 'Begrüßung',
        Q_B: 'Bedarf',
        Q_A: 'Abschluss',
        Q_AV: 'Verabschiedung',
        Q_ERR: 'Fehler'
    }
    
    def __init__(self):
        self.current_state = self.Q0
        self.history = []
        self.reset()
    
    def reset(self):
        """Setzt den Automaten in den Startzustand zurück"""
        self.current_state = self.Q0
        self.history = [(self.Q0, None, 'Initialisierung')]
    
    def get_state_name(self, state):
        """Gibt den lesbaren Namen eines Zustands zurück"""
        return self.STATE_NAMES.get(state, state)
    
    def transition(self, code):
        """
        Führt einen Übergang basierend auf dem 5-Bit-Code durch.
        Gibt (neuer_zustand, akzeptiert, erklärung) zurück.
        """
        state = self.current_state
        
        # Prüfe, ob Code gültig ist
        if len(code) != 5 or not all(c in '01' for c in code):
            self.current_state = self.Q_ERR
            explanation = f"Ungültiger Code: {code}"
            self.history.append((self.current_state, code, explanation))
            return self.current_state, False, explanation
        
        # Sprecherbit extrahieren
        speaker = 'Kunde' if code[0] == '0' else 'Verkäufer'
        phase_bits = code[1:3]
        sub_bits = code[3:5]
        
        # Phasenbestimmung
        phase_map = {'00': 'BG', '01': 'B', '10': 'A', '11': 'AV'}
        phase = phase_map.get(phase_bits, 'UNBEKANNT')
        
        # Übergangstabelle
        if state == self.Q0:
            if code == '00000':  # KBG
                self.current_state = self.Q_BG
                explanation = f"Start → Begrüßung: {speaker} eröffnet Gespräch"
            else:
                self.current_state = self.Q_ERR
                explanation = f"Start: Erwarte KBG (00000), erhielt {code}"
        
        elif state == self.Q_BG:
            if code == '10000':  # VBG
                self.current_state = self.Q_BG
                explanation = f"Begrüßung fortgesetzt: {speaker} erwidert Gruß"
            elif code == '00100':  # KBBd
                self.current_state = self.Q_B
                explanation = f"Begrüßung → Bedarf: {speaker} äußert Bedarf"
            else:
                self.current_state = self.Q_ERR
                explanation = f"Begrüßung: Unerwartetes Symbol {code}"
        
        elif state == self.Q_B:
            if code in ['00100', '10100', '00101', '10101']:  # KBBd, VBBd, KBA, VBA
                self.current_state = self.Q_B
                explanation = f"Bedarf fortgesetzt: {speaker} in Phase {phase}"
            elif code == '01000':  # KAE
                self.current_state = self.Q_A
                explanation = f"Bedarf → Abschluss: {speaker} leitet Abschluss ein"
            else:
                self.current_state = self.Q_ERR
                explanation = f"Bedarf: Unerwartetes Symbol {code}"
        
        elif state == self.Q_A:
            if code in ['01000', '11000']:  # KAE, VAE
                self.current_state = self.Q_A
                explanation = f"Abschluss fortgesetzt: {speaker} in Phase {phase}"
            elif code == '01001':  # KAA
                self.current_state = self.Q_AV
                explanation = f"Abschluss → Verabschiedung: {speaker} schließt ab"
            else:
                self.current_state = self.Q_ERR
                explanation = f"Abschluss: Unerwartetes Symbol {code}"
        
        elif state == self.Q_AV:
            if code in ['01100', '11100', '11001']:  # KAV, VAV, VAA
                self.current_state = self.Q_AV
                explanation = f"Verabschiedung: {speaker} in Phase {phase}"
            else:
                self.current_state = self.Q_ERR
                explanation = f"Verabschiedung: Unerwartetes Symbol {code}"
        
        else:  # Fehlerzustand
            self.current_state = self.Q_ERR
            explanation = f"Bereits im Fehlerzustand"
        
        self.history.append((self.current_state, code, explanation))
        
        is_accepting = self.current_state in self.ACCEPTING
        return self.current_state, is_accepting, explanation
    
    def validate_chain(self, coded_chain):
        """
        Validiert eine ganze kodierte Kette.
        Gibt (gültig, letzter_zustand, protokoll) zurück.
        """
        self.reset()
        protocol = []
        
        for i, code in enumerate(coded_chain):
            state, accepting, explanation = self.transition(code)
            protocol.append({
                'position': i + 1,
                'code': code,
                'symbol': TerminalCoding.decode(code),
                'state': self.get_state_name(state),
                'explanation': explanation,
                'is_accepting': accepting
            })
        
        valid = self.current_state in self.ACCEPTING
        return valid, self.get_state_name(self.current_state), protocol
    
    def get_history_string(self):
        """Gibt den gesamten Entscheidungspfad als String zurück"""
        lines = []
        for i, (state, code, explanation) in enumerate(self.history):
            if i == 0:
                lines.append(f"Start: {self.get_state_name(state)}")
            else:
                sym = TerminalCoding.decode(code) if code else "-"
                lines.append(f"  {i}. {code} ({sym}) → {self.get_state_name(state)}")
                lines.append(f"     {explanation}")
        return "\n".join(lines)


# ============================================================================
# ARS 2.0 - BASIS-GRAMMATIK (wie gehabt)
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
        self.chains = chains
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
        probabilities = {}
        for start in transitions:
            total = sum(transitions[start].values())
            if total > 0:
                probabilities[start] = {end: count / total 
                                       for end, count in transitions[start].items()}
        return probabilities
    
    def print_grammar(self):
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
    
    def optimize(self, max_iterations=500, tolerance=0.005, target_correlation=0.9,
                 progress_callback=None):
        probs = {}
        for start, p in self.probabilities.items():
            probs[start] = p.copy()
            
        empirical_freqs = self.compute_frequencies(self.chains)
        
        best_correlation = 0
        best_probabilities = None
        history = []
        
        for iteration in range(max_iterations):
            generated = [self.generate_chain(max_length=20) for _ in range(8)]
            generated = [g for g in generated if g]
            
            if not generated:
                continue
                
            gen_freqs = self.compute_frequencies(generated)
            
            try:
                if len(empirical_freqs) == len(gen_freqs) and len(empirical_freqs) > 1:
                    corr, p_val = pearsonr(empirical_freqs, gen_freqs)
                else:
                    corr, p_val = 0, 1
            except:
                corr, p_val = 0, 1
            
            history.append((iteration, corr, p_val))
            
            if progress_callback and iteration % 10 == 0:
                progress_callback(iteration, max_iterations, corr, p_val)
            
            if corr >= target_correlation and p_val < 0.05:
                best_correlation = corr
                best_probabilities = {s: p.copy() for s, p in probs.items()}
                break
            
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
    
    def compute_frequencies(self, chains):
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


# ============================================================================
# ARS 3.0 - GRAMMATIK MIT NONTERMINALEN
# ============================================================================

class MethodologicalReflection:
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
        
        aktionen = [self._interpretiere_symbol(s) for s in sequence if isinstance(s, str)]
        self.sequence_meaning_mapping[tuple(sequence)] = {
            'bedeutung': ' → '.join(aktionen),
            'typ': self._klassifiziere_sequenz(sequence)
        }
    
    def _interpretiere_symbol(self, symbol):
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
            'VAV': 'Verkäufer-Verabschiedung'
        }
        return bedeutungen.get(symbol, str(symbol))
    
    def _klassifiziere_sequenz(self, sequence):
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
    def __init__(self):
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
    
    def load_chains(self, chains, user_start_symbol=None):
        self.chains = [list(chain) for chain in chains]
        self.user_start_symbol = user_start_symbol
        all_symbols = set()
        for chain in chains:
            for symbol in chain:
                all_symbols.add(symbol)
        self.terminals = all_symbols
        return True
    
    def find_best_repetition(self, chains, min_length=2, max_length=5):
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
        
        best_seq = max(repeated.items(), 
                      key=lambda x: x[1] * len(x[0]) / max(1, len(set(x[0]))))
        return best_seq[0]
    
    def generate_nonterminal_name(self, sequence):
        if all(isinstance(s, str) and s.startswith(('K', 'V')) for s in sequence):
            first = sequence[0]
            last = sequence[-1]
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
            return f"NT_{'_'.join(str(s) for s in sequence)}"
    
    def _describe_sequence(self, sequence):
        if len(sequence) == 2:
            if all(isinstance(s, str) and len(s) <= 4 for s in sequence):
                return f"{self.reflection._interpretiere_symbol(sequence[0])} → {self.reflection._interpretiere_symbol(sequence[1])}"
            else:
                return f"{sequence[0]} → {sequence[1]}"
        else:
            return f"Sequenz mit {len(sequence)} Schritten"
    
    def compress_sequences(self, chains, sequence, new_nonterminal):
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
        if not chains:
            return False
        first = chains[0]
        return all(len(chain) == 1 and chain[0] == first[0] for chain in chains)
    
    def find_top_level_nonterminal(self):
        if not self.rules:
            return None
        
        symbols_in_productions = set()
        for nt, productions in self.rules.items():
            for prod, _ in productions:
                for sym in prod:
                    symbols_in_productions.add(sym)
        
        top_level = [nt for nt in self.rules if nt not in symbols_in_productions]
        
        if top_level:
            if len(top_level) > 1:
                top_level.sort(key=lambda nt: self.hierarchy_levels.get(nt, 0), reverse=True)
            return top_level[0]
        
        if self.hierarchy_levels:
            return max(self.hierarchy_levels.items(), key=lambda x: x[1])[0]
        
        return list(self.rules.keys())[0] if self.rules else None
    
    def induce_grammar(self, max_iterations=50, progress_callback=None):
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
        
        while iteration < max_iterations:
            best_seq = self.find_best_repetition(current_chains)
            
            if best_seq is None:
                print(f"\nKeine weiteren Wiederholungen nach {iteration} Iterationen.")
                break
            
            new_nonterminal = self.generate_nonterminal_name(best_seq)
            beschreibung = self._describe_sequence(best_seq)
            
            base_name = new_nonterminal
            while new_nonterminal in self.nonterminals:
                new_nonterminal = f"{base_name}_{rule_counter}"
                rule_counter += 1
            
            rationale = f"Erkanntes Dialogmuster: {beschreibung}"
            self.reflection.log_interpretation(best_seq, new_nonterminal, rationale)
            
            seq_str = ' → '.join([str(s) for s in best_seq])
            print(f"\nIteration {iteration + 1}:")
            print(f"  Erkanntes Muster: {seq_str}")
            print(f"  → Neue Kategorie: {new_nonterminal}")
            
            self.rules[new_nonterminal] = [(list(best_seq), 1.0)]
            self.nonterminals.add(new_nonterminal)
            self.hierarchy_levels[new_nonterminal] = iteration
            
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
            
            current_chains = self.compress_sequences(current_chains, best_seq, new_nonterminal)
            
            if current_chains and current_chains[0]:
                example = ' → '.join([str(s) for s in current_chains[0][:10]])
                print(f"  Beispiel: {example}...")
            
            iteration += 1
            self.iteration_count = iteration
            
            if self.all_chains_identical(current_chains):
                if current_chains and current_chains[0]:
                    unique_symbol = current_chains[0][0]
                    if self.user_start_symbol and self.user_start_symbol in self.rules:
                        self.start_symbol = self.user_start_symbol
                    elif unique_symbol in self.rules:
                        self.start_symbol = unique_symbol
                    else:
                        self.start_symbol = self.find_top_level_nonterminal()
                    break
        
        if self.start_symbol is None:
            if self.user_start_symbol and self.user_start_symbol in self.rules:
                self.start_symbol = self.user_start_symbol
            elif self.rules:
                self.start_symbol = self.find_top_level_nonterminal()
        
        all_symbols = set()
        for chain in self.chains:
            for sym in chain:
                all_symbols.add(sym)
        
        self.terminals = all_symbols - self.nonterminals
        self._calculate_probabilities()
        
        return current_chains
    
    def _calculate_probabilities(self):
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
    
    def print_grammar(self):
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
        
        return "\n".join(lines)
    
    def generate_chain(self, start_symbol=None, max_depth=20):
        if not start_symbol:
            start_symbol = self.start_symbol
        
        if not start_symbol:
            return []
        
        if start_symbol not in self.rules:
            if self.rules:
                start_symbol = self.find_top_level_nonterminal()
            else:
                return []
        
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


# ============================================================================
# PETRI-NETZE (gekürzt, da nicht im Fokus)
# ============================================================================

if NETWORKX_AVAILABLE:
    class ARSPetriNet:
        def __init__(self, name="ARS_PetriNet"):
            self.name = name
            self.places = {}
            self.transitions = {}
            self.arcs = []
            self.tokens = {}
            self.hierarchy = {}
            self.firing_history = []
            self.reached_markings = set()
        
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
            if subnet:
                self.hierarchy[name] = subnet
        
        def add_arc(self, source, target, weight=1):
            self.arcs.append({'source': source, 'target': target, 'weight': weight})
        
        def get_preset(self, transition):
            preset = {}
            for arc in self.arcs:
                if arc['target'] == transition and arc['source'] in self.places:
                    preset[arc['source']] = arc['weight']
            return preset
        
        def get_postset(self, transition):
            postset = {}
            for arc in self.arcs:
                if arc['source'] == transition and arc['target'] in self.places:
                    postset[arc['target']] = arc['weight']
            return postset
        
        def is_enabled(self, transition):
            if transition not in self.transitions:
                return False
            preset = self.get_preset(transition)
            for place, weight in preset.items():
                if self.tokens.get(place, 0) < weight:
                    return False
            trans_data = self.transitions[transition]
            if trans_data['guard'] and not trans_data['guard'](self):
                return False
            return True
        
        def fire(self, transition):
            if not self.is_enabled(transition):
                return False
            preset = self.get_preset(transition)
            for place, weight in preset.items():
                self.tokens[place] -= weight
            postset = self.get_postset(transition)
            for place, weight in postset.items():
                self.tokens[place] = self.tokens.get(place, 0) + weight
            self.firing_history.append({'transition': transition, 'marking': self.get_marking_copy()})
            self.reached_markings.add(self.get_marking_tuple())
            return True
        
        def get_marking_copy(self):
            return self.tokens.copy()
        
        def get_marking_tuple(self):
            return tuple(sorted([(p, self.tokens[p]) for p in self.places]))
        
        def reset(self):
            for place_name, place_data in self.places.items():
                self.tokens[place_name] = place_data['initial_tokens']
            self.firing_history = []
        
        def simulate(self, transition_sequence):
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
        def __init__(self, terminal_chains, grammar_rules=None):
            self.chains = terminal_chains
            self.grammar = grammar_rules
            self.petri_net = None
        
        def build_basic_net(self):
            self.petri_net = ARSPetriNet("ARS_PetriNet_Basic")
            all_symbols = set()
            for chain in self.chains:
                for sym in chain:
                    all_symbols.add(sym)
            self.petri_net.add_place("p_start", initial_tokens=1)
            self.petri_net.add_place("p_end", initial_tokens=0)
            for i, sym in enumerate(sorted(all_symbols)):
                self.petri_net.add_place(f"p_{sym}_ready", initial_tokens=0)
                self.petri_net.add_transition(f"t_{sym}")
                if i == 0:
                    self.petri_net.add_arc("p_start", f"t_{sym}")
                self.petri_net.add_arc(f"t_{sym}", f"p_{sym}_ready")
            return self.petri_net
        
        def build_resource_net(self):
            self.petri_net = ARSPetriNet("ARS_PetriNet_Resource")
            self.petri_net.add_place("p_customer_present", initial_tokens=1, place_type="customer")
            self.petri_net.add_place("p_customer_ready", initial_tokens=1, place_type="customer")
            self.petri_net.add_place("p_seller_ready", initial_tokens=1, place_type="seller")
            self.petri_net.add_place("p_goods_available", initial_tokens=10, place_type="resource")
            self.petri_net.add_place("p_goods_selected", initial_tokens=0, place_type="resource")
            self.petri_net.add_place("p_money_customer", initial_tokens=20, place_type="resource")
            self.petri_net.add_place("p_money_register", initial_tokens=0, place_type="resource")
            phases = ["Greeting", "Need", "Consult", "Completion", "Farewell"]
            for phase in phases:
                self.petri_net.add_place(f"p_phase_{phase}", initial_tokens=0, place_type="phase")
            self.petri_net.add_place("p_phase_start", initial_tokens=1, place_type="phase")
            all_symbols = set()
            for chain in self.chains:
                for sym in chain:
                    all_symbols.add(sym)
            for sym in sorted(all_symbols):
                self.petri_net.add_transition(f"t_{sym}")
                if sym.startswith('K'):
                    self.petri_net.add_arc("p_customer_ready", f"t_{sym}")
                    self.petri_net.add_arc(f"t_{sym}", "p_customer_ready")
                else:
                    self.petri_net.add_arc("p_seller_ready", f"t_{sym}")
                    self.petri_net.add_arc(f"t_{sym}", "p_seller_ready")
                if sym.endswith('A'):
                    self.petri_net.add_arc("p_goods_selected", f"t_{sym}")
                    self.petri_net.add_arc("p_money_customer", f"t_{sym}")
                    self.petri_net.add_arc(f"t_{sym}", "p_goods_available")
                    self.petri_net.add_arc(f"t_{sym}", "p_money_register")
            return self.petri_net
        
        def simulate_chain(self, chain):
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
# GUI - HAUPTFENSTER (erweitert um Kodierung und Automaten-Tab)
# ============================================================================

class ARSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ARS 4.0 - Algorithmic Recursive Sequence Analysis")
        self.root.geometry("1400x900")
        
        self.plot_thread = PlotThread(root)
        self.update_queue = queue.Queue()
        self.process_updates()
        
        # Daten
        self.chains = []
        self.terminals = []
        self.delimiter = tk.StringVar(value=",")
        self.start_symbol = tk.StringVar(value="")
        
        # Kodierte Ketten
        self.coded_chains = []
        
        # ARS-Objekte
        self.ars20 = ARS20()
        self.ars30 = GrammarInducer()
        self.petri_builder = None
        self.automaton = DialogueAutomaton()
        
        # Verfügbarkeit der optionalen Module
        self.module_status = {
            'networkx': NETWORKX_AVAILABLE,
            'hmmlearn': HMM_AVAILABLE,
            'crf': CRF_AVAILABLE,
            'transformer': TRANSFORMER_AVAILABLE,
            'seaborn': SEABORN_AVAILABLE
        }
        
        self.create_menu()
        self.create_main_panels()
        self.status_var = tk.StringVar(value="Bereit")
        self.create_statusbar()
        self.show_module_status()
    
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
        file_menu.add_command(label="Öffnen", command=self.load_file)
        file_menu.add_command(label="Beispiel laden", command=self.load_example)
        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self.root.quit)
        
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
        ttk.Radiobutton(delim_frame, text="Benutzer", variable=self.delimiter, 
                       value="custom").pack(side=tk.LEFT, padx=2)
        
        self.custom_delimiter = ttk.Entry(delim_frame, width=5)
        self.custom_delimiter.pack(side=tk.LEFT, padx=2)
        self.custom_delimiter.insert(0, "|")
        
        ttk.Label(parent, text="Terminalzeichenketten (eine pro Zeile):").pack(anchor=tk.W, pady=5)
        
        self.text_input = scrolledtext.ScrolledText(parent, height=12, font=('Courier', 10))
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=5)
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Datei laden", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Parsen", command=self.parse_input).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Beispiel", command=self.load_example).pack(side=tk.LEFT, padx=2)
        
        start_frame = ttk.Frame(parent)
        start_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(start_frame, text="Startzeichen:").pack(side=tk.LEFT)
        self.start_entry = ttk.Entry(start_frame, textvariable=self.start_symbol, width=10)
        self.start_entry.pack(side=tk.LEFT, padx=5)
        
        self.info_var = tk.StringVar(value="Keine Daten geladen")
        ttk.Label(parent, textvariable=self.info_var, foreground="blue").pack(anchor=tk.W, pady=5)
    
    def create_output_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab20 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab20, text="ARS 2.0 (Basis)")
        self.create_ars20_tab()
        
        self.tab30 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab30, text="ARS 3.0 (Nonterminale)")
        self.create_ars30_tab()
        
        self.tab_code = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_code, text="5-Bit-Kodierung")
        self.create_code_tab()
        
        self.tab_auto = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_auto, text="Entscheidungsautomat")
        self.create_automaton_tab()
        
        self.tab_petri = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_petri, text="Petri-Netze")
        self.create_petri_tab()
        
        self.tab_gen = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_gen, text="Generierung")
        self.create_generation_tab()
    
    def create_ars20_tab(self):
        control = ttk.Frame(self.tab20)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="ARS 2.0 berechnen", 
                  command=self.run_ars20).pack(side=tk.LEFT, padx=5)
        
        self.text20 = scrolledtext.ScrolledText(self.tab20, font=('Courier', 10))
        self.text20.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_ars30_tab(self):
        control = ttk.Frame(self.tab30)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Grammatik induzieren", 
                  command=self.run_ars30).pack(side=tk.LEFT, padx=5)
        
        self.ars30_progress = ttk.Progressbar(control, length=200, mode='indeterminate')
        self.ars30_progress.pack(side=tk.LEFT, padx=10)
        
        self.text30 = scrolledtext.ScrolledText(self.tab30, font=('Courier', 10))
        self.text30.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_code_tab(self):
        """Neuer Tab für 5-Bit-Kodierung"""
        control = ttk.Frame(self.tab_code)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Kodieren", 
                  command=self.encode_chains).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Dekodieren", 
                  command=self.decode_chains).pack(side=tk.LEFT, padx=5)
        
        self.text_code = scrolledtext.ScrolledText(self.tab_code, font=('Courier', 10))
        self.text_code.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_automaton_tab(self):
        """Neuer Tab für Entscheidungsautomaten"""
        control = ttk.Frame(self.tab_auto)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Validiere Transkript 1", 
                  command=self.validate_transcript_1).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Validiere alle", 
                  command=self.validate_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Automaten zurücksetzen", 
                  command=self.reset_automaton).pack(side=tk.LEFT, padx=5)
        
        self.text_auto = scrolledtext.ScrolledText(self.tab_auto, font=('Courier', 10))
        self.text_auto.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_petri_tab(self):
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
    
    def create_generation_tab(self):
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
        status = ttk.Frame(self.root)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(status, length=100, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
    
    def show_module_status(self):
        status_text = "Modulstatus:\n"
        status_text += f"✓ networkx: {'verfügbar' if self.module_status['networkx'] else 'nicht verfügbar'}\n"
        status_text += f"✓ hmmlearn: {'verfügbar' if self.module_status['hmmlearn'] else 'nicht verfügbar'}\n"
        status_text += f"✓ sklearn-crfsuite: {'verfügbar' if self.module_status['crf'] else 'nicht verfügbar'}\n"
        status_text += f"✓ sentence-transformers: {'verfügbar' if self.module_status['transformer'] else 'nicht verfügbar'}\n"
        status_text += f"✓ seaborn: {'verfügbar' if self.module_status['seaborn'] else 'nicht verfügbar'}"
        messagebox.showinfo("Modulstatus", status_text)
    
    def get_actual_delimiter(self):
        delim = self.delimiter.get()
        if delim == "custom":
            return self.custom_delimiter.get()
        return delim
    
    def parse_line(self, line):
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
        self.text_input.update()
        text = self.text_input.get("1.0", tk.END)
        lines = text.strip().split('\n')
        
        self.chains = []
        for line in lines:
            chain = self.parse_line(line)
            if chain:
                self.chains.append(chain)
        
        if self.chains:
            all_symbols = set()
            for chain in self.chains:
                for symbol in chain:
                    all_symbols.add(symbol)
            self.terminals = sorted(all_symbols)
            
            self.info_var.set(f"{len(self.chains)} Ketten, {len(self.terminals)} Terminale")
            self.status_var.set(f"{len(self.chains)} Ketten geladen")
            
            self.ars20.load_chains(self.chains, self.start_symbol.get() or None)
            self.ars30.load_chains(self.chains, self.start_symbol.get() or None)
            
            if self.module_status['networkx']:
                self.petri_builder = PetriNetBuilder(self.chains, self.ars30.rules)
            
            # Kodierte Ketten berechnen
            self.encode_chains()
            
            self.show_ars20_preview()
        else:
            messagebox.showwarning("Warnung", "Keine gültigen Ketten gefunden!")
    
    def encode_chains(self):
        """Wandelt alle Ketten in 5-Bit-Kodierung um"""
        if not self.chains:
            return
        
        self.coded_chains = []
        self.text_code.delete("1.0", tk.END)
        self.text_code.insert(tk.END, "5-BIT-KODIERUNG DER TERMINALZEICHEN\n")
        self.text_code.insert(tk.END, "=" * 70 + "\n\n")
        self.text_code.insert(tk.END, "Schema: [Sprecher][Phase][Unterphase]\n")
        self.text_code.insert(tk.END, "S: 0=Kunde, 1=Verkäufer\n")
        self.text_code.insert(tk.END, "Phase: 00=BG, 01=B, 10=A, 11=AV\n")
        self.text_code.insert(tk.END, "Unterphase: 00=Basis, 01=Folge\n\n")
        
        for i, chain in enumerate(self.chains, 1):
            coded = TerminalCoding.encode_chain(chain)
            self.coded_chains.append(coded)
            
            self.text_code.insert(tk.END, f"Transkript {i}:\n")
            self.text_code.insert(tk.END, f"  Original: {', '.join(chain)}\n")
            self.text_code.insert(tk.END, f"  Kodiert:  {', '.join(coded)}\n\n")
        
        self.status_var.set(f"{len(self.chains)} Ketten kodiert")
    
    def decode_chains(self):
        """Zeigt die dekodierten Ketten an (nur zur Bestätigung)"""
        if not self.coded_chains:
            messagebox.showinfo("Info", "Keine kodierten Ketten vorhanden")
            return
        
        self.text_code.insert(tk.END, "\n" + "=" * 70 + "\n")
        self.text_code.insert(tk.END, "DEKODIERTE KETTEN (Kontrolle)\n")
        self.text_code.insert(tk.END, "=" * 70 + "\n\n")
        
        for i, coded in enumerate(self.coded_chains, 1):
            decoded = TerminalCoding.decode_chain(coded)
            self.text_code.insert(tk.END, f"Transkript {i} (dekodiert): {', '.join(decoded)}\n")
    
    def reset_automaton(self):
        """Setzt den Automaten zurück"""
        self.automaton.reset()
        self.text_auto.delete("1.0", tk.END)
        self.text_auto.insert(tk.END, "Automaten zurückgesetzt.\n")
        self.text_auto.insert(tk.END, self.automaton.get_history_string())
        self.status_var.set("Automaten zurückgesetzt")
    
    def validate_transcript_1(self):
        """Validiert Transkript 1 mit dem Automaten"""
        if not self.coded_chains or len(self.coded_chains) < 1:
            messagebox.showerror("Fehler", "Keine kodierten Ketten vorhanden!")
            return
        
        self.validate_chain(0, "Transkript 1")
    
    def validate_all(self):
        """Validiert alle kodierten Ketten"""
        if not self.coded_chains:
            messagebox.showerror("Fehler", "Keine kodierten Ketten vorhanden!")
            return
        
        self.text_auto.delete("1.0", tk.END)
        valid_count = 0
        
        for i, coded in enumerate(self.coded_chains):
            valid, state, protocol = self.automaton.validate_chain(coded)
            
            self.text_auto.insert(tk.END, f"\n{'='*50}\n")
            self.text_auto.insert(tk.END, f"VALIDIERUNG TRANSKRIPT {i+1}\n")
            self.text_auto.insert(tk.END, f"{'='*50}\n")
            self.text_auto.insert(tk.END, f"Ergebnis: {'✓ GÜLTIG' if valid else '✗ UNGÜLTIG'}\n")
            self.text_auto.insert(tk.END, f"Endzustand: {state}\n\n")
            self.text_auto.insert(tk.END, "ENTSCHEIDUNGSPFAD:\n")
            
            for step in protocol:
                self.text_auto.insert(tk.END, f"  Schritt {step['position']}: {step['code']} ({step['symbol']})\n")
                self.text_auto.insert(tk.END, f"    → {step['state']}\n")
                self.text_auto.insert(tk.END, f"    {step['explanation']}\n")
            
            if valid:
                valid_count += 1
        
        self.text_auto.insert(tk.END, f"\n{'='*50}\n")
        self.text_auto.insert(tk.END, f"GESAMTERGEBNIS: {valid_count}/{len(self.coded_chains)} gültig\n")
        self.status_var.set(f"{valid_count}/{len(self.coded_chains)} Ketten gültig")
    
    def validate_chain(self, index, name):
        """Validiert eine einzelne Kette"""
        coded = self.coded_chains[index]
        valid, state, protocol = self.automaton.validate_chain(coded)
        
        self.text_auto.delete("1.0", tk.END)
        self.text_auto.insert(tk.END, f"{'='*50}\n")
        self.text_auto.insert(tk.END, f"VALIDIERUNG {name}\n")
        self.text_auto.insert(tk.END, f"{'='*50}\n")
        self.text_auto.insert(tk.END, f"Ergebnis: {'✓ GÜLTIG' if valid else '✗ UNGÜLTIG'}\n")
        self.text_auto.insert(tk.END, f"Endzustand: {state}\n\n")
        self.text_auto.insert(tk.END, "ENTSCHEIDUNGSPFAD:\n")
        
        for step in protocol:
            self.text_auto.insert(tk.END, f"  Schritt {step['position']}: {step['code']} ({step['symbol']})\n")
            self.text_auto.insert(tk.END, f"    → {step['state']}\n")
            self.text_auto.insert(tk.END, f"    {step['explanation']}\n")
        
        self.status_var.set(f"{name}: {'gültig' if valid else 'ungültig'}")
    
    def show_ars20_preview(self):
        self.text20.delete("1.0", tk.END)
        self.text20.insert(tk.END, self.ars20.print_grammar())
    
    def run_ars20(self):
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        self.show_ars20_preview()
        self.status_var.set("ARS 2.0 abgeschlossen")
    
    def run_ars30(self):
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
            self.status_var.set("Simulation abgeschlossen")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei der Simulation:\n{str(e)}")
    
    def generate_chains(self):
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
        example = """KBG, VBG, KBBd, VBBd, KBA, VBA, KBBd, VBBd, KBA, VAA, KAA, VAV, KAV
VBG, KBBd, VBBd, VAA, KAA, VBG, KBBd, VAA, KAA
KBBd, VBBd, VAA, KAA
KBBd, VBBd, KBA, VBA, KBBd, VBA, KAE, VAE, KAA, VAV, KAV
KBG, VBG, KBBd, VBBd, KAA
KBBd, VBBd, KBA, VAA, KAA
KBG, VBBd, KBBd, VBA, VAA, KAA, VAV, KAV"""
        
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", example)
        self.parse_input()
    
    def show_about(self):
        about = """ARS 4.0 - Algorithmic Recursive Sequence Analysis

Erweiterte Version mit:
- 5-Bit-Kodierung der Terminalzeichen
- Formalem Entscheidungsautomaten
- Expliziten, rekonstruierbaren Validierungspfaden
- Trennung von struktureller und statistischer Analyse

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
