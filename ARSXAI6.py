"""
ARS GUI - Algorithmic Recursive Sequence Analysis with Graphical User Interface
Erweiterte Version 5.1 mit:
- 5-Bit-Kodierung der Terminalzeichen
- Deterministischem Entscheidungsautomaten (strukturelle Korrektheit)
- Bayes'schen Netzen (HMM) mit korrekter one-hot-encoding für hmmlearn >= 0.3.0
- Statistischer Analyse empirischer Wahrscheinlichkeiten
- Expliziter Trennung von Struktur (deterministisch) und Empirie (probabilistisch)

Kernkonzept:
- Strukturelle Ebene: Definiert, was PRINZIPIELL möglich ist
- Empirische Ebene: Beschreibt, was TATSÄCHLICH vorkommt (Wahrscheinlichkeiten 0.0-1.0)
- Ein Terminalzeichen ist NUR DANN FALSCH, wenn es nicht im Alphabet ist oder strukturelle Regeln verletzt
"""

import sys
import subprocess
import importlib
import warnings
import traceback
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
    print("ARS 5.1 - PAKETPRÜFUNG")
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
import json
from datetime import datetime

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
    
    Wichtig: Diese Kodierung definiert die STRUKTURELLE Ebene.
    Ein Code ist genau dann gültig, wenn er in SYMBOL_TO_CODE vorkommt.
    """
    
    # Mapping von Symbolen auf 5-Bit-Codes (das definierte Alphabet)
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
    
    # Das vollständige Alphabet (alle gültigen Codes)
    ALPHABET = set(SYMBOL_TO_CODE.values())
    
    # Phasennamen für Ausgabe
    PHASE_NAMES = {
        '00': 'BG (Begrüßung)',
        '01': 'B (Bedarf)',
        '10': 'A (Abschluss)',
        '11': 'AV (Verabschiedung)'
    }
    
    @classmethod
    def is_valid_code(cls, code):
        """Prüft, ob ein Code zum definierten Alphabet gehört"""
        return code in cls.ALPHABET
    
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
        unknown = []
        for sym in chain:
            code = cls.encode(sym)
            if code:
                encoded.append(code)
            else:
                encoded.append(sym)  # Fallback für unbekannte Symbole
                unknown.append(sym)
        return encoded, unknown
    
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
    
    @classmethod
    def get_phase(cls, code):
        """Extrahiert die Phase (Bits 2-3) aus einem Code"""
        if len(code) >= 3 and cls.is_valid_code(code):
            return code[1:3]
        return None
    
    @classmethod
    def get_speaker(cls, code):
        """Extrahiert den Sprecher (Bit 1) aus einem Code"""
        if len(code) >= 1 and cls.is_valid_code(code):
            return 'Kunde' if code[0] == '0' else 'Verkäufer'
        return None
    
    @classmethod
    def get_phase_name(cls, code):
        """Gibt den lesbaren Phasennamen zurück"""
        phase = cls.get_phase(code)
        return cls.PHASE_NAMES.get(phase, phase)
    
    @classmethod
    def get_alphabet_display(cls):
        """Gibt eine lesbare Darstellung des Alphabets zurück"""
        lines = ["Definiertes Alphabet (strukturell gültige Codes):"]
        for code, sym in sorted(cls.CODE_TO_SYMBOL.items()):
            phase = cls.get_phase_name(code)
            speaker = cls.get_speaker(code)
            lines.append(f"  {code} = {sym} ({speaker}, {phase})")
        return "\n".join(lines)


# ============================================================================
# FORMALER ENTSCHEIDUNGSAUTOMAT (STRUKTURELLE EBENE)
# ============================================================================

class DialogueAutomaton:
    """
    Deterministischer endlicher Automat zur Prüfung der strukturellen
    Wohlgeformtheit von Dialogsequenzen.
    
    Wichtig: Dieser Automat entscheidet NUR über strukturelle Korrektheit.
    Ein Symbol ist genau dann falsch, wenn:
    1. Es nicht im Alphabet ist (unbekannter Code) -> sofortiger Fehler
    2. Es die Phasenregeln verletzt (falsche Position) -> Fehler
    """
    
    # Zustände
    Q0 = 'q0'        # Start
    Q_BG = 'q_BG'    # Begrüßung
    Q_B = 'q_B'      # Bedarf
    Q_A = 'q_A'      # Abschluss
    Q_AV = 'q_AV'    # Verabschiedung
    Q_ERR = 'q_err'  # Fehler (strukturell ungültig)
    
    # Akzeptierende Zustände
    ACCEPTING = {Q_AV}
    
    # Zustandsnamen für Ausgabe
    STATE_NAMES = {
        Q0: 'Start',
        Q_BG: 'Begrüßung',
        Q_B: 'Bedarf',
        Q_A: 'Abschluss',
        Q_AV: 'Verabschiedung',
        Q_ERR: 'FEHLER (strukturell ungültig)'
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
        Korrigierte Version: Erlaubt Übergang von Bedarf zu Abschluss über VAA (11001)
        und bleibt dann in der Abschlussphase für KAA (01001)
        """
        state = self.current_state
        
        # 1. Prüfe, ob Code zum Alphabet gehört
        if not TerminalCoding.is_valid_code(code):
            self.current_state = self.Q_ERR
            explanation = f"FEHLER: Code {code} ist nicht im definierten Alphabet"
            self.history.append((self.current_state, code, explanation))
            return self.current_state, False, explanation
        
        # Sprecherbit extrahieren
        speaker = 'Kunde' if code[0] == '0' else 'Verkäufer'
        phase_bits = code[1:3]
        sub_bits = code[3:5]
        
        # Phasenbestimmung
        phase_map = {'00': 'BG', '01': 'B', '10': 'A', '11': 'AV'}
        phase = phase_map.get(phase_bits, 'UNBEKANNT')
        
        # 2. Strukturelle Übergangsregeln (KORRIGIERT)
        if state == self.Q0:
            if code == '00000':  # KBG (nur Kunde kann Gespräch eröffnen)
                self.current_state = self.Q_BG
                explanation = f"Start → Begrüßung: {speaker} eröffnet Gespräch (strukturell korrekt)"
            else:
                self.current_state = self.Q_ERR
                explanation = f"FEHLER: Start muss mit KBG (00000) beginnen, erhielt {code}"
        
        elif state == self.Q_BG:
            if code == '10000':  # VBG (Verkäufer erwidert Gruß)
                self.current_state = self.Q_BG
                explanation = f"Begrüßung fortgesetzt: {speaker} erwidert Gruß (strukturell korrekt)"
            elif code == '00100':  # KBBd (Kunde äußert Bedarf)
                self.current_state = self.Q_B
                explanation = f"Begrüßung → Bedarf: {speaker} äußert Bedarf (strukturell korrekt)"
            else:
                self.current_state = self.Q_ERR
                explanation = f"FEHLER: In Begrüßung unerwartetes Symbol {code}"
        
        elif state == self.Q_B:
            if code in ['00100', '10100', '00101', '10101']:  # Alle Bedarfssymbole
                self.current_state = self.Q_B
                explanation = f"Bedarf fortgesetzt: {speaker} in Phase {phase} (strukturell korrekt)"
            elif code == '01000':  # KAE (Übergang zum Abschluss über Beratung)
                self.current_state = self.Q_A
                explanation = f"Bedarf → Abschluss (mit Beratung): {speaker} leitet Abschluss ein (strukturell korrekt)"
            elif code == '11001':  # VAA (Übergang zum Abschluss)
                self.current_state = self.Q_A  # Gehe in Abschlussphase, nicht direkt in Verabschiedung!
                explanation = f"Bedarf → Abschluss: {speaker} leitet Abschluss ein (strukturell korrekt)"
            else:
                self.current_state = self.Q_ERR
                explanation = f"FEHLER: In Bedarfsphase unerwartetes Symbol {code}"
        
        elif state == self.Q_A:
            if code in ['01000', '11000']:  # KAE, VAE (Beratung fortsetzen)
                self.current_state = self.Q_A
                explanation = f"Abschluss/Beratung fortgesetzt: {speaker} in Phase {phase} (strukturell korrekt)"
            elif code == '11001':  # VAA (Abschluss durch Verkäufer)
                self.current_state = self.Q_A
                explanation = f"Abschluss fortgesetzt: {speaker} in Phase {phase} (strukturell korrekt)"
            elif code == '01001':  # KAA (Abschluss durch Kunden)
                self.current_state = self.Q_AV
                explanation = f"Abschluss → Verabschiedung: {speaker} schließt ab (strukturell korrekt)"
            else:
                self.current_state = self.Q_ERR
                explanation = f"FEHLER: In Abschlussphase unerwartetes Symbol {code}"
        
        elif state == self.Q_AV:
            if code in ['01100', '11100']:  # KAV, VAV (Verabschiedung)
                self.current_state = self.Q_AV
                explanation = f"Verabschiedung: {speaker} in Phase {phase} (strukturell korrekt)"
            else:
                self.current_state = self.Q_ERR
                explanation = f"FEHLER: In Verabschiedung unerwartetes Symbol {code}"
        
        else:  # Bereits im Fehlerzustand
            self.current_state = self.Q_ERR
            explanation = f"Bereits im Fehlerzustand"
        
        self.history.append((self.current_state, code, explanation))
        
        is_accepting = self.current_state in self.ACCEPTING
        return self.current_state, is_accepting, explanation
    
    def validate_chain(self, coded_chain):
        """
        Validiert eine ganze kodierte Kette auf strukturelle Korrektheit.
        Gibt (gültig, letzter_zustand, protokoll, fehler) zurück.
        """
        self.reset()
        protocol = []
        first_error = None
        
        for i, code in enumerate(coded_chain):
            state, accepting, explanation = self.transition(code)
            protocol.append({
                'position': i + 1,
                'code': code,
                'symbol': TerminalCoding.decode(code) if TerminalCoding.is_valid_code(code) else "UNBEKANNT",
                'phase': TerminalCoding.get_phase_name(code) if TerminalCoding.is_valid_code(code) else "unbekannt",
                'speaker': TerminalCoding.get_speaker(code) if TerminalCoding.is_valid_code(code) else "unbekannt",
                'state': self.get_state_name(state),
                'explanation': explanation,
                'is_accepting': accepting
            })
            
            if state == self.Q_ERR and first_error is None:
                first_error = {
                    'position': i + 1,
                    'code': code,
                    'explanation': explanation
                }
        
        valid = self.current_state in self.ACCEPTING and first_error is None
        return valid, self.get_state_name(self.current_state), protocol, first_error
    
    def get_history_string(self):
        """Gibt den gesamten Entscheidungspfad als String zurück"""
        lines = []
        for i, (state, code, explanation) in enumerate(self.history):
            if i == 0:
                lines.append(f"Start: {self.get_state_name(state)}")
            else:
                if TerminalCoding.is_valid_code(code):
                    sym = TerminalCoding.decode(code)
                    phase = TerminalCoding.get_phase_name(code)
                    lines.append(f"  {i}. {code} ({sym}, {phase}) → {self.get_state_name(state)}")
                else:
                    lines.append(f"  {i}. {code} (UNBEKANNT) → {self.get_state_name(state)}")
                lines.append(f"     {explanation}")
        return "\n".join(lines)


# ============================================================================
# BAYESSCHE NETZE (ARS 4.0 - SZENARIO B) - KORRIGIERT FÜR HMMLEARN >= 0.3.0
# ============================================================================

if HMM_AVAILABLE:
    class ARSHiddenMarkovModel:
        """
        Hidden-Markov-Modell für ARS 5.1
        Modelliert latente Gesprächsphasen basierend auf den kodierten Terminalzeichen.
        
        Wichtig: Dieses Modell arbeitet auf der EMPIRISCHEN Ebene.
        Es beschreibt Wahrscheinlichkeiten, nicht strukturelle Korrektheit.
        
        Angepasst für hmmlearn >= 0.3.0 (MultinomialHMM erwartet one-hot-encoding)
        """
        
        def __init__(self, n_states=5):
            self.n_states = n_states
            self.model = None
            self.code_to_idx = {}
            self.idx_to_code = {}
            self.state_names = {
                0: "Greeting",
                1: "Need Determination",
                2: "Consultation",
                3: "Completion",
                4: "Farewell"
            }
            self.n_features = None
            self.trained = False
            self.hmm_warning_shown = False
            
        def prepare_data(self, coded_chains):
            """
            Bereitet kodierte Ketten für HMM vor (one-hot-encoding für MultinomialHMM).
            Überspringt unbekannte Codes (strukturell ungültige) automatisch.
            """
            # Alle gültigen Codes sammeln (nur aus dem definierten Alphabet)
            all_codes = set()
            for chain in coded_chains:
                for code in chain:
                    if TerminalCoding.is_valid_code(code):
                        all_codes.add(code)
            
            if not all_codes:
                return np.array([]).reshape(-1, 1), np.array([])
            
            # Mapping für gültige Codes
            self.code_to_idx = {code: i for i, code in enumerate(sorted(all_codes))}
            self.idx_to_code = {i: code for code, i in self.code_to_idx.items()}
            self.n_features = len(all_codes)
            
            # Daten in one-hot-encoding konvertieren für MultinomialHMM
            X_list = []
            lengths = []
            
            for chain in coded_chains:
                # Sammle alle gültigen Codes in dieser Kette
                valid_codes = [code for code in chain if code in self.code_to_idx]
                
                if valid_codes:
                    # One-hot-Encoding für jede Position
                    seq_length = len(valid_codes)
                    one_hot_seq = np.zeros((seq_length, self.n_features))
                    for i, code in enumerate(valid_codes):
                        one_hot_seq[i, self.code_to_idx[code]] = 1
                    X_list.append(one_hot_seq)
                    lengths.append(seq_length)
            
            if not X_list:
                return np.array([]), np.array([])
            
            # Vertikal stapeln
            X = np.vstack(X_list)
            
            return X, np.array(lengths)
        
        def initialize_from_structure(self, coded_chains):
            """
            Initialisiert HMM-Parameter basierend auf der Struktur.
            """
            if not self.hmm_warning_shown:
                print("\n=== Initialisiere HMM aus strukturellen Vorgaben ===")
                print("(Hinweis: MultinomialHMM in hmmlearn >= 0.3.0 verwendet one-hot-encoding)")
                self.hmm_warning_shown = True
            
            X, lengths = self.prepare_data(coded_chains)
            
            if len(X) == 0:
                print("Warnung: Keine gültigen Daten für HMM-Initialisierung")
                return None
            
            # Startwahrscheinlichkeiten (gleichverteilt, da empirisch)
            startprob = np.ones(self.n_states) / self.n_states
            
            # Übergangsmatrix (gleichverteilt initial)
            transmat = np.ones((self.n_states, self.n_states)) / self.n_states
            
            # Emissionswahrscheinlichkeiten (gleichverteilt initial)
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
            
            print(f"HMM initialisiert: {self.n_states} Zustände, {self.n_features} gültige Codes")
            self.trained = False
            
            return self.model
        
        def fit(self, coded_chains, n_iter=100):
            """
            Trainiert das HMM mit Baum-Welch auf den EMPIRISCHEN Daten.
            Die resultierenden Wahrscheinlichkeiten spiegeln die empirische Realität wider.
            """
            X, lengths = self.prepare_data(coded_chains)
            
            if len(X) == 0:
                raise ValueError("Keine gültigen Daten zum Trainieren vorhanden")
            
            print(f"\n=== Trainiere HMM mit {len(coded_chains)} Sequenzen ===")
            print(f"Gesamtlänge: {len(X)} Beobachtungen (one-hot-encoding, nur strukturell gültige Codes)")
            
            if self.model is None:
                self.model = hmm.MultinomialHMM(
                    n_components=self.n_states,
                    n_iter=n_iter,
                    random_state=42
                )
            
            self.model.fit(X, lengths)
            self.trained = True
            print(f"Training abgeschlossen nach {n_iter} Iterationen")
            
            return self.model
        
        def decode(self, coded_chain):
            """
            Viterbi-Dekodierung einer kodierten Kette.
            Findet die wahrscheinlichste Sequenz latenter Zustände.
            """
            if self.model is None or not self.trained:
                return None, None
            
            # Sammle alle gültigen Codes
            valid_codes = [code for code in coded_chain if code in self.code_to_idx]
            
            if not valid_codes:
                return [-1] * len(coded_chain), 0.0
            
            # One-hot-Encoding für die gültigen Codes
            X = np.zeros((len(valid_codes), self.n_features))
            for i, code in enumerate(valid_codes):
                X[i, self.code_to_idx[code]] = 1
            
            try:
                logprob, states = self.model.decode(X, algorithm="viterbi")
                
                # Rekonstruiere vollständige Zustandssequenz (mit -1 für ungültige Positionen)
                full_states = [-1] * len(coded_chain)
                valid_idx = 0
                for i, code in enumerate(coded_chain):
                    if code in self.code_to_idx:
                        full_states[i] = states[valid_idx]
                        valid_idx += 1
                
                return full_states, np.exp(logprob)
            except Exception as e:
                print(f"Fehler bei Viterbi-Dekodierung: {e}")
                return None, None
        
        def get_phase_probabilities(self):
            """
            Gibt die empirischen Übergangswahrscheinlichkeiten zwischen Phasen zurück.
            Dies sind Wahrscheinlichkeiten basierend auf den Trainingsdaten.
            """
            if self.model is None or not self.trained:
                return {}
            
            probs = {}
            phase_names = ['Greeting', 'Need', 'Consultation', 'Completion', 'Farewell']
            
            for i in range(self.n_states):
                probs[phase_names[i]] = {
                    phase_names[j]: self.model.transmat_[i, j]
                    for j in range(self.n_states)
                }
            
            return probs
        
        def get_emission_probabilities(self):
            """
            Gibt die empirischen Emissionswahrscheinlichkeiten zurück.
            Dies zeigt, mit welcher Wahrscheinlichkeit ein Code in einem latenten Zustand auftritt.
            """
            if self.model is None or not self.trained:
                return {}
            
            probs = {}
            phase_names = ['Greeting', 'Need', 'Consultation', 'Completion', 'Farewell']
            
            for i in range(self.n_states):
                code_probs = {}
                for j in range(self.n_features):
                    code = self.idx_to_code[j]
                    code_probs[code] = self.model.emissionprob_[i, j]
                # Sortiere nach Wahrscheinlichkeit
                code_probs = dict(sorted(code_probs.items(), key=lambda x: -x[1]))
                probs[phase_names[i]] = code_probs
            
            return probs
        
        def get_parameters_string(self):
            """Gibt die HMM-Parameter als String zurück"""
            if self.model is None:
                return "Kein HMM trainiert"
            
            lines = []
            lines.append("Startwahrscheinlichkeiten (empirisch):")
            for i in range(self.n_states):
                lines.append(f"  {self.state_names[i]}: {self.model.startprob_[i]:.3f}")
            
            lines.append("\nÜbergangsmatrix (empirisch):")
            for i in range(self.n_states):
                row = "  " + " ".join([f"{self.model.transmat_[i,j]:.3f}" 
                                       for j in range(self.n_states)])
                lines.append(f"{self.state_names[i]}: {row}")
            
            lines.append("\nTop-3 Emissionswahrscheinlichkeiten pro Zustand (empirisch):")
            for i in range(self.n_states):
                probs = self.model.emissionprob_[i]
                top_indices = np.argsort(probs)[-3:][::-1]
                top_symbols = []
                for idx in top_indices:
                    code = self.idx_to_code[idx]
                    sym = TerminalCoding.decode(code)
                    top_symbols.append(f"{sym}({code}): {probs[idx]:.3f}")
                lines.append(f"  {self.state_names[i]}: {', '.join(top_symbols)}")
            
            return '\n'.join(lines)
else:
    class ARSHiddenMarkovModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("hmmlearn nicht installiert")


# ============================================================================
# STATISTISCHE ERWEITERUNG (EMPIRISCHE WAHRSCHEINLICHKEITEN)
# ============================================================================

class StatisticalExtension:
    """
    Statistische Erweiterung für empirische Wahrscheinlichkeiten.
    
    Wichtig: Diese Klasse arbeitet NUR mit strukturell gültigen Codes.
    Strukturell ungültige Codes werden in der Statistik ignoriert (sie sind "falsch").
    """
    
    def __init__(self, automaton):
        self.automaton = automaton
        self.terminal_transitions = defaultdict(Counter)
        self.phase_transitions = defaultdict(Counter)
        self.terminal_counts = defaultdict(int)
        self.phase_counts = defaultdict(int)
        self.loops = []
        self.missing_elements = defaultdict(int)
        self.transcript_results = []
        self.unknown_codes = defaultdict(int)  # Zählt unbekannte Codes (strukturell falsch)
    
    def analyze_coded_chain(self, coded_chain, transcript_id):
        """
        Analysiert eine kodierte Kette statistisch.
        Strukturell ungültige Codes werden gezählt, aber nicht in der Statistik verwendet.
        """
        # 1. Strukturelle Validierung (unabhängig)
        valid, state, protocol, first_error = self.automaton.validate_chain(coded_chain)
        
        # 2. Zähle unbekannte Codes
        for code in coded_chain:
            if not TerminalCoding.is_valid_code(code):
                self.unknown_codes[code] += 1
        
        # 3. Statistische Analyse NUR für gültige Codes
        valid_codes = [code for code in coded_chain if TerminalCoding.is_valid_code(code)]
        
        if valid_codes:
            self._count_transitions(valid_codes)
            self._count_phases(valid_codes)
            self._detect_loops(valid_codes, transcript_id)
            self._check_missing_elements(valid_codes)
        
        result = {
            'transcript_id': transcript_id,
            'valid': valid,
            'state': state,
            'length': len(coded_chain),
            'valid_length': len(valid_codes),
            'unknown_codes': len(coded_chain) - len(valid_codes),
            'first_error': first_error,
            'protocol': protocol
        }
        self.transcript_results.append(result)
        
        return result
    
    def _count_transitions(self, valid_codes):
        """Zählt Übergänge zwischen Terminalzeichen (nur gültige Codes)"""
        for i in range(len(valid_codes) - 1):
            curr = valid_codes[i]
            next_sym = valid_codes[i + 1]
            self.terminal_transitions[curr][next_sym] += 1
            self.terminal_counts[curr] += 1
    
    def _count_phases(self, valid_codes):
        """Zählt Übergänge zwischen Phasen (nur gültige Codes)"""
        phases = [code[1:3] for code in valid_codes]
        for i in range(len(phases) - 1):
            curr_phase = phases[i]
            next_phase = phases[i + 1]
            self.phase_transitions[curr_phase][next_phase] += 1
            self.phase_counts[curr_phase] += 1
    
    def _detect_loops(self, valid_codes, transcript_id):
        """Erkennt Schleifen in der Sequenz (nur gültige Codes)"""
        for length in range(2, min(5, len(valid_codes) // 2 + 1)):
            for i in range(len(valid_codes) - 2 * length + 1):
                pattern = valid_codes[i:i+length]
                if valid_codes[i+length:i+2*length] == pattern:
                    pattern_tuple = tuple(pattern)
                    existing = False
                    for loop in self.loops:
                        if loop['pattern'] == pattern_tuple:
                            existing = True
                            loop['occurrences'] += 1
                            if transcript_id not in loop['transcripts']:
                                loop['transcripts'].append(transcript_id)
                            break
                    
                    if not existing:
                        self.loops.append({
                            'position': i,
                            'length': length,
                            'pattern': pattern_tuple,
                            'transcripts': [transcript_id],
                            'occurrences': 1
                        })
    
    def _check_missing_elements(self, valid_codes):
        """Prüft auf fehlende Elemente (nur gültige Codes)"""
        if not valid_codes:
            return
        
        # Fehlende Begrüßung?
        first = valid_codes[0]
        if first not in ['00000', '10000']:
            self.missing_elements['greeting'] += 1
        
        # Fehlende Verabschiedung?
        last = valid_codes[-1]
        if last not in ['01100', '11100']:
            self.missing_elements['farewell'] += 1
        
        # Ungewöhnliche Phasenfolgen
        phases = [code[1:3] for code in valid_codes]
        for i in range(len(phases) - 1):
            curr = phases[i]
            next_phase = phases[i + 1]
            phase_order = {'00': 0, '01': 1, '10': 2, '11': 3}
            if curr in phase_order and next_phase in phase_order:
                if phase_order[next_phase] < phase_order[curr]:
                    self.missing_elements['phase_regression'] += 1
    
    def get_terminal_probabilities(self):
        """
        Berechnet empirische Übergangswahrscheinlichkeiten auf Terminalebene.
        Dies sind Wahrscheinlichkeiten von 0.0 bis 1.0 basierend auf den Daten.
        """
        probs = {}
        for curr, targets in self.terminal_transitions.items():
            total = self.terminal_counts[curr]
            if total > 0:
                probs[curr] = {
                    next_sym: count / total 
                    for next_sym, count in targets.items()
                }
        return probs
    
    def get_phase_probabilities(self):
        """
        Berechnet empirische Übergangswahrscheinlichkeiten auf Phasenebene.
        Dies zeigt, mit welcher Wahrscheinlichkeit Phasenübergänge auftreten.
        """
        probs = {}
        for curr, targets in self.phase_transitions.items():
            total = self.phase_counts[curr]
            if total > 0:
                probs[curr] = {
                    next_phase: count / total 
                    for next_phase, count in targets.items()
                }
        return probs
    
    def get_terminal_frequencies(self):
        """
        Berechnet die relative Häufigkeit jedes Terminalzeichens.
        Ein Wert von 0.0 bedeutet: kommt nie vor (obwohl strukturell möglich)
        Ein Wert von 1.0 bedeutet: kommt immer vor (in jeder Position)
        """
        if not self.terminal_counts:
            return {}
        
        total = sum(self.terminal_counts.values())
        if total == 0:
            return {}
        
        return {
            code: count / total
            for code, count in self.terminal_counts.items()
        }
    
    def get_unknown_code_report(self):
        """Gibt einen Bericht über unbekannte Codes (strukturell falsch) aus"""
        if not self.unknown_codes:
            return "Keine unbekannten Codes gefunden."
        
        lines = ["Unbekannte Codes (nicht im Alphabet, daher strukturell falsch):"]
        for code, count in sorted(self.unknown_codes.items(), key=lambda x: -x[1]):
            lines.append(f"  {code}: {count}x")
        return "\n".join(lines)
    
    def get_loop_statistics(self):
        """Gibt Statistiken über Schleifen zurück"""
        return sorted(self.loops, key=lambda x: -x['occurrences'])
    
    def get_statistics(self):
        """Gibt alle statistischen Kennzahlen zurück"""
        return {
            'terminal_probabilities': self.get_terminal_probabilities(),
            'phase_probabilities': self.get_phase_probabilities(),
            'loops': self.get_loop_statistics(),
            'missing_elements': dict(self.missing_elements),
            'total_sequences': len(self.terminal_counts),
            'transcript_results': self.transcript_results
        }
    
    def reset(self):
        """Setzt alle statistischen Daten zurück"""
        self.terminal_transitions = defaultdict(Counter)
        self.phase_transitions = defaultdict(Counter)
        self.terminal_counts = defaultdict(int)
        self.phase_counts = defaultdict(int)
        self.loops = []
        self.missing_elements = defaultdict(int)
        self.transcript_results = []
        self.unknown_codes = defaultdict(int)
    
    def print_report(self):
        """Gibt einen statistischen Bericht aus"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("STATISTISCHE ANALYSE (EMPIRISCHE WAHRSCHEINLICHKEITEN)")
        lines.append("=" * 70)
        
        # 0. Unbekannte Codes (strukturell falsch)
        lines.append("\n0. UNBEKANNTE CODES (STRUKTURELL FALSCH):")
        if self.unknown_codes:
            for code, count in sorted(self.unknown_codes.items(), key=lambda x: -x[1]):
                lines.append(f"   {code}: {count}x")
        else:
            lines.append("   Keine unbekannten Codes gefunden.")
        
        # 1. Validierungsergebnisse
        lines.append("\n1. VALIDIERUNGSERGEBNISSE:")
        valid_count = sum(1 for r in self.transcript_results if r['valid'])
        total = len(self.transcript_results)
        lines.append(f"   Strukturell gültige Ketten: {valid_count}/{total} ({valid_count/total*100:.1f}%)")
        
        for result in self.transcript_results:
            status = "✓" if result['valid'] else "✗"
            error_info = ""
            if not result['valid'] and result['first_error']:
                error_info = f" (Fehler bei Pos {result['first_error']['position']}: {result['first_error']['code']})"
            lines.append(f"   Transkript {result['transcript_id']}: {status} "
                        f"(Länge: {result['length']}, gültige Codes: {result['valid_length']}, "
                        f"unbekannte: {result['unknown_codes']}{error_info})")
        
        # 2. Fehlende Elemente
        lines.append("\n2. FEHLENDE ELEMENTE (empirische Häufigkeit):")
        if self.missing_elements:
            for elem, count in self.missing_elements.items():
                if elem == 'greeting':
                    lines.append(f"   Fehlende Begrüßung: {count}x")
                elif elem == 'farewell':
                    lines.append(f"   Fehlende Verabschiedung: {count}x")
                elif elem == 'phase_regression':
                    lines.append(f"   Phasenrücksprünge: {count}x")
        else:
            lines.append("   Keine fehlenden Elemente")
        
        # 3. Terminal-Häufigkeiten (empirische Verteilung)
        lines.append("\n3. TERMINAL-HÄUFIGKEITEN (empirisch, 0.0-1.0):")
        frequencies = self.get_terminal_frequencies()
        if frequencies:
            for code, freq in sorted(frequencies.items(), key=lambda x: -x[1]):
                sym = TerminalCoding.decode(code)
                phase = TerminalCoding.get_phase_name(code)
                lines.append(f"   {sym} ({code}, {phase}): {freq:.3f}")
        
        # 4. Schleifen
        lines.append("\n4. ERKANNTE SCHLEIFEN (empirische Wiederholungen):")
        loops = self.get_loop_statistics()
        if loops:
            for i, loop in enumerate(loops[:5]):
                pattern_str = ' → '.join([TerminalCoding.decode(c) for c in loop['pattern']])
                codes_str = ','.join(loop['pattern'])
                lines.append(f"   {i+1}. Muster: {pattern_str}")
                lines.append(f"      Codes: {codes_str}")
                lines.append(f"      Länge: {loop['length']}, Vorkommen: {loop['occurrences']}x")
                lines.append(f"      in Transkripten: {', '.join(map(str, loop['transcripts']))}")
        else:
            lines.append("   Keine Schleifen erkannt")
        
        # 5. Phasen-Übergangswahrscheinlichkeiten (empirisch)
        lines.append("\n5. PHASEN-ÜBERGANGSWAHRSCHEINLICHKEITEN (empirisch):")
        phase_probs = self.get_phase_probabilities()
        phase_names = {'00': 'BG (Begrüßung)', '01': 'B (Bedarf)', 
                      '10': 'A (Abschluss)', '11': 'AV (Verabschiedung)'}
        
        for curr, targets in phase_probs.items():
            curr_name = phase_names.get(curr, curr)
            transitions = []
            for next_phase, prob in sorted(targets.items(), key=lambda x: -x[1]):
                next_name = phase_names.get(next_phase, next_phase)
                transitions.append(f"{next_name}: {prob:.3f}")
            lines.append(f"   {curr_name} → {', '.join(transitions)}")
        
        # 6. Terminal-Übergangswahrscheinlichkeiten (Top 10)
        lines.append("\n6. TERMINAL-ÜBERGANGSWAHRSCHEINLICHKEITEN (Top 10, empirisch):")
        term_probs = self.get_terminal_probabilities()
        all_transitions = []
        for curr, targets in term_probs.items():
            for next_sym, prob in targets.items():
                all_transitions.append((curr, next_sym, prob))
        
        all_transitions.sort(key=lambda x: -x[2])
        for i, (curr, next_sym, prob) in enumerate(all_transitions[:10]):
            curr_sym = TerminalCoding.decode(curr)
            next_sym_dec = TerminalCoding.decode(next_sym)
            lines.append(f"   {i+1}. {curr_sym} ({curr}) → {next_sym_dec} ({next_sym}): {prob:.3f}")
        
        return "\n".join(lines)
    
    def export_json(self, filename="statistik_export.json"):
        """Exportiert die Statistiken als JSON"""
        stats = {
            'unknown_codes': dict(self.unknown_codes),
            'terminal_frequencies': self.get_terminal_frequencies(),
            'phase_probabilities': self.get_phase_probabilities(),
            'loops': [
                {
                    'position': l['position'],
                    'length': l['length'],
                    'pattern': list(l['pattern']),
                    'transcripts': l['transcripts'],
                    'occurrences': l['occurrences']
                }
                for l in self.loops
            ],
            'missing_elements': dict(self.missing_elements),
            'transcript_results': [
                {
                    'transcript_id': r['transcript_id'],
                    'valid': r['valid'],
                    'length': r['length'],
                    'valid_length': r['valid_length'],
                    'unknown_codes': r['unknown_codes']
                }
                for r in self.transcript_results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        return filename


# ============================================================================
# ARS 2.0 - BASIS-GRAMMATIK
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
        self.optimization_running = False
    
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


# ============================================================================
# ARS 3.0 - GRAMMATIK MIT NONTERMINALEN (mit Flag zur Vermeidung doppelter Ausführung)
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
        self.induction_done = False  # Flag zur Vermeidung doppelter Ausführung
    
    def load_chains(self, chains, user_start_symbol=None):
        self.chains = [list(chain) for chain in chains]
        self.user_start_symbol = user_start_symbol
        all_symbols = set()
        for chain in chains:
            for symbol in chain:
                all_symbols.add(symbol)
        self.terminals = all_symbols
        self.induction_done = False
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
        if self.induction_done:
            print("Grammatik bereits induziert. Überspringe...")
            return self.chains
        
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
        
        self.induction_done = True
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
# PETRI-NETZE (gekürzt, aus ARSXAI2 übernommen)
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
# HYBRIDE INTEGRATION (gekürzt, aus ARSXAI2)
# ============================================================================

if CRF_AVAILABLE:
    class ARSCRFModel:
        def __init__(self):
            self.crf = CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
            )
        
        def extract_features(self, sequence, i):
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
            
            for offset in [-2, -1, 1, 2]:
                if 0 <= i + offset < len(sequence):
                    sym = sequence[i + offset]
                    features[f'context_{offset:+d}'] = sym
            
            if i > 0:
                features['bigram'] = f"{sequence[i-1]}_{sequence[i]}"
            
            return features
        
        def prepare_data(self, sequences):
            X = []
            y = []
            for seq in sequences:
                X_seq = [self.extract_features(seq, i) for i in range(len(seq))]
                y_seq = [sym for sym in seq]
                X.append(X_seq)
                y.append(y_seq)
            return X, y
        
        def fit(self, sequences):
            X, y = self.prepare_data(sequences)
            self.crf.fit(X, y)
            return self
        
        def predict(self, sequence):
            X = [self.extract_features(sequence, i) for i in range(len(sequence))]
            return self.crf.predict([X])[0]
        
        def get_top_features(self, n=20):
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
        def __init__(self):
            self.model = None
            self.embeddings = {}
            self.symbol_to_texts = self._create_text_mapping()
        
        def _create_text_mapping(self):
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
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                return True
            except:
                return False
        
        def compute_embeddings(self):
            if self.model is None:
                if not self.load_model():
                    return False
            for symbol, texts in self.symbol_to_texts.items():
                embeddings = self.model.encode(texts)
                self.embeddings[symbol] = np.mean(embeddings, axis=0)
            return True
        
        def similarity_matrix(self):
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
        def __init__(self, grammar_rules):
            self.grammar = grammar_rules
            self.graph = nx.DiGraph()
            self.build_graph()
        
        def build_graph(self):
            for nt, productions in self.grammar.items():
                for prod, prob in productions:
                    for sym in prod:
                        self.graph.add_edge(nt, sym, weight=prob)
        
        def centrality(self):
            return nx.degree_centrality(self.graph)
else:
    class GrammarGraph:
        def __init__(self, *args, **kwargs):
            raise ImportError("networkx nicht installiert")


class AttentionVisualizer:
    def __init__(self, chains):
        self.chains = chains
        self.bigram_probs = self._compute_bigram_probs()
    
    def _compute_bigram_probs(self):
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
# PLOT-FUNKTIONEN
# ============================================================================

def plot_petri_net(petri_net, filename="petri_net.png"):
    if not NETWORKX_AVAILABLE:
        print("networkx nicht verfügbar")
        return
    
    G = nx.DiGraph()
    for place in petri_net.places:
        G.add_node(place, type='place', shape='circle')
    for trans in petri_net.transitions:
        G.add_node(trans, type='transition', shape='box')
    for arc in petri_net.arcs:
        G.add_edge(arc['source'], arc['target'], weight=arc['weight'])
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(15, 10))
    
    place_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'place']
    nx.draw_networkx_nodes(G, pos, nodelist=place_nodes, 
                          node_color='lightblue', node_shape='o', 
                          node_size=1000)
    
    trans_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'transition']
    nx.draw_networkx_nodes(G, pos, nodelist=trans_nodes, 
                          node_color='lightgreen', node_shape='s', 
                          node_size=800)
    
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
    
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
        self.root.title("ARS 5.1 - Algorithmic Recursive Sequence Analysis")
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
        self.unknown_symbols = []  # Speichert unbekannte Symbole pro Kette
        
        # ARS-Objekte
        self.ars20 = ARS20()
        self.ars30 = GrammarInducer()
        self.petri_builder = None
        self.automaton = DialogueAutomaton()
        self.hmm_model = None
        self.crf_model = None
        self.semantic_validator = None
        self.grammar_graph = None
        self.attention_viz = None
        self.stats_extension = StatisticalExtension(self.automaton)
        
        # Flags für laufende Operationen
        self.optimization_running = False
        self.ars30_running = False
        
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
        
        self.tab_bayes = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bayes, text="Bayessche Netze")
        self.create_bayes_tab()
        
        self.tab_stats = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_stats, text="Statistische Analyse")
        self.create_statistics_tab()
        
        self.tab_petri = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_petri, text="Petri-Netze")
        self.create_petri_tab()
        
        self.tab_hybrid = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_hybrid, text="Hybrid")
        self.create_hybrid_tab()
        
        self.tab_gen = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_gen, text="Generierung")
        self.create_generation_tab()
    
    def create_ars20_tab(self):
        control = ttk.Frame(self.tab20)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="ARS 2.0 berechnen", 
                  command=self.run_ars20).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Optimierung starten", 
                  command=self.run_optimization).pack(side=tk.LEFT, padx=5)
        
        self.opt_progress = ttk.Progressbar(control, length=200, mode='determinate')
        self.opt_progress.pack(side=tk.LEFT, padx=10)
        
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
        control = ttk.Frame(self.tab_code)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Kodieren", 
                  command=self.encode_chains).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Alphabet anzeigen", 
                  command=self.show_alphabet).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Dekodieren", 
                  command=self.decode_chains).pack(side=tk.LEFT, padx=5)
        
        self.text_code = scrolledtext.ScrolledText(self.tab_code, font=('Courier', 10))
        self.text_code.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_automaton_tab(self):
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
    
    def create_bayes_tab(self):
        control = ttk.Frame(self.tab_bayes)
        control.pack(fill=tk.X, pady=5)
        
        if self.module_status['hmmlearn']:
            ttk.Button(control, text="HMM initialisieren", 
                      command=self.init_hmm).pack(side=tk.LEFT, padx=5)
            ttk.Button(control, text="HMM trainieren (empirisch)", 
                      command=self.train_hmm).pack(side=tk.LEFT, padx=5)
            ttk.Button(control, text="Dekodiere Transkript 1", 
                      command=self.decode_hmm).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Label(control, text="hmmlearn nicht verfügbar", 
                     foreground="red").pack(side=tk.LEFT, padx=5)
        
        self.text_bayes = scrolledtext.ScrolledText(self.tab_bayes, font=('Courier', 10))
        self.text_bayes.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_statistics_tab(self):
        control = ttk.Frame(self.tab_stats)
        control.pack(fill=tk.X, pady=5)
        
        ttk.Button(control, text="Statistische Analyse starten", 
                  command=self.run_statistical_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Bericht exportieren (JSON)", 
                  command=self.export_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Statistik zurücksetzen", 
                  command=self.reset_statistics).pack(side=tk.LEFT, padx=5)
        
        self.text_stats = scrolledtext.ScrolledText(self.tab_stats, font=('Courier', 10))
        self.text_stats.pack(fill=tk.BOTH, expand=True, pady=5)
    
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
    
    def create_hybrid_tab(self):
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
        self.unknown_symbols = []
        
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
    
    def show_alphabet(self):
        """Zeigt das definierte Alphabet an"""
        self.text_code.insert(tk.END, "\n" + "=" * 70 + "\n")
        self.text_code.insert(tk.END, TerminalCoding.get_alphabet_display())
        self.text_code.insert(tk.END, "\n\n")
    
    def encode_chains(self):
        """Wandelt alle Ketten in 5-Bit-Kodierung um und erkennt unbekannte Symbole"""
        if not self.chains:
            return
        
        self.coded_chains = []
        self.unknown_symbols = []
        
        self.text_code.delete("1.0", tk.END)
        self.text_code.insert(tk.END, "5-BIT-KODIERUNG DER TERMINALZEICHEN\n")
        self.text_code.insert(tk.END, "=" * 70 + "\n\n")
        self.text_code.insert(tk.END, "Schema: [Sprecher][Phase][Unterphase]\n")
        self.text_code.insert(tk.END, "S: 0=Kunde, 1=Verkäufer\n")
        self.text_code.insert(tk.END, "Phase: 00=BG, 01=B, 10=A, 11=AV\n")
        self.text_code.insert(tk.END, "Unterphase: 00=Basis, 01=Folge\n\n")
        
        for i, chain in enumerate(self.chains, 1):
            coded, unknown = TerminalCoding.encode_chain(chain)
            self.coded_chains.append(coded)
            self.unknown_symbols.append(unknown)
            
            self.text_code.insert(tk.END, f"Transkript {i}:\n")
            self.text_code.insert(tk.END, f"  Original: {', '.join(chain)}\n")
            self.text_code.insert(tk.END, f"  Kodiert:  {', '.join(coded)}\n")
            if unknown:
                self.text_code.insert(tk.END, f"  ⚠️ Unbekannte Symbole (nicht im Alphabet): {', '.join(unknown)}\n")
            self.text_code.insert(tk.END, "\n")
        
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
            valid, state, protocol, first_error = self.automaton.validate_chain(coded)
            
            self.text_auto.insert(tk.END, f"\n{'='*50}\n")
            self.text_auto.insert(tk.END, f"VALIDIERUNG TRANSKRIPT {i+1}\n")
            self.text_auto.insert(tk.END, f"{'='*50}\n")
            self.text_auto.insert(tk.END, f"Ergebnis: {'✓ STRUKTURELL GÜLTIG' if valid else '✗ STRUKTURELL UNGÜLTIG'}\n")
            self.text_auto.insert(tk.END, f"Endzustand: {state}\n\n")
            
            if first_error:
                self.text_auto.insert(tk.END, f"⚠️ Erster Fehler bei Position {first_error['position']}: {first_error['code']}\n")
                self.text_auto.insert(tk.END, f"   {first_error['explanation']}\n\n")
            
            self.text_auto.insert(tk.END, "ENTSCHEIDUNGSPFAD:\n")
            
            for step in protocol:
                self.text_auto.insert(tk.END, f"  Schritt {step['position']}: {step['code']} ({step['symbol']}, {step['phase']})\n")
                self.text_auto.insert(tk.END, f"    → {step['state']}\n")
                self.text_auto.insert(tk.END, f"    {step['explanation']}\n")
            
            if valid:
                valid_count += 1
        
        self.text_auto.insert(tk.END, f"\n{'='*50}\n")
        self.text_auto.insert(tk.END, f"GESAMTERGEBNIS: {valid_count}/{len(self.coded_chains)} strukturell gültig\n")
        self.status_var.set(f"{valid_count}/{len(self.coded_chains)} Ketten strukturell gültig")
    
    def validate_chain(self, index, name):
        """Validiert eine einzelne Kette"""
        coded = self.coded_chains[index]
        valid, state, protocol, first_error = self.automaton.validate_chain(coded)
        
        self.text_auto.delete("1.0", tk.END)
        self.text_auto.insert(tk.END, f"{'='*50}\n")
        self.text_auto.insert(tk.END, f"VALIDIERUNG {name}\n")
        self.text_auto.insert(tk.END, f"{'='*50}\n")
        self.text_auto.insert(tk.END, f"Ergebnis: {'✓ STRUKTURELL GÜLTIG' if valid else '✗ STRUKTURELL UNGÜLTIG'}\n")
        self.text_auto.insert(tk.END, f"Endzustand: {state}\n\n")
        
        if first_error:
            self.text_auto.insert(tk.END, f"⚠️ Erster Fehler bei Position {first_error['position']}: {first_error['code']}\n")
            self.text_auto.insert(tk.END, f"   {first_error['explanation']}\n\n")
        
        self.text_auto.insert(tk.END, "ENTSCHEIDUNGSPFAD:\n")
        
        for step in protocol:
            self.text_auto.insert(tk.END, f"  Schritt {step['position']}: {step['code']} ({step['symbol']}, {step['phase']})\n")
            self.text_auto.insert(tk.END, f"    → {step['state']}\n")
            self.text_auto.insert(tk.END, f"    {step['explanation']}\n")
        
        self.status_var.set(f"{name}: {'gültig' if valid else 'ungültig'}")
    
    def init_hmm(self):
        """Initialisiert HMM"""
        if not self.module_status['hmmlearn']:
            messagebox.showerror("Fehler", "hmmlearn nicht installiert!")
            return
        
        if not self.coded_chains:
            messagebox.showerror("Fehler", "Keine kodierten Ketten vorhanden!")
            return
        
        try:
            self.hmm_model = ARSHiddenMarkovModel(n_states=5)
            result = self.hmm_model.initialize_from_structure(self.coded_chains)
            
            if result is None:
                messagebox.showerror("Fehler", "HMM-Initialisierung fehlgeschlagen - keine gültigen Daten?")
                return
            
            self.text_bayes.delete("1.0", tk.END)
            self.text_bayes.insert(tk.END, "HMM initialisiert (basierend auf Struktur):\n\n")
            self.text_bayes.insert(tk.END, self.hmm_model.get_parameters_string())
            self.status_var.set("HMM initialisiert")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei HMM-Initialisierung:\n{str(e)}")
    
    def train_hmm(self):
        """Trainiert HMM auf empirischen Daten"""
        if not self.module_status['hmmlearn']:
            messagebox.showerror("Fehler", "hmmlearn nicht installiert!")
            return
        
        if not self.coded_chains:
            messagebox.showerror("Fehler", "Keine kodierten Ketten vorhanden!")
            return
        
        if not self.hmm_model:
            self.hmm_model = ARSHiddenMarkovModel(n_states=5)
            self.hmm_model.initialize_from_structure(self.coded_chains)
        
        self.status_var.set("Trainiere HMM auf empirischen Daten...")
        self.progress_bar.start()
        
        def run():
            try:
                self.hmm_model.fit(self.coded_chains, n_iter=100)
                
                def update_display():
                    self.text_bayes.insert(tk.END, "\n" + "="*50 + "\n")
                    self.text_bayes.insert(tk.END, "HMM NACH EMPIRISCHEM TRAINING:\n\n")
                    self.text_bayes.insert(tk.END, self.hmm_model.get_parameters_string())
                    self.status_var.set("HMM-Training abgeschlossen (empirische Wahrscheinlichkeiten)")
                    self.progress_bar.stop()
                
                self.safe_gui_update(update_display)
            except Exception as e:
                def error_display(err_msg):
                    messagebox.showerror("Fehler", f"HMM-Training fehlgeschlagen:\n{err_msg}")
                    self.progress_bar.stop()
                
                self.safe_gui_update(lambda: error_display(str(e)))
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def decode_hmm(self):
        """Dekodiert Transkript 1 mit HMM"""
        if not self.module_status['hmmlearn']:
            messagebox.showerror("Fehler", "hmmlearn nicht installiert!")
            return
        
        if not self.hmm_model or not self.hmm_model.model or not self.hmm_model.trained:
            messagebox.showerror("Fehler", "Kein trainiertes HMM vorhanden!")
            return
        
        if not self.coded_chains:
            return
        
        try:
            states, prob = self.hmm_model.decode(self.coded_chains[0])
            
            if states is None:
                messagebox.showerror("Fehler", "Dekodierung fehlgeschlagen")
                return
            
            self.text_bayes.insert(tk.END, "\n" + "="*50 + "\n")
            self.text_bayes.insert(tk.END, f"VITERBI-DEKODIERUNG TRANSKRIPT 1 (p={prob:.4f}):\n")
            self.text_bayes.insert(tk.END, "="*50 + "\n")
            
            for i, (code, state) in enumerate(zip(self.coded_chains[0], states)):
                if state == -1:
                    state_name = "UNBEKANNT (strukturell ungültig)"
                else:
                    state_name = self.hmm_model.state_names.get(state, f"State {state}")
                
                if TerminalCoding.is_valid_code(code):
                    sym = TerminalCoding.decode(code)
                    self.text_bayes.insert(tk.END, f"{i+1:2d}: {code} ({sym}) -> {state_name}\n")
                else:
                    self.text_bayes.insert(tk.END, f"{i+1:2d}: {code} (UNBEKANNT) -> {state_name}\n")
                
        except Exception as e:
            messagebox.showerror("Fehler", f"Dekodierung fehlgeschlagen:\n{str(e)}")
    
    def run_statistical_analysis(self):
        """Führt die statistische Analyse durch"""
        if not hasattr(self, 'coded_chains') or not self.coded_chains:
            messagebox.showerror("Fehler", "Keine kodierten Ketten vorhanden!")
            return
        
        self.stats_extension.reset()
        self.text_stats.delete("1.0", tk.END)
        self.text_stats.insert(tk.END, "STATISTISCHE ANALYSE LÄUFT...\n")
        self.root.update()
        
        for i, coded in enumerate(self.coded_chains):
            result = self.stats_extension.analyze_coded_chain(coded, i+1)
            self.text_stats.insert(tk.END, f"✓ Transkript {i+1} analysiert\n")
            self.root.update()
        
        self.text_stats.delete("1.0", tk.END)
        self.text_stats.insert(tk.END, self.stats_extension.print_report())
        self.status_var.set("Statistische Analyse abgeschlossen")
    
    def export_statistics(self):
        """Exportiert die Statistiken als JSON"""
        if not self.stats_extension.transcript_results:
            messagebox.showerror("Fehler", "Keine statistischen Daten vorhanden!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"statistik_{timestamp}.json"
        
        try:
            exported_file = self.stats_extension.export_json(filename)
            self.text_stats.insert(tk.END, f"\n\nStatistik exportiert als: {exported_file}\n")
            messagebox.showinfo("Export erfolgreich", f"Statistik gespeichert als:\n{exported_file}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Export fehlgeschlagen:\n{str(e)}")
    
    def reset_statistics(self):
        """Setzt die statistische Analyse zurück"""
        self.stats_extension.reset()
        self.text_stats.delete("1.0", tk.END)
        self.text_stats.insert(tk.END, "Statistik zurückgesetzt.\n")
        self.status_var.set("Statistik zurückgesetzt")
    
    def show_ars20_preview(self):
        self.text20.delete("1.0", tk.END)
        self.text20.insert(tk.END, self.ars20.print_grammar())
    
    def run_ars20(self):
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        self.show_ars20_preview()
        self.status_var.set("ARS 2.0 abgeschlossen")
    
    def run_optimization(self):
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
                def error_display(err_msg):
                    messagebox.showerror("Fehler", f"Optimierung fehlgeschlagen:\n{err_msg}")
                    self.optimization_running = False
                
                self.safe_gui_update(lambda: error_display(str(e)))
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def run_ars30(self):
        if not self.chains:
            messagebox.showerror("Fehler", "Keine Daten geladen!")
            return
        
        if self.ars30_running:
            messagebox.showinfo("Info", "Grammatikinduktion läuft bereits")
            return
        
        self.ars30_running = True
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
                    self.ars30_running = False
                
                self.safe_gui_update(update_display)
            except Exception as e:
                def error_display(err_msg):
                    messagebox.showerror("Fehler", f"Grammatikinduktion fehlgeschlagen:\n{err_msg}")
                    self.ars30_progress.stop()
                    self.ars30_running = False
                
                self.safe_gui_update(lambda: error_display(str(e)))
        
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
    
    def train_crf(self):
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
                
                matrix, symbols = self.semantic_validator.similarity_matrix()
                if matrix is not None:
                    self.plot_thread.plot(plot_similarity_matrix, matrix, symbols)
                
                self.status_var.set("Semantische Validierung abgeschlossen")
            else:
                self.text_hybrid.insert(tk.END, "Fehler beim Laden des Modells\n")
        except Exception as e:
            messagebox.showerror("Fehler", f"Semantische Validierung fehlgeschlagen:\n{str(e)}")
    
    def build_grammar_graph(self):
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
            
            self.plot_thread.plot(plot_grammar_graph, self.grammar_graph.graph)
            
            self.status_var.set("Grammatik-Graph erstellt")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Erstellen des Grammatik-Graphen:\n{str(e)}")
    
    def visualize_attention(self):
        if not self.chains:
            return
        
        try:
            self.attention_viz = AttentionVisualizer(self.chains)
            
            self.text_hybrid.insert(tk.END, "\n" + "="*50 + "\n")
            self.text_hybrid.insert(tk.END, "Attention visualisiert (siehe plot)\n")
            
            attention = self.attention_viz.attention_weights(self.chains[0])
            self.plot_thread.plot(plot_attention, attention, self.chains[0])
            
            self.status_var.set("Attention visualisiert")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei Attention-Visualisierung:\n{str(e)}")
    
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
KBG, VBG, KBBd, VBBd, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV
KBG, VBG, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV
KBG, VBG, KBBd, VBBd, KBA, VBA, KAE, VAE, KBA, VBA, VAA, KAA, VAV, KAV
KBG, VBG, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV
KBG, VBG, KBBd, VBBd, KBBd, VBBd, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV
KBG, VBG, KBBd, VBBd, KBA, VBA, KAE, VAE, VAA, KAA, VAV, KAV
KBG, VBG, KBBd, VBBd, KBA, VBA, KBBd, VBBd, KBA, VBA, KAE, VAE, KBA, VBA, VAA, KAA, VAV, KAV"""
        
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", example)
        self.parse_input()
    
    def show_about(self):
        about = """ARS 5.1 - Algorithmic Recursive Sequence Analysis

Erweiterte Version mit:
- 5-Bit-Kodierung der Terminalzeichen
- Formalem Entscheidungsautomaten (strukturelle Korrektheit)
- Bayes'schen Netzen (HMM) mit korrekter one-hot-encoding für hmmlearn >= 0.3.0
- Statistischer Analyse empirischer Wahrscheinlichkeiten (0.0-1.0)
- Expliziter Trennung von Struktur (deterministisch) und Empirie (probabilistisch)

Ein Terminalzeichen ist NUR DANN FALSCH, wenn es:
1. Nicht im definierten Alphabet vorkommt
2. Die strukturellen Regeln verletzt (Phasenfolge, Sprecherwechsel)

Die Wahrscheinlichkeit sagt nur etwas über die empirische Häufigkeit aus.

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
