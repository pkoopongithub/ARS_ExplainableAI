# ARSXAI7 - Algorithmic Recursive Sequence Analysis with Explainable AI


## 📖 Inhaltsverzeichnis

1. [Einführung und Zielsetzung](#-einführung-und-zielsetzung)
2. [Installation](#-installation)
3. [Theoretische Grundlagen](#-theoretische-grundlagen)
4. [Algorithmen und Datenstrukturen](#-algorithmen-und-datenstrukturen)
5. [Dateiformate](#-dateiformate)
6. [Benutzeroberfläche](#-benutzeroberfläche)
7. [Bedienungsanleitung](#-bedienungsanleitung)
8. [Ausgabeformate](#-ausgabeformate)
9. [Beispiele](#-beispiele)
10. [Fehlerbehebung](#-fehlerbehebung)
11. [Mitwirken](#-mitwirken)
12. [Lizenz](#-lizenz)

## 🎯 Einführung und Zielsetzung

ARSXAI7 ist eine universelle Analyseplattform für sequenzielle Daten mit Fokus auf **Explainable AI (XAI)**. Das Programm wurde entwickelt, um aus beliebigen Terminalzeichenketten (z.B. Dialogtranskripte, Prozessabläufe, Verhaltenssequenzen) automatisch strukturelle Regeln abzuleiten und diese für den Menschen nachvollziehbar zu erklären.

### Kernziele

- **Universalität**: Analyse beliebiger Sequenzen, nicht nur vordefinierter Domänen
- **Automatische Strukturableitung**: Extraktion von Kodierungen und Regeln aus rohen Daten
- **Erklärbarkeit**: Transparente Darstellung der Ableitungsprozesse und Entscheidungen
- **Multiple Perspektiven**: Vergleich verschiedener Ableitungsstrategien
- **Progressive Learning**: Kontinuierliche Anpassung an neue Daten

### Hauptanwendungsgebiete

- **Dialoganalyse**: Untersuchung von Gesprächsstrukturen (Verkauf, Beratung, Therapie)
- **Prozessanalyse**: Workflow-Muster, Produktionsabläufe
- **Verhaltensanalyse**: Aktionssequenzen, Interaktionsmuster
- **Code-Analyse**: Programmabläufe, API-Aufrufsequenzen
- **Wissenschaftliche Forschung**: Sequenzmuster in beliebigen Domänen

## 💻 Installation

### Systemvoraussetzungen

- Python 3.8 oder höher
- 4 GB RAM (empfohlen)
- Betriebssystem: Windows, macOS, Linux

### Abhängigkeiten

Das Programm installiert automatisch folgende Pakete:

```bash
# Kernpakete
numpy          # Numerische Berechnungen
scipy          # Statistische Funktionen
matplotlib     # Visualisierung
tkinter        # GUI (standardmäßig in Python enthalten)

# Optionale ML-Pakete (für erweiterte Analysen)
hmmlearn                    # Hidden Markov Models
sklearn-crfsuite           # Conditional Random Fields
sentence-transformers      # Semantische Embeddings
networkx                    # Graph-Analyse
torch                       # Deep Learning Backend
seaborn                     # Erweiterte Visualisierung
graphviz                    # Automaten-Visualisierung
tabulate                    # Tabellen-Export
```

### Installation

```bash
# Repository klonen
git clone https://github.com/yourusername/ARSXAI7.git
cd ARSXAI7

# Programm starten (Pakete werden automatisch installiert)
python ARSXAI7.py
```

## 📚 Theoretische Grundlagen

### 1. **5-Bit-Kodierung**

Die 5-Bit-Kodierung ist ein flexibles Schema zur Strukturierung von Symbolen:

```
Bit-Struktur: [S][P1P2][U1U2]

- Bit 1 (Sprecher):     0 = Kunde / 1 = Verkäufer (oder andere Dichotomie)
- Bits 2-3 (Phase):     00, 01, 10, 11 für verschiedene Phasen
- Bits 4-5 (Subphase):  00 = Basis / 01 = Folge (für Wiederholungen)
```

Die konkrete Bedeutung der Bits wird automatisch aus den Daten abgeleitet und kann je nach Domäne variieren.

### 2. **Ableitungsstrategien**

ARSXAI7 verwendet drei komplementäre Strategien:

| Strategie | Prinzip | Stärke |
|-----------|---------|--------|
| **Positionsbasiert** | Analyse der durchschnittlichen Position in Sequenzen | Gut für lineare Prozesse |
| **Musterbasiert** | Erkennung wiederkehrender Pattern | Gut für zyklische Strukturen |
| **Statistisch** | Häufigkeits- und Korrelationsanalyse | Gut für probabilistische Modelle |

### 3. **Konfidenzmetriken**

Jede Ableitung wird mit einer Konfidenz versehen:

- **Hohe Konfidenz** (>0.7): Konsistent in vielen Daten beobachtet
- **Mittlere Konfidenz** (0.4-0.7): Mehrfach, aber nicht vollständig konsistent
- **Niedrige Konfidenz** (<0.4): Basierend auf wenigen Beobachtungen

### 4. **Generischer Automat**

Der Automat lernt Zustände und Übergänge aus den Daten:

- Zustände werden aus Positionen abgeleitet
- Übergänge aus beobachteten Symbolfolgen
- Akzeptierende Zustände aus Kettenenden
- Konfidenz für jede Transition

## 🔧 Algorithmen und Datenstrukturen

### Kernklassen

#### `DataValidator`
```python
- validate_chains(chains)      # Prüft Datenqualität
- group_similar_symbols()      # Erkennt Tippfehler
- suggest_corrections()        # Generiert Korrekturvorschläge
```

#### `CodingStrategy` (Basisklasse)
```python
- derive(chains)               # Leitet Kodierung ab
- explain(symbol, code_data)   # Erklärt die Ableitung
- confidence                   # Konfidenzwert (0-1)
```

#### `PositionBasedCoding`
```python
# Algorithmus
1. Sammle Positionsstatistiken pro Symbol
2. Berechne durchschnittliche Position
3. Normalisiere auf Phasen (0-4)
4. Bestimme Subphase aus Wiederholungen
5. Generiere 5-Bit-Code
```

#### `PatternBasedCoding`
```python
# Algorithmus
1. Finde wiederkehrende Pattern (Länge 2-5)
2. Analysiere Nachbarschaftsbeziehungen
3. Bestimme Phase aus typischen Nachbarn
4. Subphase aus Pattern-Teilnahme
5. Generiere 5-Bit-Code
```

#### `StatisticalBasedCoding`
```python
# Algorithmus
1. Berechne Häufigkeiten pro Symbol
2. Analysiere Erst-/Letztpositionen
3. Erstelle Übergangsmatrix
4. Phase aus Positionsverteilung
5. Subphase aus Transitionsvielfalt
```

#### `GenericDialogueAutomaton`
```python
- learn_from_chains(chains)    # Lernt Regeln aus Daten
- create_states_from_positions() # Generiert Zustände
- learn_transitions()           # Lernt Übergänge
- validate_chain(chain)         # Validiert eine Kette
- transition(symbol)            # Führt Übergang aus
```

#### `InteractiveExplainer`
```python
- why_this_coding(symbol)       # Erklärt Kodierung
- why_this_transition(from, sym, to) # Erklärt Übergang
- what_if(symbol, alt_code)     # Simuliert Alternativen
```

#### `ProgressiveLearner`
```python
- incorporate_new_data(chains, coding, rules) # Lernt dazu
- detect_changes(prev, current) # Erkennt Änderungen
- show_evolution()               # Zeigt Versionshistorie
```

### Datenstrukturen

```python
# Kodierungsergebnis
{
    'code': '01001',           # 5-Bit-Code
    'agreement': 0.85,         # Übereinstimmung der Strategien
    'confidence': 0.92,        # Konfidenzwert
    'evidence': [               # Belege pro Strategie
        {
            'strategy': 'Positionsbasiert',
            'avg_position': 5.3,
            'phase_norm': 0.45
        }
    ]
}

# Automaten-Regel
{
    ('q_phase_2', 'KBG'): 'q_phase_3',  # (Zustand, Symbol) -> Folgezustand
    'confidence': 0.87                    # Konfidenz der Regel
}

# Versionsinformation
{
    'timestamp': '2024-01-15T10:30:00',
    'n_chains': 25,
    'coding': {...},           # Kodierung zu diesem Zeitpunkt
    'changes': [...]            # Änderungen zur Vorversion
}
```

## 📁 Dateiformate

### Eingabeformat (Transkriptdatei)

```txt
# Kommentare beginnen mit #
# Trennzeichen: Komma, Semikolon oder Leerzeichen

# Transkript 1: Beispiel
KBG, VBG, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV

# Transkript 2: Mit Wiederholung
KBG, VBG, KBBd, VBBd, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV

# Leerzeilen werden ignoriert
```

**Formatregeln:**
- Eine Kette pro Zeile
- Kommentare mit `#` am Zeilenanfang
- Symbole durch Trennzeichen getrennt
- Leere Zeilen werden ignoriert

### Exportformate

#### JSON (maschinenlesbar)
```json
{
  "coding": {
    "KBG": {
      "code": "00000",
      "agreement": 1.0,
      "confidence": 0.95,
      "evidence": [...]
    }
  },
  "terminals": ["KBG", "VBG", ...],
  "chains": [...],
  "timestamp": "2024-01-15T10:30:00"
}
```

#### CSV (Tabellenkalkulation)
```csv
Symbol,Code,Konfidenz
KBG,00000,0.95
VBG,10000,0.92
KBBd,00100,0.88
```

#### HTML (Interaktiver Bericht)
Generiert einen vollständigen Analysebericht mit:
- Kodierungstabellen
- Konfidenzvisualisierung
- Statistiken
- Export-Datum

#### LaTeX (Wissenschaftlich)
```latex
\documentclass{article}
\begin{document}
\section{ARSXAI7 Analyseergebnisse}
\begin{tabular}{lll}
Symbol & Code & Konfidenz \\
KBG & 00000 & 95\% \\
...
\end{tabular}
\end{document}
```

## 🖥️ Benutzeroberfläche

### Hauptfenster

```
┌─────────────────────────────────────────────────────────────┐
│ ARSXAI7 - Algorithmic Recursive Sequence Analysis           │
├──────────────────────┬──────────────────────────────────────┤
│ EINGABE              │ AUSGABE (Notebook-Tabs)              │
│                      │                                      │
│ [Trennzeichen-Auswahl│ ┌─────────────────────────────────┐ │
│  ○ Komma ○ Semikolon │ │ Kodierung | Automat | XAI | ... │ │
│  ○ Leerzeichen ○ | ] │ └─────────────────────────────────┘ │
│                      │                                      │
│ Transkripte:         │ Kodierungsergebnisse:               │
│ ┌──────────────────┐ │ KBG: 00000 (Konsens: 100%)         │
│ │ KBG, VBG, KBBd,  │ │ VBG: 10000 (Konsens: 100%)         │
│ │ KBBd, VBBd, KBA, │ │ KBBd: 00100 (Konsens: 67%) ⚠️       │
│ │ VAA, KAA, VAV    │ │ ...                                 │
│ └──────────────────┘ │                                      │
│                      │                                      │
│ [Datei laden] [Parsen]                                      │
├─────────────────────────────────────────────────────────────┤
│ Status: 15 Ketten geladen                     [======>  ]  │
└─────────────────────────────────────────────────────────────┘
```

### Menüstruktur

```
Datei
├── Transkripte laden
├── Beispiel laden
├── Exportieren
│   ├── JSON
│   ├── CSV
│   ├── HTML (Bericht)
│   └── LaTeX
└── Beenden

Analyse
├── Alle Strategien anwenden
├── Automaten lernen
└── Validierung durchführen

XAI
├── Erklärung für Symbol
├── Regeln erklären
└── Was-wäre-wenn Simulation

Visualisierung
├── Kodierungsvergleich
├── Konfidenz-Heatmap
└── Automaten-Graph

Hilfe
├── Modulstatus
├── Evolution anzeigen
└── Über
```

### Tabs

| Tab | Funktion |
|-----|----------|
| **Kodierung** | Ergebnisse aller Ableitungsstrategien |
| **Automat** | Gelernte Automaten-Regeln und Validierung |
| **XAI** | Interaktive Erklärungen und Simulationen |
| **Statistiken** | Kennzahlen und Verteilungen |
| **Evolution** | Versionshistorie und Änderungen |

## 📘 Bedienungsanleitung

### Schritt-für-Schritt

#### 1. **Daten laden**

**Methode A - Datei laden:**
- Klicken Sie auf "Datei laden"
- Wählen Sie eine Textdatei mit Transkripten
- Das Programm parst automatisch

**Methode B - Direkteingabe:**
- Geben Sie Ketten direkt ins Textfeld ein
- Eine Kette pro Zeile
- Kommentare mit `#` möglich

**Methode C - Beispiel:**
- Klicken Sie auf "Beispiel"
- Lädt vordefinierte Verkaufsgespräche

#### 2. **Trennzeichen wählen**

Wählen Sie das in Ihrer Datei verwendete Trennzeichen:
- **Komma (,)**: `KBG, VBG, KBBd`
- **Semikolon (;)**: `KBG; VBG; KBBd`
- **Leerzeichen**: `KBG VBG KBBd`
- **Benutzerdefiniert**: z.B. `|`

#### 3. **Analyse starten**

Nach dem Laden startet die Analyse automatisch:

1. **Validierung**: Prüft Datenqualität
2. **Kodierungsstrategien**: Alle drei Strategien werden angewandt
3. **Konsensbildung**: Mehrheitsentscheidung über Kodierung
4. **Automaten lernen**: Regeln werden extrahiert

#### 4. **Ergebnisse erkunden**

**Kodierungs-Tab:**
- Vergleichen Sie die Ergebnisse der drei Strategien
- Sehen Sie den Konsens mit Übereinstimmungswerten
- ⚠️ markiert unsichere Kodierungen

**XAI-Tab:**
- Geben Sie ein Symbol ein
- Klicken Sie "Warum diese Kodierung?"
- Erhalten Sie detaillierte Erklärung mit Belegen

**Was-wäre-wenn Simulation:**
1. Klicken Sie "Was-wäre-wenn"
2. Geben Sie Symbol und alternativen Code ein
3. Sehen Sie die Auswirkungen der Änderung

#### 5. **Visualisierungen**

**Kodierungsvergleich:**
- Menü: Visualisierung → Kodierungsvergleich
- Zeigt Heatmap der 5-Bit-Codes aller Strategien

**Konfidenz-Heatmap:**
- Menü: Visualisierung → Konfidenz-Heatmap
- Visualisiert Sicherheit der Automaten-Regeln

**Automaten-Graph:**
- Menü: Visualisierung → Automaten-Graph
- Zeigt Zustände und Übergänge mit Konfidenzen

#### 6. **Exportieren**

1. Menü: Datei → Exportieren
2. Wählen Sie Format:
   - **JSON**: Für Weiterverarbeitung
   - **CSV**: Für Excel/Tabellenkalkulation
   - **HTML**: Interaktiver Bericht
   - **LaTeX**: Für wissenschaftliche Publikationen

#### 7. **Progressive Learning**

Bei jeder neuen Analyse:
- Version wird gespeichert
- Änderungen werden erkannt
- Evolution unter "Evolution"-Tab einsehbar

### Tastaturkürzel

| Aktion | Kurzbefehl |
|--------|------------|
| Datei laden | `Strg+O` |
| Parsen | `Strg+Enter` |
| Alle Strategien | `Strg+A` |
| Erklärung | `Strg+E` |
| Export | `Strg+S` |
| Beenden | `Strg+Q` |

## 📊 Ausgabeformate

### Kodierungsergebnisse

```
ERGEBNISSE DER KODIERUNGSSTRATEGIEN
======================================================================

Positionsbasierte Kodierung:
----------------------------------------
Konfidenz: 78%

  KBG: 00000
  VBG: 10000
  KBBd: 00100
  VBBd: 10100
  KBA: 00101
  VBA: 10101
  ...

Musterbasierte Kodierung:
----------------------------------------
Konfidenz: 82%

  KBG: 00000
  VBG: 10000
  KBBd: 00101
  VBBd: 10101
  ...

KONSENS-KODIERUNG (Mehrheitsentscheidung)
======================================================================
✓ KBG: 00000 (Übereinstimmung: 100%)
✓ VBG: 10000 (Übereinstimmung: 100%)
⚠️ KBBd: 00100 (Übereinstimmung: 67%)
```

### Automaten-Regeln

```
GELERNTE AUTOMATEN-REGELN
======================================================================
  q_start + KBG → q_phase_0 (Konfidenz: 95%)
  q_phase_0 + VBG → q_phase_0 (Konfidenz: 88%)
  q_phase_0 + KBBd → q_phase_1 (Konfidenz: 92%)
  q_phase_1 + VBBd → q_phase_1 (Konfidenz: 85%)
  q_phase_1 + KBA → q_phase_1 (Konfidenz: 78%)
  q_phase_1 + VAA → q_phase_2 (Konfidenz: 72%)
  ...

Akzeptierende Zustände: q_phase_4
```

### XAI-Erklärung

```
🔍 Erklärung für Kodierung von 'KBBd':
======================================================================
Konsens-Kodierung: 00100
Übereinstimmung: 67%

Einzelne Strategien:

  📊 Positionsbasierte Kodierung:
    Code: 00100
    → Sprecher: Kunde | Phase: Bedarf (durchschnittliche Position 3.2) | Subphase: Basis (seltene Wiederholungen)

  📊 Musterbasierte Kodierung:
    Code: 00101
    → Sprecher: Kunde | Phase: Bedarf (häufige Nachbarn: VBBd(5x)) | Subphase: Folge (Teil von 3 wiederkehrenden Mustern)

  📊 Statistisch basierte Kodierung:
    Code: 00100
    → Sprecher: Kunde | Phase: Bedarf (beginnt oft, häufig (22%)) | Subphase: Basis (3 Folgesymbole)
```

### Statistiken

```
STATISTISCHE KENNZAHLEN
======================================================================

Anzahl Ketten: 15
Anzahl Terminale: 12
Durchschnittliche Länge: 11.4
Minimale Länge: 8
Maximale Länge: 18

Häufigste Symbole:
  KBG: 15x
  VBG: 15x
  KBBd: 28x
  VBBd: 28x
  KBA: 22x
  ...

Kommentare: 3
```

## 💡 Beispiele

### Beispiel 1: Einfache Dialogsequenz

**Eingabe:**
```
# Standard-Verkaufsgespräch
KBG, VBG, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV
```

**Erwartete Ausgabe:**
- Kodierung wird automatisch abgeleitet
- Automat erkennt Struktur: Begrüßung → Bedarf → Abschluss → Verabschiedung
- Hohe Konfidenz (>90%) für alle Ableitungen

### Beispiel 2: Mit Wiederholungen

**Eingabe:**
```
# Mit Schleife in der Bedarfsphase
KBG, VBG, KBBd, VBBd, KBBd, VBBd, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV
```

**Erwartete Ausgabe:**
- Musterbasierte Strategie erkennt Wiederholungen
- KBBd/VBBd erhalten Subphase "01" (Folge)
- Automat lernt, dass Wiederholungen in Phase 1 erlaubt sind

### Beispiel 3: Unvollständige Sequenz

**Eingabe:**
```
# Fehlende Verabschiedung
KBG, VBG, KBBd, VBBd, KBA, VBA, VAA, KAA
```

**Erwartete Ausgabe:**
- Validierung warnt vor fehlendem Abschluss
- Automat endet in nicht-akzeptierendem Zustand
- Statistik zeigt Muster "fehlende Verabschiedung"

## 🔍 Fehlerbehebung

### Häufige Probleme

#### 1. **"Keine gültigen Ketten gefunden"**
- **Ursache**: Falsches Trennzeichen oder leere Datei
- **Lösung**: Trennzeichen überprüfen, Dateiformat kontrollieren

#### 2. **Warnung: "Sehr kurze Ketten"**
- **Ursache**: Ketten mit <2 Symbolen
- **Lösung**: Prüfen, ob Daten vollständig sind

#### 3. **Warnung: "Ähnliche Symbole"**
- **Ursache**: Mögliche Tippfehler (z.B. "KBG" und "KBG ")
- **Lösung**: Symbole auf Konsistenz prüfen

#### 4. **Niedrige Konfidenz bei Kodierung**
- **Ursache**: Zu wenige Daten oder inkonsistente Muster
- **Lösung**: Mehr Daten sammeln oder Domäne überprüfen

### Modul-Fehler

| Fehler | Lösung |
|--------|--------|
| "hmmlearn nicht installiert" | Automatische Installation abwarten oder manuell: `pip install hmmlearn` |
| "graphviz nicht verfügbar" | Graphviz installieren: https://graphviz.org/download/ |
| "sentence-transformers fehlt" | `pip install sentence-transformers` |

### Performance-Probleme

- **Bei großen Dateien (>1000 Ketten)**: Export als JSON für spätere Analyse
- **Bei komplexen Mustern**: Progressive Learning nutzen
- **Bei Speicherproblemen**: Datei in kleinere Teile aufteilen

## 🤝 Mitwirken

Beiträge sind willkommen! Bitte beachten Sie:

1. **Issues**: Nutzen Sie GitHub Issues für Fehlerberichte und Feature-Wünsche
2. **Pull Requests**: Bitte mit Beschreibung der Änderungen
3. **Tests**: Stellen Sie sicher, dass alle Tests bestehen
4. **Dokumentation**: Aktualisieren Sie bei Bedarf die README

### Entwicklungsumgebung einrichten

```bash
# Repository forken und klonen
git clone https://github.com/yourusername/ARSXAI7.git
cd ARSXAI7

# Virtuelle Umgebung (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Code-Stil

- PEP 8 für Python-Code
- Docstrings für alle öffentlichen Methoden
- Typannotationen wo sinnvoll
- Kommentare auf Deutsch oder Englisch
