# ARSXAI8 - Algorithmic Recursive Sequence Analysis with Explainable AI


## 📖 Inhaltsverzeichnis

1. [Einführung und Zielsetzung](#-einführung-und-zielsetzung)
2. [Hauptmerkmale](#-hauptmerkmale)
3. [Installation](#-installation)
4. [Integrierte Modelle](#-integrierte-modelle)
5. [XAI-Komponenten](#-xai-komponenten)
6. [Dateiformate](#-dateiformate)
7. [Benutzeroberfläche](#-benutzeroberfläche)
8. [Bedienungsanleitung](#-bedienungsanleitung)
9. [Beispiele](#-beispiele)
10. [Exportformate](#-exportformate)
11. [Fehlerbehebung](#-fehlerbehebung)
12. [Mitwirken](#-mitwirken)
13. [Lizenz](#-lizenz)

---

## 🎯 Einführung und Zielsetzung

**ARSXAI8** ist eine universelle Analyseplattform für sequenzielle Daten mit Fokus auf **Explainable AI (XAI)**. Das Programm integriert sechs verschiedene Modellierungsansätze, um aus beliebigen Terminalzeichenketten (z.B. Dialogtranskripte, Prozessabläufe, Verhaltenssequenzen) automatisch strukturelle Regeln abzuleiten und diese für den Menschen **nachvollziehbar zu erklären**.

### Kernziele

- **Universalität**: Analyse beliebiger Sequenzen, nicht nur vordefinierter Domänen
- **Multi-Modell-Ansatz**: Sechs verschiedene Modelle für unterschiedliche Perspektiven
- **Erklärbarkeit**: Transparente Darstellung der Ableitungsprozesse und Entscheidungen
- **Vergleichbarkeit**: Konsens- und Diskrepanzanalyse zwischen Modellen
- **Generierung**: Synthetische Erzeugung neuer Sequenzen aus gelernten Strukturen

### Hauptanwendungsgebiete

| Bereich | Anwendung |
|---------|-----------|
| **Dialoganalyse** | Verkaufsgespräche, Beratungen, Therapiegespräche |
| **Prozessanalyse** | Workflow-Muster, Produktionsabläufe, Geschäftsprozesse |
| **Verhaltensanalyse** | Aktionssequenzen, Interaktionsmuster, Nutzerverhalten |
| **Code-Analyse** | Programmabläufe, API-Aufrufsequenzen, Log-Analyse |
| **Wissenschaft** | Sequenzmuster in beliebigen Domänen, Modellvergleiche |

---

## 🌟 Hauptmerkmale

### 📊 **Multi-Modell-Architektur**
- **6 integrierte Modelle** mit einheitlichem XAI-Interface
- Gleichzeitiges Training aller Modelle
- Modellvergleich mit Konsensanalyse

### 🔍 **Mehrere Ableitungsstrategien**
- Positionsbasierte Kodierung
- Musterbasierte Kodierung
- Statistisch basierte Kodierung
- Konsensbildung über Mehrheitsentscheidung

### 💡 **XAI-Komponenten**
- Interaktiver Erklärer ("Warum?")
- Modellvergleich für beliebige Symbole
- Konfidenzmetriken für alle Ableitungen
- Was-wäre-wenn Simulationen

### 🎨 **Umfangreiche Visualisierungen**
- Kodierungsvergleich als Heatmap
- Modell-Konfidenzen im Vergleich
- Automaten-Graph (mit Graphviz)
- Textuelle Alternativen bei fehlender Grafik

### 📤 **Multi-Format Export**
- JSON (maschinenlesbar)
- CSV (Tabellenkalkulation)
- HTML (interaktiver Bericht)
- LaTeX (wissenschaftliche Publikationen)

---

## 💻 Installation

### Systemvoraussetzungen

- **Python**: 3.8 oder höher
- **RAM**: 4 GB (empfohlen)
- **Festplatte**: 500 MB für Abhängigkeiten
- **OS**: Windows, macOS, Linux

### Automatische Installation

```bash
# Repository klonen
git clone https://github.com/yourusername/ARSXAI8.git
cd ARSXAI8

# Programm starten (Pakete werden automatisch installiert)
python ARSXAI8.py
```

### Manuelle Installation

```bash
# Alle Abhängigkeiten installieren
pip install numpy scipy matplotlib hmmlearn sklearn-crfsuite sentence-transformers networkx torch seaborn tabulate graphviz
```

### Graphviz (für Automaten-Visualisierung)

**Windows:**
```bash
# Mit Chocolatey (als Administrator)
choco install graphviz

# Oder manuell: https://graphviz.org/download/
# Installationspfad muss im SYSTEM-PATH sein
```

**Linux:**
```bash
sudo apt-get install graphviz  # Ubuntu/Debian
```

**macOS:**
```bash
brew install graphviz
```

---

## 🧠 Integrierte Modelle

| Modell | Klasse | Beschreibung | XAI-Fokus |
|--------|--------|--------------|-----------|
| **ARS 2.0** | `ARS20` | Einfache Bigramm-Übergangswahrscheinlichkeiten | Transparente Wahrscheinlichkeiten |
| **ARS 3.0** | `GrammarInducer` | Hierarchische Grammatik mit Nonterminalen | Mustererkennung und Kompression |
| **HMM** | `ARSHiddenMarkovModel` | Bayessche Netze für latente Phasen | Verborgene Zustände |
| **CRF** | `ARSCRFModel` | Conditional Random Fields | Feature-Wichtigkeit |
| **Petri-Netz** | `ARSPetriNet` | Ressourcen-basierte Prozessmodellierung | Token und Aktivierung |
| **Generator** | `ChainGenerator` | Synthetische Ketten aus Modellen | Generierungserklärung |

### Modell-Vergleich

```python
# Jedes Modell liefert einheitliche Erklärungen
modell.explain(symbol) -> {
    'model': "ARS 2.0",
    'confidence': 0.85,
    'content': ["Erklärung Zeile 1", "Erklärung Zeile 2"]
}
```

---

## 🔬 XAI-Komponenten

### 1. **Interaktiver Erklärer** (`InteractiveExplainer`)

```python
# Warum wurde Symbol so kodiert?
explainer.why_symbol("KBBd")
>>> 🔍 Erklärung für Symbol 'KBBd':
    ============================================================
    Konsens-Kodierung: 00100
    Übereinstimmung: 67%
    
    Positionsbasiert: Kunde | Phase: Bedarf (Pos.3.2) | Basis
    Musterbasiert: Kunde | Phase: Bedarf (Nachbarn: VBBd) | Basis
    Statistisch: Kunde | Phase: Bedarf (Häufigkeit: 22%) | Basis
```

### 2. **Modellvergleich** (`XAIModelManager`)

```python
# Alle aktiven Modelle vergleichen
manager.compare_models("KBBd")
>>> ARS 2.0:  Übergang KBBd→VBBd mit 85%
    ARS 3.0:  KBBd ist Teil von NT_BEDARF
    HMM:      KBBd in Phase 1 mit 78%
```

### 3. **Konfidenzmetriken**

Jede Ableitung wird mit einer Konfidenz versehen:

| Konfidenz | Bedeutung | Darstellung |
|-----------|-----------|-------------|
| > 0.7 | Hohe Konfidenz | ★★★★☆ (grün) |
| 0.4 - 0.7 | Mittlere Konfidenz | ★★★☆☆ (orange) |
| < 0.4 | Niedrige Konfidenz | ★★☆☆☆ (rot) |

### 4. **Was-wäre-wenn Simulation**

Simulation alternativer Kodierungen und deren Auswirkungen auf die Modelle.

---

## 📁 Dateiformate

### Eingabeformat (Transkriptdatei)

```txt
# Kommentare beginnen mit #
# Trennzeichen: Komma, Semikolon oder Leerzeichen

# Transkript 1: Standard
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

### Beispiel: Verkaufsgespräche

Die Beispieltranskripte verwenden folgende Symbole:

| Symbol | Bedeutung |
|--------|-----------|
| KBG | Kunden-Begrüßung |
| VBG | Verkäufer-Begrüßung |
| KBBd | Kunden-Bedarf (konkret) |
| VBBd | Verkäufer-Nachfrage |
| KBA | Kunden-Antwort |
| VBA | Verkäufer-Reaktion |
| KAE | Kunden-Erkundigung |
| VAE | Verkäufer-Auskunft |
| KAA | Kunden-Abschluss |
| VAA | Verkäufer-Abschluss |
| KAV | Kunden-Verabschiedung |
| VAV | Verkäufer-Verabschiedung |

---

## 🖥️ Benutzeroberfläche

### Hauptfenster

```
┌─────────────────────────────────────────────────────────────────┐
│ ARSXAI8 - Algorithmic Recursive Sequence Analysis with Explainable AI │
├──────────────────────┬──────────────────────────────────────────┤
│ EINGABE              │ AUSGABE (Notebook-Tabs)                  │
│                      │                                          │
│ Trennzeichen:        │ ┌─────────────────────────────────────┐ │
│ ○ Komma ○ Semikolon  │ │ Kodierung | Modelle | Automat | XAI | │
│ ○ Leerzeichen ○ |    │ └─────────────────────────────────────┘ │
│                      │                                          │
│ Transkripte:         │ Kodierungsergebnisse:                   │
│ ┌──────────────────┐ │ ╔═════════════════════════════════════╗ │
│ │ KBG, VBG, KBBd,  │ │ ║ POSITIONSBASIERTE KODIERUNG        ║ │
│ │ KBBd, VBBd, KBA, │ │ ║ Konfidenz: 78%                     ║ │
│ │ VAA, KAA, VAV    │ │ ║   KBG: 00000                        ║ │
│ └──────────────────┘ │ ║   VBG: 10000                        ║ │
│                      │ ║   KBBd: 00100                       ║ │
│ [Datei laden] [Parsen]│ ╚═════════════════════════════════════╝ │
│                      │                                          │
│ ✓ 15 Ketten geladen  │                                          │
├──────────────────────┴──────────────────────────────────────────┤
│ Status: Alle Modelle trainiert                     [=====▶   ] │
└─────────────────────────────────────────────────────────────────┘
```

### Tabs im Überblick

| Tab | Funktion | Inhalt |
|-----|----------|--------|
| **Kodierung** | Ableitungsstrategien | Ergebnisse von Position, Muster, Statistik |
| **Modelle** | Modellverwaltung | Checkboxen, Trainingsstatus, Konfidenzen |
| **Automat** | Automaten-Lernen | Gelernte Regeln, Validierung |
| **XAI** | Erklärungen | Interaktive Fragen, Modellvergleich |
| **Generierung** | Synthese | Neue Ketten aus Modellen |
| **Statistiken** | Kennzahlen | Verteilungen, Häufigkeiten |

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
├── Alle Modelle trainieren
├── Automaten lernen
└── Validierung durchführen

XAI
├── Erklärung für Symbol
├── Modellvergleich
├── Konfidenzen vergleichen
└── Was-wäre-wenn

Generierung
├── Mit ARS 2.0 generieren
├── Mit ARS 3.0 generieren
└── Mit HMM generieren

Visualisierung
├── Kodierungsvergleich
├── Modell-Konfidenzen
└── Automaten-Graph

Hilfe
├── Modulstatus
└── Über
```

---

## 📘 Bedienungsanleitung

### Schritt-für-Schritt

#### 1. **Daten laden**

**Option A - Datei:**
- Klicken Sie auf "Datei laden"
- Wählen Sie eine Textdatei mit Transkripten
- Das Programm parst automatisch

**Option B - Beispiel:**
- Klicken Sie auf "Beispiel"
- Lädt vordefinierte Verkaufsgespräche

**Option C - Direkteingabe:**
- Geben Sie Ketten direkt ins Textfeld ein
- Eine Kette pro Zeile
- Kommentare mit `#` möglich

#### 2. **Trennzeichen wählen**

Wählen Sie das in Ihrer Datei verwendete Trennzeichen:
- **Komma (,)**: `KBG, VBG, KBBd`
- **Semikolon (;)**: `KBG; VBG; KBBd`
- **Leerzeichen**: `KBG VBG KBBd`
- **Benutzerdefiniert**: z.B. `|`

#### 3. **Analyse starten**

Nach dem Laden startet die Analyse automatisch:

1. **Validierung** - Prüft Datenqualität
2. **Kodierungsstrategien** - Drei Strategien werden angewandt
3. **Modell-Training** - Alle sechs Modelle werden trainiert
4. **Konsensbildung** - Mehrheitsentscheidung über Kodierung

#### 4. **Modelle erkunden**

**Modelle-Tab:**
- Sehen Sie alle registrierten Modelle
- Aktivieren/deaktivieren Sie Modelle per Checkbox
- Trainingsstatus und Konfidenzen werden angezeigt

**Kodierung-Tab:**
- Vergleichen Sie die drei Ableitungsstrategien
- Sehen Sie den Konsens mit Übereinstimmungswerten
- ⚠️ markiert unsichere Kodierungen

#### 5. **XAI-Fragen stellen**

**XAI-Tab:**
- Geben Sie ein Symbol ein (z.B. "KBBd")
- Klicken Sie "Warum?" für detaillierte Erklärung
- Klicken Sie "Modelle vergleichen" für Multi-Perspektive

**Was-wäre-wenn:**
1. Klicken Sie "Was-wäre-wenn"
2. Wählen Sie Symbol, alternativen Code und Modell
3. Sehen Sie simulierte Auswirkungen

#### 6. **Generierung**

**Generierung-Tab:**
- Wählen Sie ein Quellmodell (ARS 2.0, ARS 3.0, HMM)
- Stellen Sie die Anzahl ein
- Klicken Sie "Generieren"
- Neue Ketten werden basierend auf gelernten Strukturen erzeugt

#### 7. **Visualisierungen**

**Kodierungsvergleich:**
- Menü: Visualisierung → Kodierungsvergleich
- Zeigt Heatmap der 5-Bit-Codes aller Strategien

**Modell-Konfidenzen:**
- Menü: Visualisierung → Modell-Konfidenzen
- Balkendiagramm aller Modell-Konfidenzen

**Automaten-Graph:**
- Menü: Visualisierung → Automaten-Graph
- Zeigt Zustände und Übergänge (mit Graphviz)

#### 8. **Exportieren**

1. Menü: Datei → Exportieren
2. Wählen Sie Format:
   - **JSON**: Für Weiterverarbeitung
   - **CSV**: Für Tabellenkalkulation
   - **HTML**: Interaktiver Bericht
   - **LaTeX**: Für wissenschaftliche Publikationen

---

## 💡 Beispiele

### Beispiel 1: Einfache Analyse

**Eingabe:**
```
# Einfaches Gespräch
KBG, VBG, KBBd, VBBd, KBA, VBA, VAA, KAA, VAV, KAV
```

**Erwartete Ausgabe:**
- Alle Modelle werden trainiert
- Hohe Konfidenz (>85%) für alle Ableitungen
- Konsens-Kodierung wird gebildet

### Beispiel 2: Modellvergleich

**Frage:** "Warum ist KBBd = 00100?"

**ARS 2.0:** "KBBd geht zu 85% in VBBd über"
**ARS 3.0:** "KBBd ist Teil des Musters KBBd→VBBd→KBA"
**HMM:** "KBBd ist mit 92% in Phase 1 (Bedarf)"
**CRF:** "Feature 'prefix_K' (+2.3) unterstützt diese Zuordnung"

### Beispiel 3: Generierung

**Mit ARS 2.0 generiert:**
```
1. KBG → VBG → KBBd → VBBd → KBA → VBA → VAA → KAA → VAV → KAV
2. KBG → VBG → KBBd → VBBd → KBBd → VBBd → KBA → VAA → KAA → VAV → KAV
3. KBG → VBG → KBBd → VBBd → KBA → VBA → KAE → VAE → VAA → KAA → VAV → KAV
```

---

## 📊 Exportformate

### JSON (maschinenlesbar)
```json
{
  "coding": {
    "KBG": {
      "code": "00000",
      "agreement": 1.0,
      "confidence": 0.95
    }
  },
  "model_comparison": {
    "ARS20": {
      "name": "ARS 2.0",
      "confidence": 0.85,
      "trained": true
    }
  }
}
```

### CSV (Tabellenkalkulation)
```csv
Symbol,Code,Konfidenz,Modell
KBG,00000,95%,ARS20
VBG,10000,92%,ARS20
```

### HTML (Interaktiver Bericht)
- Kodierungstabellen mit farbigen Konfidenzen
- Modell-Infoboxen mit Beschreibungen
- Export-Datum und Statistiken

### LaTeX (Wissenschaftlich)
```latex
\documentclass{article}
\begin{document}
\section{ARSXAI8 Analyseergebnisse}
\begin{tabular}{lll}
Symbol & Code & Konfidenz \\
KBG & 00000 & 95\% \\
\end{tabular}
\end{document}
```

---

## 🔧 Fehlerbehebung

### Häufige Probleme

#### 1. **"Keine gültigen Ketten gefunden"**
- **Ursache**: Falsches Trennzeichen oder leere Datei
- **Lösung**: Trennzeichen überprüfen, Dateiformat kontrollieren

#### 2. **Warnung: "Ähnliche Symbole"**
- **Ursache**: Mögliche Tippfehler (z.B. "KBG" und "KBG ")
- **Lösung**: Symbole auf Konsistenz prüfen

#### 3. **Graphviz-Fehler**
```
failed to execute WindowsPath('dot')
```
- **Ursache**: Graphviz Systembibliothek fehlt
- **Lösung**: Graphviz installieren und PATH setzen

#### 4. **Niedrige Konfidenz bei Modellen**
- **Ursache**: Zu wenige Daten oder inkonsistente Muster
- **Lösung**: Mehr Daten sammeln oder Modelle deaktivieren

### Modul-Fehler

| Fehler | Lösung |
|--------|--------|
| "hmmlearn nicht installiert" | `pip install hmmlearn` |
| "sklearn-crfsuite nicht installiert" | `pip install sklearn-crfsuite` |
| "sentence-transformers fehlt" | `pip install sentence-transformers` |

---

## 🤝 Mitwirken

Beiträge sind willkommen!

### Entwicklungsumgebung

```bash
# Repository forken
git clone https://github.com/yourusername/ARSXAI8.git
cd ARSXAI8

# Virtuelle Umgebung
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Entwicklungspakete
pip install -r requirements-dev.txt
```

### Code-Stil

- PEP 8 für Python-Code
- Docstrings für alle öffentlichen Methoden
- Typannotationen wo sinnvoll
- Kommentare auf Deutsch oder Englisch

### Issue-Template

```markdown
**Beschreibung:**
Kurze Beschreibung des Problems

**Schritte zu reproduzieren:**
1. ...
2. ...

**Erwartetes Verhalten:**
...

**Tatsächliches Verhalten:**
...

**Umgebung:**
- OS: ...
- Python: ...
- ARSXAI8 Version: ...
```

---

## 📄 Lizenz

ARSXAI8 ist unter der **MIT-Lizenz** veröffentlicht.

```
MIT License

Copyright (c) 2024 Explainable AI Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 📚 Zitation

Wenn Sie ARSXAI8 in wissenschaftlichen Arbeiten verwenden, bitte wie folgt zitieren:

```bibtex
@software{ARSXAI8,
  author = {Koop, Paul},
  title = {ARSXAI8: Algorithmic Recursive Sequence Analysis with Explainable AI},
  year = {2024},
  url = {https://github.com/yourusername/ARSXAI8}
}
```

---

## 🙏 Danksagung

- **hmmlearn** - Für die HMM-Implementierung
- **sklearn-crfsuite** - Für die CRF-Implementierung
- **Graphviz** - Für Automaten-Visualisierung
- **Sentence-Transformers** - Für semantische Analysen

---

**Entwickelt mit ❤️ für erklärbare Künstliche Intelligenz**

