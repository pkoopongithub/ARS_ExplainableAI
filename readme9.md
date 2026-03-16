# ARSXAI9 - Algorithmic Recursive Sequence Analysis mit Explainable AI

## 📖 Inhaltsverzeichnis

1. [Einführung und Vision](#-einführung-und-vision)
2. [Hauptmerkmale](#-hauptmerkmale)
3. [Neu in Version 9.0](#-neu-in-version-90)
4. [Installation](#-installation)
5. [Die zentrale Idee: PCFG-basierte XAI](#-die-zentrale-idee-pcfg-basierte-xai)
6. [Integrierte Modelle](#-integrierte-modelle)
7. [Dateiformate](#-dateiformate)
8. [Benutzeroberfläche](#-benutzeroberfläche)
9. [Bedienungsanleitung](#-bedienungsanleitung)
10. [XAI-Erklärungen verstehen](#-xai-erklärungen-verstehen)
11. [Beispiele](#-beispiele)
12. [Exportformate](#-exportformate)
13. [Fehlerbehebung](#-fehlerbehebung)
14. [Mitwirken](#-mitwirken)
15. [Lizenz](#-lizenz)

---

## 🎯 Einführung und Vision

**ARSXAI9** ist eine universelle Analyseplattform für sequenzielle Daten mit Fokus auf **Explainable AI (XAI)**. Die Kernidee: Statt willkürlicher Kodierungen (wie der früheren 5-Bit-Kodierung) basieren alle Erklärungen auf einer **Probabilistischen Kontextfreien Grammatik (PCFG)**, die automatisch aus den Daten induziert wird.

### Die zentrale Erkenntnis
Wiederkehrende Muster in Sequenzen sind die natürliche Grundlage für Erklärungen. ARSXAI9 abstrahiert diese Muster zu **Nonterminalen** und macht sie damit explizit und erklärbar.

### Hauptanwendungsgebiete

| Bereich | Anwendung |
|---------|-----------|
| **Dialoganalyse** | Verkaufsgespräche, Beratungen, Therapiegespräche |
| **Prozessanalyse** | Workflow-Muster, Produktionsabläufe |
| **Verhaltensanalyse** | Aktionssequenzen, Interaktionsmuster |
| **Code-Analyse** | Programmabläufe, API-Aufrufsequenzen |
| **Wissenschaft** | Sequenzmuster in beliebigen Domänen |

---

## 🌟 Hauptmerkmale

### 🧠 **PCFG-basierte Mustererkennung**
- Automatische Induktion einer hierarchischen Grammatik
- Wiederholte Sequenzen werden zu Nonterminalen abstrahiert
- **Keine willkürlichen Kodierungen** oder domänenspezifischen Annahmen

### 💬 **Natürlichsprachliche Erklärungen**
```text
🔍 Erklärung für Symbol 'CBG':
============================================================
🔤 CBG ist ein grundlegendes Symbol.

📊 Es kommt in folgenden wiederkehrenden Mustern vor:

  • P_CBG_BBG_2 (100% der Ketten):
    P_CBG_BBG_2 → CBG → BBG
    Position: nach nichts, vor BBG

🏗️ Hierarchische Einbettung:
└─ in P_CBG_BBG_2 → CBG → BBG
  └─ in P_GESAMT_10 → P_CBG_BBG_2 ...
```

### 📊 **Hierarchische Musterübersicht**
- Alle erkannten Muster mit Häufigkeiten
- Begründung, warum ein Muster erkannt wurde
- Kontext-Informationen (Vorher/Nachher)

### 🔄 **Multiple Modell-Perspektiven**
- ARS 2.0: Einfache Bigramm-Wahrscheinlichkeiten
- ARS 3.0: Hierarchische Grammatik (Hauptmodell)
- HMM: Latente Phasen
- CRF: Kontext-sensitive Features
- Petri-Netze: Ressourcen-basierte Modellierung

---

## ✨ Neu in Version 9.0

### ❌ **Entfernt**
- Die 5-Bit-Kodierung (mit ihren willkürlichen Annahmen)
- Positionsbasierte, musterbasierte und statistische Kodierungsstrategien
- Künstliche Aufteilung in "Sprechergruppen" (A-M vs. N-Z)

### ✅ **Hinzugefügt**
- **GrammarInducer** als zentrale Wissensbasis
- **NaturalLanguageExplainer** für menschenlesbare Erklärungen
- Hierarchische Musterübersicht mit Häufigkeiten
- Sequenz-Erklärungen mit Zerlegung in Teil-Muster
- Kontext-Informationen für Symbole in Mustern

### 🎯 **XAI-Verbesserungen**
- Erklärungen basieren auf **tatsächlich gelernten Strukturen**
- **Keine versteckten Annahmen** mehr
- **Natürlichsprachliche** Ausgabe
- **Hierarchische Einbettung** sichtbar gemacht
- **Häufigkeitsangaben** für alle Muster

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
git clone https://github.com/yourusername/ARSXAI9.git
cd ARSXAI9

# Programm starten (Pakete werden automatisch installiert)
python ARSXAI9.py
```

### Manuelle Installation

```bash
# Alle Abhängigkeiten installieren
pip install numpy scipy matplotlib hmmlearn sklearn-crfsuite sentence-transformers networkx torch seaborn tabulate graphviz
```

### Graphviz (für Visualisierung)

**Windows:**
```bash
# Mit Chocolatey (als Administrator)
choco install graphviz

# Oder manuell: https://graphviz.org/download/
# Bei Installation HAKEN SETZEN bei "Add Graphviz to PATH"
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

## 🧠 Die zentrale Idee: PCFG-basierte XAI

### 1. **Mustererkennung statt willkürlicher Kodierung**

In Version 8.x gab es noch drei Kodierungsstrategien, die versuchten, jedem Symbol eine 5-Bit-Bedeutung zuzuweisen:
- Bit 1: "Gruppe A-M" vs. "Gruppe N-Z" (völlig willkürlich!)
- Bits 2-3: "Phase 0-3" (ebenfalls willkürlich)
- Bits 4-5: "Basis/Folge" (auch willkürlich)

**Das Problem:** Diese Kodierung musste immer interpretiert werden und traf Annahmen über die Daten.

### 2. **Die Lösung: Lerne die Struktur aus den Daten**

ARSXAI9 geht einen völlig anderen Weg:

```python
# Wiederholtes Muster gefunden: [CBG, BBG] kommt in 100% der Ketten vor
→ Neues Nonterminal: P_CBG_BBG_2

# Nächstes Muster: [CBBd, BBBd] kommt häufig vor
→ Neues Nonterminal: P_CBBd_BBBd_2

# Die Grammatik wächst hierarchisch:
P_GESAMT_10 → P_CBG_BBG_2, P_CBBd_BBBd_2, P_CBA_BBA_2, ...
```

### 3. **Erklärungen direkt aus der Grammatik**

```python
# Für Symbol 'CBG':
"CBG ist Teil des Musters P_CBG_BBG_2 (kommt in 100% der Ketten vor)"
```

**Das ist echte XAI:** Die Erklärung basiert auf der tatsächlich gelernten Struktur, nicht auf einer zusätzlichen Interpretationsschicht.

---

## 📦 Integrierte Modelle

| Modell | Klasse | Beschreibung | Status |
|--------|--------|--------------|--------|
| **ARS 3.0** | `GrammarInducer` | Hierarchische PCFG (ZENTRAL) | ⭐ Hauptmodell |
| **ARS 2.0** | `ARS20` | Einfache Bigramm-Wahrscheinlichkeiten | 🔧 Optional |
| **HMM** | `ARSHiddenMarkovModel` | Latente Phasen | 🔧 Optional |
| **CRF** | `ARSCRFModel` | Kontext-sensitive Features | 🔧 Optional |
| **Petri-Netz** | `ARSPetriNet` | Ressourcen-basierte Modellierung | 🔧 Optional |
| **Generator** | `ChainGenerator` | Synthetische Ketten | 🔧 Optional |

### Fokus auf ARS 3.0

ARS 3.0 ist das **Hauptmodell** und die Grundlage aller XAI-Erklärungen. Die anderen Modelle dienen als Vergleich und zur Validierung.

---

## 📁 Dateiformate

### Eingabeformat (Transkriptdatei)

```txt
# Kommentare beginnen mit #
# Trennzeichen: Komma, Semikolon oder Leerzeichen (einstellbar)

# Transkript 1: Standard
CBG, BBG, CBBd, BBBd, CBA, BBA, CBBd, BBBd, CBA, BAA, CAA, BAB, CAB

# Transkript 2: Mit Wiederholungen
CBG, BBG, CBBd, BBBd, CBBd, BBBd, CBA, BBA, BAA, CAA, BAB, CAB

# Transkript 3: Kurz
CBG, BBG, CBBd, BBBd, CBA, BBA, BAA, CAA, BAB, CAB

# Leerzeilen werden ignoriert
```

**Formatregeln:**
- Eine Kette pro Zeile
- Kommentare mit `#` am Zeilenanfang
- Symbole durch Trennzeichen getrennt
- Leere Zeilen werden ignoriert

### Trennzeichen

Das Trennzeichen ist **frei wählbar** und wird für ALLE Eingaben verwendet:
- Haupttranskripte
- Symbol-Eingabe (XAI-Tab)
- Sequenz-Eingabe (XAI-Tab)

---

## 🖥️ Benutzeroberfläche

### Hauptfenster

```
┌─────────────────────────────────────────────────────────────────┐
│ ARSXAI9 - PCFG-basierte Musteranalyse mit XAI                        │
├──────────────────────┬──────────────────────────────────────────┤
│ EINGABE              │ AUSGABE (Notebook-Tabs)                  │
│                      │                                          │
│ Trennzeichen:        │ ┌─────────────────────────────────────┐ │
│ ○ Komma ○ Semikolon  │ │ Grammatik | Muster | XAI | Modelle  │ │
│ ○ Leerzeichen ○ |    │ └─────────────────────────────────────┘ │
│                      │                                          │
│ Transkripte:         │ ERKANNTE MUSTER:                        │
│ ┌──────────────────┐ │ ╔═════════════════════════════════════╗ │
│ │ CBG, BBG, CBBd,  │ │ ║ 📌 P_CBG_BBG_2 (100% der Ketten):  ║ │
│ │ BBBd, CBA, BBA,  │ │ ║    CBG → BBG                        ║ │
│ │ CBBd, BBBd, CBA, │ │ ║    Grund: Zweierfolge in 100%       ║ │
│ │ BAA, CAA, BAB,   │ │ ║                                    ║ │
│ │ CAB               │ │ ║ 📌 P_CBBd_BBBd_2 (88%):           ║ │
│ └──────────────────┘ │ ║    CBBd → BBBd                      ║ │
│                      │ ╚═════════════════════════════════════╝ │
│ [Datei laden] [Parsen]│                                          │
│ [Grammatik induzieren]│                                          │
├──────────────────────┴──────────────────────────────────────────┤
│ Status: Grammatik induziert - 5 Muster gefunden      [====▶   ] │
└─────────────────────────────────────────────────────────────────┘
```

### Tabs im Überblick

| Tab | Funktion | Inhalt |
|-----|----------|--------|
| **Grammatik** | Vollständige PCFG | Alle Produktionsregeln, Terminale, Nonterminale |
| **Erkannte Muster** | Übersicht | Alle Muster mit Häufigkeiten und Begründungen |
| **XAI-Erklärungen** | Interaktiv | Erklärungen für Symbole und Sequenzen |
| **Weitere Modelle** | Vergleich | ARS 2.0, HMM, CRF, Petri-Netze |
| **Statistiken** | Kennzahlen | Verteilungen, Häufigkeiten, Kompressionsrate |

---

## 📘 Bedienungsanleitung

### 1. **Daten laden**

**Option A - Datei:**
- Klicken Sie auf "Datei laden"
- Wählen Sie eine Textdatei mit Transkripten
- Das Programm parst automatisch

**Option B - Beispiel:**
- Klicken Sie auf "Beispiel"
- Lädt vordefinierte C-Symbol-Transkripte

**Option C - Direkteingabe:**
- Geben Sie Ketten direkt ins Textfeld ein
- Eine Kette pro Zeile
- Kommentare mit `#` möglich

### 2. **Trennzeichen wählen**

Wählen Sie das in Ihrer Datei verwendete Trennzeichen:
- **Komma (,)**: `CBG, BBG, CBBd`
- **Semikolon (;)**: `CBG; BBG; CBBd`
- **Leerzeichen**: `CBG BBG CBBd`
- **Benutzerdefiniert**: z.B. `|`

> **Wichtig:** Dieses Trennzeichen wird für ALLE Eingaben verwendet (auch für Sequenzen im XAI-Tab)!

### 3. **Grammatik induzieren**

Nach dem Laden:
1. Klicken Sie auf "Grammatik induzieren" (oder es startet automatisch)
2. Das Programm findet wiederkehrende Muster
3. Die Grammatik wird aufgebaut
4. Nonterminale werden automatisch benannt (z.B. `P_CBG_BBG_2`)

### 4. **Muster erkunden**

**Grammatik-Tab:**
- Sehen Sie die vollständige PCFG
- Alle Produktionsregeln mit Wahrscheinlichkeiten
- Terminale und Nonterminale

**Muster-Tab:**
- Übersicht aller erkannten Muster
- Häufigkeit in Prozent
- Begründung, warum das Muster erkannt wurde

### 5. **XAI-Fragen stellen**

**XAI-Tab - Symbol erklären:**
- Geben Sie ein Symbol ein (z.B. "CBG")
- Klicken Sie "Symbol erklären"
- Sie erhalten:
  - Information, ob es Terminal oder Nonterminal ist
  - In welchen Mustern es vorkommt
  - Hierarchische Einbettung
  - Kontext (Vorher/Nachher)

**XAI-Tab - Sequenz erklären:**
- Geben Sie eine Sequenz ein (mit dem eingestellten Trennzeichen)
- Beispiele (bei Komma):
  - `CBG, BBG, CBBd`
  - `CBG,BBG,CBBd` (ohne Leerzeichen)
  - `CBG , BBG , CBBd` (mit Leerzeichen)
- Klicken Sie "Sequenz erklären"
- Sie erhalten:
  - Hierarchische Zerlegung der Sequenz
  - Information, ob sie ein eigenständiges Muster bildet
  - Vorkommenshäufigkeit

### 6. **Weitere Modelle vergleichen**

**Modelle-Tab:**
- Aktivieren/deaktivieren Sie optionale Modelle
- Trainieren Sie alle Modelle
- Vergleichen Sie die Erklärungen verschiedener Modelle für dasselbe Symbol

### 7. **Visualisierungen**

**Grammatik-Hierarchie:**
- Menü: Visualisierung → Grammatik-Hierarchie
- Zeigt die hierarchische Struktur der Grammatik als Graph

**Muster-Häufigkeiten:**
- Menü: Visualisierung → Muster-Häufigkeiten
- Balkendiagramm der häufigsten Muster

### 8. **Exportieren**

1. Menü: Datei → Exportieren
2. Wählen Sie Format:
   - **JSON**: Für Weiterverarbeitung
   - **HTML**: Interaktiver Bericht mit allen Mustern
   - **LaTeX**: Für wissenschaftliche Publikationen

---

## 🔍 XAI-Erklärungen verstehen

### Symbol-Erklärung

```text
🔍 **Erklärung für Symbol 'CBBd'**
================================================================

🔤 **CBBd** ist ein **grundlegendes Symbol**.

📊 Es kommt in folgenden wiederkehrenden Mustern vor:

  • **P_CBBd_BBBd_2** (88% der Ketten):
    P_CBBd_BBBd_2 → CBBd → BBBd
    Position: nach nichts, vor BBBd

  • **P_Doppel_4** (75% der Ketten):
    P_Doppel_4 → P_CBBd_BBBd_2 → P_CBBd_BBBd_2
    Position: nach P_CBBd_BBBd_2, vor P_CBBd_BBBd_2

🏗️ **Hierarchische Einbettung:**
└─ in P_CBBd_BBBd_2 → CBBd → BBBd
  └─ in P_Doppel_4 → P_CBBd_BBBd_2 → P_CBBd_BBBd_2
    └─ in P_GESAMT_10 → P_CBG_BBG_2 → P_Doppel_4 ...

✅ **Konfidenz dieser Analyse**: 95%
```

### Sequenz-Erklärung

```text
🔍 **Erklärung für Sequenz:** CBBd → BBBd → CBBd → BBBd
================================================================

**Hierarchische Struktur:**
└─ **P_CBBd_BBBd_2** = CBBd → BBBd (in 88% der Ketten)
  └─ **P_CBBd_BBBd_2** = CBBd → BBBd (in 88% der Ketten)
    └─ **P_Doppel_4** = P_CBBd_BBBd_2 → P_CBBd_BBBd_2 (in 75% der Ketten)

📊 **Vorkommen**: 75% der Ketten (12 von 16)
```

### Bedeutung der Symbole

| Symbol | Bedeutung |
|--------|-----------|
| `P_...` | Nonterminal (erkanntes Muster) |
| `P_CBG_BBG_2` | Muster aus CBG und BBG (Länge 2) |
| `P_GESAMT_10` | Gesamtstruktur der Länge 10 |
| `🔤` | Terminales Symbol (grundlegend) |
| `📦` | Nonterminal (abstraktes Muster) |
| `📊` | Statistische Information |
| `🏗️` | Hierarchische Einbettung |

---

## 💡 Beispiele

### Beispiel 1: C-Symbol-Datensatz

**Eingabe:**
```
CBG, BBG, CBBd, BBBd, CBA, BBA, CBBd, BBBd, CBA, BAA, CAA, BAB, CAB
CBG, BBG, CBBd, BBBd, CBBd, BBBd, CBA, BBA, BAA, CAA, BAB, CAB
CBG, BBG, CBBd, BBBd, CBA, BBA, BAA, CAA, BAB, CAB
```

**Erkannte Muster:**
- `P_CBG_BBG_2` → CBG → BBG (100%)
- `P_CBBd_BBBd_2` → CBBd → BBBd (100%) 
- `P_CBA_BBA_2` → CBA → BBA (100%)
- `P_BAA_CAA_2` → BAA → CAA (100%)
- `P_BAB_CAB_2` → BAB → CAB (100%)
- `P_Doppel_4` → P_CBBd_BBBd_2 → P_CBBd_BBBd_2 (67%)

### Beispiel 2: D-Symbol-Datensatz (analoge Struktur)

**Eingabe:**
```
DBG, BBG, DBBd, BBBd, DBA, BBA, DBBd, BBBd, DBA, BAA, DAA, BAB, DAB
DBG, BBG, DBBd, BBBd, DBBd, BBBd, DBA, BBA, BAA, DAA, BAB, DAB
DBG, BBG, DBBd, BBBd, DBA, BBA, BAA, DAA, BAB, DAB
```

**Erkannte Muster:**
- `P_DBG_BBG_2` → DBG → BBG (100%)
- `P_DBBd_BBBd_2` → DBBd → BBBd (100%)
- `P_DBA_BBA_2` → DBA → BBA (100%)
- ... (analoge Struktur, andere Symbole)

### Beispiel 3: Sequenz-Erklärung

**Eingabe:** `CBG, BBG, CBBd, BBBd` (bei Komma als Trennzeichen)

**Ausgabe:**
```
🔍 **Erklärung für Sequenz:** CBG → BBG → CBBd → BBBd
================================================================

**Hierarchische Struktur:**
└─ **P_CBG_BBG_2** = CBG → BBG (in 100% der Ketten)
  └─ **P_CBBd_BBBd_2** = CBBd → BBBd (in 88% der Ketten)

📊 **Vorkommen**: 88% der Ketten (14 von 16)
```

---

## 📊 Exportformate

### JSON (maschinenlesbar)
```json
{
  "grammar": {
    "patterns": [
      {
        "name": "P_CBG_BBG_2",
        "sequence": ["CBG", "BBG"],
        "frequency": 100.0,
        "rationale": "Die Zweierfolge CBG → BBG kommt in 100% aller Ketten vor"
      }
    ]
  }
}
```

### HTML (Interaktiver Bericht)
```html
<div class="pattern-box">
  <h3>P_CBG_BBG_2</h3>
  <p><strong>Sequenz:</strong> CBG → BBG</p>
  <p><strong>Vorkommen:</strong> 100% der Ketten</p>
  <p><strong>Begründung:</strong> Die Zweierfolge CBG → BBG kommt in 100% aller Ketten vor</p>
</div>
```

### LaTeX (Wissenschaftlich)
```latex
\begin{tabular}{lll}
\toprule
Muster & Sequenz & Häufigkeit \\
\midrule
P\_CBG\_BBG\_2 & CBG $\rightarrow$ BBG & 100\% \\
\bottomrule
\end{tabular}
```

---

## 🔧 Fehlerbehebung

### Häufige Probleme

#### 1. **"Keine gültigen Ketten gefunden"**
- **Ursache**: Falsches Trennzeichen oder leere Datei
- **Lösung**: Trennzeichen überprüfen, Dateiformat kontrollieren

#### 2. **"Symbol nicht gefunden" bei XAI-Anfrage**
- **Ursache**: Symbol existiert nicht in den Daten
- **Lösung**: Groß-/Kleinschreibung prüfen, Tippfehler korrigieren

#### 3. **Sequenz wird nicht erkannt**
- **Ursache**: Falsches Trennzeichen verwendet
- **Lösung**: Das gleiche Trennzeichen wie in der Haupteingabe verwenden

#### 4. **Graphviz-Fehler**
```
failed to execute WindowsPath('dot')
```
- **Ursache**: Graphviz Systembibliothek fehlt
- **Lösung**: Graphviz installieren und PATH setzen

### Tipps

- **Trennzeichen-Konsistenz**: Für ALLE Eingaben das GLEICHE Trennzeichen verwenden
- **Leerzeichen**: Sind erlaubt, werden aber ignoriert
- **Groß-/Kleinschreibung**: Wichtig! "CBG" ≠ "cbg"
- **Lange Sequenzen**: Bei Komma als Trennzeichen: `CBG,BBG,CBBd,BBBd` (ohne Leerzeichen) ist am sichersten

---

## 🤝 Mitwirken

Beiträge sind willkommen!

### Entwicklungsumgebung

```bash
# Repository forken
git clone https://github.com/yourusername/ARSXAI9.git
cd ARSXAI9

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

---

## 📄 Lizenz

ARSXAI9 ist unter der **MIT-Lizenz** veröffentlicht.

```
MIT License

Copyright (c) 2024 Explainable AI Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 📚 Zitation

Wenn Sie ARSXAI9 in wissenschaftlichen Arbeiten verwenden, bitte wie folgt zitieren:

```bibtex
@software{ARSXAI9,
  author = {Koop, Paul},
  title = {ARSXAI9: PCFG-based Sequence Analysis with Explainable AI},
  year = {2024},
  url = {https://github.com/yourusername/ARSXAI9}
}
```

---

## 🙏 Danksagung

- **hmmlearn** - Für die HMM-Implementierung
- **sklearn-crfsuite** - Für die CRF-Implementierung
- **Graphviz** - Für Visualisierungen
- **NetworkX** - Für Graph-Analysen

---

**Entwickelt mit ❤️ für erklärbare Künstliche Intelligenz**
