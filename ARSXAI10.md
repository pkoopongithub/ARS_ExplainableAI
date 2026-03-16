# ARSXAI10 - Algorithmic Recursive Sequence Analysis mit Explainable AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XAI](https://img.shields.io/badge/XAI-Depth--Bounded_PCFG-green.svg)](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)
[![Version](https://img.shields.io/badge/version-10.0-red.svg)](.)

---

## 📖 Inhaltsverzeichnis

1. [Einführung und Vision](#-einführung-und-vision)
2. [Hauptmerkmale](#-hauptmerkmale)
3. [Neu in Version 10.0](#-neu-in-version-100)
4. [Installation](#-installation)
5. [Die zentrale Idee: Depth-Bounded PCFG](#-die-zentrale-idee-depth-bounded-pcfg)
6. [Integrierte Modelle](#-integrierte-modelle)
7. [Neue Verfahren im Detail](#-neue-verfahren-im-detail)
8. [Dateiformate](#-dateiformate)
9. [Benutzeroberfläche](#-benutzeroberfläche)
10. [Bedienungsanleitung](#-bedienungsanleitung)
11. [XAI-Erklärungen verstehen](#-xai-erklärungen-verstehen)
12. [Erweiterte Induktion](#-erweiterte-induktion)
13. [Beispiele](#-beispiele)
14. [Exportformate](#-exportformate)
15. [Fehlerbehebung](#-fehlerbehebung)
16. [Mitwirken](#-mitwirken)
17. [Lizenz](#-lizenz)

---

## 🎯 Einführung und Vision

**ARSXAI10** ist die evolutionäre Weiterentwicklung von ARSXAI9 und eine universelle Analyseplattform für sequenzielle Daten mit Fokus auf **Explainable AI (XAI)**. Die Kernidee bleibt: Wiederkehrende Muster in Sequenzen werden zu Nonterminalen abstrahiert und bilden die Grundlage für natürliche Erklärungen.

### Die zentrale Erkenntnis von Version 10
Nicht alle Muster sind gleich wichtig - und nicht alle sollten unbegrenzt tief geschachtelt werden. ARSXAI10 führt **Tiefenbeschränkung** und **MDL-Optimierung** ein, um die Interpretierbarkeit zu maximieren und Overfitting zu vermeiden.

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

### 🧠 **Depth-Bounded PCFG**
- Beschränkung der Hierarchietiefe (einstellbar 1-10)
- Verhindert Überanpassung an seltene Muster
- Kognitiv plausibel (beschränktes Arbeitsgedächtnis)

### 📊 **MDL-Optimierung (Minimum Description Length)**
- Kompression als Gütekriterium für Grammatiken
- Automatische Erkennung des optimalen Iterationsstopps
- Vergleich verschiedener Grammatiken nach Kompressionsrate

### 🔍 **SemInfo-Maximierung** (optional)
- Semantische Namen für Nonterminale mit Sentence-Transformers
- Kohärenzmessung für erkannte Muster
- Beispiel: `KOHÄRENT_2` statt `P_CBG_BBG_2`

### ⚡ **PrefixSpan für große Daten** (optional)
- Effiziente Mustersuche bei >1000 Ketten
- On-demand Installation über GUI
- Skaliert auf große Korpora

### 💬 **Natürlichsprachliche Erklärungen** (verbessert)
```text
🔍 Erklärung für Symbol 'CBBd':
============================================================
🔤 CBBd ist ein grundlegendes Symbol.

📊 Es kommt in folgenden wiederkehrenden Mustern vor:
  • P_CBBd_BBBd_2_d1 (Tiefe 1, 88% der Ketten):
    Position: nach nichts, vor BBBd

🏗️ Hierarchische Einbettung (max. Tiefe 5):
└─ in P_CBBd_BBBd_2_d1 (Tiefe 1)
  └─ in P_Doppel_4_d2 (Tiefe 2)
    └─ in P_GESAMT_10_d3 (Tiefe 3)

✅ Konfidenz: 95%
```

---

## ✨ Neu in Version 10.0

### ❌ **Optimiert**
- Die alte `GrammarInducer`-Klasse bleibt als Fallback erhalten
- Alle XAI-Erklärungen zeigen jetzt Tiefeninformationen
- Exportformate enthalten Tiefenstatistik

### ✅ **Hinzugefügt**
- **`DepthBoundedGrammarInducer`** - Tiefenbeschränkte Grammatikinduktion
- **`MDLOptimizer`** - Minimum Description Length Optimierung
- **`SemInfoMaximizer`** - Semantische Namen (optional)
- **Neuer GUI-Tab** "Erweiterte Induktion"
- **PrefixSpan-Integration** für große Daten
- **Tiefenstatistik** mit Verteilungsanalyse
- **Vergleichsfunktion** Depth-Bounded vs. Standard

### 🎯 **XAI-Verbesserungen**
- Erklärungen zeigen jetzt **Tiefe** jedes Nonterminals
- **Übersprungene Muster** werden dokumentiert
- **MDL-Scores** als Qualitätsmaß
- **Semantische Namen** wenn verfügbar

---

## 💻 Installation

### Systemvoraussetzungen

- **Python**: 3.8 oder höher
- **RAM**: 4 GB (empfohlen, 8 GB für SemInfo)
- **Festplatte**: 1 GB für Abhängigkeiten
- **OS**: Windows, macOS, Linux

### Automatische Installation

```bash
# Repository klonen
git clone https://github.com/yourusername/ARSXAI10.git
cd ARSXAI10

# ARSXAI9.py muss im gleichen Verzeichnis sein!
# Programm starten (Pakete werden automatisch installiert)
python ARSXAI10.py
```

### Wichtiger Hinweis
ARSXAI10.py **importiert ARSXAI9.py** und erweitert es. Stellen Sie sicher, dass beide Dateien im gleichen Verzeichnis liegen!

### Optionale Pakete

| Paket | Funktion | Installation |
|-------|----------|--------------|
| **prefixspan** | Effiziente Mustersuche für >1000 Ketten | `pip install prefixspan` |
| **sentence-transformers** | Semantische Namen | `pip install sentence-transformers` |

Beide können auch **während der Laufzeit** über den "Erweiterte Induktion"-Tab installiert werden.

---

## 🧠 Die zentrale Idee: Depth-Bounded PCFG

### 1. **Das Problem unbegrenzter Tiefe**

In ARSXAI9 konnte die Hierarchie theoretisch unbegrenzt wachsen:
```
P_CBG_BBG_2 → CBG, BBG
P_Doppel_4 → P_CBBd_BBBd_2, P_CBBd_BBBd_2
P_GESAMT_10 → P_CBG_BBG_2, P_Doppel_4, ...
P_SUPER_20 → P_GESAMT_10, P_GESAMT_10, ...
```

**Probleme:**
- Schwer interpretierbar (was bedeutet Tiefe 7?)
- Overfitting auf seltene Muster
- Kognitiv nicht plausibel (Menschen denken in begrenzten Hierarchien)

### 2. **Die Lösung: Tiefenbeschränkung**

ARSXAI10 führt eine **maximale Tiefe** ein (einstellbar 1-10):

```python
# Tiefe 1 (max_depth=1)
P_CBG_BBG_2_d1 → CBG, BBG
P_CBBd_BBBd_2_d1 → CBBd, BBBd

# Tiefe 2 (max_depth=2) - erlaubt, wenn max_depth=2
P_Doppel_4_d2 → P_CBBd_BBBd_2_d1, P_CBBd_BBBd_2_d1

# Tiefe 3 (max_depth=3) - erlaubt, wenn max_depth=3
P_GESAMT_10_d3 → P_CBG_BBG_2_d1, P_Doppel_4_d2, ...
```

### 3. **MDL-Optimierung: Wann stoppen?**

Das MDL-Prinzip (Minimum Description Length) besagt: Die beste Grammatik ist die, die die Daten am stärksten komprimiert.

```python
# Kompressionsgewinn pro Iteration
Iteration 1: 50% Kompression
Iteration 2: 30% Kompression (zusätzlich)
Iteration 3: 10% Kompression  
Iteration 4: 2% Kompression   ← Stopp hier (Elbow)
```

Der **optimale Stopppunkt** wird automatisch erkannt.

---

## 📦 Integrierte Modelle

| Modell | Klasse | Beschreibung | Status |
|--------|--------|--------------|--------|
| **Depth-Bounded PCFG** | `DepthBoundedGrammarInducer` | Tiefenbeschränkte Grammatik | ⭐ Hauptmodell (neu) |
| **ARS 3.0** | `GrammarInducer` | Hierarchische PCFG (unbegrenzt) | 🔧 Fallback |
| **ARS 2.0** | `ARS20` | Einfache Bigramm-Wahrscheinlichkeiten | 🔧 Optional |
| **HMM** | `ARSHiddenMarkovModel` | Latente Phasen | 🔧 Optional |
| **CRF** | `ARSCRFModel` | Kontext-sensitive Features | 🔧 Optional |
| **Petri-Netz** | `ARSPetriNet` | Ressourcen-basierte Modellierung | 🔧 Optional |
| **Generator** | `ChainGenerator` | Synthetische Ketten | 🔧 Optional |

---

## 🔬 Neue Verfahren im Detail

### 1. **Depth-BoundedGrammarInducer**

```python
class DepthBoundedGrammarInducer(GrammarInducer):
    """
    Parameter:
        max_depth=5: Maximale Hierarchietiefe
        use_mdl=True: MDL-Optimierung aktivieren
        use_prefixspan=False: PrefixSpan für große Daten
        use_seminfo=False: Semantische Namen (benötigt sentence-transformers)
    """
    
    # Neue Methoden:
    - get_depth_statistics()  # Tiefenverteilung anzeigen
    - get_mdl_statistics()    # MDL-Kompressionsstatistik
    - compare_with_standard() # Vergleich mit Standard-Grammatik
```

### 2. **MDL-Optimizer**

```python
mdl = MDLOptimizer()

# Kompressionsrate berechnen
ratio = mdl.calculate_compression_ratio(chains, grammar)

# Zwei Grammatiken vergleichen
comparison = mdl.compare_grammars(grammar1, grammar2, chains)

# Optimalen Stopppunkt finden
cutoff = mdl.optimal_cutoff(compression_gains)
```

### 3. **SemInfo-Maximizer**

```python
seminfo = SemInfoMaximizer()

# Semantische Kohärenz einer Sequenz
coherence = seminfo.semantic_coherence(["CBG", "BBG", "CBBd"])

# Semantischen Namen vorschlagen
name = seminfo.suggest_name(["CBG", "BBG"])  # → "KOHÄRENT_2"
```

---

## 📁 Dateiformate

### Eingabeformat (wie in ARSXAI9)

```txt
# Kommentare beginnen mit #
# Trennzeichen: Komma, Semikolon oder Leerzeichen

# Transkript 1
CBG, BBG, CBBd, BBBd, CBA, BBA, CBBd, BBBd, CBA, BAA, CAA, BAB, CAB

# Transkript 2  
CBG, BBG, CBBd, BBBd, CBBd, BBBd, CBA, BBA, BAA, CAA, BAB, CAB
```

### Tiefeninformation im Export

```json
{
  "grammar": {
    "patterns": [...],
    "depth_statistics": {
      "max_depth": 5,
      "depth_distribution": {"1": 12, "2": 5, "3": 2}
    }
  }
}
```

---

## 🖥️ Benutzeroberfläche

### Neuer Tab: "Erweiterte Induktion"

```
┌─────────────────────────────────────────────────────────────────┐
│ ARSXAI10 - Depth-Bounded PCFG mit XAI                                │
├──────────────────────┬──────────────────────────────────────────┤
│ EINGABE              │ AUSGABE (Notebook-Tabs)                  │
│                      │ ┌─────────────────────────────────────┐ │
│ Trennzeichen: [Komma]│ │ Grammatik | Muster | XAI | Modelle  │ │
│                      │ │ Erweiterte Induktion [NEU]          │ │
├──────────────────────┴──────────────────────────────────────────┤
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ ▤ ERWEITERTE INDUKTION                                       │ │
│ │                                                              │ │
│ │ [Param
