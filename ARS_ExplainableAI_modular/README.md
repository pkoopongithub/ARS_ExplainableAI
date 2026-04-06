# ARSXAI - Algorithmic Recursive Sequence Analysis with Explainable AI

## English Version

### Overview

ARSXAI is a comprehensive framework for pattern discovery and explainable AI in sequential data. It induces hierarchical context-free grammars from sequences of symbols, identifies recurring patterns, and provides natural language explanations of the discovered structures.

**Version:** 10.0 (Modular Architecture)

### Core Features

- **Hierarchical Grammar Induction** - Automatically discovers repeating patterns and abstracts them into nonterminals
- **Natural Language Explanations** - Human-readable explanations of patterns and their roles
- **Multiple AI Models** - HMM, CRF, Petri Nets, and baseline models for comparison
- **Interactive GUI** - Full graphical interface for analysis and visualization
- **Multi-format Export** - JSON, HTML, and LaTeX reports

### Architecture

```
arsxai/
├── ARSXAI.py                   # Complete reference implementation
├── arsxai_ext_depth.py         # Depth-bounded PCFG extension
├── arsxai_ext_mdl.py           # MDL optimization extension
├── arsxai_ext_prefixspan.py    # PrefixSpan for large corpora
└── arsxai_ext_seminfo.py       # Semantic naming extension
```

### Installation

```bash
# Clone or download the package
pip install numpy scipy matplotlib hmmlearn scikit-learn networkx

# For extensions (optional)
pip install sentence-transformers   # for semantic names
pip install prefixspan               # for large datasets
```

### Quick Start

```python
from arsxai import GrammarInducer, NaturalLanguageExplainer

# Prepare your sequences
chains = [
    ["A", "B", "C", "A", "B", "C"],
    ["A", "B", "C", "D"],
    ["A", "B", "C", "A", "B", "C", "D"]
]

# Induce grammar
grammar = GrammarInducer()
grammar.train(chains)

# Get explanations
explainer = NaturalLanguageExplainer(grammar)
print(explainer.explain_symbol("A"))
print(explainer.explain_sequence(["A", "B", "C"]))

# View grammar
print(grammar.get_grammar_string())
```

### GUI Usage

```bash
python -m arsxai.ARSXAI
```

The GUI provides:
- Input panel for sequence data
- Grammar visualization
- Pattern summary
- XAI explanations
- Model comparison
- Statistics and export

### Extensions

#### Depth-Bounded PCFG (`arsxai_ext_depth.py`)

Limits hierarchical depth during grammar induction to prevent overly complex structures.

```python
from arsxai_ext_depth import DepthBoundedGrammarInducer

inducer = DepthBoundedGrammarInducer(max_depth=5)
inducer.train(chains)
print(inducer.get_depth_statistics())
```

#### MDL Optimization (`arsxai_ext_mdl.py`)

Uses Minimum Description Length principle to evaluate grammar quality and find optimal compression.

```python
from arsxai_ext_mdl import MDLOptimizer, MDLGrammarInducer

optimizer = MDLOptimizer()
ratio = optimizer.calculate_compression_ratio(chains, grammar)
print(f"Compression: {ratio:.1%}")
```

#### PrefixSpan Extension (`arsxai_ext_prefixspan.py`)

Optimized pattern mining for large datasets (>1000 sequences).

```python
from arsxai_ext_prefixspan import PrefixSpanGrammarInducer

inducer = PrefixSpanGrammarInducer(min_support=2)
inducer.train(large_chains)  # Efficient for large corpora
```

#### Semantic Naming (`arsxai_ext_seminfo.py`)

Uses sentence-transformers to generate meaningful names for discovered patterns.

```python
from arsxai_ext_seminfo import SemInfoGrammarInducer

inducer = SemInfoGrammarInducer(model_name='paraphrase-multilingual-MiniLM-L12-v2')
inducer.train(chains)
print(inducer.get_seminfo_status())
```

### Input Format

Sequences are provided as lines with symbols separated by delimiter:

```
# Comments start with hash
A, B, C, D, E
A, B, C, A, B, C
X, Y, Z
```

Supported delimiters: comma `,`, semicolon `;`, space, or custom.

### Output Formats

- **JSON** - Machine-readable complete analysis
- **HTML** - Human-readable report with styling
- **LaTeX** - Academic paper ready format

### Model Comparison

ARSXAI includes multiple models for comparative XAI:

| Model | Description | Best for |
|-------|-------------|----------|
| ARS 3.0 | Hierarchical Grammar | Pattern discovery, explanations |
| ARS 2.0 | Bigram transitions | Simple sequence prediction |
| HMM | Hidden Markov Model | Latent phase detection |
| CRF | Conditional Random Fields | Context-sensitive labeling |
| Petri Net | Resource-based model | Process validation |

### XAI Features

- **Symbol Explanations** - Understand what any symbol represents in the grammar
- **Sequence Explanations** - See hierarchical structure of any sequence
- **Pattern Summaries** - Overview of all discovered recurring patterns
- **Model Comparison** - Compare explanations across different AI models
- **Confidence Scores** - Quantitative measure of each analysis

### Requirements

**Core:**
- Python 3.7+
- numpy, scipy
- matplotlib
- tkinter (included with Python)

**Optional:**
- hmmlearn (for HMM)
- sklearn-crfsuite (for CRF)
- networkx (for graph visualization)
- sentence-transformers (for semantic names)
- prefixspan (for large datasets)
- graphviz (for automaton visualization)

### License

© 2024 - Explainable AI Research

---

## Deutsche Version

### Überblick

ARSXAI ist ein umfassendes Framework für Mustererkennung und erklärbare KI in sequenziellen Daten. Es induziert hierarchische kontextfreie Grammatiken aus Symbolsequenzen, identifiziert wiederkehrende Muster und liefert natürlichsprachliche Erklärungen der entdeckten Strukturen.

**Version:** 10.0 (Modulare Architektur)

### Kernfunktionen

- **Hierarchische Grammatikinduktion** - Automatische Erkennung wiederholter Muster und Abstraktion zu Nonterminalen
- **Natürlichsprachliche Erklärungen** - Menschenlesbare Erklärungen von Mustern und ihrer Rolle
- **Mehrere KI-Modelle** - HMM, CRF, Petri-Netze und Basis-Modelle zum Vergleich
- **Interaktive GUI** - Vollständige grafische Oberfläche für Analyse und Visualisierung
- **Multi-Format Export** - JSON-, HTML- und LaTeX-Berichte

### Architektur

```
arsxai/
├── ARSXAI.py                   # Vollständige Referenzimplementierung
├── arsxai_ext_depth.py         # Tiefenbeschränkte PCFG-Erweiterung
├── arsxai_ext_mdl.py           # MDL-Optimierungs-Erweiterung
├── arsxai_ext_prefixspan.py    # PrefixSpan für große Korpora
└── arsxai_ext_seminfo.py       # Semantische Namens-Erweiterung
```

### Installation

```bash
# Paket herunterladen oder klonen
pip install numpy scipy matplotlib hmmlearn scikit-learn networkx

# Für Erweiterungen (optional)
pip install sentence-transformers   # für semantische Namen
pip install prefixspan               # für große Datenmengen
```

### Schnellstart

```python
from arsxai import GrammarInducer, NaturalLanguageExplainer

# Ihre Sequenzen vorbereiten
chains = [
    ["A", "B", "C", "A", "B", "C"],
    ["A", "B", "C", "D"],
    ["A", "B", "C", "A", "B", "C", "D"]
]

# Grammatik induzieren
grammar = GrammarInducer()
grammar.train(chains)

# Erklärungen erhalten
explainer = NaturalLanguageExplainer(grammar)
print(explainer.explain_symbol("A"))
print(explainer.explain_sequence(["A", "B", "C"]))

# Grammatik anzeigen
print(grammar.get_grammar_string())
```

### GUI-Nutzung

```bash
python -m arsxai.ARSXAI
```

Die GUI bietet:
- Eingabefeld für Sequenzdaten
- Grammatik-Visualisierung
- Musterübersicht
- XAI-Erklärungen
- Modellvergleich
- Statistiken und Export

### Erweiterungen

#### Tiefenbeschränkte PCFG (`arsxai_ext_depth.py`)

Begrenzt die hierarchische Tiefe bei der Grammatikinduktion, um übermäßig komplexe Strukturen zu vermeiden.

```python
from arsxai_ext_depth import DepthBoundedGrammarInducer

inducer = DepthBoundedGrammarInducer(max_depth=5)
inducer.train(chains)
print(inducer.get_depth_statistics())
```

#### MDL-Optimierung (`arsxai_ext_mdl.py`)

Verwendet das Minimum Description Length Prinzip zur Bewertung der Grammatikqualität und optimalen Kompression.

```python
from arsxai_ext_mdl import MDLOptimizer, MDLGrammarInducer

optimizer = MDLOptimizer()
ratio = optimizer.calculate_compression_ratio(chains, grammar)
print(f"Kompression: {ratio:.1%}")
```

#### PrefixSpan-Erweiterung (`arsxai_ext_prefixspan.py`)

Optimierte Mustererkennung für große Datensätze (>1000 Sequenzen).

```python
from arsxai_ext_prefixspan import PrefixSpanGrammarInducer

inducer = PrefixSpanGrammarInducer(min_support=2)
inducer.train(grosse_ketten)  # Effizient für große Korpora
```

#### Semantische Namen (`arsxai_ext_seminfo.py`)

Verwendet Sentence-Transformers zur Generierung bedeutungsvoller Namen für entdeckte Muster.

```python
from arsxai_ext_seminfo import SemInfoGrammarInducer

inducer = SemInfoGrammarInducer(model_name='paraphrase-multilingual-MiniLM-L12-v2')
inducer.train(chains)
print(inducer.get_seminfo_status())
```

### Eingabeformat

Sequenzen werden als Zeilen mit durch Trennzeichen getrennten Symbolen eingegeben:

```
# Kommentare beginnen mit Doppelkreuz
A, B, C, D, E
A, B, C, A, B, C
X, Y, Z
```

Unterstützte Trennzeichen: Komma `,`, Semikolon `;`, Leerzeichen oder benutzerdefiniert.

### Ausgabeformate

- **JSON** - Maschinenlesbare vollständige Analyse
- **HTML** - Menschenlesbarer Bericht mit Formatierung
- **LaTeX** - Für wissenschaftliche Arbeiten geeignet

### Modellvergleich

ARSXAI enthält mehrere Modelle für vergleichende XAI:

| Modell | Beschreibung | Am besten für |
|--------|--------------|---------------|
| ARS 3.0 | Hierarchische Grammatik | Mustererkennung, Erklärungen |
| ARS 2.0 | Bigramm-Übergänge | Einfache Sequenzvorhersage |
| HMM | Hidden Markov Model | Erkennung latenter Phasen |
| CRF | Conditional Random Fields | Kontextsensitive Markierung |
| Petri-Netz | Ressourcenbasiertes Modell | Prozessvalidierung |

### XAI-Funktionen

- **Symbolerklärungen** - Verstehen, was jedes Symbol in der Grammatik repräsentiert
- **Sequenzerklärungen** - Hierarchische Struktur jeder Sequenz sehen
- **Musterübersichten** - Überblick über alle entdeckten wiederkehrenden Muster
- **Modellvergleich** - Erklärungen verschiedener KI-Modelle vergleichen
- **Konfidenzwerte** - Quantitative Maßzahl jeder Analyse

### Voraussetzungen

**Kern:**
- Python 3.7+
- numpy, scipy
- matplotlib
- tkinter (in Python enthalten)

**Optional:**
- hmmlearn (für HMM)
- sklearn-crfsuite (für CRF)
- networkx (für Graphvisualisierung)
- sentence-transformers (für semantische Namen)
- prefixspan (für große Datensätze)
- graphviz (für Automatenvisualisierung)

