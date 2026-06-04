# ARS_ExplainableAI

**Algorithmic Recursive Sequence Analysis for Explainable AI in Qualitative Social Research**

---

## 🔑 Kernbotschaft / Core Message

> **"Explainability is not a luxury – neither in AI nor in qualitative research."**
>
> *„Erklärbarkeit ist kein Luxus – weder in der KI noch in der qualitativen Forschung.“*

---

## 📋 Overview (English)

**ARS_ExplainableAI** is a methodological and software-based framework for **Algorithmic Recursive Sequence Analysis (ARS)**. It integrates qualitative hermeneutics with formal modeling and contributes to **Explainable Artificial Intelligence (XAI)** in text analysis.

### What problem does it solve?

Qualitative social research faces a methodological dilemma: Generative AI systems promise scalability but evade classical validation due to their opacity. ARS bridges this gap by making interpretation processes **explicit, decidable, and reproducible**.

### This repository contains:

| Category | Content |
|----------|---------|
| **Scientific Papers** | Complete publications on ARS methodology (German / English) |
| **Python Code** | Grammar induction from terminal symbol sequences |
| **Network Models** | Transformation into Petri nets and Bayesian networks |
| **Compression Principles** | Repetition, recursion, symmetry, hierarchy |
| **Optimization** | Iterative adjustment of transition probabilities |
| **Empirical Data** | Eight transcripts of sales conversations (Aachen market, 1994) |

---

## 📋 Überblick (Deutsch)

**ARS_ExplainableAI** ist ein methodologisches und softwaretechnisches Framework zur **Algorithmisch Rekursiven Sequenzanalyse (ARS)**. Es verbindet qualitative Hermeneutik mit formaler Modellierung und leistet einen Beitrag zur **erklärbaren Künstlichen Intelligenz (XAI)** in der Textanalyse.

### Welches Problem wird gelöst?

Die qualitative Sozialforschung steht vor einem methodologischen Dilemma: Generative KI-Systeme versprechen Skalierung, entziehen sich jedoch aufgrund ihrer Opazität der klassischen Validierung. Die ARS überbrückt diese Lücke, indem sie Interpretationsprozesse **explizit, entscheidbar und reproduzierbar** macht.

### Dieses Repository enthält:

| Kategorie | Inhalt |
|-----------|--------|
| **Wissenschaftliche Aufsätze** | Vollständige Publikationen zur ARS-Methodologie (Deutsch/Englisch) |
| **Python-Code** | Grammatikinduktion aus Terminalzeichenketten |
| **Netzmodelle** | Transformation in Petri-Netze und Bayessche Netze |
| **Komprimierungsprinzipien** | Wiederholung, Rekursion, Symmetrie, Hierarchie |
| **Optimierung** | Iterative Anpassung von Übergangswahrscheinlichkeiten |
| **Empirische Daten** | Acht Transkripte von Verkaufsgesprächen (Aachener Markt, 1994) |

---

## 🎯 Objectives (English)

Qualitative social research faces a methodological dilemma: Generative AI systems promise scalability but evade classical validation due to their opacity.

**ARS_ExplainableAI** addresses this challenge through:

- **Transparent model construction** – every interpretative step is explicitly documented
- **Formalization of qualitative processes** – transformation of interpretations into terminal symbol sequences
- **Explainable network models** – compressive transformation into Petri and Bayesian networks
- **Recursive self-application** – AI as an epistemic agent reflecting on its own interpretations

---

## 🎯 Zielsetzung (Deutsch)

Die qualitative Sozialforschung steht vor einem methodologischen Dilemma: Generative KI-Systeme versprechen Skalierung, entziehen sich jedoch aufgrund ihrer Opazität der klassischen Validierung.

**ARS_ExplainableAI** begegnet diesem Problem durch:

- **Transparente Modellbildung** – jeder Interpretationsschritt wird explizit dokumentiert
- **Formalisierung qualitativer Prozesse** – Überführung von Lesarten in Terminalzeichenketten
- **Erklärbare Netzmodelle** – komprimierende Transformation in Petri- und Bayessche Netze
- **Rekursive Selbstanwendung** – KI als epistemischer Akteur, der eigene Interpretationen reflektiert

---

## 📊 Methodological Transparency

> **Note on Intercoder Reliability (1994 study):**  
> The original ARS study achieved a Cohen's Kappa of **κ ≈ 0.55** – a value that highlights the limits of purely qualitative coding. ARS does not hide this weakness; it makes it the **starting point of methodological reflection**. Formal procedures make these limits visible and tractable.

---

## 🧩 How ARS Works (Mini Demo)

A sales conversation is transcribed and each speech act is assigned a terminal symbol:

```
KBG → VBG → KBBd → VBBd → KBA → VBA → KBBd → VBBd → KBA → VAA → KAA → VAV → KAV
```

| Symbol | Meaning |
|--------|---------|
| KBG | Customer greeting |
| VBG | Seller greeting |
| KBBd | Customer needs (concrete) |
| VBBd | Seller inquiry |
| KBA | Customer response |
| VBA | Seller reaction |
| KAA | Customer closing |
| VAA | Seller closing |
| KAV | Customer farewell |
| VAV | Seller farewell |

From this sequence, ARS induces a **probabilistic context-free grammar (PCFG)**. Every decision is documented, traceable, and formally verifiable.

---

## 📁 Repository Structure

```
ARS_ExplainableAI/
├── docs/                       # Scientific papers (PDF + TeX)
│   ├── ARS_XAI_Ger.pdf
│   ├── ARS_XAI_Eng.pdf
│   ├── ARS_XAI_PCFG_Ger.pdf
│   ├── ARS_XAI_Petri_Ger.pdf
│   ├── ARS_XAI_Bayes_Ger.pdf
│   ├── ARS_XAI_CL_Ger.pdf
│   └── ...
├── src/                        # Python implementations
│   ├── grammar_inducer.py
│   ├── petri_net_transformer.py
│   ├── bayesian_network.py
│   └── ...
├── data/                       # Empirical transcripts
│   └── transcripts_1994.json
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required packages: `numpy`, `scikit-learn`, `networkx`, `torch` (for CL components)

### Installation

```bash
git clone https://github.com/pkoopongithub/ARS_ExplainableAI.git
cd ARS_ExplainableAI
pip install -r requirements.txt
```

### Basic Usage

```python
from src.grammar_inducer import GrammarInducer

# Load empirical terminal chains
chains = [...]  # Your sequences

# Induce grammar
inducer = GrammarInducer()
compressed = inducer.induce_grammar(chains)

# View induced rules
print(inducer.rules)
```

---

## 📚 Documentation

All scientific papers are available in `docs/` as **PDF (print-ready)** and **TeX (source code)**. The TeX files allow full traceability and adaptation for your own research.

| Document | Content | Language |
|----------|---------|----------|
| `ARS_XAI` | Main framework: Between interpretation and computation | DE/EN |
| `ARS_XAI_PCFG` | Hierarchical grammar induction (ARS 3.0) | DE/EN |
| `ARS_XAI_Petri` | Concurrency modeling with Petri nets (ARS 4.0) | DE/EN |
| `ARS_XAI_Bayes` | HMM and dynamic Bayesian networks (ARS 4.0) | DE/EN |
| `ARS_XAI_CL` | Didactic exploration of Transformers, CRF, Attention | DE/EN |
| `ARS_XAI_Hybrid` | Complementary integration of CL methods | DE/EN |

---

## 🤝 Contributing / Collaboration

**This framework is methodologically mature but empirically underdetermined.**

If you have access to larger datasets, are interested in methodological development, or want to apply ARS to new domains (doctor-patient interactions, classroom discourse, online conversations) – I warmly invite you to collaborate.

- **Open Issues**: Check the [issue tracker](https://github.com/pkoopongithub/ARS_ExplainableAI/issues)
- **Contact**: [post@paul-koop.org](mailto:post@paul-koop.org)

---

## 📖 Citation

If you use ARS_ExplainableAI in your research, please cite:

```bibtex
@misc{koop2024ars,
  author = {Koop, Paul},
  title = {Algorithmic Recursive Sequence Analysis (ARS) as a Framework for Explainable AI},
  year = {2024/2026},
  url = {https://the-last-freedom.org/algorithmisch-rekursive-sequenzanalyse/ARS_ExplainableAI/},
  note = {Open access: PDF and TeX available}
}
```

---

## 📄 License

**Creative Commons BY-NC-SA 4.0** – Free use for non‑commercial research and education with attribution and share‑alike.

---

## 🔗 Links

| Platform | Link |
|----------|------|
| 🌐 Project Website | [arsxai.org](https://the-last-freedom.org/algorithmisch-rekursive-sequenzanalyse/ARS_ExplainableAI/) |
| 🐙 GitHub | [pkoopongithub/ARS_ExplainableAI](https://github.com/pkoopongithub/ARS_ExplainableAI) |
| 🦊 GitLab | [pkoop/algorithmisch-rekursive-sequenzanalyse](https://gitlab.com/pkoop/algorithmisch-rekursive-sequenzanalyse) |
| 📄 OverLeaf | [Read-only project](https://www.overleaf.com/read/hvktxktfkzmx#4629e6) |

---

## 📅 Historical Note

The empirical foundation of this project consists of **eight transcripts of sales conversations** recorded at Aachen market square in **June/July 1994**. The original coding sheets with handwritten codings by two independent coders are included in `docs/fallstruktur.pdf`. This historical material serves as a transparent basis for reliability calculations (κ ≈ 0.55) and methodological reflection.

---

*„Explainability is not a luxury – neither in AI nor in qualitative research.“*
