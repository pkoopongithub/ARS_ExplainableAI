---
abstract: |
  The integration of connectionist and symbolic methods---neuro-symbolic AI---represents one of the most promising research programs in contemporary artificial intelligence. At the same time, the Algorithmic Recursive Sequence Analysis (ARS) has developed a formal framework that transforms qualitative interpretive processes into explainable, intersubjectively verifiable models (PCFG, Petri nets, Bayesian networks, finite automata). This paper examines the reciprocal relationship between these two paradigms. It argues that neuro-symbolic AI can benefit from ARS as a methodologically controlled procedure for rule induction and symbol grounding, while ARS---particularly in its XAI-oriented versions---can benefit from neuro-symbolic methods for scaling, learning under uncertainty, and the integration of subsymbolic representations. The synthesis developed here does not blur the boundaries between paradigms but sharpens them: ARS provides the *symbolic scaffolding*, neuro-symbolic methods provide the *learning dynamics*. Methodological control remains with the human researcher.

  Die Integration von konnektionistischen und symbolischen Methoden -- neuro-symbolische KI -- ist eines der vielversprechendsten Forschungsprogramme der gegenwärtigen künstlichen Intelligenz. Zugleich hat die Algorithmisch Rekursive Sequenzanalyse (ARS) ein formales Framework entwickelt, das qualitative Interpretationsprozesse in erklärbare, intersubjektiv prüfbare Modelle (PCFG, Petri-Netze, Bayessche Verfahren, endliche Automaten) überführt. Der vorliegende Beitrag untersucht das wechselseitige Verhältnis dieser beiden Paradigmen. Er argumentiert, dass die neuro-symbolische KI von der ARS als methodologisch kontrolliertem Verfahren der Regelinduktion und Symbolverankerung profitieren kann, während die ARS -- insbesondere in ihren XAI-orientierten Versionen -- von neuro-symbolischen Methoden durch Skalierung, Lernen unter Unsicherheit und die Integration subsymbolischer Repräsentationen profitieren kann. Die hier entwickelte Synthese verwischt die Grenzen zwischen den Paradigmen nicht, sondern schärft sie: Die ARS liefert das *symbolische Gerüst*, neuro-symbolische Methoden liefern die *Lern dynamik*. Die methodologische Kontrolle verbleibt beim menschlichen Forscher.
author:
- Paul Koop
date: 2026
title: |
  **Neuro-Symbolic KI und ARS**\
  Eine methodologische Synthese von\
  maschinellem Lernen und erklärbarer Sequenzanalyse
---

# Introduction: Two Paradigms, One Problem

## The Neuro-Symbolic Research Program

Neuro-symbolic AI has established itself as a research program that integrates neural methods (deep learning, pattern recognition, subsymbolic representations) with symbolic methods (formal logic, knowledge representation, rule-based reasoning) [@hitzler2022neuro; @garcez2020neurosymbolic]. The fundamental insight is that neither paradigm alone is sufficient:

- **Neural methods** excel at pattern recognition, learning from noisy data, and generalization but suffer from opacity, lack of explainability, and hallucination.

- **Symbolic methods** excel at reasoning, planning, and explainability but suffer from brittleness, the knowledge acquisition bottleneck, and difficulties with noisy or ambiguous data.

The synthesis promises systems that combine the learning capabilities of neural networks with the reasoning capabilities of symbolic systems. Gary Marcus argues that \"hybrid architectures combining learning and symbol manipulation are necessary---though not sufficient---for robust intelligence\" [@marcus2020next]. Henry Kautz's taxonomy of neuro-symbolic architectures [@kautz2020third] provides a framework for understanding the different integration modalities:

- **Neural \| Symbolic**: Neural perception, symbolic reasoning

- **Neural: Symbolic → Neural**: Symbolic generation of training data

- **NeuralSymbolic**: Neural networks generated from symbolic rules

- **Neural\[Symbolic\]**: Symbolic reasoning embedded in neural networks

## The ARS Research Program

Algorithmic Recursive Sequence Analysis (ARS), in its versions 2.0 to 4.0, has developed a formal framework for the analysis of sequential interactions [@koop2024ars]. The core innovation is the transformation of qualitative hermeneutic interpretation into formal, explainable models:

- **ARS 2.0/3.0**: Induction of probabilistic context-free grammars (PCFG) from terminal symbol strings through hierarchical compression

- **ARS 4.0 (Petri)**: Modeling of concurrency and resources through Petri nets

- **ARS 4.0 (Bayes)**: Modeling of uncertainty and latent variables through Hidden Markov Models

- **ARS 4.0 (Hybrid)**: Complementary integration of computational linguistics methods (CRF, Transformer embeddings, GNN, Attention)

A distinctive feature of ARS is its commitment to **explainability by design**: every interpretive decision is documented, every formal model is semantically meaningful, and the entire process is intersubjectively traceable. This fulfills the XAI criteria of meaningfulness, accuracy, and knowledge limits [@ortigossa2024xai].

## The Question of This Paper

Despite their different origins---neuro-symbolic AI from computer science, ARS from qualitative social research---both paradigms share a fundamental interest: the integration of statistical learning (or pattern recognition) with symbolic structures (or interpretative categories). This paper asks two reciprocal questions:

1.  **How can neuro-symbolic AI benefit from ARS?** Specifically: What does ARS offer as a methodologically controlled procedure for rule induction, symbol grounding, and XAI-oriented validation?

2.  **How can ARS benefit from neuro-symbolic AI?** Specifically: How can ARS overcome its limitations---small sample size, manual effort, lack of scalability---through neuro-symbolic integration?

# The Relationship Between ARS and Neuro-Symbolic AI

## Common Ground: The Integration of Pattern and Rule

Both ARS and neuro-symbolic AI address the same fundamental challenge: the integration of *pattern-based* and *rule-based* cognition. Daniel Kahneman's distinction between System 1 (fast, intuitive, pattern-based) and System 2 (slow, deliberative, rule-based) provides a useful framework [@kahneman2011thinking]:

::: {#tab:kahneman}
  **Dimension**        **ARS**                                                                  **Neuro-Symbolic AI**
  -------------------- ------------------------------------------------------------------------ -------------------------------------------------------------------
  System 1 (Pattern)   Empirical transition frequencies, Transformer embeddings, CRF features   Neural networks, pattern recognition, subsymbolic representations
  System 2 (Rule)      PCFG grammar rules, Petri net transitions, DFA states                    Symbolic logic, rule bases, knowledge graphs
  Integration          Hierarchical compression (ARS 3.0), hybrid modeling (ARS 4.0)            Kautz taxonomies (Neural\|Symbolic, NeuralSymbolic, etc.)
  Explainability       Explanation by design (ad-hoc)                                           Post-hoc or hybrid

  : System 1 and System 2 in ARS and Neuro-Symbolic AI
:::

## Key Differences: Epistemology and Methodology

Despite common ground, significant differences remain:

1.  **Epistemology**: Neuro-symbolic AI typically assumes that symbolic rules are *discovered* from data. ARS assumes that rules are *constructed* through interpretation and must be validated against empirical material. This difference is not merely philosophical but has methodological consequences.

2.  **Role of the human**: In most neuro-symbolic systems, the human is external---designing architectures, providing training data, evaluating results. In ARS, the human is *constitutively* part of the method: interpretation is a human act that cannot be fully automated.

3.  **Validation criteria**: Neuro-symbolic systems are typically validated through accuracy metrics on held-out data. ARS is validated through intersubjective traceability, communicative validation, and structural fit.

## Complementarity Rather Than Competition

These differences suggest that ARS and neuro-symbolic AI are not competitors but *complements*. Neuro-symbolic AI excels at the automatic extraction of patterns from large datasets. ARS excels at the methodologically controlled construction of symbolic models from small datasets. Their integration is therefore not a zero-sum game but a win-win.

# How Neuro-Symbolic AI Benefits from ARS

## Methodologically Controlled Rule Induction

One of the central problems of neuro-symbolic AI is the **symbol grounding problem**---the question of how symbols acquire meaning. ARS offers a solution: symbols (terminal symbols, nonterminals) are not arbitrary labels but are *interpretively grounded*. Each terminal symbol (KBG, KBBd, VAA, etc.) has a documented qualitative meaning derived from the interpretive analysis.

For neuro-symbolic AI, this means that the ARS procedure can serve as a methodologically controlled **rule induction engine**:

1.  Interpretive formation of terminal symbols (ARS Phase 1-2)

2.  Hierarchical compression into nonterminals (ARS 3.0)

3.  Formal modeling as PCFG, Petri net, or DFA (ARS 4.0)

4.  XAI validation of the induced rules

This contrasts with purely data-driven rule extraction, which often produces rules that are statistically correct but semantically meaningless or even misleading.

## XAI-Grounded Symbolic Scaffolding

Neuro-symbolic systems often suffer from what Dreyfus called the \"illusion of cognitive transparency\" [@dreyfus1972what]: the assumption that looking deeply enough into a system's internal calculations reveals its understanding. ARS counters this by providing **XAI-grounded symbolic scaffolding**:

- **Meaningfulness**: Every symbol is semantically interpretable

- **Transparency**: Every rule is documented with its rationale

- **Traceability**: Every inference step can be reconstructed

For neuro-symbolic AI, adopting ARS principles means that the symbolic component is not just formally correct but also *interpretively valid*. This is particularly important for applications in the social sciences, medicine, law, and other areas where decisions must be justified to human stakeholders.

## The DFA as a Neuro-Symbolic Interface

The deterministic finite automaton (DFA) developed in `ARS_XAI_Aut_Ger.tex` offers a particularly clean interface between neural and symbolic components:

``` {caption="DFA as Neuro-Symbolic Interface"}
# The DFA defines the symbolic structure
class ARSDFA:
    def __init__(self):
        self.states = ['q0', 'qBG', 'qB', 'qA', 'qAV', 'q_perp']
        self.accepting = ['qAV']
        self.transitions = {
            ('q0', 'KBG'): 'qBG', ('qBG', 'VBG'): 'qBG',
            ('qBG', 'KBBd'): 'qB', ('qB', 'VBBd'): 'qB',
            # ... complete transition function
        }
    
    def accepts(self, sequence):
        state = 'q0'
        for symbol in sequence:
            state = self.transitions.get((state, symbol), 'q_perp')
        return state in self.accepting
```

In a neuro-symbolic architecture, this DFA can serve as:

- A **constraint** for neural predictions (filtering invalid sequences)

- A **training signal** for neural sequence models (rewarding well-formedness)

- An **explanation interface** for neural decisions (mapping predictions to symbolic states)

## Validation Through ARS Quality Criteria

Neuro-symbolic systems are typically evaluated through accuracy, F1-score, or other quantitative metrics. ARS offers a complementary validation framework based on qualitative quality criteria:

1.  **Intersubjective traceability**: Can another researcher follow the reasoning?

2.  **Reflexivity**: Are the interpretation decisions documented and justified?

3.  **Structural fit**: Does the symbolic model reproduce the observed structure?

4.  **Communicative validation**: Do domain experts agree with the interpretation?

These criteria can be applied to the symbolic component of a neuro-symbolic system, providing a richer validation than accuracy metrics alone.

# How ARS Benefits from Neuro-Symbolic AI

## Scaling Through Neural Pattern Recognition

A central limitation of ARS (particularly in its CGTI and XAI versions) is the high manual effort required for sequential microanalysis. Phase 2 (interpretation) and Phase 4 (systematic case comparison) are labor-intensive and limit the scalability of the method to large corpora.

Neuro-symbolic methods can address this limitation through **neural pattern recognition**:

1.  **Neural pre-labeling**: A neural network (e.g., a fine-tuned transformer) can propose terminal symbols for each utterance.

2.  **Symbolic validation**: The ARS DFA or PCFG checks the well-formedness of the proposed sequence.

3.  **Discrepancy resolution**: Cases where the neural proposal violates structural rules are flagged for human review.

This creates a **human-in-the-loop neuro-symbolic system** that maintains methodological control while scaling to larger datasets. The neural component does not replace the human interpreter but works as a heuristic assistant.

## Learning Under Uncertainty

ARS 4.0 already incorporates Bayesian methods (HMM, DBN) for modeling uncertainty [@koop2024bayes]. However, these models are estimated from small samples (n = 8 in the empirical example). Neuro-symbolic methods can enhance this:

- **Neural estimation of transition probabilities**: A neural network can learn transition probabilities from larger datasets while respecting the symbolic structure defined by ARS.

- **DeepProbLog integration**: ARS grammars could be represented as probabilistic logic programs, combining neural predicate learning with symbolic inference [@manhaeve2018deepproblog].

- **Abductive learning**: Neural and symbolic components can cooperate in a balanced loop, where the neural component proposes patterns and the symbolic component abduces explanations [@zhou2022abductive].

## From Small Samples to Large Corpora

The empirical foundation of ARS is currently small (8 transcripts). While this is methodologically defensible (depth over breadth), it limits the generalizability of findings. Neuro-symbolic methods offer a path toward scalable ARS:

1.  **Seed ARS model** induced from a small, manually analyzed corpus

2.  **Neural transfer** of the symbolic structure to a larger corpus

3.  **ARS validation** of neural predictions on a representative sample

4.  **Iterative refinement** of both components

This approach preserves the methodological rigor of ARS while leveraging the scalability of neural methods---a classic neuro-symbolic synergy.

## Semantic Enrichment of Symbolic Categories

ARS 4.0 (Hybrid) already uses Transformer embeddings for semantic validation [@koop2024hybrid]. Intra-category similarities (0.83-0.95) confirm that interpretively formed categories are semantically coherent. Neuro-symbolic methods can take this further:

- **Neural concept learning**: Learn vector representations of ARS categories that capture semantic relationships

- **Symbolic abstraction from embeddings**: Extract symbolic rules from learned embeddings through concept activation vectors (TCAV)

- **Dynamic category refinement**: Use neural similarity to suggest splits or mergers of existing categories

## Attention Mechanisms for Explanation

ARS 4.0 implements simplified attention mechanisms to identify relevant predecessors. Neuro-symbolic systems can provide **more sophisticated attention-based explanations**:

1.  Train a transformer on ARS-labeled data

2.  Extract attention weights for each prediction

3.  Map attention weights back to ARS symbolic categories

4.  Generate explanations of the form: \"The prediction of symbol X at position i is primarily based on symbols Y and Z at positions j and k, which is consistent with ARS rule R.\"

This bridges the gap between neural opacity and symbolic explainability.

# Toward a Synthesized Methodology

## The ARS-Neuro-Symbolic Pipeline

Based on the analysis above, we propose the following integrated pipeline:

::: {#tab:pipeline}
  **Phase**                 **ARS Component**                                       **Neuro-Symbolic Component**
  ------------------------- ------------------------------------------------------- -------------------------------------------------------
  1\. Seed interpretation   Manual sequential microanalysis (small corpus)          Neural pre-training on similar domains
  2\. Symbol grounding      Terminal symbol formation, interpretive documentation   Neural proposal of symbols, symbolic validation (DFA)
  3\. Rule induction        Hierarchical compression (ARS 3.0)                      Neural estimation of transition probabilities
  4\. Formal modeling       PCFG, Petri net, DFA, HMM                               Neural refinement of parameters, DeepProbLog
  5\. Scaling               Validation on representative sample                     Neural transfer to large corpus, attention extraction
  6\. XAI validation        Communicative validation, reflexivity                   Attention-based explanations, concept activation

  : ARS-Neuro-Symbolic Integration Pipeline
:::

## Epistemic Roles Revisited

The threefold division of epistemic roles developed in the AQSA proposal [@koop2026aqsa] can be extended to neuro-symbolic integration:

::: {#tab:roles}
  **Role**             **Function**                                                    **ARS/Neuro-Symbolic Correspondence**
  -------------------- --------------------------------------------------------------- --------------------------------------------------------------
  Neural proposer      Pattern recognition, symbol proposals, probability estimation   Transformer, CRF, GNN (neuro-symbolic System 1)
  Human interpreter    Hermeneutic interpretation, validation, justification           Phase 2 (sequential microanalysis), communicative validation
  Symbolic validator   Structural well-formedness, rule checking                       ARS DFA, PCFG, Petri net (System 2)
  Formal modeler       Construction of symbolic models from validated patterns         ARS 3.0/4.0 (hierarchical compression, PCFG, Bayes)

  : Extended Epistemic Roles in Neuro-Symbolic ARS
:::

## Methodological Safeguards

The integration must not compromise the methodological standards of ARS. We propose five safeguards:

1.  **Primacy of interpretation**: Neural proposals must be validated by human interpretation before becoming part of the symbolic model.

2.  **Separation of structure and statistics**: As developed in `ARS_XAI_Aut2_Ger.tex`, structural rules must be decidable independently of empirical frequencies.

3.  **XAI validation of neural components**: Neural components must be evaluated not only by accuracy but also by XAI criteria (meaningfulness, transparency, knowledge limits).

4.  **Reflexive documentation of neuro-symbolic decisions**: Every integration decision must be documented, including why a neural component was used, how it was trained, and what its limitations are.

5.  **Human final authority**: The human researcher retains the authority to override neural proposals and to reject model outputs that violate interpretive plausibility.

# Discussion

## Comparison with Purely Neural Approaches

Compared to purely neural approaches (e.g., end-to-end transformer models for conversation analysis), the ARS-neuro-symbolic synthesis offers:

- **Explainability**: Every decision is traceable to symbolic rules

- **Small sample capability**: ARS works with n=8; purely neural methods require thousands of examples

- **Methodological control**: The human interpreter remains in charge

- **Generalizability**: Symbolic rules generalize beyond the training distribution

The price is higher upfront effort and the need for interpretive expertise.

## Comparison with Purely Symbolic Approaches

Compared to purely symbolic approaches (e.g., manual grammar writing), the ARS-neuro-symbolic synthesis offers:

- **Scalability**: Neural components can process large corpora

- **Learning under uncertainty**: Probabilistic models capture empirical variation

- **Pattern discovery**: Neural components can suggest patterns that might be overlooked by human interpreters

- **Semantic enrichment**: Embeddings provide semantic relationships

The price is increased complexity and the need for technical expertise.

## Limitations

The synthesis proposed here has limitations that must be acknowledged:

1.  **Technical complexity**: Implementing a full ARS-neuro-symbolic pipeline requires expertise in both qualitative methods and machine learning.

2.  **Resource requirements**: Large-scale application requires significant computational resources.

3.  **Validation challenges**: Mixed-method validation (qualitative + quantitative) is methodologically demanding.

4.  **Risk of automation**: There is a risk that neural components become substitutes for, rather than supplements to, human interpretation.

# Conclusion and Outlook

This paper has examined the reciprocal relationship between neuro-symbolic AI and the Algorithmic Recursive Sequence Analysis (ARS). We have argued that:

1.  **Neuro-symbolic AI can benefit from ARS** as a methodologically controlled procedure for rule induction, symbol grounding, and XAI-grounded symbolic scaffolding.

2.  **ARS can benefit from neuro-symbolic AI** for scaling, learning under uncertainty, semantic enrichment, and attention-based explanation.

The synthesis developed here does not blur the boundaries between paradigms but sharpens them: ARS provides the *symbolic scaffolding* (explicit, interpretable, verifiable), while neuro-symbolic methods provide the *learning dynamics* (pattern recognition, probabilistic inference, scalability). Methodological control remains with the human researcher.

For future research, we identify four desiderata:

1.  **Implementation of the ARS-neuro-symbolic pipeline**: A prototype system that integrates neural symbol proposers, ARS symbolic validators, and human interpreters.

2.  **Empirical evaluation**: Application of the integrated system to larger corpora (e.g., hundreds of transcripts) with comparative evaluation of purely neural, purely symbolic, and integrated approaches.

3.  **Extension to additional neuro-symbolic architectures**: Beyond Neural\|Symbolic, implement NeuralSymbolic (e.g., logic tensor networks) and Neural\[Symbolic\] (e.g., neural theorem proving) variants.

4.  **Methodological reflection**: Systematic analysis of the conditions under which neuro-symbolic integration is beneficial versus problematic, with particular attention to the risk of automation.

In conclusion: The question is not whether ARS and neuro-symbolic AI can be integrated---they can. The question is how to integrate them without compromising the methodological standards of either tradition. The present paper has offered a preliminary answer.

# Zusammenfassung auf Deutsch {#zusammenfassung-auf-deutsch .unnumbered}

Die vorliegende Arbeit untersucht das wechselseitige Verhältnis zwischen neuro-symbolischer KI und der Algorithmisch Rekursiven Sequenzanalyse (ARS). Die neuro-symbolische KI strebt die Integration von neuronalen (musterbasierten) und symbolischen (regelbasierten) Methoden an. Die ARS hat ein formales Framework entwickelt, das qualitative Interpretationsprozesse in erklärbare, intersubjektiv prüfbare Modelle überführt -- darunter probabilistische kontextfreie Grammatiken (PCFG), Petri-Netze, Bayessche Verfahren und deterministische endliche Automaten (DFA).

Der Beitrag zeigt, dass beide Paradigmen voneinander profitieren können. Die neuro-symbolische KI kann von der ARS profitieren durch: (1) methodologisch kontrollierte Regelinduktion, (2) XAI-fundiertes symbolisches Gerüst, (3) den DFA als Schnittstelle für neuro-symbolische Systeme, (4) Validierung durch qualitative Gütekriterien.

Die ARS kann von neuro-symbolischer KI profitieren durch: (1) Skalierung durch neuronale Mustererkennung, (2) Lernen unter Unsicherheit (DeepProbLog, abduktives Lernen), (3) Transfer von kleinen zu großen Korpora, (4) semantische Anreicherung symbolischer Kategorien, (5) Attention-basierte Erklärungen.

Die vorgeschlagene Synthese verwischt die Grenzen zwischen den Paradigmen nicht, sondern schärft sie: Die ARS liefert das symbolische Gerüst, neuro-symbolische Methoden liefern die Lerndynamik. Die methodologische Kontrolle verbleibt beim menschlichen Forscher. Fünf Sicherungsmechanismen werden vorgeschlagen, um die methodologischen Standards der ARS zu wahren: Primat der Interpretation, Trennung von Struktur und Statistik, XAI-Validierung neuronaler Komponenten, reflexive Dokumentation neuro-symbolischer Entscheidungen und letzte Autorität des menschlichen Interpreten.

::: thebibliography
99

Dreyfus, H. L. (1972). *What computers can't do: A critique of artificial reason*. Harper & Row.

Garcez, A. d'Avila, & Lamb, L. C. (2020). Neurosymbolic AI: The 3rd wave. *arXiv preprint arXiv:2012.05876*.

Hitzler, P., & Sarker, M. K. (Eds.). (2022). *Neuro-Symbolic Artificial Intelligence: The State of the Art*. IOS Press.

Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

Kautz, H. (2020). The third AI summer: AAAI Robert S. Engelmore Memorial Award Lecture. *AI Magazine*, 43(1), 93-104.

Koop, P. (2024/2026). *Between Interpretation and Computation: Algorithmic Recursive Sequence Analysis as a Bridge between Qualitative Hermeneutics and Formal Modeling*. the-last-freedom.org.

Koop, P. (2026). *Algorithmic Recursive Sequence Analysis 4.0: Integration of Bayesian Methods for Probabilistic Modeling of Sales Conversations*. the-last-freedom.org.

Koop, P. (2026). *Algorithmic Recursive Sequence Analysis 4.0: Hybrid Integration of Computational Linguistics Methods as a Complementary Extension of ARS 3.0*. the-last-freedom.org.

Koop, P. (2026). *Benchmarks as Epistemic Operators in ARS: Bridging Processual KI Evaluation and Qualitative Sequence Analysis*. the-last-freedom.org.

Manhaeve, R., Dumancic, S., Kimmig, A., Demeester, T., & De Raedt, L. (2018). DeepProbLog: Neural probabilistic logic programming. *Advances in Neural Information Processing Systems*, 31.

Marcus, G. (2020). The next decade in AI: Four steps towards robust artificial intelligence. *arXiv preprint arXiv:2002.06177*.

Ortigossa, E. S., Gonçalves, T., & Nonato, L. G. (2024). Explainable Artificial Intelligence (XAI)---From Theory to Methods and Applications. *IEEE Access*, 12, 80799-80846.

Zhou, Z.-H., & Huang, Y.-X. (2022). Abductive learning. In P. Hitzler & M. K. Sarker (Eds.), *Neuro-Symbolic Artificial Intelligence: The State of the Art* (pp. 353-379). IOS Press.
:::

# Glossary of Key Terms

  **Term**              **Definition**
  --------------------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ARS                   Algorithmic Recursive Sequence Analysis -- A formal framework for the analysis of sequential interactions that combines qualitative interpretation with formal modeling.
  Neuro-symbolic AI     A research program integrating neural methods (pattern recognition, learning) with symbolic methods (logic, rules, reasoning).
  XAI                   Explainable Artificial Intelligence -- Methods for making AI decisions transparent and interpretable.
  PCFG                  Probabilistic Context-Free Grammar -- A grammar where each production rule has a probability.
  DFA                   Deterministic Finite Automaton -- A finite state machine that accepts or rejects sequences of symbols.
  HMM                   Hidden Markov Model -- A statistical model for systems with hidden states and observable emissions.
  Symbol grounding      The problem of how symbols acquire meaning; in ARS, solved through interpretive documentation.
  System 1 / System 2   Kahneman's distinction between fast, intuitive (System 1) and slow, deliberative (System 2) cognition.
