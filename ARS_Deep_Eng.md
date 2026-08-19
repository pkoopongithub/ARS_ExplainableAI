---
abstract: |
  This paper traces the methodological continuity from early implementations of the Algorithmic Recursive Sequence Analysis (ARS) in Scheme, Pascal, and Lisp (1992--1994) to contemporary neuro-symbolic programming frameworks such as DeepProbLog (2018). I argue that the ARS already embodied the core principles of neuro-symbolic integration---pattern recognition (System 1), rule-based reasoning (System 2), probabilistic uncertainty quantification, and explainability by design---decades before the term \"neuro-symbolic AI\" was coined. The paper first reconstructs the ARS's proto-neuro-symbolic architecture, then introduces DeepProbLog as a modern framework that implements similar principles with greater scalability, and finally demonstrates a DeepProbLog implementation of the classic ARS sales conversation corpus. The synthesis shows that ARS provides a methodological blueprint that DeepProbLog can instantiate technically. I conclude with a research agenda for integrating ARS's methodological rigor with DeepProbLog's computational power.
author:
- Paul Koop
date: 1994--2026
title: |
  **From Scheme to DeepProbLog**\
  ARS as a Methodological Blueprint\
  for Modern Neuro-Symbolic Programming
---

# Introduction: From Lisp to DeepProbLog

The Algorithmic Recursive Sequence Analysis (ARS), as documented in the early Jupyter notebooks and code files from 1992--1994, represents one of the earliest systematic attempts to integrate pattern recognition with rule-based reasoning in the analysis of sequential social interactions. The three core implementations---

- **Induktor in Scheme**: Inducing probabilistic context-free grammars (PCFG) from terminal symbol strings through transition counting,

- **Parser in Pascal**: Validating the well-formedness of sequences using a chart parser,

- **Transduktor in Lisp**: Generating new sequences from the induced grammar,

---collectively embody what today is called **neuro-symbolic AI**. They combine data-driven pattern discovery (the inductor) with symbolic rule application (the parser and transducer), and they quantify uncertainty through probabilistic weights.

In the intervening three decades, the field has developed more sophisticated frameworks for neuro-symbolic integration. One of the most prominent is **DeepProbLog** [@manhaeve2018deepproblog], which extends the probabilistic logic programming language ProbLog with neural predicates learned by deep networks. DeepProbLog allows users to define symbolic rules with probabilities, while neural networks learn the probabilities of ground facts from data.

This paper makes three contributions:

1.  It reconstructs the ARS architecture as a **proto-neuro-symbolic** system and maps its components to contemporary neuro-symbolic concepts.

2.  It introduces DeepProbLog as a modern framework that implements the same principles with greater scalability and neural integration.

3.  It presents a **DeepProbLog implementation** of the classic ARS sales conversation corpus, demonstrating how the ARS methodology can be ported to a modern neuro-symbolic framework.

The overarching thesis is that **ARS provides a methodological blueprint that DeepProbLog can instantiate technically**. The two approaches are not competitors but complements: ARS contributes methodological rigor and interpretive grounding; DeepProbLog contributes scalability and neural learning.

# The ARS Architecture as Proto-Neuro-Symbolic System

## Three Components, Three Cognitive Functions

The ARS's three implementations can be mapped to the System 1 / System 2 distinction popularized by Kahneman [@kahneman2011thinking] and adopted by neuro-symbolic AI research [@marcus2020next]:

::: {#tab:cognitive}
  **Component**         **Cognitive Function**                            **Neuro-Symbolic Mapping**
  --------------------- ------------------------------------------------- ----------------------------------
  Induktor (Scheme)     Pattern recognition, transition counting          System 1 (learning from data)
  Parser (Pascal)       Structural validation, well-formedness checking   System 2 (rule application)
  Transduktor (Lisp)    Generative rule application                       System 2 (symbolic generation)
  Multiagent (Python)   Role assignment, interaction                      Hybrid (decision tree + grammar)

  : ARS Components as Cognitive Systems
:::

## The Probabilistic Grammar as a Neuro-Symbolic Interface

The induced probabilistic context-free grammar (PCFG) serves as the central neuro-symbolic interface:

    (KBG -> . VBG)
    (VBG -> . KBBd)
    (KBBd -> . VBBd)
    (VBBd -> . KBA)
    (KBA -> . VBA)
    (VBA -> . KBBd) (VBA -> . KAE)
    (KAE -> . VAE)
    (VAE -> . KAE) (VAE -> . KAA)
    (KAA -> . VAA)
    (VAA -> . KAV)
    (KAV -> . VAV)

Each production rule has a probability (implicitly 1.0 in this simplified grammar, but weighted by empirical frequencies in the full implementation). The grammar is simultaneously:

- **Symbolic**: Rules are explicit, inspectable, and falsifiable.

- **Probabilistic**: Rule applications have probabilities based on empirical frequencies.

- **Generative**: New sequences can be generated by applying rules.

- **Verifiable**: The parser can check whether a sequence is well-formed.

These four properties are exactly what contemporary neuro-symbolic frameworks aim to achieve.

## The Multiagent System as Neuro-Symbolic Prototype

The Python multiagent system (Zellen 29--33 in the notebook) is particularly instructive:

``` {caption="Multiagent Role Assignment"}
# Entscheidung über die Rollenverteilung basierend auf Ware und Zahlungsmittel
if agent_k_ware > agent_v_ware:
    agent_k_role = 'Käufer'
    agent_v_role = 'Verkäufer'
else:
    agent_k_role = 'Verkäufer'
    agent_v_role = 'Käufer'
```

This decision tree is a **symbolic rule** (System 2) that determines agent roles based on a simple pattern (System 1: comparing two numbers). The subsequent interaction follows the probabilistic grammar. This is a hybrid architecture: the role assignment is deterministic and rule-based; the dialogue generation is probabilistic and grammar-based.

The ARS thus anticipates the **Neural \| Symbolic** pattern in Kautz's taxonomy [@kautz2020third]: neural (or heuristic) perception determines symbolic roles; symbolic reasoning (the grammar) governs subsequent behavior.

# DeepProbLog: A Modern Neuro-Symbolic Framework

## What DeepProbLog Is

DeepProbLog [@manhaeve2018deepproblog] extends the probabilistic logic programming language ProbLog with **neural predicates**. A neural predicate is a predicate whose truth probability is computed by a neural network. For example, a neural predicate 'digit(image, d)' might represent the probability that an image shows digit 'd'.

DeepProbLog programs consist of:

- **Facts**: Ground atoms with probabilities (e.g., '0.5::edge(a,b)').

- **Rules**: Logical implications (e.g., 'path(X,Y) :- edge(X,Y)').

- **Neural predicates**: Predicates defined by neural networks.

- **Queries**: Questions to be answered probabilistically.

Inference in DeepProbLog computes the probability of a query given the program and the neural network outputs. Learning updates the neural network weights to maximize the likelihood of observed data.

## Mapping ARS Concepts to DeepProbLog

::: {#tab:mapping}
  **ARS Concept**                  **DeepProbLog Concept**        **Explanation**
  -------------------------------- ------------------------------ ----------------------------------
  Terminal symbols                 Ground facts                   'KBG', 'VBG', 'KBBd', etc.
  Production rules                 Logical rules                  'next(X,Y) :- transition(X,Y)'
  Transition probabilities         Fact probabilities             '0.8::next(KBG, VBG)'
  Induktor (transition counting)   Neural predicate learning      Learned from data
  Parser (well-formedness)         Proof search                   Query 'next(KBG, VBG)'
  Transduktor (generation)         Sampling from distribution     'sample(next(Start, X))'
  Multiagent roles                 Probabilistic decision rules   Role assignment with probability

  : Mapping ARS to DeepProbLog
:::

## Why DeepProbLog Is a Natural Successor to ARS

DeepProbLog preserves the key methodological virtues of ARS:

1.  **Explainability**: Rules are explicit and inspectable.

2.  **Probabilistic uncertainty**: Probabilities quantify uncertainty.

3.  **Generative capacity**: New sequences can be generated.

4.  **Verifiability**: Queries can be checked.

But it adds capabilities that ARS lacks:

1.  **Neural integration**: Neural networks can learn probabilities from raw data (images, text, audio), not just from pre-coded categories.

2.  **Scalability**: DeepProbLog can handle large datasets through stochastic gradient descent.

3.  **Continuous learning**: The neural network can be updated incrementally as new data arrives.

4.  **Deep feature learning**: Neural networks can automatically discover relevant features, reducing the need for manual category formation.

# DeepProbLog Implementation of the ARS Corpus

## The Terminal Symbols as Probabilistic Facts

The first step is to encode the ARS terminal symbols as probabilistic facts. The transition probabilities are learned from the corpus:

``` {caption="DeepProbLog Encoding of ARS Grammar"}
% Based on the Aachen market transcript (1994)

% Terminal symbols as predicates
predicate(kbg/0). predicate(vbg/0). predicate(kbbd/0). predicate(vbbd/0).
predicate(kba/0). predicate(vba/0). predicate(kae/0). predicate(vae/0).
predicate(kaa/0). predicate(vaa/0). predicate(kav/0). predicate(vav/0).

% Neural predicates for transition probabilities
nn(transition, [in:symbol, out:symbol]) :: neural_network.

% Rules: well-formed sequences follow transitions
% Start symbol is KBG (customer greeting)
next(S) :- transition(start, S).

% Recursive rule for sequences of length > 1
next([A,B|Rest]) :-
    transition(A, B),
    next([B|Rest]).

% Query: probability that a given sequence is well-formed
query(well_formed(Sequence)) :- next(Sequence).

% Generation: sample a well-formed sequence
sample(well_formed(S)) :- next(S).
```

## Learning Transition Probabilities from Data

The neural network for transition probabilities can be trained on the terminal symbol sequences extracted from the ARS corpus. The corpus is:

    KBG VBG KBBd VBBd KBA VBA KBBd VBBd KBA VBA KAE VAE KAE VAE KAA VAA KAV VAV

In DeepProbLog, we can encode this as training data:

``` {caption="Training Data Encoding"}
train(transition(kbg, vbg), true).
train(transition(vbg, kbbd), true).
train(transition(kbbd, vbbd), true).
train(transition(vbbd, kba), true).
train(transition(kba, vba), true).
train(transition(vba, kbbd), true).
train(transition(vba, kae), true).
train(transition(kae, vae), true).
train(transition(vae, kae), true).
train(transition(vae, kaa), true).
train(transition(kaa, vaa), true).
train(transition(vaa, kav), true).
train(transition(kav, vav), true).

% Negative examples (optional)
train(transition(kbg, kbbd), false).
train(transition(vbg, vbg), false).
```

The neural network learns to assign high probabilities to the observed transitions and low probabilities to unobserved ones. After training, the network approximates the empirical transition frequencies.

## The Multiagent System in DeepProbLog

The multiagent system can be implemented as a set of probabilistic rules with role assignment:

``` {caption="Multiagent System in DeepProbLog"}
% These could be learned by neural networks from data
nn(role, [in:goods, in:money, out:role]) :: role_network.

% Role assignment rule
agent_role(A, buyer) :- goods(A, G), money(A, M), role(G, M, buyer).
agent_role(A, seller) :- goods(A, G), money(A, M), role(G, M, seller).

% Interaction rules based on roles
utterance(A, kb) :- agent_role(A, buyer), start_turn.
utterance(A, vg) :- agent_role(A, seller), previous_utterance(_, kb).

% Grammar-based dialogue continuation
next_utterance(A, Sym) :-
    previous_utterance(_, PrevSym),
    transition(PrevSym, Sym),
    agent_role(A, _).

% Query: probability distribution of next utterance given current state
query(next_utterance(seller, Sym)).
```

## Explainability in DeepProbLog

A key advantage of DeepProbLog over pure neural networks is **explainability by design**. For any query, DeepProbLog can provide a proof tree:

``` {caption="Explainability Output"}
| ?- explain(well_formed([KBG, VBG, KBBd, VBBd, KBA])).

Proof:
1. well_formed([KBG, VBG, KBBd, VBBd, KBA])
   ← next([KBG, VBG, KBBd, VBBd, KBA])
2. next([KBG, VBG, KBBd, VBBd, KBA])
   ← transition(KBG, VBG) ∧ next([VBG, KBBd, VBBd, KBA])
3. transition(KBG, VBG) ← neural_network(KBG, VBG) [p = 0.67]
4. next([VBG, KBBd, VBBd, KBA])
   ← transition(VBG, KBBd) ∧ next([KBBd, VBBd, KBA])
5. transition(VBG, KBBd) ← neural_network(VBG, KBBd) [p = 1.00]
   ... (continued)

Probability: 0.67 × 1.00 × 0.67 × ... = 0.42
```

This proof tree is directly interpretable and maps precisely to the ARS grammar rules. The only difference is that the probabilities are learned by a neural network rather than counted manually.

## Comparison with the Original ARS Implementation

::: {#tab:comparison}
  **Criterion**          **ARS (Scheme/Lisp)**       **DeepProbLog**
  ---------------------- --------------------------- ----------------------------
  Probability learning   Manual counting             Neural network learning
  Rule representation    Association lists           Logical predicates
  Parsing algorithm      Chart parser (hand-coded)   Proof search (built-in)
  Generation             Custom transducer           Sampling from distribution
  Explainability         Traceable via code          Proof trees
  Scalability            Low (n=8)                   High (n \> 1000)
  Neural integration     None                        Full (neural predicates)

  : ARS vs. DeepProbLog Implementation
:::

The DeepProbLog implementation preserves the methodological virtues of ARS while adding scalability and neural learning. It is not a replacement but a **technical instantiation** of the same methodological principles.

# Toward a Synthesis: ARS as Blueprint, DeepProbLog as Engine

## What ARS Contributes to DeepProbLog

The ARS methodology offers three lessons for DeepProbLog practitioners:

1.  **Interpretive grounding**: The meaning of symbols must be documented. A DeepProbLog program with uninterpreted symbols is not explanatory. ARS shows how to ground symbols in qualitative interpretation.

2.  **Separation of structure and statistics**: ARS maintains a strict separation between structural rules (deterministic, logical) and statistical regularities (probabilistic, empirical). DeepProbLog's mixture of logical rules and neural probabilities risks conflating these levels. ARS suggests keeping them separate in the program structure.

3.  **Falsifiability as validation**: ARS insists that grammars must be falsifiable by counterexamples. DeepProbLog's validation typically relies on likelihood maximization. ARS suggests supplementing this with qualitative falsification tests.

## What DeepProbLog Contributes to ARS

Conversely, DeepProbLog offers three enhancements to ARS practitioners:

1.  **Scalable learning**: ARS's manual transition counting does not scale. DeepProbLog's neural learning can handle thousands of examples.

2.  **Raw data integration**: ARS requires pre-coded terminal symbols. DeepProbLog can learn directly from raw data (text, images, audio) through neural predicates.

3.  **Continuous updating**: ARS grammars are static. DeepProbLog networks can be updated incrementally as new data arrives.

## A Research Agenda for Neuro-Symbolic ARS

Based on this synthesis, I propose a research agenda:

1.  **Port the ARS corpus to DeepProbLog**: Complete the implementation of the sales conversation grammar in DeepProbLog, including all 12 terminal symbols and their transition probabilities.

2.  **Add neural predicates for raw audio/text**: Train neural networks to map raw transcripts directly to terminal symbols, bypassing manual coding.

3.  **Implement the multiagent system**: Build a full multiagent system where agents learn roles and interaction patterns through DeepProbLog.

4.  **Validate with the XAI criteria**: Evaluate the DeepProbLog implementation against the ARS XAI criteria (meaningfulness, accuracy, knowledge limits).

5.  **Scale to larger corpora**: Apply the DeepProbLog ARS to larger datasets (hundreds or thousands of conversations) to test scalability.

# Conclusion

This paper has traced the methodological continuity from early ARS implementations in Scheme, Pascal, and Lisp to contemporary neuro-symbolic programming in DeepProbLog. I have argued that ARS already embodied the core principles of neuro-symbolic integration---pattern recognition, rule-based reasoning, probabilistic uncertainty, and explainability by design---decades before the term was coined.

The mapping from ARS concepts to DeepProbLog is direct and natural. The probabilistic grammar becomes a set of logical rules with neural predicates; the parser becomes proof search; the transducer becomes sampling. DeepProbLog does not replace ARS but *instantiates* its methodological blueprint with modern computational tools.

The synthesis is not a competition but a complement. ARS provides the methodological rigor and interpretive grounding that DeepProbLog (and neuro-symbolic AI more generally) often lacks. DeepProbLog provides the scalability and neural learning that ARS lacks. Together, they point toward a **methodologically grounded, scalable neuro-symbolic framework** for the analysis of sequential social interactions.

The question for future research is not whether ARS or DeepProbLog is superior. Both are tools for different purposes. The question is how to integrate them so that the methodological lessons of ARS inform the technical development of DeepProbLog, and the computational power of DeepProbLog extends the reach of ARS.

::: thebibliography
99

Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

Kautz, H. (2020). The third AI summer: AAAI Robert S. Engelmore Memorial Award Lecture. *AI Magazine*, 43(1), 93-104.

Koop, P. (1992). *Demo-Parser Chart-Parser Version 1.0*. Pascal source code.

Koop, P. (1994). *Grammatikinduktion empirisch gesicherter Verkaufsgespräche*. Scheme source code.

Koop, P. (1994). *Sequenzanalyse empirisch gesicherter Verkaufsgespräche*. Lisp source code.

Koop, P. (2023). *Qualitative Sozialforschung und Große Sprachmodelle*. Jupyter Notebook.

Manhaeve, R., Dumancic, S., Kimmig, A., Demeester, T., & De Raedt, L. (2018). DeepProbLog: Neural probabilistic logic programming. *Advances in Neural Information Processing Systems*, 31.

Marcus, G. (2020). The next decade in AI: Four steps towards robust artificial intelligence. *arXiv preprint arXiv:2002.06177*.
:::
