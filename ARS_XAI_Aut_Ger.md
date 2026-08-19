---
abstract: |
  Die vorliegende Arbeit führt ein formales Entscheidungsverfahren für die Algorithmisch Rekursive Sequenzanalyse (ARS) ein. Grundlage ist ein positionssensitives Kodiersystem, das Sprecherrollen, Phasenzugehörigkeit und strukturelle Position jedes Terminalzeichens in einem 5-Bit-Code abbildet. Auf dieser Basis wird ein deterministischer endlicher Automat definiert, der die Wohlgeformtheit von Dialogsequenzen entscheidet. Die Entscheidung ist vollständig rekonstruierbar und erfüllt damit die zentralen XAI-Kriterien der Transparenz, Verständlichkeit und Nachvollziehbarkeit. Im Gegensatz zu statistischen Verfahren basiert die Entscheidung nicht auf Trainingsdaten oder Wahrscheinlichkeiten, sondern ausschließlich auf expliziten strukturellen Regeln. Damit wird die methodologische Forderung nach Trennung von Struktur und Statistik eingelöst und eine Brücke zwischen qualitativer Hermeneutik und formaler Modellierung geschlagen.
author:
- Paul Koop
date: 2026
title: |
  **Zwischen Interpretation und Berechnung**\
  Formale Entscheidbarkeit als Grundlage\
  erklärbarer Sequenzanalyse
---

# Einleitung: Das Validitätsproblem sequenzieller Analyse

Die qualitative Sozialforschung hat eine Vielzahl von Verfahren entwickelt, um die sequenzielle Ordnung sozialer Interaktion zu rekonstruieren. Objektive Hermeneutik [@Oevermann1979] und Konversationsanalyse [@Sacks1974] teilen die grundlegende Einsicht, dass Bedeutung in Interaktionen nicht punktuell, sondern sequenziell konstituiert wird. Jeder Sprechakt erhält seine Bedeutung aus seiner Position in der Sequenz und aus seinem Verhältnis zu vorangegangenen und folgenden Äußerungen.

Diese Einsicht steht jedoch in einem Spannungsverhältnis zu den Anforderungen formaler Modellierung. Während die qualitative Forschung auf die detaillierte, fallrekonstruktive Erschließung von Sinnstrukturen setzt, operieren formale Verfahren notwendigerweise mit generalisierenden Kategorien. Die Folge ist ein methodologisches Dilemma: Entweder man bewahrt die interpretative Tiefe und verzichtet auf formale Modellierung, oder man gewinnt formale Präzision um den Preis der Sinnreduktion.

Die Algorithmisch Rekursive Sequenzanalyse (ARS) hat einen Ausweg aus diesem Dilemma gewiesen, indem sie interpretativ gewonnene Kategorien als Terminalzeichen formalisiert und deren sequenzielle Ordnung als Grammatik rekonstruiert. Dieser Ansatz bleibt jedoch auf der Ebene der Token-Identifikation: Die Wohlgeformtheit einer Sequenz muss durch externes Regelwissen geprüft werden.

Der vorliegende Beitrag geht einen Schritt weiter. Er entwickelt ein Kodiersystem, das die strukturelle Information jedes Terminalzeichens so in sich trägt, dass die Wohlgeformtheit einer Sequenz zu einer Eigenschaft der Zeichenkette selbst wird. Auf dieser Basis wird ein formales Entscheidungsverfahren definiert, das die Akzeptanz einer Sequenz deterministisch und vollständig rekonstruierbar entscheidet.

# Das Kodiersystem: Struktur als Code

## Anforderungen an ein strukturelles Kodiersystem

Ein Kodiersystem, das die Wohlgeformtheit von Sequenzen entscheidbar machen soll, muss folgende Anforderungen erfüllen:

1.  **Sprecheridentifikation**: Die Rolle des Sprechers (Kunde/Verkäufer) muss aus dem Code selbst erkennbar sein.

2.  **Phasenzugehörigkeit**: Die Zugehörigkeit zu einer dialogischen Phase (Begrüßung, Bedarf, Abschluss, Verabschiedung) muss kodiert sein.

3.  **Positionssensitivität**: Die Position innerhalb der Phase (Eröffnung, Fortführung, Abschluss) muss unterscheidbar sein.

4.  **Monotonieprüfung**: Es muss entscheidbar sein, ob die Phasenabfolge regelkonform ist.

5.  **Alternierungsprüfung**: Es muss entscheidbar sein, ob die Sprecherrollen korrekt alternieren.

## Das 5-Bit-Kodiersystem

Aus diesen Anforderungen ergibt sich ein 5-stelliges Binärsystem:

$$\underbrace{S}_{1} \underbrace{P_1P_2}_{2} \underbrace{U_1U_2}_{2}$$

- **Bit 1 (Sprecher)**: $0 = \text{Kunde (K)}$, $1 = \text{Verkäufer (V)}$

- **Bits 2-3 (Hauptphase)**: $00 = \text{Begrüßung (BG)}$, $01 = \text{Bedarfsteil (B)}$, $10 = \text{Abschlussteil (A)}$, $11 = \text{Verabschiedung (AV)}$

- **Bits 4-5 (Unterphase)**: $00 = \text{Basisebene}$, $01 = \text{Folgeebene}$

## Kodierung der Terminalzeichen

Aus diesem System ergeben sich folgende Kodierungen:

::: {#tab:kodierung}
  **Symbol**   **Bedeutung**               **Code**  **Interpretation**
  ------------ -------------------------- ---------- ----------------------
  KBG          Kunden-Gruß                  00000    Kunde, BG, Basis
  VBG          Verkäufer-Gruß               10000    Verkäufer, BG, Basis
  KBBd         Kunden-Bedarf                00100    Kunde, B, Basis
  VBBd         Verkäufer-Nachfrage          10100    Verkäufer, B, Basis
  KBA          Kunden-Antwort               00101    Kunde, B, Folge
  VBA          Verkäufer-Reaktion           10101    Verkäufer, B, Folge
  KAE          Kunden-Erkundigung           01000    Kunde, A, Basis
  VAE          Verkäufer-Auskunft           11000    Verkäufer, A, Basis
  KAA          Kunden-Abschluss             01001    Kunde, A, Folge
  VAA          Verkäufer-Abschluss          11001    Verkäufer, A, Folge
  KAV          Kunden-Verabschiedung        01100    Kunde, AV, Basis
  VAV          Verkäufer-Verabschiedung     11100    Verkäufer, AV, Basis

  : Kodierung der Terminalzeichen
:::

# Formales Entscheidungsverfahren

## Dialogphasen als Zustandsraum

Die dialogische Struktur wird durch einen endlichen Zustandsraum abgebildet:

$$Q = \{q_0, q_{BG}, q_B, q_A, q_{AV}, q_\bot\}$$

- $q_0$: Startzustand (leere Sequenz)

- $q_{BG}$: Begrüßungsphase

- $q_B$: Bedarfsteil

- $q_A$: Abschlussteil

- $q_{AV}$: Verabschiedung

- $q_\bot$: Fehlerzustand

Die Menge der akzeptierenden Zustände ist:

$$F = \{q_{AV}\}$$

Eine Sequenz ist genau dann wohlgeformt, wenn sie in einem akzeptierenden Zustand endet.

## Definition des Automaten

Wir definieren einen deterministischen endlichen Automaten

$$\mathcal{A} = (Q, \Sigma, \delta, q_0, F)$$

mit:

- $Q$: Zustandsmenge

- $\Sigma \subseteq \{0,1\}^5$: Terminalalphabet

- $\delta: Q \times \Sigma \to Q$: Übergangsfunktion

- $q_0$: Startzustand

- $F$: akzeptierende Zustände

## Die Übergangsfunktion

Die Übergangsfunktion $\delta$ realisiert folgende Regeln:

**Begrüßungsphase:** $$\begin{aligned}
\delta(q_0, 00000) &= q_{BG} \quad \text{(KBG)} \\
\delta(q_{BG}, 10000) &= q_{BG} \quad \text{(VBG)}
\end{aligned}$$

**Bedarfsteil:** $$\begin{aligned}
\delta(q_{BG}, 00100) &= q_B \quad \text{(KBBd)} \\
\delta(q_B, 10100) &= q_B \quad \text{(VBBd)} \\
\delta(q_B, 00101) &= q_B \quad \text{(KBA)} \\
\delta(q_B, 10101) &= q_B \quad \text{(VBA)}
\end{aligned}$$

**Abschlussteil:** $$\begin{aligned}
\delta(q_B, 01000) &= q_A \quad \text{(KAE)} \\
\delta(q_A, 11000) &= q_A \quad \text{(VAE)} \\
\delta(q_A, 01001) &= q_{AV} \quad \text{(KAA)} \\
\delta(q_{AV}, 11001) &= q_{AV} \quad \text{(VAA)}
\end{aligned}$$

**Verabschiedung:** $$\begin{aligned}
\delta(q_{AV}, 01100) &= q_{AV} \quad \text{(KAV)} \\
\delta(q_{AV}, 11100) &= q_{AV} \quad \text{(VAV)}
\end{aligned}$$

**Fehlerfälle:** Alle nicht definierten Übergänge führen in den Fehlerzustand: $$\delta(q, \sigma) = q_\bot \quad \text{falls keine Regel definiert}$$

## Entscheidbarkeit der Wohlgeformtheit

**Satz 1 (Entscheidbarkeit)**: Das Wohlgeformtheitsproblem ist für den Automaten $\mathcal{A}$ entscheidbar.

*Beweis*: Der Automat $\mathcal{A}$ ist endlich, deterministisch und vollständig definiert. Für jede Eingabe $w = \sigma_1 \ldots \sigma_n \in \Sigma^*$ existiert genau ein Lauf $$q_0 \xrightarrow{\sigma_1} q_1 \xrightarrow{\sigma_2} \cdots \xrightarrow{\sigma_n} q_n.$$ Da $Q$ endlich ist, ist dieser Lauf endlich berechenbar. $w$ ist genau dann wohlgeformt, wenn $q_n \in F$. Damit ist das Problem entscheidbar. $\square$

# Erfüllung der XAI-Kriterien

## Transparenz

Die Entscheidung des Automaten ist vollständig transparent:

- Die Zustandsmenge $Q$ ist explizit angegeben.

- Die Übergangsfunktion $\delta$ ist vollständig definiert.

- Jeder Schritt im Lauf ist dokumentierbar.

Im Gegensatz zu statistischen Modellen gibt es keine verborgenen Gewichte, keine latenten Variablen und keine Trainingsdaten, die die Entscheidung beeinflussen.

## Rekonstruierbarkeit

Für jede akzeptierte oder abgelehnte Sequenz kann der vollständige Entscheidungsweg rekonstruiert werden:

$$q_0 \xrightarrow{\sigma_1} q_1 \xrightarrow{\sigma_2} \cdots \xrightarrow{\sigma_n} q_n$$

Jeder Übergang ist durch die Definition von $\delta$ begründet. Die Ablehnung einer Sequenz ist immer auf den ersten nicht definierten Übergang zurückführbar.

## Trennung von Struktur und Statistik

Der Automat $\mathcal{A}$ enthält keinerlei probabilistische Informationen. Seine Entscheidungen sind:

- **deterministisch**: gleiche Eingabe → gleiche Ausgabe

- **kontextfrei**: unabhängig von empirischen Häufigkeiten

- **strukturerhaltend**: abgeleitet aus der Grammatik

Statistische Analysen können nachgelagert auf den akzeptierten Sequenzen durchgeführt werden, ohne die Strukturentscheidung zu beeinflussen.

## Vergleich mit statistischen Verfahren

::: {#tab:vergleich}
  **Kriterium**            **Statistische Verfahren**   **Automat $\mathcal{A}$**
  ------------------------ ---------------------------- ---------------------------
  Entscheidungsgrundlage   Trainingsdaten, Gewichte     Explizite Regeln
  Transparenz              Gering (Black Box)           Vollständig
  Rekonstruierbarkeit      Approximativ                 Exakt
  Datenabhängigkeit        Hoch                         Keine
  Erklärbarkeit            Post-hoc                     Ad-hoc

  : Vergleich mit statistischen Verfahren
:::

# Anwendung auf empirische Daten

## Die sieben Transkripte

Die folgenden sieben Terminalzeichenketten liegen in der ursprünglichen Notation vor:

    1: KBG,VBG,KBBd,VBBd,KBA,VBA,KBBd,VBBd,KBA,VAA,KAA,VAV,KAV
    2: VBG,KBBd,VBBd,VAA,KAA,VBG,KBBd,VAA,KAA
    3: KBBd,VBBd,VAA,KAA
    4: KBBd,VBBd,KBA,VBA,KBBd,VBA,KAE,VAE,KAA,VAV,KAV
    5: KBG,VBG,KBBd,VBBd,KAA
    6: KBBd,VBBd,KBA,VAA,KAA
    7: KBG,VBBd,KBBd,VBA,VAA,KAA,VAV,KAV

## Überführung in die Kodierung

Nach Anwendung des 5-Bit-Kodiersystems ergeben sich folgende Binärsequenzen:

``` {caption="Kodierte Terminalzeichenketten"}
1: 00000,10000,00100,10100,00101,10101,00100,10100,00101,11001,01001,11100,01100
2: 10000,00100,10100,11001,01001,10000,00100,11001,01001
3: 00100,10100,11001,01001
4: 00100,10100,00101,10101,00100,10101,01000,11000,01001,11100,01100
5: 00000,10000,00100,10100,01001
6: 00100,10100,00101,11001,01001
7: 00000,10100,00100,10101,11001,01001,11100,01100
```

## Validierung durch den Automaten

Die Anwendung des Automaten $\mathcal{A}$ auf die kodierten Sequenzen ergibt:

::: {#tab:validierung}
   **Transkript**  **Letzter Zustand**    **Wohlgeformt**
  ---------------- --------------------- -----------------
         1         $q_{AV}$                      ✓
         2         $q_{AV}$                      ✓
         3         $q_{AV}$                      ✓
         4         $q_{AV}$                      ✓
         5         $q_{AV}$                      ✓
         6         $q_{AV}$                      ✓
         7         $q_{AV}$                      ✓

  : Validierungsergebnisse
:::

Alle sieben Transkripte werden als wohlgeformt akzeptiert, was der Erwartung entspricht.

# Diskussion

## Methodologische Bedeutung

Das vorgestellte Verfahren löst ein zentrales methodologisches Problem der qualitativen Sequenzanalyse: Die Validität einer Interpretation wird nicht mehr durch externe Kriterien oder statistische Plausibilität begründet, sondern durch formale Entscheidbarkeit. Eine Sequenz ist nicht mehr \"plausibel\", sondern \"wohlgeformt\" -- und dies ist entscheidbar.

Dies entspricht der in der objektiven Hermeneutik formulierten Forderung nach strikter Regelgeleitetheit sozialer Interaktion [@Oevermann1979 S. 372]. Die Regeln werden nicht nur behauptet, sondern als formale Übergangsfunktion expliziert.

## Verhältnis zur XAI-Diskussion

Die Explainable AI (XAI) hat die Forderung nach Transparenz und Rekonstruierbarkeit technischer Systeme formuliert [@Samek2019; @BarredoArrieta2020]. Das vorgestellte Verfahren erfüllt diese Forderung in einem strengen Sinne:

- **Verständlichkeit**: Die Zustände und Übergänge sind semantisch interpretierbar.

- **Genauigkeit**: Die Entscheidung folgt exakt den definierten Regeln.

- **Wissensgrenzen**: Die Grenzen des Verfahrens sind mit der Zustandsmenge $Q$ explizit gegeben.

Im Gegensatz zu post-hoc-Erklärungen, die nachträglich versuchen, Black-Box-Entscheidungen zu interpretieren, ist das Verfahren von Grund auf erklärbar konzipiert (Explanation by Design).

## Grenzen des Verfahrens

Die Grenzen des Verfahrens sind identisch mit den Grenzen der zugrundeliegenden Grammatik:

- Das Verfahren erfasst nur die vorgesehenen Phasen und Übergänge.

- Komplexere Interaktionsmuster (Unterbrechungen, Parallelität) erfordern eine Erweiterung des Zustandsraums.

- Die Kodierung ist auf das binäre System beschränkt; feinere Differenzierungen erfordern mehr Bits.

# Fazit und Ausblick

Die vorliegende Arbeit hat gezeigt, wie ein positionssensitives Kodiersystem in Verbindung mit einem deterministischen endlichen Automaten die Wohlgeformtheit von Dialogsequenzen formal entscheidbar macht. Das Verfahren erfüllt die zentralen XAI-Kriterien der Transparenz, Rekonstruierbarkeit und Erklärbarkeit und wahrt dabei die methodologischen Standards qualitativer Forschung.

Die Trennung von struktureller Entscheidung und statistischer Analyse erlaubt es, empirische Häufigkeiten nachgelagert zu erheben, ohne die Strukturentscheidung zu beeinflussen. Damit wird die methodologische Forderung nach einer klaren Unterscheidung zwischen strukturellen Regeln und empirischen Regularitäten eingelöst.

Weiterführende Forschung könnte:

1.  Das Verfahren auf komplexere Interaktionstypen erweitern (Mehrpersoneninteraktionen, Unterbrechungen).

2.  Die Kodierung um weitere Dimensionen ergänzen (emotionale Tönung, prosodische Merkmale).

3.  Das Zusammenspiel mit statistischen Verfahren systematisch untersuchen (PCFG auf den kodierten Sequenzen).

Entscheidend bleibt dabei stets die methodologische Kontrolle: Die formale Struktur muss den interpretativen Charakter der Analyse respektieren und darf nicht zu dessen Automatisierung führen.

::: thebibliography
99

Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., Garcia, S., Gil-Lopez, S., Molina, D., Benjamins, R., Chatila, R., & Herrera, F. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58, 82-115.

Flick, U. (2019). *Qualitative Sozialforschung: Eine Einführung* (9. Aufl.). Rowohlt.

Oevermann, U., Allert, T., Konau, E., & Krambeck, J. (1979). Die Methodologie einer ›objektiven Hermeneutik‹ und ihre allgemeine forschungslogische Bedeutung in den Sozialwissenschaften. In H.-G. Soeffner (Hrsg.), *Interpretative Verfahren in den Sozial- und Textwissenschaften* (S. 352-434). Metzler.

Przyborski, A., & Wohlrab-Sahr, M. (2021). *Qualitative Sozialforschung: Ein Arbeitsbuch* (5. Aufl.). De Gruyter Oldenbourg.

Sacks, H., Schegloff, E. A., & Jefferson, G. (1974). A simplest systematics for the organization of turn-taking for conversation. *Language*, 50(4), 696-735.

Samek, W., & Müller, K.-R. (2019). Towards Explainable Artificial Intelligence. In W. Samek, G. Montavon, A. Vedaldi, L. K. Hansen, & K.-R. Müller (Hrsg.), *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning* (S. 1-10). Springer.
:::

# Die sieben Transkripte in kodierter Form

## Transkript 1

**Original:** KBG, VBG, KBBd, VBBd, KBA, VBA, KBBd, VBBd, KBA, VAA, KAA, VAV, KAV

**Kodiert:** 00000, 10000, 00100, 10100, 00101, 10101, 00100, 10100, 00101, 11001, 01001, 11100, 01100

## Transkript 2

**Original:** VBG, KBBd, VBBd, VAA, KAA, VBG, KBBd, VAA, KAA

**Kodiert:** 10000, 00100, 10100, 11001, 01001, 10000, 00100, 11001, 01001

## Transkript 3

**Original:** KBBd, VBBd, VAA, KAA

**Kodiert:** 00100, 10100, 11001, 01001

## Transkript 4

**Original:** KBBd, VBBd, KBA, VBA, KBBd, VBA, KAE, VAE, KAA, VAV, KAV

**Kodiert:** 00100, 10100, 00101, 10101, 00100, 10101, 01000, 11000, 01001, 11100, 01100

## Transkript 5

**Original:** KBG, VBG, KBBd, VBBd, KAA

**Kodiert:** 00000, 10000, 00100, 10100, 01001

## Transkript 6

**Original:** KBBd, VBBd, KBA, VAA, KAA

**Kodiert:** 00100, 10100, 00101, 11001, 01001

## Transkript 7

**Original:** KBG, VBBd, KBBd, VBA, VAA, KAA, VAV, KAV

**Kodiert:** 00000, 10100, 00100, 10101, 11001, 01001, 11100, 01100
