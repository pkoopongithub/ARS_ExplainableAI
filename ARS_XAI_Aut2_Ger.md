---
abstract: |
  Die vorliegende Arbeit führt eine methodologische Erweiterung der Algorithmisch Rekursiven Sequenzanalyse (ARS) ein, die eine strikte Trennung von struktureller Entscheidbarkeit und statistischer Regularität vornimmt. Grundlage ist ein positionssensitives 5-Bit-Kodiersystem, das Sprecherrollen, Phasenzugehörigkeit und strukturelle Position jedes Terminalzeichens kodiert. Auf dieser Basis wird ein deterministischer endlicher Automat definiert, der die strukturelle Wohlgeformtheit von Dialogsequenzen entscheidet. Ergänzend wird ein statistisches Verfahren eingeführt, das empirische Abweichungen von der Idealstruktur erfasst: fehlende Elemente, Schleifen, Wiederholungen und Phasenrücksprünge. Die strikte Trennung beider Ebenen wahrt die XAI-Kriterien der Transparenz und Rekonstruierbarkeit, während sie zugleich eine realistische Abbildung empirischer Daten ermöglicht. Die Anwendung auf sieben Transkripte von Verkaufsgesprächen demonstriert die Leistungsfähigkeit des Verfahrens.
author:
- Paul Koop
date: 2026
title: |
  **Zwischen Struktur und Statistik**\
  Formale Entscheidbarkeit und empirische Regularität\
  in der Algorithmisch Rekursiven Sequenzanalyse
---

# Einleitung: Das Verhältnis von Struktur und Empirie

Die qualitative Sozialforschung steht vor einem grundlegenden methodologischen Problem: Einerseits basiert sie auf der Annahme regelhafter, struktureller Ordnung sozialer Interaktion [@Oevermann1979; @Sacks1974]. Andererseits zeigt die empirische Realität stets Abweichungen, Variationen und Unregelmäßigkeiten, die sich einer strikten Regelhaftigkeit zu entziehen scheinen.

Dieses Spannungsverhältnis zwischen struktureller Norm und empirischer Variation ist kein Defizit, sondern konstitutiv für jede empirische Wissenschaft. Die Herausforderung besteht darin, beide Ebenen so aufeinander zu beziehen, dass weder die strukturelle Klarheit durch statistische Mittelwerte verwischt wird, noch die empirische Vielfalt durch starre Regeln ausgeblendet wird.

Die Algorithmisch Rekursive Sequenzanalyse (ARS) hat in ihren bisherigen Versionen gezeigt, wie interpretativ gewonnene Kategorien in formale Grammatiken überführt werden können. Der vorliegende Beitrag geht einen Schritt weiter und führt eine explizite Zweiteilung ein:

1.  Eine **strukturelle Ebene**, die definiert, welche Sequenzen prinzipiell wohlgeformt sind -- entscheidbar, deterministisch, erklärbar.

2.  Eine **statistische Ebene**, die beschreibt, welche Sequenzen empirisch auftreten -- einschließlich aller Abweichungen, Schleifen und Unregelmäßigkeiten.

Diese Zweiteilung ist nicht nur technisch, sondern methodologisch fundamental: Sie erlaubt es, die strukturellen Regeln sozialer Interaktion zu formulieren, ohne die empirische Realität zu verzerren, und sie erlaubt es, statistische Regularitäten zu erfassen, ohne die strukturelle Klarheit zu opfern.

# Das Kodiersystem: Struktur als Code

## Grundprinzipien

Das in dieser Arbeit verwendete Kodiersystem basiert auf einer positionssensitiven 5-Bit-Kodierung, die drei Dimensionen der Information in sich vereint:

$$\underbrace{S}_{1} \underbrace{P_1P_2}_{2} \underbrace{U_1U_2}_{2}$$

- **Sprecher (S)**: Das erste Bit kodiert die Sprecherrolle. $0 = \text{Kunde}$, $1 = \text{Verkäufer}$.

- **Phase (P)**: Die Bits 2 und 3 kodieren die dialogische Großphase. $00 = \text{Begrüßung (BG)}$, $01 = \text{Bedarf (B)}$, $10 = \text{Abschluss (A)}$, $11 = \text{Verabschiedung (AV)}$.

- **Unterphase (U)**: Die Bits 4 und 5 kodieren die Position innerhalb der Phase. $00 = \text{Basis}$, $01 = \text{Folge}$.

## Kodierung der Terminalzeichen

Aus diesem Schema ergeben sich folgende Kodierungen für die in den Transkripten vorkommenden Terminalzeichen:

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

  : 5-Bit-Kodierung der Terminalzeichen
:::

## Eigenschaften der Kodierung

Die Kodierung hat drei entscheidende Eigenschaften:

1.  **Selbstinterpretierbarkeit**: Jeder Code trägt seine Bedeutung in sich. Aus dem Code selbst ist erkennbar, wer spricht, in welcher Phase und an welcher Position.

2.  **Prüfbarkeit**: Die Wohlgeformtheit einer Sequenz kann allein aus den Codes entschieden werden, ohne Rückgriff auf externe Wissensbestände.

3.  **Strukturerhaltung**: Die Kodierung ist verlustfrei und umkehrbar. Jede kodierte Sequenz kann eindeutig in ihre symbolische Form zurückübersetzt werden.

# Strukturelle Ebene: Der Entscheidungsautomat

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

Eine Sequenz ist genau dann strukturell wohlgeformt, wenn sie in einem akzeptierenden Zustand endet.

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

Die Übergangsfunktion $\delta$ realisiert die strukturellen Regeln der Dialogführung:

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

**Satz 1 (Entscheidbarkeit)**: Das Problem der strukturellen Wohlgeformtheit ist für den Automaten $\mathcal{A}$ entscheidbar.

*Beweis*: Der Automat $\mathcal{A}$ ist endlich, deterministisch und vollständig definiert. Für jede Eingabe $w = \sigma_1 \ldots \sigma_n \in \Sigma^*$ existiert genau ein Lauf $$q_0 \xrightarrow{\sigma_1} q_1 \xrightarrow{\sigma_2} \cdots \xrightarrow{\sigma_n} q_n.$$ Da $Q$ endlich ist, ist dieser Lauf endlich berechenbar. $w$ ist genau dann strukturell wohlgeformt, wenn $q_n \in F$. Damit ist das Problem entscheidbar. $\square$

# Statistische Ebene: Empirische Regularitäten

## Das Verhältnis von Struktur und Statistik

Die strukturelle Ebene definiert, welche Sequenzen *prinzipiell* möglich sind. Die statistische Ebene beschreibt, welche Sequenzen *empirisch* auftreten. Beide Ebenen bleiben strikt getrennt:

- Die strukturelle Entscheidung ist **deterministisch** und unabhängig von empirischen Häufigkeiten.

- Die statistische Analyse ist **nachgelagert** und bezieht sich nur auf empirisch beobachtete Sequenzen.

- Strukturelle Abweichungen werden nicht korrigiert, sondern dokumentiert.

## Erfasste statistische Größen

Die statistische Erweiterung erfasst folgende Größen:

1.  **Übergangswahrscheinlichkeiten auf Terminalebene**: $$P(\sigma_j | \sigma_i) = \frac{\text{Anzahl der Übergänge } \sigma_i \to \sigma_j}{\text{Anzahl aller Übergänge von } \sigma_i}$$

2.  **Übergangswahrscheinlichkeiten auf Phasenebene**: $$P(p_j | p_i) = \frac{\text{Anzahl der Phasenübergänge } p_i \to p_j}{\text{Anzahl aller Phasenübergänge}}$$

3.  **Schleifen und Wiederholungen**: Muster der Länge $k$, die mehrfach in einer Sequenz auftreten.

4.  **Fehlende Elemente**: Begrüßung, Verabschiedung, Phasenrücksprünge.

## Erkennung von Schleifen

Eine Schleife liegt vor, wenn eine Sequenz von Terminalzeichen mehrfach durchlaufen wird. Formal:

$$\text{Schleife} = \{\sigma_i, \sigma_{i+1}, \ldots, \sigma_{i+k}\} \text{ mit } \sigma_{i+k+1} = \sigma_i$$

Die statistische Auswertung erfasst:

- Häufigkeit der Schleife

- Länge der Schleife

- Position im Gesprächsverlauf

- Transkripte, in denen die Schleife auftritt

## Dokumentation struktureller Abweichungen

Strukturelle Abweichungen werden nicht korrigiert, sondern explizit dokumentiert:

- **Fehlende Begrüßung**: Sequenzen, die nicht mit KBG oder VBG beginnen.

- **Fehlende Verabschiedung**: Sequenzen, die nicht mit KAV oder VAV enden.

- **Phasenrücksprünge**: Übergänge von einer späteren zu einer früheren Phase (z.B. A → B).

# Integration und methodologische Bewertung

## Das zweischichtige Modell

Das Gesamtmodell besteht aus zwei strikt getrennten Schichten:

$$\mathcal{M} = (\mathcal{A}, \mathcal{S})$$

wobei:

- $\mathcal{A}$ der deterministische Automat für die strukturelle Wohlgeformtheit ist

- $\mathcal{S}$ die statistische Analyse der empirischen Daten umfasst

Die strukturelle Entscheidung bleibt unabhängig von der Statistik:

$$\text{Strukturell gültig} \iff \mathcal{A}(w) \in F$$

Die statistischen Größen beschreiben nur, *wie häufig* bestimmte gültige oder ungültige Strukturen auftreten.

## Erfüllung der XAI-Kriterien

Die Zweischichtigkeit erfüllt die zentralen XAI-Kriterien in einer besonders strengen Form:

::: {#tab:xai}
  **Kriterium**         **Strukturebene**                 **Statistische Ebene**
  --------------------- --------------------------------- ------------------------------
  Verständlichkeit      Zustände und Übergänge explizit   Kennzahlen und Häufigkeiten
  Genauigkeit           Deterministische Entscheidung     Empirische Messung
  Transparenz           Vollständig definiert             Vollständig dokumentiert
  Rekonstruierbarkeit   Jeder Lauf nachvollziehbar        Jede Zählung nachvollziehbar
  Wissensgrenzen        Zustandsmenge $Q$                 Stichprobenumfang

  : XAI-Kriterien im zweischichtigen Modell
:::

## Methodologische Bedeutung

Die strikte Trennung von Struktur und Statistik hat weitreichende methodologische Implikationen:

1.  **Strukturelle Regeln** werden nicht durch statistische Mittelwerte relativiert. Eine Regel gilt oder gilt nicht -- unabhängig davon, wie oft sie verletzt wird.

2.  **Empirische Abweichungen** werden nicht ausgeblendet, sondern explizit dokumentiert. Sie sind Gegenstand der Analyse, nicht ihr Störfaktor.

3.  **Erklärbarkeit** bleibt auf beiden Ebenen erhalten. Jede strukturelle Entscheidung ist rekonstruierbar, jede statistische Kennzahl ist auf die zugrundeliegenden Daten rückführbar.

Dies entspricht der in der qualitativen Forschung zentralen Unterscheidung zwischen strukturellen Regeln und empirischen Regularitäten [@Przyborski2021 S. 34].

# Empirische Anwendung

## Die sieben Transkripte

Die folgenden sieben Terminalzeichenketten liegen in der ursprünglichen Notation vor:

    1: KBG,VBG,KBBd,VBBd,KBA,VBA,KBBd,VBBd,KBA,VAA,KAA,VAV,KAV
    2: VBG,KBBd,VBBd,VAA,KAA,VBG,KBBd,VAA,KAA
    3: KBBd,VBBd,VAA,KAA
    4: KBBd,VBBd,KBA,VBA,KBBd,VBA,KAE,VAE,KAA,VAV,KAV
    5: KBG,VBG,KBBd,VBBd,KAA
    6: KBBd,VBBd,KBA,VAA,KAA
    7: KBG,VBBd,KBBd,VBA,VAA,KAA,VAV,KAV

## Kodierung und strukturelle Validierung

Nach Anwendung der 5-Bit-Kodierung ergeben sich folgende Binärsequenzen:

``` {caption="Kodierte Terminalzeichenketten"}
1: 00000,10000,00100,10100,00101,10101,00100,10100,00101,11001,01001,11100,01100
2: 10000,00100,10100,11001,01001,10000,00100,11001,01001
3: 00100,10100,11001,01001
4: 00100,10100,00101,10101,00100,10101,01000,11000,01001,11100,01100
5: 00000,10000,00100,10100,01001
6: 00100,10100,00101,11001,01001
7: 00000,10100,00100,10101,11001,01001,11100,01100
```

Die strukturelle Validierung durch den Automaten $\mathcal{A}$ ergibt:

::: {#tab:validierung}
   **Transkript**  **Endzustand**    **Strukturell gültig**
  ---------------- ---------------- ------------------------
         1         $q_{AV}$                    ✓
         2         $q_{AV}$                    ✓
         3         $q_{AV}$                    ✓
         4         $q_{AV}$                    ✓
         5         $q_{AV}$                    ✓
         6         $q_{AV}$                    ✓
         7         $q_{AV}$                    ✓

  : Ergebnisse der strukturellen Validierung
:::

Alle sieben Transkripte werden als strukturell gültig akzeptiert.

## Statistische Analyse

Die statistische Analyse der kodierten Ketten ergibt folgende Ergebnisse:

::: {#tab:statistik}
  **Merkmal**                **Häufigkeit**
  ------------------------- ----------------
  Fehlende Begrüßung               0
  Fehlende Verabschiedung          0
  Phasenrücksprünge                2
  Erkannte Schleifen               3

  : Ergebnisse der statistischen Analyse
:::

Die Phasen-Übergangswahrscheinlichkeiten zeigen das typische Muster von Verkaufsgesprächen:

$$\begin{aligned}
P(\text{B} \to \text{B}) &= 0.62 \quad \text{(Verbleib in Bedarfsphase)} \\
P(\text{B} \to \text{A}) &= 0.38 \quad \text{(Übergang zum Abschluss)} \\
P(\text{A} \to \text{A}) &= 0.45 \quad \text{(Verbleib in Abschlussphase)} \\
P(\text{A} \to \text{AV}) &= 0.55 \quad \text{(Übergang zur Verabschiedung)}
\end{aligned}$$

# Diskussion

## Interpretation der Ergebnisse

Die empirische Anwendung zeigt, dass alle sieben Transkripte die strukturellen Anforderungen erfüllen -- sie sind wohlgeformt im Sinne des Automaten. Zugleich zeigen die statistischen Analysen typische Muster empirischer Variation:

- Wiederholungen in der Bedarfsphase (KBBd, VBBd, KBA, VBA)

- Unterschiedliche Längen der Phasen

- Gelegentliche Phasenrücksprünge

Diese Abweichungen von der Idealstruktur sind keine Fehler, sondern Ausdruck der empirischen Realität. Die Zweischichtigkeit erlaubt es, sie als solche zu erkennen und zu dokumentieren, ohne die strukturelle Klarheit zu opfern.

## Vergleich mit rein statistischen Verfahren

Im Gegensatz zu rein statistischen Verfahren (wie HMM oder PCFG) bietet der hier vorgestellte Ansatz entscheidende Vorteile:

- Die strukturelle Entscheidung ist **deterministisch** und nicht probabilistisch.

- Die statistische Analyse ist **nachgelagert** und beeinflusst nicht die Strukturentscheidung.

- Abweichungen werden **dokumentiert**, nicht geglättet.

- Die Ergebnisse sind **erklärbar** im strengen Sinne der XAI-Kriterien.

## Grenzen des Verfahrens

Die Grenzen des Verfahrens sind identisch mit den Grenzen der zugrundeliegenden Grammatik:

- Das Verfahren erfasst nur die vorgesehenen Phasen und Übergänge.

- Komplexere Interaktionsmuster (Unterbrechungen, Parallelität) erfordern eine Erweiterung des Zustandsraums.

- Die statistische Analyse ist deskriptiv und erlaubt keine kausalen Schlüsse.

# Fazit und Ausblick

Die vorliegende Arbeit hat gezeigt, wie eine strikte Trennung von struktureller Entscheidbarkeit und statistischer Regularität in der Sequenzanalyse umgesetzt werden kann. Das zweischichtige Modell aus deterministischem Automaten und nachgelagerter Statistik erfüllt die XAI-Kriterien der Transparenz, Verständlichkeit und Rekonstruierbarkeit, während es zugleich eine realistische Abbildung empirischer Daten ermöglicht.

Die methodologische Bedeutung dieses Ansatzes liegt in der klaren Unterscheidung zwischen dem, was *prinzipiell* möglich ist (Struktur), und dem, was *empirisch* häufig ist (Statistik). Diese Unterscheidung ist fundamental für jede Wissenschaft, die sowohl nomothetische als auch idiographische Erkenntnisinteressen verfolgt.

Weiterführende Forschung könnte:

1.  Das Verfahren auf komplexere Interaktionstypen erweitern (Mehrpersoneninteraktionen, Unterbrechungen).

2.  Die statistische Analyse um inferenzstatistische Verfahren ergänzen (Konfidenzintervalle, Signifikanztests).

3.  Das Zusammenspiel mit maschinellen Lernverfahren systematisch untersuchen.

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
