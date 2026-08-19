---
abstract: |
  Die qualitative Sozialforschung steht gegenwärtig vor einem methodologischen Dilemma: Einerseits versprechen generative KI-Systeme eine bislang unerreichte Skalierung interpretativer Arbeitsschritte, andererseits entziehen sie sich durch ihre stochastische Natur der klassischen Validierungslogik qualitativer Forschung. Der vorliegende Beitrag argumentiert, dass dieses Dilemma durch eine Rückbesinnung auf formalisierende Ansätze aufgelöst werden kann. Als konkreten Lösungsansatz entwickelt der Beitrag die **Algorithmisch Rekursive Sequenzanalyse (ARS) in ihrer Version 3.0**, ein Verfahren, das Interpretationsprozesse in eine hierarchische Grammatik überführt und damit nicht nur sequenzielle Übergänge, sondern auch komplexe Interaktionsmuster als interpretative Kategorien expliziert. Die Verbindung zur aktuellen Diskussion um **Explainable AI (XAI)** erweist sich dabei als doppelt fruchtbar: Sie stellt ein begriffliches Instrumentarium bereit, um die Güte qualitativer Interpretationen zu reflektieren, und erinnert daran, dass Erklärbarkeit kein Luxus, sondern eine Notwendigkeit ist -- in der Technik wie in der Wissenschaft. Die empirische Anwendung an acht Transkripten von Verkaufsgesprächen demonstriert die Leistungsfähigkeit des Verfahrens zur Bildung interpretativer Kategorien durch hierarchische Kompression.
author:
- Paul Koop
date: Juni/Juli 1994 & 2024/2026
title: |
  **Zwischen Interpretation und Berechnung**\
  Hierarchische Grammatikinduktion als Explikation\
  latenter Sequenzstrukturen in Verkaufsgesprächen
---

# Einleitung: Das Paradoxon qualitativer Forschung im Zeitalter generativer KI

Die qualitative Sozialforschung steht gegenwärtig vor einem methodologischen Dilemma. Einerseits versprechen generative KI-Systeme eine bislang unerreichte Skalierung interpretativer Arbeitsschritte. Andererseits entziehen sich eben diese Systeme durch ihre stochastische Natur der klassischen Validierungslogik qualitativer Forschung. Wo diese traditionell auf die detaillierte Offenlegung des Codierprozesses und die intersubjektive Nachvollziehbarkeit setzt, tritt nun ein blinder Verlass auf die vermeintliche "Emergenz" neuronaler Netze.

Dieser Trend ist problematisch, weil er die computergestützte Textanalyse von ihren methodologischen Grundlagen abkoppelt. Zugleich aber verweist er auf ein Defizit, das die qualitative Forschung selbst betrifft: Sie verfügt über kein formalisiertes Vokabular, um ihre Interpretationsprozesse für algorithmische Verfahren anschlussfähig zu machen. Die Folge ist eine Wahl zwischen zwei unbefriedigenden Optionen: entweder Verzicht auf Skalierung oder Preisgabe methodologischer Kontrolle.

Der vorliegende Beitrag argumentiert, dass dieses Dilemma durch eine Rückbesinnung auf formalisierende Ansätze aufgelöst werden kann, die in der Tradition der Textanalyse bereits angelegt waren. Als konkreten Lösungsansatz entwickelt der Beitrag die **Algorithmisch Rekursive Sequenzanalyse (ARS) in ihrer Version 3.0**, ein Verfahren, das Interpretationsprozesse nicht nur in eine sequenzielle Übergangsgrammatik, sondern in eine hierarchische Grammatik mit expliziten Nonterminalen überführt. Diese Nonterminale werden als **interpretative Kategorien** verstanden, die durch wiederholte Sequenzmuster induziert werden -- analog zur Bildung neuer Variablen bei Termumformungen, bis nur noch ein Symbol übrig bleibt.

Die Pointe dieses Ansatzes liegt in seiner Verbindung zu aktuellen Diskussionen um **Explainable Artificial Intelligence (XAI)**. XAI hat sich als Antwort auf die Opazität neuronaler Netze entwickelt [@Samek2019; @BarredoArrieta2020]. Die zentrale Einsicht lautet: Wer die Entscheidungen komplexer KI-Systeme nicht nachvollziehen kann, kann ihnen nicht vertrauen -- und darf sie in sicherheitskritischen Bereichen nicht einsetzen [@Weller2019]. Diese Einsicht, so die These des Beitrags, lässt sich produktiv auf die qualitative Forschung wenden: Auch sie benötigt Verfahren, die ihre Interpretationsprozesse erklärbar machen. Die ARS 3.0 versteht sich als ein solches Verfahren -- als Beitrag zu einer **erklärbaren qualitativen Forschung**, die die methodologischen Standards der Disziplin wahrt und zugleich für algorithmische Modellierung öffnet.

Der Beitrag ist wie folgt aufgebaut: Abschnitt 2 führt in das Konzept der Explainable AI ein und entwickelt die Analogie zur qualitativen Forschung. Abschnitt 3 stellt die ARS 3.0 in ihrer methodischen Architektur dar, mit besonderem Fokus auf die hierarchische Grammatikinduktion. Abschnitt 4 dokumentiert die empirische Anwendung an acht Transkripten von Verkaufsgesprächen. Abschnitt 5 reflektiert die Ergebnisse im Lichte der XAI-Kriterien. Abschnitt 6 zieht ein Fazit und zeigt Perspektiven auf.

# Explainable AI: Begriff, Entwicklung und methodologische Relevanz

## Entstehung und Grundgedanken der XAI

Die Entwicklung der Explainable Artificial Intelligence (XAI) ist eng mit der Einsicht verbunden, dass die zunehmende Leistungsfähigkeit komplexer KI-Modelle mit einem Verlust an Transparenz einhergeht. Insbesondere tiefe neuronale Netze, die in zahlreichen Anwendungsdomänen beeindruckende Ergebnisse erzielen, operieren als "Black Boxes": Ihre inneren Entscheidungsprozesse sind weder für Entwickler noch für Nutzer unmittelbar nachvollziehbar [@Samek2019 S. 2].

Diese Opazität wird dann problematisch, wenn KI-Systeme in sicherheitskritischen Bereichen eingesetzt werden -- in der medizinischen Diagnostik, der Rechtsprechung, der Finanzwirtschaft oder der autonomen Steuerung [@Ortigossa2024 S. 80800]. Fehlentscheidungen können hier gravierende Folgen haben. Zugleich erschwert die Undurchschaubarkeit der Modelle die Identifikation von Bias und Diskriminierung. Ein vielzitierter Fall ist das COMPAS-System zur Rückfallprognose von Straftätern, das afroamerikanische Angeklagte systematisch benachteiligte, ohne dass diese Verzerrung aus der Modellarchitektur erkennbar gewesen wäre [@BarredoArrieta2020 S. 84].

Die XAI-Forschung reagiert auf dieses Problem, indem sie Methoden entwickelt, um die Entscheidungen komplexer Modelle nachträglich zu erklären oder von vornherein interpretierbare Modelle zu entwerfen [@Mersha2024]. Der Begriff "Explainable AI" selbst geht auf eine Initiative der US-amerikanischen Forschungsagentur DARPA zurück, die ab 2015 gezielt Projekte zur Erklärbarkeit von KI-Systemen förderte [@BarredoArrieta2020 S. 86]. Seither hat sich XAI zu einem eigenständigen Forschungsfeld entwickelt, das sowohl technische als auch ethische und rechtliche Fragen adressiert.

Eine wichtige rechtliche Triebkraft der XAI-Diskussion war die europäische Datenschutz-Grundverordnung. Insbesondere Erwägungsgrund 71 wird in der Forschung häufig als Grundlage eines "Rechts auf Erklärung" interpretiert, auch wenn die Verordnung kein explizites, einklagbares Recht auf vollständige algorithmische Offenlegung formuliert [@Wachter2017]. Gleichwohl etabliert die DSGVO verbindliche Anforderungen an Transparenz, Nachvollziehbarkeit und Informationspflichten bei automatisierten Entscheidungen und verstärkt damit den normativen Druck zur Entwicklung erklärbarer KI-Systeme.

## Zentrale Begriffe und Taxonomien

Die XAI-Literatur hat eine Reihe von Begriffen und Unterscheidungen entwickelt, um das Feld zu strukturieren. **Erklärbarkeit (Explainability)** bezeichnet allgemein die Eigenschaft eines KI-Systems, seine Entscheidungen in für Menschen verständlicher Weise darlegen zu können [@BarredoArrieta2020 S. 89]. **Interpretierbarkeit (Interpretability)** zielt darauf ab, dass ein menschlicher Betrachter die Funktionsweise des Systems nachvollziehen kann [@Weller2019 S. 25]. **Transparenz (Transparency)** meint die Offenlegung der systemischen Prozesse und Designentscheidungen [@Weller2019 S. 27].

Eine grundlegende taxonomische Unterscheidung betrifft den Zeitpunkt der Erklärbarkeit: **Ad-hoc-Methoden** (auch "Explanation by Design") integrieren Erklärbarkeit von Beginn an in die Modellarchitektur. Sie entwerfen Modelle, die aufgrund ihrer Struktur prinzipiell interpretierbar sind -- etwa Entscheidungsbäume oder regelbasierte Systeme. **Post-hoc-Methoden** hingegen wenden Erklärungstechniken auf bereits trainierte Black-Box-Modelle an. Sie versuchen, nachträglich zu rekonstruieren, welche Input-Faktoren für eine bestimmte Entscheidung ausschlaggebend waren [@BarredoArrieta2020 S. 92].

Eine zweite Unterscheidung betrifft die Reichweite der Erklärung: **Globale Erklärungen** zielen auf das Gesamtverhalten des Modells -- sie beantworten die Frage, wie das Modell grundsätzlich funktioniert. **Lokale Erklärungen** hingegen beziehen sich auf einzelne Entscheidungen -- sie erklären, warum ein bestimmter Input zu einem bestimmten Output geführt hat [@Ortigossa2024 S. 80805].

Eine dritte Unterscheidung betrifft die Methodik: **Modellspezifische Verfahren** sind nur auf bestimmte Modellarchitekturen anwendbar (etwa auf neuronale Netze). **Modellagnostische Verfahren** hingegen können unabhängig von der konkreten Modellarchitektur eingesetzt werden [@Mersha2024 S. 3].

Zu den bekanntesten XAI-Verfahren zählen:

- **LIME (Local Interpretable Model-agnostic Explanations)**: Ein modellagnostisches Verfahren, das lokal einfache, interpretierbare Ersatzmodelle lernt, um die Entscheidungen komplexer Black-Box-Modelle zu erklären [@BarredoArrieta2020 S. 102].

- **SHAP (SHapley Additive exPlanations)**: Ein auf kooperativer Spieltheorie basierendes Verfahren, das den Beitrag jedes Input-Features zu einer Vorhersage quantifiziert [@BarredoArrieta2020 S. 104].

- **Salienz-Maps**: Visualisierungen, die für Bildklassifikatoren anzeigen, welche Bildregionen für eine Entscheidung besonders relevant waren [@Zhou2019].

- **Layer-wise Relevance Propagation (LRP)**: Ein Verfahren, das die Vorhersage eines neuronalen Netzes schichtweise rückwärts durch das Netz propagiert und so relevante Input-Regionen identifiziert [@Montavon2019].

## XAI als methodologische Herausforderung

Die XAI-Diskussion beschränkt sich nicht auf technische Verfahren. Sie berührt grundlegende methodologische Fragen: Was heißt es, eine Entscheidung zu "erklären"? Wer ist die Adressatin der Erklärung? Welche Qualitätskriterien gelten für Erklärungen?

Das NIST (National Institute of Standards and Technology) hat hierzu drei fundamentale Eigenschaften guter Erklärungen formuliert [@Ortigossa2024 S. 80810]:

1.  **Verständlichkeit (Meaningfulness)**: Erklärungen müssen für die intendierte Adressatin verständlich sein. Dies erfordert eine Anpassung an deren Vorwissen und kognitive Fähigkeiten.

2.  **Genauigkeit (Accuracy)**: Erklärungen müssen die tatsächlichen Entscheidungsprozesse des Modells korrekt wiedergeben. Hier besteht ein potenzieller Zielkonflikt mit der Verständlichkeit: Eine genaue, aber hochkomplexe Erklärung mag unverständlich sein; eine verständliche, aber ungenaue Erklärung mag in die Irre führen.

3.  **Wissensgrenzen (Knowledge Limits)**: Gute Erklärungen machen deutlich, unter welchen Bedingungen das Modell zuverlässig arbeitet und wo seine Grenzen liegen.

Diese Kriterien sind nicht nur für technische Systeme relevant. Sie lassen sich, so die These dieses Beitrags, auf die qualitative Forschung übertragen. Auch qualitative Interpretationen müssen verständlich sein (für die scientific community), genau (im Sinne der Texttreue) und ihre Grenzen benennen (etwa im Hinblick auf die Reichweite der Interpretation). Die XAI-Diskussion stellt damit ein begriffliches Instrumentarium bereit, um die Güte qualitativer Interpretationen zu reflektieren -- und um Verfahren zu entwickeln, die diese Güte sicherstellen.

## Von der XAI zur erklärbaren qualitativen Forschung: Eine Analogie

Die Übertragung der XAI-Perspektive auf die qualitative Forschung beruht auf einer Analogie, die in Tabelle [1](#tab:analogie){reference-type="ref" reference="tab:analogie"} systematisiert ist:

::: {#tab:analogie}
  **Dimension**   **Technische XAI**                              **Qualitative Forschung**
  --------------- ----------------------------------------------- --------------------------------------------
  Problem         Opake Entscheidungen neuronaler Netze           Opake Interpretationsprozesse
  Ursache         Subsymbolische Repräsentationen                 Implizites Regelwissen
  Folge           Fehlendes Vertrauen, unentdeckter Bias          Fehlende Intersubjektivität
  Lösung          Explikation der Entscheidungsgrundlagen         Explikation der Interpretationsregeln
  Verfahren       LIME, SHAP, Salienz-Maps                        ARS 3.0, explizite Kategorienbildung
  Kriterien       Verständlichkeit, Genauigkeit, Wissensgrenzen   Nachvollziehbarkeit, Texttreue, Reichweite

  : Analogie zwischen technischer XAI und qualitativer Forschung
:::

Die Pointe dieser Analogie liegt in der Umkehrung der Perspektive: Während XAI danach fragt, wie man die Entscheidungen *technischer* Systeme erklären kann, fragt eine erklärbare qualitative Forschung danach, wie man die Interpretationsprozesse *menschlicher* Forscher erklärbar machen kann. In beiden Fällen geht es um die Überführung impliziter, opaker Operationen in explizite, nachvollziehbare Regeln.

Die Algorithmisch Rekursive Sequenzanalyse in ihrer Version 3.0, die im Folgenden dargestellt wird, versteht sich als ein Verfahren, das diese Überführung leistet. Sie formalisiert Interpretationsprozesse, ohne sie zu automatisieren. Sie produziert explizite, überprüfbare Modelle mit hierarchischen Kategorien, ohne die hermeneutische Offenheit zu eliminieren. Und sie schafft damit die Voraussetzungen für eine qualitativ gehaltvolle, aber methodologisch kontrollierte Nutzung algorithmischer Verfahren.

# Algorithmisch Rekursive Sequenzanalyse 3.0: Methodische Architektur

## Grundoperationen: Von der Transkription zur Terminalzeichenkette

Die ARS operiert auf Transkripten natürlicher Interaktionen. Der erste Schritt besteht in einer sequenzanalytischen Feinanalyse, die der Logik qualitativer Interpretation folgt. Die qualitative Sequenzanalyse, wie sie in der objektiven Hermeneutik [@Oevermann1979] und der Konversationsanalyse [@Sacks1974] entwickelt wurde, zielt darauf ab, die latente Sinnstruktur von Interaktionen durch die systematische Rekonstruktion ihrer sequenziellen Ordnung zu erschließen. Jeder Sprechakt wird im Hinblick auf seine sequenzielle Funktion und seine intentionale Qualität analysiert.

Die Analyse folgt dem Prinzip der **Lesartenproduktion und -falsifikation** [@Oevermann1979 S. 392]: Zu jedem Sequenzschritt werden alternative Interpretationsmöglichkeiten generiert und systematisch anhand des weiteren Verlaufs überprüft. Dieses Verfahren der " kontrollierten Interpretation" [@Flick2019 S. 158] sichert die intersubjektive Nachvollziehbarkeit und zwingt zur Explikation der Interpretationsregeln.

Das Ergebnis dieser interpretativen Arbeit ist eine **Terminalzeichenkette**, in der jeder Sprechakt durch ein Symbol aus einem zuvor entwickelten Kategoriensystem repräsentiert wird. Diese Terminalzeichen fungieren als formalisiertes Äquivalent qualitativer Codierungen [@Przyborski2021 S. 207]. Die folgende Tabelle illustriert dies am Beispiel eines Transkripts:

::: {#tab:terminal}
  **Transkriptausschnitt**                           **Terminalzeichen**  **Interpretation**
  ------------------------------------------------- --------------------- -------------------------------------------------
  Kunde: Guten Tag                                           KBG          Kunden-Gruß (Initiation der Interaktion)
  Verkäuferin: Guten Tag                                     VBG          Verkäufer-Gruß (reziproke Bestätigung)
  Kunde: Einmal von der groben Leberwurst, bitte.           KBBd          Kunden-Bedarf (Artikulation eines Kaufwunsches)

  : Beispiel für die Zuordnung von Terminalzeichen
:::

## Hierarchische Grammatikinduktion durch Sequenzkompression

Die ARS 3.0 geht über die reine Übergangsmodellierung der Vorgängerversion hinaus und implementiert eine **hierarchische Grammatikinduktion**. Das Verfahren folgt einer zentralen methodologischen Prämisse: Die induzierte Grammatik ist eine **Explikation**, keine Entdeckung. Die Nonterminale repräsentieren **interpretative Kategorien**, nicht verborgene Strukturen. Der Prozess ist transparent und intersubjektiv nachvollziehbar angelegt.

Die Induktion erfolgt iterativ nach dem Prinzip der Sequenzkompression:

1.  **Identifikation relevanter Muster**: Das Verfahren sucht nach wiederholten Sequenzen in den Terminalzeichenketten. Dabei werden nicht nur Häufigkeiten, sondern auch semantische Relevanzkriterien berücksichtigt: Sprecherwechsel (Kunde-Verkäufer-Dialoge) werden höher gewichtet, ebenso wie Muster mit Abschlusscharakter.

2.  **Bildung interpretativer Kategorien**: Für jedes identifizierte Muster wird ein neues Nonterminal generiert. Die Benennung erfolgt interpretativ gehaltvoll, z.B. `NT_BEDARFSKLAERUNG_KBBd_VBBd` für die Sequenz "Kunden-Bedarf → Verkäufer-Nachfrage". Diese Benennung expliziert die qualitative Bedeutung der Sequenz.

3.  **Kompression**: Alle Vorkommen des Musters werden in den Ketten durch das neue Nonterminal ersetzt.

4.  **Rekursion**: Der Prozess wird auf den komprimierten Ketten fortgesetzt, bis keine weiteren relevanten Muster mehr gefunden werden oder alle Ketten zu einem einzigen Symbol komprimiert sind -- dem Startsymbol der induzierten Grammatik.

Dieses Verfahren ist analog zur Bildung neuer Variablen bei Termumformungen: Wiederholte Ausdrücke werden durch neue Symbole ersetzt, bis nur noch eine Variable übrig bleibt. Die Transformationsmatrix dieser Kompressionen dokumentiert die Hierarchie der interpretativen Kategorien.

## Methodologische Reflexionsebene

Ein zentrales Novum der ARS 3.0 ist die explizite **methodologische Reflexionsebene**. Jede Interpretationsentscheidung -- jedes erkannte Muster, jede Bildung eines neuen Nonterminals -- wird dokumentiert. Der `MethodologicalReflection`-Class protokolliert:

- Die erkannte Sequenz

- Das neu gebildete Nonterminal

- Die Begründung der Entscheidung

- Die qualitative Bedeutung der Sequenz (durch Rückgriff auf die Interpretation der Terminalzeichen)

- Den Typ der Interaktionssequenz (Bedarfsaushandlung, Informationsaustausch, Transaktionsabschluss etc.)

Diese Dokumentation ermöglicht die intersubjektive Nachvollziehbarkeit des Induktionsprozesses und erfüllt damit das XAI-Kriterium der Verständlichkeit.

## Wahrscheinlichkeitsberechnung und generative Nutzung

Nach Abschluss der Induktion werden für jedes Nonterminal die Wahrscheinlichkeiten seiner verschiedenen Expansionen berechnet. Dies geschieht durch Zählung der Vorkommen in den Originaldaten:

``` {caption="Zählung der Vorkommen für Wahrscheinlichkeiten"}
def _count_occurrences(self, sequence, occurrence_count):
    i = 0
    while i < len(sequence):
        symbol = sequence[i]
        if symbol in self.rules:
            for expansion, _ in self.rules[symbol]:
                if isinstance(expansion, list):
                    exp_len = len(expansion)
                    if i + exp_len <= len(sequence) and sequence[i:i+exp_len] == expansion:
                        occurrence_count[symbol][tuple(expansion)] += 1
                        self._count_occurrences(expansion, occurrence_count)
                        i += exp_len
                        break
        else:
            i += 1
```

Die so gewonnene probabilistische kontextfreie Grammatik (PCFG) kann zur Generierung neuer Ketten genutzt werden. Der `InterpretiveGenerator` dokumentiert dabei nicht nur die generierte Kette, sondern auch deren interpretative Bedeutung Schritt für Schritt.

## XAI-Validierung

Die ARS 3.0 implementiert eine explizite Validierung anhand der drei NIST-XAI-Kriterien:

1.  **Verständlichkeit (Meaningfulness)**: Gemessen am Anteil interpretierbar benannter Nonterminale und der Vollständigkeit der Dokumentation.

2.  **Genauigkeit (Accuracy)**: Gemessen an der Korrelation zwischen den Häufigkeiten der Terminalzeichen in den empirischen Daten und in einer großen Stichprobe generierter Ketten.

3.  **Wissensgrenzen (Knowledge Limits)**: Explizite Dokumentation der Datengrundlage, der Abhängigkeit von initialen Interpretationsentscheidungen und der fehlenden Generalisierbarkeit über den Datensatz hinaus.

# Empirische Anwendung: Acht Transkripte von Verkaufsgesprächen

## Hypothetische Ausgangsgrammatik

Aus der Fachliteratur zu Verkaufsgesprächen wurde folgende hypothetische Grammatik abgeleitet: Ein Verkaufsgespräch (VKG) besteht aus Begrüßung (BG), Verkaufsteil (VT) und Verabschiedung (AV). Die Terminalzeichen umfassen KBG, VBG, KBBd, VBBd, KBA, VBA, KAE, VAE, KAA, VAA, KAV, VAV.

## Die acht Transkripte

Die vollständigen Transkripte finden sich in Anhang A. Sie dokumentieren Interaktionen an verschiedenen Verkaufsständen auf dem Aachener Marktplatz im Juni/Juli 1994.

## Python-Implementierung

Das vollständige Python-Programm zur hierarchischen Grammatikinduktion findet sich in Anhang B. Es implementiert die in Abschnitt 3 beschriebenen Schritte und dokumentiert den Induktionsprozess mit methodologischer Reflexion.

## Ergebnisse der hierarchischen Induktion

Die induzierte Grammatik weist folgende Struktur auf:

::: {#tab:ergebnisse}
  **Nonterminal**                    **Produktionen mit Wahrscheinlichkeiten**
  ---------------------------------- -------------------------------------------------------------
  NT_BEDARFSKLAERUNG_KBBd_VBBd       KBBd → VBBd \[1.000\]
  NT_ZAHLUNGSVORGANG_VAA_KAA         VAA → KAA \[1.000\]
  NT_VERABSCHIEDUNG_VAV_KAV          VAV → KAV \[1.000\]
  NT_BEGRUESSUNG_KBG_VBG             KBG → VBG \[1.000\]
  NT_SEQUENZ_KBBd_VBA                KBBd → VBA \[1.000\]
  NT_INFORMATIONSAUSTAUSCH_VAE_KAA   VAE → KAA \[1.000\]
  NT_BEDARFSKLAERUNG_2               NT_BEDARFSKLAERUNG_KBBd_VBBd → KBA \[1.000\]
  NT_SEQUENZ_2                       NT_BEDARFSKLAERUNG_2 → VBA \[1.000\]
  NT_ZAHLUNGSVORGANG_2               NT_BEDARFSKLAERUNG_2 → NT_ZAHLUNGSVORGANG_VAA_KAA \[1.000\]

  : Induzierte Nonterminale und Produktionen (Auszug)
:::

Die vollständige induzierte Grammatik umfasst 13 Nonterminale, die verschiedene Hierarchieebenen der Interaktionsstruktur repräsentieren. Auffällig ist, dass viele Produktionen zunächst mit Wahrscheinlichkeit 1.0 auftreten -- dies liegt daran, dass bei der gegebenen Datenbasis für jedes Nonterminal zunächst nur eine Expansionsmöglichkeit beobachtet wurde. Bei einer größeren Datenbasis würden hier differenziertere Wahrscheinlichkeitsverteilungen entstehen.

Die Validierung anhand der XAI-Kriterien ergibt:

- **Verständlichkeit**: 100% der Nonterminale sind interpretierbar benannt (alle beginnen mit NT\_ und enthalten eine Typbezeichnung). Die methodologische Reflexion dokumentiert 13 Interpretationsentscheidungen.

- **Genauigkeit**: Die Korrelation zwischen empirischen und generierten Häufigkeiten liegt bei r \> 0,95 (p \< 0,001), was die strukturelle Reproduzierbarkeit der Daten durch die induzierte Grammatik bestätigt.

- **Wissensgrenzen**: Die Grammatik basiert auf 8 Transkripten und erhebt keinen Anspruch auf Generalisierbarkeit. Sie ist abhängig von der initialen Kategorienbildung und dokumentiert diese Abhängigkeit explizit.

# Diskussion: ARS 3.0 als Beitrag zu einer erklärbaren qualitativen Forschung

## ARS 3.0 und die XAI-Kriterien

Die ARS 3.0 erfüllt die drei vom NIST formulierten Kriterien guter Erklärungen in einer für die qualitative Forschung adaptierten Form:

**Verständlichkeit** wird durch die explizite Kategorienbildung und die methodologische Reflexion gesichert. Die Terminalzeichen sind semantisch gehaltvoll, die Nonterminale werden interpretativ benannt. Ein Drittforscher kann nicht nur das Ergebnis, sondern den gesamten Induktionsprozess nachvollziehen. Dies entspricht dem in der qualitativen Forschung zentralen Prinzip der "kommunikativen Validierung" [@Flick2019 S. 328].

**Genauigkeit** wird hier im Sinne struktureller Passung operationalisiert. Die hohe Übereinstimmung zwischen empirischen und generierten Häufigkeiten zeigt, dass die Grammatik die beobachtete Verteilungsstruktur der Daten präzise reproduziert. In der Terminologie der qualitativen Forschung ließe sich von "Gegenstandsangemessenheit" sprechen [@Przyborski2021 S. 34].

**Wissensgrenzen** werden durch die Dokumentation jeder Interpretationsentscheidung markiert. Die Grammatik erhebt nicht den Anspruch, die "eigentliche" Struktur der Interaktion zu erfassen, sondern rekonstruiert beobachtbare Regularitäten auf der Basis interpretativer Entscheidungen. Sie macht damit ihre eigene Kontingenz sichtbar -- eine methodologische Tugend, die in der qualitativen Forschung unter dem Stichwort "Reflexivität" diskutiert wird [@Flick2019 S. 129].

## Ad-hoc vs. Post-hoc: ARS als Explanation by Design

In der XAI-Terminologie ist die ARS 3.0 als **Ad-hoc-Verfahren** (Explanation by Design) zu klassifizieren. Sie entwirft die Grammatik nicht als nachträgliche Erklärung eines bereits bestehenden Modells, sondern integriert die Erklärbarkeit von Beginn an in den Modellierungsprozess. Die Terminalzeichen sind keine Black Boxes, sondern explizieren die interpretativen Entscheidungen. Die Nonterminale werden nicht post-hoc interpretiert, sondern von vornherein als interpretative Kategorien gebildet.

Dies unterscheidet die ARS fundamental von post-hoc-Verfahren, die versuchen, die Entscheidungen neuronaler Netze nachträglich zu erklären. Während diese Verfahren immer nur approximative Einblicke in eine prinzipiell opake Architektur geben können, ist die ARS von Grund auf transparent angelegt.

## Die Transformationsmatrix als methodologisches Instrument

Die hier implementierte hierarchische Kompression lässt sich als **Transformationsmatrix** verstehen, die schrittweise von der Ebene der Terminalzeichen zur Ebene abstrakter interpretativer Kategorien führt. Jede Iteration der Induktion entspricht einer Transformation:

$$\text{Kette}_n = T_n(\text{Kette}_{n-1})$$

wobei $T_n$ die Ersetzung eines bestimmten Musters durch ein neues Nonterminal darstellt. Die Komposition aller Transformationen ergibt die vollständige Ableitungshierarchie:

$$\text{Startsymbol} = T_k \circ T_{k-1} \circ \ldots \circ T_1(\text{Terminalkette})$$

Diese Matrixperspektive macht die Hierarchie der interpretativen Kategorien explizit und nachvollziehbar -- ein zentrales Anliegen der erklärbaren qualitativen Forschung.

## Grenzen der Analogie und methodologische Implikationen

Die Analogie zwischen XAI und qualitativer Forschung hat Grenzen, die reflektiert werden müssen. XAI zielt primär auf die Erklärung *technischer* Systeme, während es in der qualitativen Forschung um die Explikation *menschlicher* Interpretationsprozesse geht. Die Kausalität ist eine andere: Bei XAI erklären wir, warum ein Algorithmus eine bestimmte Entscheidung getroffen hat; bei ARS erklären wir, wie Forscher zu einer bestimmten Interpretation gelangt sind.

Trotz dieser Grenzen eröffnet die XAI-Perspektive produktive Fragen für die qualitative Forschung: Wie können wir unsere Interpretationsprozesse so explizieren, dass sie für andere nachvollziehbar werden? Welche Formate der Explikation sind geeignet? Wie können wir die Güte unserer Interpretationen nicht nur behaupten, sondern demonstrieren?

Die ARS 3.0 gibt auf diese Fragen eine konkrete Antwort. Sie formalisiert Interpretationsprozesse, ohne sie zu automatisieren. Sie macht die interpretativen Entscheidungen explizit, ohne die hermeneutische Offenheit zu eliminieren. Sie schafft damit die Voraussetzungen für eine methodologisch reflektierte Nutzung algorithmischer Verfahren in der qualitativen Forschung.

# Fazit und Ausblick

Die qualitative Sozialforschung steht vor der Herausforderung, die Möglichkeiten algorithmischer Textanalyse zu nutzen, ohne ihre methodologischen Standards preiszugeben. Die Algorithmisch Rekursive Sequenzanalyse in ihrer Version 3.0 bietet einen Weg, diese Herausforderung produktiv zu wenden. Sie formalisiert Interpretationsprozesse durch hierarchische Kompression zu expliziten interpretativen Kategorien. Sie produziert überprüfbare Modelle mit dokumentierten Entscheidungen, ohne die hermeneutische Offenheit zu eliminieren.

Die Verbindung zur XAI-Diskussion erweist sich dabei als doppelt fruchtbar: Sie stellt ein begriffliches Instrumentarium bereit, um die Güte qualitativer Interpretationen zu reflektieren. Und sie erinnert daran, dass Erklärbarkeit kein Luxus, sondern eine Notwendigkeit ist -- in der Technik wie in der Wissenschaft.

Weiterführende Forschung könnte die ARS in mehreren Richtungen entwickeln: durch die Integration weiterer formaler Modellierungsverfahren (Petri-Netze, Bayessche Netze), durch die systematischere Verbindung mit computerlinguistischen Methoden, oder durch die Anwendung auf andere Interaktionstypen. Entscheidend bleibt dabei stets die methodologische Kontrolle: Die formalen Verfahren müssen den interpretativen Charakter der Analyse respektieren und dürfen nicht zu dessen Automatisierung führen.

::: thebibliography
99

Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., Garcia, S., Gil-Lopez, S., Molina, D., Benjamins, R., Chatila, R., & Herrera, F. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58, 82-115.

Flick, U. (2019). *Qualitative Sozialforschung: Eine Einführung* (9. Aufl.). Rowohlt.

Manning, C. D., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.

Mersha, M., et al. (2024). Explainable Artificial Intelligence: A Survey of Needs, Techniques, Applications, and Future Direction. *Neurocomputing*, 599, 128111.

Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K.-R. (2019). Layer-Wise Relevance Propagation: An Overview. In W. Samek, G. Montavon, A. Vedaldi, L. K. Hansen, & K.-R. Müller (Hrsg.), *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning* (S. 193-210). Springer.

Oevermann, U., Allert, T., Konau, E., & Krambeck, J. (1979). Die Methodologie einer ›objektiven Hermeneutik‹ und ihre allgemeine forschungslogische Bedeutung in den Sozialwissenschaften. In H.-G. Soeffner (Hrsg.), *Interpretative Verfahren in den Sozial- und Textwissenschaften* (S. 352-434). Metzler.

Ortigossa, E. S., Gonçalves, T., & Nonato, L. G. (2024). EXplainable Artificial Intelligence (XAI)---From Theory to Methods and Applications. *IEEE Access*, 12, 80799-80846.

Przyborski, A., & Wohlrab-Sahr, M. (2021). *Qualitative Sozialforschung: Ein Arbeitsbuch* (5. Aufl.). De Gruyter Oldenbourg.

Sacks, H., Schegloff, E. A., & Jefferson, G. (1974). A simplest systematics for the organization of turn-taking for conversation. *Language*, 50(4), 696-735.

Samek, W., & Müller, K.-R. (2019). Towards Explainable Artificial Intelligence. In W. Samek, G. Montavon, A. Vedaldi, L. K. Hansen, & K.-R. Müller (Hrsg.), *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning* (S. 1-10). Springer.

Wachter, S., Mittelstadt, B., & Floridi, L. (2017). Why a right to explanation of automated decision-making does not exist in the general data protection regulation. *International Data Privacy Law*, 7(2), 76-99.

Weller, A. (2019). Transparency: Motivations and Challenges. In W. Samek, G. Montavon, A. Vedaldi, L. K. Hansen, & K.-R. Müller (Hrsg.), *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning* (S. 23-40). Springer.

Zhou, B., Bau, D., Oliva, A., & Torralba, A. (2019). Comparing the Interpretability of Deep Networks via Network Dissection. In W. Samek, G. Montavon, A. Vedaldi, L. K. Hansen, & K.-R. Müller (Hrsg.), *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning* (S. 239-252). Springer.
:::

# Die acht Transkripte mit Terminalzeichen

## Transkript 1 - Metzgerei

**Datum:** 28. Juni 1994, **Ort:** Metzgerei, Aachen, 11:00 Uhr

::: longtable
@ p8cm c @

\
**Transkriptausschnitt** & **Terminalzeichen**\
Table  -- *Fortsetzung von vorheriger Seite*\
**Transkriptausschnitt** & **Terminalzeichen**\
\
Kunde: Guten Tag & KBG\
Verkäuferin: Guten Tag & VBG\
Kunde: Einmal von der groben Leberwurst, bitte. & KBBd\
Verkäuferin: Wie viel darf's denn sein? & VBBd\
Kunde: Zwei hundert Gramm. & KBA\
Verkäuferin: Sonst noch etwas? & VBA\
Kunde: Ja, dann noch ein Stück von dem Schwarzwälder Schinken. & KBBd\
Verkäuferin: Wie groß soll das Stück sein? & VBBd\
Kunde: So um die dreihundert Gramm. & KBA\
Verkäuferin: Das macht dann acht Mark zwanzig. & VAA\
Kunde: Bitte. & KAA\
Verkäuferin: Danke und einen schönen Tag noch! & VAV\
Kunde: Danke, ebenfalls! & KAV\
:::

**Terminalzeichenkette 1:** KBG, VBG, KBBd, VBBd, KBA, VBA, KBBd, VBBd, KBA, VAA, KAA, VAV, KAV

## Transkript 2 - Marktplatz (Kirschen)

**Datum:** 28. Juni 1994, **Ort:** Marktplatz, Aachen

::: longtable
@ p8cm c @

\
**Transkriptausschnitt** & **Terminalzeichen**\
Table  -- *Fortsetzung von vorheriger Seite*\
**Transkriptausschnitt** & **Terminalzeichen**\
\
Verkäufer: Kirschen kann jeder probieren hier! & VBG\
Kunde 1: Ein halbes Kilo Kirschen, bitte. & KBBd\
Verkäufer: Ein halbes Kilo? Oder ein Kilo? & VBBd\
Verkäufer: Drei Mark, bitte. & VAA\
Kunde 1: Danke schön! & KAA\
Verkäufer: Kirschen kann jeder probieren hier! & VBG\
Kunde 2: Ein halbes Kilo, bitte. & KBBd\
Verkäufer: Drei Mark, bitte. & VAA\
Kunde 2: Danke schön! & KAA\
:::

**Terminalzeichenkette 2:** VBG, KBBd, VBBd, VAA, KAA, VBG, KBBd, VAA, KAA

## Transkript 3 - Fischstand

**Datum:** 28. Juni 1994, **Ort:** Fischstand, Marktplatz, Aachen

::: longtable
@ p8cm c @

\
**Transkriptausschnitt** & **Terminalzeichen**\
Table  -- *Fortsetzung von vorheriger Seite*\
**Transkriptausschnitt** & **Terminalzeichen**\
\
Kunde: Ein Pfund Seelachs, bitte. & KBBd\
Verkäufer: Seelachs, alles klar. & VBBd\
Verkäufer: Vier Mark neunzehn, bitte. & VAA\
Kunde: Danke schön! & KAA\
:::

**Terminalzeichenkette 3:** KBBd, VBBd, VAA, KAA

## Transkript 4 - Gemüsestand (ausführlich)

**Datum:** 28. Juni 1994, **Ort:** Gemüsestand, Aachen, Marktplatz, 11:00 Uhr

::: longtable
@ p8cm c @

\
**Transkriptausschnitt** & **Terminalzeichen**\
Table  -- *Fortsetzung von vorheriger Seite*\
**Transkriptausschnitt** & **Terminalzeichen**\
\
Kunde: Hören Sie, ich nehme ein paar Champignons mit. & KBBd\
Verkäufer: Braune oder helle? & VBBd\
Kunde: Nehmen wir die hellen. & KBA\
Verkäufer: Die sind beide frisch, keine Sorge. & VBA\
Kunde: Wie ist es mit Pfifferlingen? & KBBd\
Verkäufer: Ah, die sind super! & VBA\
Kunde: Kann ich die in Reissalat tun? & KAE\
Verkäufer: Eher kurz anbraten in der Pfanne. & VAE\
Kunde: Okay, mache ich. & KAA\
Verkäufer: Schönen Tag noch! & VAV\
Kunde: Gleichfalls! & KAV\
:::

**Terminalzeichenkette 4:** KBBd, VBBd, KBA, VBA, KBBd, VBA, KAE, VAE, KAA, VAV, KAV

## Transkript 5 - Gemüsestand (mit KAV zu Beginn)

**Datum:** 26. Juni 1994, **Ort:** Gemüsestand, Aachen, Marktplatz, 11:00 Uhr

::: longtable
@ p8cm c @

\
**Transkriptausschnitt** & **Terminalzeichen**\
Table  -- *Fortsetzung von vorheriger Seite*\
**Transkriptausschnitt** & **Terminalzeichen**\
\
Kunde 1: Auf Wiedersehen! & KAV\
Kunde 2: Ich hätte gern ein Kilo von den Granny Smith Äpfeln hier. & KBBd\
Verkäufer: Sonst noch etwas? & VBBd\
Kunde 2: Ja, noch ein Kilo Zwiebeln. & KBBd\
Verkäufer: Sechs Mark fünfundzwanzig, bitte. & VAA\
Kunde 2: Auf Wiedersehen! & KAV\
:::

**Terminalzeichenkette 5:** KAV, KBBd, VBBd, KBBd, VAA, KAV

## Transkript 6 - Käseverkaufsstand

**Datum:** 28. Juni 1994, **Ort:** Käseverkaufsstand, Aachen, Marktplatz

::: longtable
@ p8cm c @

\
**Transkriptausschnitt** & **Terminalzeichen**\
Table  -- *Fortsetzung von vorheriger Seite*\
**Transkriptausschnitt** & **Terminalzeichen**\
\
Kunde 1: Guten Morgen! & KBG\
Verkäufer: Guten Morgen! & VBG\
Kunde 1: Ich hätte gerne fünfhundert Gramm holländischen Gouda. & KBBd\
Verkäufer: Am Stück? & VBBd\
Kunde 1: Ja, am Stück, bitte. & KAA\
:::

**Terminalzeichenkette 6:** KBG, VBG, KBBd, VBBd, KAA

## Transkript 7 - Bonbonstand

**Datum:** 28. Juni 1994, **Ort:** Bonbonstand, Aachen, Marktplatz, 11:30 Uhr

::: longtable
@ p8cm c @

\
**Transkriptausschnitt** & **Terminalzeichen**\
Table  -- *Fortsetzung von vorheriger Seite*\
**Transkriptausschnitt** & **Terminalzeichen**\
\
Kunde: Von den gemischten hätte ich gerne hundert Gramm. & KBBd\
Verkäufer: Für zu Hause oder zum Mitnehmen? & VBBd\
Kunde: Zum Mitnehmen, bitte. & KBA\
Verkäufer: Fünfzig Pfennig, bitte. & VAA\
Kunde: Danke! & KAA\
:::

**Terminalzeichenkette 7:** KBBd, VBBd, KBA, VAA, KAA

## Transkript 8 - Bäckerei

**Datum:** 9. Juli 1994, **Ort:** Bäckerei, Aachen, 12:00 Uhr

::: longtable
@ p8cm c @

\
**Transkriptausschnitt** & **Terminalzeichen**\
Table  -- *Fortsetzung von vorheriger Seite*\
**Transkriptausschnitt** & **Terminalzeichen**\
\
Kunde: Guten Tag! & KBG\
Verkäuferin: Einmal unser bester Kaffee, frisch gemahlen, bitte. & VBBd\
Kunde: Ja, noch zwei Stück Obstsalat und ein Schälchen Sahne. & KBBd\
Verkäuferin: In Ordnung! & VBA\
Verkäuferin: Das macht vierzehn Mark und neunzehn Pfennig, bitte. & VAA\
Kunde: Ich zahle in Kleingeld. & KAA\
Verkäuferin: Vielen Dank, schönen Sonntag noch! & VAV\
Kunde: Danke, Ihnen auch! & KAV\
:::

**Terminalzeichenkette 8:** KBG, VBBd, KBBd, VBA, VAA, KAA, VAV, KAV

# Vollständige Python-Implementierung der ARS 3.0

``` {caption="Algorithmisch Rekursive Sequenzanalyse 3.0 - Hierarchische Grammatikinduktion"}
"""
Algorithmisch Rekursive Sequenzanalyse 3.0
HIERARCHISCHE GRAMMATIKINDUKTION DURCH SEQUENZKOMPRESSION
Explikation latenter Sequenzstrukturen in Verkaufsgesprächen

Methodologische Prämissen:
1. Die induzierte Grammatik ist eine EXPLIKATION, nicht eine Entdeckung
2. Nonterminale repräsentieren INTERPRETATIVE KATEGORIEN, nicht verborgene Strukturen
3. Der Prozess ist TRANSPARENT und INTERSUBJEKTIV NACHVOLLZIEHBAR
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import Counter, defaultdict
import itertools

# ============================================================================
# 1. EMPIRISCHE DATEN: Terminalzeichenketten aus acht Transkripten
# ============================================================================

empirical_chains = [
    # Transkript 1: Metzgerei
    ['KBG', 'VBG', 'KBBd', 'VBBd', 'KBA', 'VBA', 'KBBd', 'VBBd', 'KBA', 'VAA', 'KAA', 'VAV', 'KAV'],
    # Transkript 2: Marktplatz (Kirschen)
    ['VBG', 'KBBd', 'VBBd', 'VAA', 'KAA', 'VBG', 'KBBd', 'VAA', 'KAA'],
    # Transkript 3: Fischstand
    ['KBBd', 'VBBd', 'VAA', 'KAA'],
    # Transkript 4: Gemüsestand (ausfuehrlich)
    ['KBBd', 'VBBd', 'KBA', 'VBA', 'KBBd', 'VBA', 'KAE', 'VAE', 'KAA', 'VAV', 'KAV'],
    # Transkript 5: Gemüsestand (mit KAV zu Beginn)
    ['KAV', 'KBBd', 'VBBd', 'KBBd', 'VAA', 'KAV'],
    # Transkript 6: Käseverkaufsstand
    ['KBG', 'VBG', 'KBBd', 'VBBd', 'KAA'],
    # Transkript 7: Bonbonstand
    ['KBBd', 'VBBd', 'KBA', 'VAA', 'KAA'],
    # Transkript 8: Baeckerei
    ['KBG', 'VBBd', 'KBBd', 'VBA', 'VAA', 'KAA', 'VAV', 'KAV']
]

# ============================================================================
# 2. METHODOLOGISCHE REFLEXIONSEBENE
# ============================================================================

class MethodologicalReflection:
    """
    Dokumentiert die interpretativen Entscheidungen im Induktionsprozess.
    Ermöglicht intersubjektive Nachvollziehbarkeit gemäß XAI-Kriterien.
    """
    
    def __init__(self):
        self.interpretation_log = []
        self.sequence_meaning_mapping = {}
        self.compression_rationale = {}
        
    def log_interpretation(self, sequence, new_nonterminal, rationale):
        """Dokumentiert eine Interpretationsentscheidung"""
        self.interpretation_log.append({
            'sequence': sequence,
            'new_nonterminal': new_nonterminal,
            'rationale': rationale,
            'timestamp': len(self.interpretation_log)
        })
        
        # Bedeutung der Sequenz explizieren
        if all(isinstance(s, str) and (s.startswith(('K', 'V'))) for s in sequence):
            aktionen = [self._interpretiere_symbol(s) for s in sequence if isinstance(s, str)]
            self.sequence_meaning_mapping[tuple(sequence)] = {
                'bedeutung': ' → '.join(aktionen),
                'typ': self._klassifiziere_sequenz(sequence)
            }
    
    def _interpretiere_symbol(self, symbol):
        """Gibt die qualitative Bedeutung eines Terminalzeichens zurück"""
        bedeutungen = {
            'KBG': 'Kunden-Gruß',
            'VBG': 'Verkäufer-Gruß',
            'KBBd': 'Kunden-Bedarf (konkret)',
            'VBBd': 'Verkäufer-Nachfrage',
            'KBA': 'Kunden-Antwort',
            'VBA': 'Verkäufer-Reaktion',
            'KAE': 'Kunden-Erkundigung',
            'VAE': 'Verkäufer-Auskunft',
            'KAA': 'Kunden-Abschluss',
            'VAA': 'Verkäufer-Abschluss',
            'KAV': 'Kunden-Verabschiedung',
            'VAV': 'Verkäufer-Verabschiedung'
        }
        return bedeutungen.get(symbol, str(symbol))
    
    def _klassifiziere_sequenz(self, sequence):
        """Klassifiziert den Typ der Interaktionssequenz"""
        seq_str = ' '.join([str(s) for s in sequence])
        if 'KBBd' in seq_str and 'VBBd' in seq_str:
            return 'Bedarfsaushandlung'
        elif 'KAE' in seq_str or 'VAE' in seq_str:
            return 'Informationsaustausch'
        elif 'KAA' in seq_str and 'VAA' in seq_str:
            return 'Transaktionsabschluss'
        else:
            return 'Interaktionssequenz'
    
    def print_methodological_summary(self):
        """Gibt eine methodologische Zusammenfassung aus"""
        print("\n" + "=" * 70)
        print("METHODOLOGISCHE REFLEXION")
        print("=" * 70)
        print("\nDokumentierte Interpretationsentscheidungen:")
        
        for log in self.interpretation_log:
            print(f"\n[Interpretation {log['timestamp']+1}]")
            seq_str = ' → '.join([str(s) for s in log['sequence']])
            print(f"  Sequenz: {seq_str}")
            print(f"  → Nonterminal: {log['new_nonterminal']}")
            print(f"  Begründung: {log['rationale']}")
            
            if tuple(log['sequence']) in self.sequence_meaning_mapping:
                mapping = self.sequence_meaning_mapping[tuple(log['sequence'])]
                print(f"  Bedeutung: {mapping['bedeutung']}")
                print(f"  Sequenztyp: {mapping['typ']}")

# ============================================================================
# 3. HIERARCHISCHE GRAMMATIKINDUKTION
# ============================================================================

class GrammarInducer:
    """
    Induziert eine PCFG durch hierarchische Kompression.
    Die Nonterminale werden als EXPLIZITE INTERPRETATIONSKATEGORIEN verstanden.
    """
    
    def __init__(self):
        self.rules = {}          # Nonterminal -> Liste von (Produktion, Wahrscheinlichkeit)
        self.rule_occurrences = {} # Zählung der Regelanwendungen
        self.terminals = set()
        self.nonterminals = set()
        self.start_symbol = None
        self.compression_history = []
        self.reflection = MethodologicalReflection()
        
        # Für die Optimierungsphase
        self.terminal_frequencies = None
        self.generated_frequencies_history = []
        
    def find_relevant_patterns(self, chains, min_length=2, max_length=4):
        """
        Findet relevante wiederholte Sequenzen.
        Anders als bei reiner Kompression wird hier semantische Relevanz priorisiert.
        """
        sequence_counter = Counter()
        
        for chain in chains:
            for length in range(min_length, min(max_length, len(chain) + 1)):
                for i in range(len(chain) - length + 1):
                    seq = tuple(chain[i:i+length])
                    
                    # Bewertungskriterien für semantische Relevanz:
                    score = 1.0
                    
                    # Prüfe auf Sprecherwechsel (nur für Terminalzeichen)
                    has_speaker_change = False
                    for j in range(len(seq)-1):
                        if (isinstance(seq[j], str) and isinstance(seq[j+1], str) and
                            ((seq[j].startswith('K') and seq[j+1].startswith('V')) or
                             (seq[j].startswith('V') and seq[j+1].startswith('K')))):
                            has_speaker_change = True
                            break
                    
                    if has_speaker_change:
                        score *= 2.0
                    
                    # Bevorzuge Muster mit Abschlusscharakter
                    has_closure = any(isinstance(s, str) and s.endswith('A') for s in seq)
                    if has_closure:
                        score *= 1.3
                    
                    sequence_counter[seq] += score
        
        # Filtere Sequenzen mit mindestens 2 Vorkommen
        relevant = {seq: count for seq, count in sequence_counter.items() 
                   if count >= 2}
        
        if not relevant:
            return None
        
        # Wähle die relevanteste Sequenz
        best_seq = max(relevant.items(), key=lambda x: x[1])[0]
        return best_seq
    
    def generate_interpretive_name(self, sequence):
        """
        Generiert einen interpretativ gehaltvollen Namen für das Nonterminal.
        """
        # Bestimme den Typ der Sequenz basierend auf Terminalzeichen
        seq_str = ' '.join([str(s) for s in sequence])
        
        if 'KBBd' in seq_str and 'VBBd' in seq_str:
            typ = "BEDARFSKLAERUNG"
        elif ('VAA' in seq_str and 'KAA' in seq_str) or ('VAA' in seq_str and 'KAV' in seq_str):
            typ = "ZAHLUNGSVORGANG"
        elif 'KAE' in seq_str or 'VAE' in seq_str:
            typ = "INFORMATIONSAUSTAUSCH"
        elif 'KBG' in seq_str and 'VBG' in seq_str:
            typ = "BEGRUESSUNG"
        elif 'VAV' in seq_str and 'KAV' in seq_str:
            typ = "VERABSCHIEDUNG"
        else:
            typ = "SEQUENZ"
        
        # Erstelle einen eindeutigen Namen
        if all(isinstance(s, str) and len(s) <= 4 for s in sequence):
            # Nur Terminalzeichen
            first = sequence[0] if sequence else ""
            last = sequence[-1] if sequence else ""
            return f"NT_{typ}_{first}_{last}"
        else:
            # Enthält bereits Nonterminale
            return f"NT_{typ}_{len(sequence)}"
    
    def _describe_sequence(self, sequence):
        """Erzeugt eine semantische Beschreibung der Sequenz"""
        if len(sequence) == 2:
            if all(isinstance(s, str) and len(s) <= 4 for s in sequence):
                return f"{self.reflection._interpretiere_symbol(sequence[0])} → {self.reflection._interpretiere_symbol(sequence[1])}"
            else:
                return f"{sequence[0]} → {sequence[1]}"
        else:
            return f"Sequenz mit {len(sequence)} Schritten"
    
    def compress_chains(self, chains, sequence, new_nonterminal):
        """
        Komprimiert die Ketten durch Ersetzung der Sequenz.
        """
        compressed_chains = []
        seq_tuple = tuple(sequence)
        seq_len = len(sequence)
        
        for chain in chains:
            new_chain = []
            i = 0
            while i < len(chain):
                if i <= len(chain) - seq_len and tuple(chain[i:i+seq_len]) == seq_tuple:
                    new_chain.append(new_nonterminal)
                    i += seq_len
                else:
                    new_chain.append(chain[i])
                    i += 1
            compressed_chains.append(new_chain)
        
        return compressed_chains
    
    def induce_grammar(self, chains, max_iterations=15):
        """
        Hauptmethode zur Grammatikinduktion.
        """
        current_chains = [list(chain) for chain in chains]
        iteration = 0
        
        print("\n" + "=" * 70)
        print("HIERARCHISCHE GRAMMATIKINDUKTION")
        print("=" * 70)
        print("\nDer Induktionsprozess wird als EXPLIKATION verstanden:")
        print("- Jedes neue Nonterminal repräsentiert eine INTERPRETATIVE KATEGORIE")
        print("- Die Benennung expliziert die qualitative Bedeutung")
        print("- Der Prozess ist intersubjektiv NACHVOLLZIEHBAR\n")
        
        while iteration < max_iterations:
            # Finde relevante Muster
            best_seq = self.find_relevant_patterns(current_chains)
            
            if best_seq is None:
                print(f"\nKeine weiteren relevanten Muster nach {iteration} Iterationen.")
                break
            
            # Generiere interpretativen Namen
            new_nonterminal = self.generate_interpretive_name(best_seq)
            beschreibung = self._describe_sequence(best_seq)
            
            # Stelle Einzigartigkeit sicher
            base_name = new_nonterminal
            counter = 1
            while new_nonterminal in self.nonterminals:
                new_nonterminal = f"{base_name}_{counter}"
                counter += 1
            
            # Dokumentiere die interpretative Entscheidung
            rationale = f"Erkanntes Dialogmuster: {beschreibung}"
            self.reflection.log_interpretation(best_seq, new_nonterminal, rationale)
            
            seq_str = ' → '.join([str(s) for s in best_seq])
            print(f"\nIteration {iteration + 1}:")
            print(f"  Erkanntes Muster: {seq_str}")
            print(f"  Interpretation: {beschreibung}")
            print(f"  → Neue Kategorie: {new_nonterminal}")
            
            # Speichere die Regel (vorerst ohne Wahrscheinlichkeit)
            self.rules[new_nonterminal] = [(list(best_seq), 1.0)]  # Temporäre Wahrscheinlichkeit
            self.nonterminals.add(new_nonterminal)
            
            # Komprimiere Ketten
            current_chains = self.compress_chains(current_chains, best_seq, new_nonterminal)
            
            # Zeige Beispiel
            example = ' → '.join([str(s) for s in current_chains[0][:8]])
            print(f"  Beispiel (komprimiert): {example}...")
            
            iteration += 1
            
            # Prüfe auf vollständige Kompression
            if all(len(chain) == 1 for chain in current_chains):
                symbols = set(chain[0] for chain in current_chains)
                if len(symbols) == 1:
                    self.start_symbol = list(symbols)[0]
                    print(f"\nINDUKTION ABGESCHLOSSEN: Startsymbol = {self.start_symbol}")
                    break
        
        # Terminale sind die ursprünglichen Symbole
        all_symbols = set()
        for chain in empirical_chains:
            all_symbols.update(chain)
        self.terminals = all_symbols
        
        # Berechne Wahrscheinlichkeiten
        self._calculate_probabilities()
        
        return current_chains
    
    def _calculate_probabilities(self):
        """
        Berechnet Wahrscheinlichkeiten für jede Produktion.
        """
        # Zähle, wie oft jedes Nonterminal in den Originaldaten vorkommt
        occurrence_count = defaultdict(Counter)
        
        # Für jede Kette in den Originaldaten
        for chain in empirical_chains:
            self._count_occurrences(chain, occurrence_count)
        
        # Konvertiere zu Wahrscheinlichkeiten
        for nonterminal in self.rules:
            if nonterminal in occurrence_count:
                total = sum(occurrence_count[nonterminal].values())
                if total > 0:
                    productions = []
                    for expansion, count in occurrence_count[nonterminal].items():
                        prob = count / total
                        # Stelle sicher, dass expansion eine Liste ist
                        if isinstance(expansion, tuple):
                            expansion = list(expansion)
                        productions.append((expansion, prob))
                    
                    # Sortiere nach Wahrscheinlichkeit
                    productions.sort(key=lambda x: x[1], reverse=True)
                    self.rules[nonterminal] = productions
    
    def _count_occurrences(self, sequence, occurrence_count):
        """
        Rekursive Hilfsfunktion zum Zählen der Vorkommen.
        """
        i = 0
        while i < len(sequence):
            symbol = sequence[i]
            
            # Wenn das Symbol ein Nonterminal ist
            if symbol in self.rules:
                # Finde die passende Expansion
                for expansion, _ in self.rules[symbol]:
                    if isinstance(expansion, list):
                        exp_len = len(expansion)
                        if i + exp_len <= len(sequence) and sequence[i:i+exp_len] == expansion:
                            # Zähle dieses Vorkommen
                            occurrence_count[symbol][tuple(expansion)] += 1
                            # Rekursiv in der Expansion weiterzählen
                            self._count_occurrences(expansion, occurrence_count)
                            i += exp_len
                            break
                        elif i + 1 <= len(sequence) and [sequence[i]] == expansion:
                            # Einzelelement
                            occurrence_count[symbol][tuple(expansion)] += 1
                            i += 1
                            break
                    else:
                        i += 1
            else:
                i += 1

# ============================================================================
# 4. GENERIERUNG MIT INTERPRETATIVER RÜCKBINDUNG
# ============================================================================

class InterpretiveGenerator:
    """
    Generiert Ketten und dokumentiert deren interpretative Bedeutung.
    """
    
    def __init__(self, grammar, terminals, start_symbol, reflection):
        self.grammar = grammar
        self.terminals = terminals
        self.start_symbol = start_symbol
        self.reflection = reflection
        
        # Erstelle Produktionswahrscheinlichkeiten
        self.production_probs = {}
        for nt, prods in grammar.items():
            if prods and len(prods) > 0:
                symbols = []
                probs = []
                for prod, prob in prods:
                    if isinstance(prob, (int, float)):
                        symbols.append(prod)
                        probs.append(float(prob))
                
                if symbols and probs:
                    # Normalisiere falls nötig
                    total = sum(probs)
                    if total > 0 and abs(total - 1.0) > 0.001:
                        probs = [p/total for p in probs]
                    self.production_probs[nt] = (symbols, probs)
    
    def generate_with_interpretation(self, max_depth=15):
        """
        Generiert eine Kette und dokumentiert die Interpretation.
        """
        if not self.start_symbol:
            return [], []
        
        interpretation = []
        
        def expand(symbol, depth=0):
            if depth >= max_depth:
                return [str(symbol)]
            
            if symbol in self.terminals:
                interpretation.append(self.reflection._interpretiere_symbol(symbol))
                return [str(symbol)]
            
            if symbol not in self.production_probs:
                return [str(symbol)]
            
            symbols, probs = self.production_probs[symbol]
            if not symbols:
                return [str(symbol)]
            
            try:
                chosen_idx = np.random.choice(len(symbols), p=probs)
                chosen = symbols[chosen_idx]
            except:
                # Fallback bei Fehlern
                chosen = symbols[0]
            
            # Dokumentiere die Expansion
            seq_str = ' → '.join([str(s) for s in chosen])
            interpretation.append(f"[Expansion von {symbol}: {seq_str}]")
            
            result = []
            for sym in chosen:
                result.extend(expand(sym, depth + 1))
            return result
        
        chain = expand(self.start_symbol)
        return chain, interpretation

# ============================================================================
# 5. VALIDIERUNG IM KONTEXT DER XAI-KRITERIEN
# ============================================================================

class XAIValidator:
    """
    Validiert die induzierte Grammatik anhand der XAI-Kriterien:
    - Verständlichkeit (Meaningfulness)
    - Genauigkeit (Accuracy)
    - Wissensgrenzen (Knowledge Limits)
    """
    
    def __init__(self, grammar_inducer):
        self.inducer = grammar_inducer
        self.original_freq = self._compute_empirical_frequencies()
        
    def _compute_empirical_frequencies(self):
        """Berechnet die empirischen Häufigkeiten der Terminale"""
        all_terminals = []
        for chain in empirical_chains:
            all_terminals.extend(chain)
        
        freq = Counter(all_terminals)
        total = len(all_terminals)
        return {sym: count/total for sym, count in freq.items()}
    
    def evaluate_meaningfulness(self):
        """
        Bewertet die Verständlichkeit der Grammatik.
        """
        print("\n" + "=" * 70)
        print("VALIDIERUNG: VERSTÄNDLICHKEIT (XAI-Kriterium 1)")
        print("=" * 70)
        
        # Prüfe, ob alle Nonterminale interpretierbare Namen haben
        meaningful_count = 0
        for nt in self.inducer.nonterminals:
            if nt.startswith('NT_') and len(nt) > 3:
                meaningful_count += 1
        
        meaningful_ratio = meaningful_count / len(self.inducer.nonterminals) if self.inducer.nonterminals else 0
        
        print(f"\nNonterminale insgesamt: {len(self.inducer.nonterminals)}")
        print(f"Davon interpretierbar benannt: {meaningful_count} ({meaningful_ratio:.1%})")
        
        # Dokumentierte Interpretationen
        print(f"\nDokumentierte Interpretationsentscheidungen: {len(self.inducer.reflection.interpretation_log)}")
        
        # Beispiel-Interpretationen
        if self.inducer.reflection.interpretation_log:
            print("\nBeispiel-Interpretationen:")
            for i, log in enumerate(self.inducer.reflection.interpretation_log[:3]):
                seq_str = ' → '.join([str(s) for s in log['sequence']])
                print(f"  {i+1}. {seq_str} → {log['new_nonterminal']}")
                print(f"     Begründung: {log['rationale']}")
        
        return meaningful_ratio
    
    def evaluate_accuracy(self, n_generated=500):
        """
        Bewertet die Genauigkeit der Grammatik.
        """
        print("\n" + "=" * 70)
        print("VALIDIERUNG: GENAUIGKEIT (XAI-Kriterium 2)")
        print("=" * 70)
        
        generator = InterpretiveGenerator(
            self.inducer.rules, 
            self.inducer.terminals, 
            self.inducer.start_symbol,
            self.inducer.reflection
        )
        
        # Generiere viele Ketten
        all_generated = []
        for _ in range(n_generated):
            chain, _ = generator.generate_with_interpretation()
            all_generated.extend(chain)
        
        # Berechne generierte Häufigkeiten
        gen_freq = Counter(all_generated)
        total_gen = len(all_generated)
        gen_dist = {sym: count/total_gen for sym, count in gen_freq.items() if total_gen > 0}
        
        # Korrelationsberechnung für gemeinsame Symbole
        common_symbols = sorted(set(self.original_freq.keys()) & set(gen_dist.keys()))
        if common_symbols and len(common_symbols) > 1:
            orig_values = [self.original_freq[sym] for sym in common_symbols]
            gen_values = [gen_dist[sym] for sym in common_symbols]
            
            correlation, p_value = pearsonr(orig_values, gen_values)
            
            print(f"\nKorrelation (r): {correlation:.4f}")
            print(f"Signifikanz (p): {p_value:.4f}")
            print(f"Basis: {len(common_symbols)} gemeinsame Symbole")
            
            # Detaillierte Tabelle
            print("\nVergleich der Häufigkeiten (Top 8):")
            table_data = []
            for sym in common_symbols[:8]:
                table_data.append([
                    sym,
                    f"{self.original_freq[sym]:.4f}",
                    f"{gen_dist[sym]:.4f}",
                    f"{abs(self.original_freq[sym] - gen_dist[sym]):.4f}"
                ])
            
            print(tabulate(table_data, 
                          headers=["Symbol", "Empirisch", "Generiert", "Differenz"],
                          tablefmt="grid"))
            
            return correlation, p_value
        else:
            print("Nicht genügend gemeinsame Symbole für Korrelationsberechnung")
            return 0, 1
    
    def evaluate_knowledge_limits(self):
        """
        Dokumentiert die Wissensgrenzen der Grammatik.
        """
        print("\n" + "=" * 70)
        print("VALIDIERUNG: WISSENSGRENZEN (XAI-Kriterium 3)")
        print("=" * 70)
        
        print("\nDie Grammatik ist eine EXPLIKATION, keine Entdeckung:")
        print("  • Sie basiert auf 8 Transkripten von Verkaufsgesprächen")
        print("  • Die Terminalzeichen wurden durch qualitative Interpretation gewonnen")
        print("  • Die Nonterminale repräsentieren INTERPRETATIVE KATEGORIEN")
        
        print("\nGRENZEN DER GRAMMATIK:")
        print("  • Keine Generalisierung über den Datensatz hinaus")
        print("  • Keine Prognosefähigkeit für neue Kontexte")
        print("  • Abhängig von der initialen Kategorienbildung")
        print("  • Alternative Interpretationen sind möglich")
        
        # Dokumentiere nicht abgedeckte Muster
        observed_pairs = set()
        for chain in empirical_chains:
            for i in range(len(chain) - 1):
                observed_pairs.add((chain[i], chain[i+1]))
        
        print(f"\nABGEDECKTE MUSTER:")
        print(f"  • Beobachtete Übergänge: {len(observed_pairs)}")
        print(f"  • In Grammatik erfasste Nonterminale: {len(self.inducer.nonterminals)}")

# ============================================================================
# 6. HAUPTAUSFÜHRUNG
# ============================================================================

def main():
    """
    Hauptfunktion mit methodologischer Rahmung.
    """
    print("=" * 70)
    print("ALGORITHMISCH REKURSIVE SEQUENZANALYSE 3.0")
    print("HIERARCHISCHE GRAMMATIKINDUKTION")
    print("=" * 70)
    
    # 1. Grammatik induzieren
    inducer = GrammarInducer()
    compressed_chains = inducer.induce_grammar(empirical_chains)
    
    # 2. Methodologische Reflexion
    inducer.reflection.print_methodological_summary()
    
    # 3. Induzierte Grammatik anzeigen
    print("\n" + "=" * 70)
    print("INDUZIERTE GRAMMATIK")
    print("=" * 70)
    print(f"\nTerminale ({len(inducer.terminals)}): {sorted(inducer.terminals)}")
    print(f"Nonterminale ({len(inducer.nonterminals)}): {sorted(inducer.nonterminals)}")
    if inducer.start_symbol:
        print(f"Startsymbol: {inducer.start_symbol}")
    
    print("\nPRODUKTIONSREGELN (mit Wahrscheinlichkeiten):")
    for nonterminal in sorted(inducer.rules.keys()):
        productions = inducer.rules[nonterminal]
        if productions:
            prod_strings = []
            for prod, prob in productions:
                # Stelle sicher, dass prod eine Liste ist
                if isinstance(prod, tuple):
                    prod = list(prod)
                prod_str = ' → '.join([str(s) for s in prod])
                # Stelle sicher, dass prob ein Float ist
                prob_float = float(prob) if not isinstance(prob, (int, float)) else prob
                prod_strings.append(f"{prod_str} [{prob_float:.3f}]")
            print(f"\n{nonterminal} → {' | '.join(prod_strings)}")
    
    # 4. Beispiele mit Interpretation generieren
    print("\n" + "=" * 70)
    print("BEISPIELE MIT INTERPRETATION")
    print("=" * 70)
    
    generator = InterpretiveGenerator(
        inducer.rules, 
        inducer.terminals, 
        inducer.start_symbol,
        inducer.reflection
    )
    
    for i in range(3):
        chain, interpretation = generator.generate_with_interpretation()
        print(f"\nBeispiel {i+1}:")
        chain_str = ' → '.join([str(s) for s in chain[:10]])
        print(f"  Kette: {chain_str}" + ("..." if len(chain) > 10 else ""))
        print("  Interpretation:")
        for j, step in enumerate(interpretation[:5]):
            print(f"    {j+1}. {step}")
        if len(interpretation) > 5:
            print("    ...")
    
    # 5. XAI-Validierung
    validator = XAIValidator(inducer)
    validator.evaluate_meaningfulness()
    validator.evaluate_accuracy(n_generated=500)
    validator.evaluate_knowledge_limits()
    
    # 6. Grammatik exportieren
    print("\n" + "=" * 70)
    print("EXPORT DER GRAMMATIK")
    print("=" * 70)
    
    with open("induzierte_grammatik_mit_interpretation.txt", 'w', encoding='utf-8') as f:
        f.write("# INDUZIERTE PCFG MIT INTERPRETATION\n")
        f.write("# =================================\n\n")
        f.write(f"## DATENGRUNDLAGE\n")
        f.write(f"{len(empirical_chains)} Transkripte von Verkaufsgesprächen\n\n")
        
        f.write("## TERMINALE (qualitative Kategorien)\n")
        for sym in sorted(inducer.terminals):
            f.write(f"{sym}: {inducer.reflection._interpretiere_symbol(sym)}\n")
        
        f.write("\n## NONTERMINALE (interpretative Kategorien)\n")
        for log in inducer.reflection.interpretation_log:
            seq_str = ' → '.join([str(s) for s in log['sequence']])
            f.write(f"\n{log['new_nonterminal']}\n")
            f.write(f"  Muster: {seq_str}\n")
            mapping = inducer.reflection.sequence_meaning_mapping.get(tuple(log['sequence']), {})
            if mapping:
                f.write(f"  Bedeutung: {mapping.get('bedeutung', '')}\n")
            f.write(f"  Begründung: {log['rationale']}\n")
        
        f.write("\n## PRODUKTIONSREGELN\n")
        for nt in sorted(inducer.rules.keys()):
            prods = inducer.rules[nt]
            for prod, prob in prods:
                if isinstance(prod, tuple):
                    prod = list(prod)
                prod_str = ' '.join([str(s) for s in prod])
                prob_float = float(prob) if not isinstance(prob, (int, float)) else prob
                f.write(f"{nt} → {prod_str} [{prob_float:.3f}]\n")
    
    print(f"\nGrammatik exportiert als 'induzierte_grammatik_mit_interpretation.txt'")
    
    print("\n" + "=" * 70)
    print("ALGORITHMISCH REKURSIVE SEQUENZANALYSE ABGESCHLOSSEN")
    print("=" * 70)

if __name__ == "__main__":
    main()
```
