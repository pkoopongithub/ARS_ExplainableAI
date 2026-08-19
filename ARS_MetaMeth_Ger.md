---
abstract: |
  Dieser Beitrag betrachtet die Algorithmisch Rekursive Sequenzanalyse (ARS) nicht nur als Methode, sondern als *Metamethodologie* -- ein Rahmenwerk, das die Bedingungen spezifiziert, unter denen ein Modell als *erklärend* und nicht nur als *beschreibend* oder *simulierend* gelten kann. Ausgehend von den Kernprinzipien der ARS -- dem Primat der Interpretation, der Trennung von Struktur und Statistik, der kontrollierten Falsifikation und der XAI-Validierung -- argumentiere ich, dass diese Prinzipien notwendige Kriterien für erklärende Modelle in jeder Disziplin darstellen, die sich mit sequenziellen sozialen Prozessen befasst. Der Beitrag setzt die ARS systematisch zu fünf zeitgenössischen Forschungsprogrammen in Beziehung: (1) Formale Verifikation und Model Checking, (2) Interpretierbares maschinelles Lernen und Rule Extraction, (3) Grounded Theory, (4) Kausale Inferenz und (5) Process Mining. Für jedes dieser Felder zeige ich, was die ARS von diesem Ansatz lernen kann und, entscheidend, was dieser Ansatz von der ARS lernen kann. Der Beitrag schließt mit der These, dass die ARS einen transdisziplinären Benchmark dafür bereitstellt, echte Erklärung von statistischer oder struktureller Beschreibung zu unterscheiden.
author:
- Paul Koop
date: 1994--2026
title: |
  **ARS als Metamethodologie**\
  Die Bedingungen erklärender Modelle\
  im Zeitalter generativer KI
---

# Einleitung: Von der Methode zur Metamethodologie

Die Algorithmisch Rekursive Sequenzanalyse (ARS) wurde in ihren verschiedenen Versionen (2.0--4.0) primär als eine Methode zur formalen Analyse sequenzieller Interaktionen präsentiert. Sie transformiert interpretativ gewonnene Kategorien in formale Modelle: probabilistische kontextfreie Grammatiken (PCFG), Petri-Netze, Bayessche Netze und deterministische endliche Automaten (DFA). Dieser Beitrag nimmt eine andere Perspektive ein. Er argumentiert, dass die ARS nicht nur eine Methode ist, sondern eine *Metamethodologie* -- ein Rahmenwerk, das die *Bedingungen* spezifiziert, unter denen ein Modell als erklärend gelten kann.

Diese Perspektivverschiebung wird durch eine wiederkehrende Verwirrung in der zeitgenössischen KI und Datenwissenschaft motiviert: die Gleichsetzung von **Simulation** mit **Erklärung**. Ein großes Sprachmodell (LLM), das auf Verkaufsgesprächen trainiert wurde, kann plausible Dialoge mit hoher Genauigkeit simulieren. Doch wie die ARS-Notizbücher gezeigt haben, kann es nicht erklären, *warum* eine bestimmte Sequenz von Sprechakten wohlgeformt ist oder welche Regeln ihre Generierung konstituieren. Das LLM liefert einen statistischen Schatten des Prozesses; die ARS liefert sein strukturelles Gerüst.

Die metamethodologische These dieses Beitrags ist, dass die ARS-Prinzipien **notwendige Bedingungen** für erklärende Modelle in jeder Disziplin darstellen, die sich mit sequenziellen sozialen Prozessen befasst. Diese Bedingungen sind:

1.  **Interpretative Verankerung**: Symbole müssen an dokumentierte qualitative Interpretationen gebunden sein, nicht nur an statistische Korrelationen.

2.  **Strukturelle Entscheidbarkeit**: Die Wohlgeformtheit von Sequenzen muss formal entscheidbar sein (z.B. durch einen DFA), unabhängig von empirischen Häufigkeiten.

3.  **Generative Transparenz**: Das Modell muss in der Lage sein, Sequenzen so zu generieren, dass jeder Schritt auf explizite Regeln zurückgeführt werden kann.

4.  **Falsifizierbarkeit**: Interpretationen müssen der kontrollierten Falsifikation unterliegen; Gegenbeispiele müssen Regeln widerlegen können.

5.  **XAI-Validierung**: Das Modell muss die NIST-XAI-Kriterien erfüllen -- Verständlichkeit, Genauigkeit, Wissensgrenzen.

Um diese These zu untermauern, setze ich die ARS systematisch zu fünf zeitgenössischen Forschungsprogrammen in Beziehung, die überlappende Anliegen teilen: formale Verifikation, interpretierbares maschinelles Lernen, Grounded Theory, kausale Inferenz und Process Mining. Für jedes stelle ich zwei reziproke Fragen:

1.  Was kann die ARS von diesem Ansatz lernen? (Technische oder konzeptionelle Verbesserungen)

2.  Was kann dieser Ansatz von der ARS lernen? (Methodologische Sicherungen, Kriterien für Erklärung)

# ARS als Metamethodologie: Die Kernprinzipien

Bevor die fünf Ansätze untersucht werden, müssen die ARS-Prinzipien dargelegt werden, die als metamethodologischer Benchmark dienen.

## Prinzip 1: Interpretative Verankerung

In der ARS ist jedes Terminalzeichen (KBG, VBBd, KAA etc.) das Produkt einer dokumentierten qualitativen Interpretation. Der Kodierungsprozess ist keine Black Box; er wird aufgezeichnet, begründet und unterliegt der intersubjektiven Validierung. Dieses Prinzip schließt rein datengetriebene Kategorienbildung (z.B. Clustering von Embeddings) als Erklärung aus.

## Prinzip 2: Trennung von Struktur und Statistik

Wie in `ARS_XAI_Aut2_Ger.tex` entwickelt, hält die ARS eine strikte Trennung zwischen strukturellen Regeln (die deterministisch und kontextfrei sind) und statistischen Regularitäten (die empirisch und kontingent sind) ein. Eine strukturelle Regel gilt oder gilt nicht -- unabhängig davon, wie oft sie verletzt wird. Diese Trennung fehlt in rein statistischen Modellen.

## Prinzip 3: Generative Transparenz

Die induzierte Grammatik muss in der Lage sein, Sequenzen auf nachvollziehbare Weise zu generieren. Der Transduktor in Lisp, der Parser in Pascal und der Induktor in Scheme bieten jeweils ein unterschiedliches Fenster in diese Transparenz. Ein Modell, das keine Exemplare aus seinen eigenen Regeln generieren kann, ist nicht erklärend.

## Prinzip 4: Kontrollierte Falsifikation

Interpretationen werden nicht ein für alle Mal behauptet. Sie werden als Lesarten produziert und dann durch nachfolgende Sequenzpositionen falsifiziert (nach Oevermanns Sequenzanalyse). Dies erzeugt eine Spirale von Interpretation und Widerlegung, die Poppers Falsifikationismus für qualitatives Material adaptiert.

## Prinzip 5: XAI-Validierung

Die drei NIST-XAI-Kriterien -- Verständlichkeit, Genauigkeit, Wissensgrenzen -- sind keine optionalen Zusätze, sondern konstitutive Elemente erklärender Modelle. Verständlichkeit erfordert semantische Interpretierbarkeit; Genauigkeit erfordert Übereinstimmung mit dem Material; Wissensgrenzen erfordern explizite Dokumentation der Modellgrenzen.

# Fünf Ansätze im Dialog mit der ARS

## Formale Verifikation und Model Checking

### Was formale Verifikation ist

Formale Verifikation, insbesondere Model Checking, ist eine Methode aus der theoretischen Informatik, die systematisch und algorithmisch prüft, ob ein formales Modell (z.B. ein endlicher Automat, ein Petri-Netz oder ein Bayessches Netz) spezifizierte Eigenschaften erfüllt. Diese Eigenschaften umfassen *Safety* (\"etwas Schlechtes passiert nie\") und *Liveness* (\"etwas Gutes passiert irgendwann\").

Model Checking ist exhaustiv: Es erkundet den gesamten Zustandsraum des Modells. Im Gegensatz zum statistischen Testen, das probabilistische Garantien liefert, bietet Model Checking *deterministische* Garantien über das Verhalten des Modells.

### Was die ARS von formaler Verifikation lernen kann

- **Spezifikationssprachen für Eigenschaften**: Die ARS könnte temporale Logiken (LTL, CTL) übernehmen, um zu spezifizieren, welche Eigenschaften die induzierte Grammatik erfüllen sollte. Zum Beispiel: $\square (KBG \rightarrow \lozenge VBG)$ (\"Immer, wenn ein Kunde grüßt, grüßt irgendwann der Verkäufer zurück\").

- **Generierung von Gegenbeispielen**: Wenn eine Eigenschaft verletzt wird, produzieren Model Checker eine Gegenbeispielspur. Dies könnte als ein mächtiges Falsifikationswerkzeug für ARS-Interpretationen dienen.

- **Bewusstsein für Zustandsraumexplosion**: ARS-Modellierer sollten sich bewusst sein, dass hierarchische Grammatiken zu großen Zustandsräumen führen können. Formale Verifikation bietet Abstraktionstechniken zur Bewältigung dieses Problems.

### Was formale Verifikation von der ARS lernen kann

- **Interpretative Verankerung von Zuständen**: Im standardmäßigen Model Checking sind Zustände abstrakte Symbole. Die ARS besteht darauf, dass jeder Zustand interpretativ verankert sein muss. Dies könnte zu einem neuen Teilgebiet führen: *interpretatives Model Checking*, bei dem Eigenschaften geprüft *und* die Bedeutung der Zustände dokumentiert wird.

- **Die Trennung von Struktur und Statistik**: Model Checking geht typischerweise von einem deterministischen Modell aus. Die ARS zeigt, wie strukturelle Regeln (die verifiziert werden können) von statistischen Variationen (die es nicht können) getrennt werden können. Dies könnte das probabilistische Model Checking bereichern.

- **XAI-Kriterien für Verifikation**: Die Ergebnisse des Model Checkings sind für Nicht-Experten oft opak. Die XAI-Kriterien der ARS könnten die Entwicklung verständlicherer Verifikationsausgaben leiten.

## Interpretierbares maschinelles Lernen und Rule Extraction

### Was IML und Rule Extraction sind

Interpretierbares maschinelles Lernen (IML) zielt darauf ab, Black-Box-Modelle (neuronale Netze, Gradient Boosting, Random Forests) für den Menschen verständlich zu machen. *Rule Extraction* (Regel-Extraktion) ist eine spezifische IML-Technik, die versucht, die gelernte Funktion approximativ durch eine Menge von Wenn-Dann-Regeln zu beschreiben (z.B. mit RIPPER, CART oder durch Analyse von Aktivierungsmustern).

Im Gegensatz zur ARS, die Regeln direkt aus Daten induziert, arbeitet Rule Extraction typischerweise *post-hoc*: Das Modell ist bereits trainiert, und Regeln werden als Erklärung extrahiert.

### Was die ARS von IML und Rule Extraction lernen kann

- **Skalierbare Regelinduktion**: Die ARS induziert derzeit Regeln aus kleinen Korpora (n = 8). IML bietet Techniken zur Extraktion von Regeln aus großen Datensätzen, wenn auch mit geringerer interpretativer Kontrolle.

- **Quantitative Regelbewertung**: IML bietet Metriken zur Bewertung extrahierter Regeln (Coverage, Fidelity, Stabilität). Die ARS könnte diese übernehmen, um zu beurteilen, wie gut eine Grammatik generalisiert.

- **Hybride Regelmengen**: Einige IML-Methoden kombinieren globale und lokale Regeln. Die ARS könnte hybride Grammatiken mit einem strukturellen Kern und lokalen statistischen Variationen erforschen.

### Was IML und Rule Extraction von der ARS lernen können

- **Explikation vs. Approximation**: Die Rule Extraction in IML ist fast immer approximativ. Die ARS besteht auf *exakten* Regeln für das gegebene Korpus. Dies wirft eine grundlegende Frage auf: Ist Approximation für Erklärung jemals akzeptabel? Die ARS legt nahe: Approximation ist nur akzeptabel, wenn der Approximationsfehler dokumentiert ist und der strukturelle Kern exakt bleibt.

- **Interpretative Validierung von Regeln**: IML extrahiert Regeln, die statistisch zu den Daten passen. Die ARS fügt eine Ebene der *interpretativen Validierung* hinzu: Regeln müssen auch für menschliche Interpreten sinnvoll sein. Dies könnte die Extraktion von statistisch korrekten, aber semantisch bedeutungslosen Regeln verhindern.

- **Falsifizierbarkeit als Kriterium**: IML diskutiert selten, wie extrahierte Regeln falsifiziert werden könnten. Die ARS macht Falsifizierbarkeit zu einem zentralen Kriterium. IML könnte dies übernehmen: Eine Regelmenge ist nicht erklärend, wenn kein denkbares Gegenbeispiel sie widerlegen könnte.

## Grounded Theory

### Was Grounded Theory ist

Die Grounded Theory (GT) nach Glaser und Strauss ist eine klassische Methodologie der qualitativen Sozialforschung zur Entwicklung von Theorien aus Daten. Sie umfasst Verfahren wie offenes Kodieren, axiales Kodieren und selektives Kodieren. Das Ziel ist die Generierung von Theorien mittlerer Reichweite, die im empirischen Material \"verankert\" (grounded) sind.

In jüngerer Zeit gibt es Versuche, Teile der GT zu formalisieren oder computergestützt zu unterstützen (z.B. durch Natural Language Processing oder Topic Modeling). Die GT bleibt jedoch in ihrer endgültigen Ausgabe weitgehend informell.

### Was die ARS von der Grounded Theory lernen kann

- **Systematische Kodierverfahren**: Die GT bietet ein reiches Vokabular und eine Reihe von Kodierverfahren, die die interpretative Phase der ARS bereichern könnten. Konzepte wie \"axiales Kodieren\" (Beziehung von Kategorien zu Unterkategorien) ähneln der hierarchischen Kompression der ARS, sind aber feinkörniger.

- **Theoretisches Sampling**: Das Prinzip des theoretischen Samplings -- Auswahl neuer Fälle basierend auf sich entwickelnden theoretischen Einsichten -- könnte die Fallauswahl der ARS in größeren Studien leiten.

- **Ständiger Vergleich**: Die Methode des ständigen Vergleichs (Vergleich jedes neuen Falls mit bereits entwickelten Kategorien) ist bereits implizit im systematischen Fallvergleich der ARS (Phase 4) enthalten, könnte aber expliziter gemacht werden.

### Was die Grounded Theory von der ARS lernen kann

- **Von der Theorie zum generativen Modell**: Die GT stoppt typischerweise auf der Ebene narrativer Theorie oder Kategoriensysteme. Die ARS geht weiter: Sie transformiert die Theorie in eine *generative Grammatik*, die neue Sequenzen produzieren kann. Die GT könnte dies übernehmen, um ihre Theorien testbar und ausführbar zu machen.

- **Formale Falsifizierbarkeit**: Die Validierungsverfahren der GT sind primär qualitativ (z.B. Member Checking, Peer Debriefing). Die ARS fügt formale Falsifizierbarkeit hinzu: Die Grammatik kann auf eine Weise falsch sein, die mechanisch demonstriert werden kann (z.B. durch einen Parser, der eine Sequenz ablehnt).

- **XAI-Kriterien für Grounded Theory**: Die XAI-Kriterien der ARS (Verständlichkeit, Genauigkeit, Wissensgrenzen) bieten eine Checkliste für die Bewertung grounded theories. Eine GT-Theorie, die ihre Wissensgrenzen nicht spezifizieren kann, ist unvollständig.

## Kausale Inferenz und kausale Graphenmodelle

### Was kausale Inferenz ist

Kausale Inferenz, insbesondere mit kausalen Graphenmodellen (z.B. DAGs, DoWhy, CausalNex), geht über bloße Korrelation hinaus und versucht, kausale Beziehungen zwischen Variablen zu identifizieren und zu quantifizieren. Sie verwendet Techniken wie den Do-Kalkül, instrumentelle Variablen und kontrafaktisches Schließen.

Eine zentrale Einsicht der kausalen Inferenz ist, dass Korrelation nicht Kausalität ist. Gerichtete azyklische Graphen (DAGs) werden verwendet, um Annahmen über kausale Strukturen zu repräsentieren.

### Was die ARS von kausaler Inferenz lernen kann

- **Kausale Interpretation von Grammatiken**: ARS-Grammatiken beschreiben sequenzielle Abhängigkeiten. Kausale Inferenz könnte helfen zu unterscheiden, ob diese Abhängigkeiten lediglich sequenziell oder genuin kausal sind. Zum Beispiel: Verursacht die Frage \"Sonst noch etwas?\" einen zusätzlichen Kauf oder ist sie nur damit korreliert?

- **Kontrafaktisches Schließen**: Kausale Inferenz exzelliert bei der Beantwortung kontrafaktischer Fragen (\"Was wäre passiert, wenn der Verkäufer nicht gefragt hätte?\"). Die ARS könnte die kontrafaktische Simulation (bereits vorhanden in Phase 3 der CGTI) als Standardvalidierungswerkzeug übernehmen.

- **Instrumentelle Variablen für sequenzielle Daten**: Die ARS befasst sich mit sequenziellen Daten, bei denen Confounding häufig ist. Techniken mit instrumentellen Variablen könnten helfen, kausale Effekte selbst in beobachteten sequenziellen Daten zu identifizieren.

### Was kausale Inferenz von der ARS lernen kann

- **Interpretative Verankerung kausaler Graphen**: In der standardmäßigen kausalen Inferenz wird der DAG oft ohne interpretative Dokumentation angenommen oder aus Daten gelernt. Die ARS besteht darauf, dass jeder Knoten und jede Kante interpretativ verankert sein muss. Dies könnte zu einer *interpretativen kausalen Inferenz* als neuem Teilgebiet führen.

- **Sequenzielle Grammatiken als kausale Strukturen**: ARS-Grammatiken sind eine Form der kausalen Struktur über Sequenzen. Die kausale Inferenz befasst sich typischerweise mit statischen oder Zeitreihendaten, nicht mit grammatikalischen Sequenzen. Die ARS könnte eine neue Klasse *grammatikalischer kausaler Modelle* inspirieren.

- **XAI für kausale Modelle**: Kausale Modelle werden oft als DAGs mit Wahrscheinlichkeiten präsentiert, die nicht selbsterklärend sind. Die XAI-Kriterien der ARS könnten die Dokumentation kausaler Modelle leiten und sie für Domänenexperten zugänglicher machen.

## Process Mining

### Was Process Mining ist

Process Mining ist ein Forschungsfeld an der Schnittstelle von Data Mining, maschinellem Lernen und Prozessmodellierung. Es zielt darauf ab, aus Ereignislogs -- sequenziellen Aufzeichnungen von Prozessschritten, z.B. in Workflow-Management-Systemen oder ERP-Systemen -- Prozessmodelle (häufig in Form von Petri-Netzen, BPMN-Diagrammen oder direkten Folgengraphen) zu entdecken, zu konformieren und zu verbessern.

Process Mining arbeitet typischerweise mit großen, anonymisierten und schwach annotierten Logs. Es entdeckt den \"durchschnittlichen Prozess\" oder die \"häufigsten Pfade\". Es zielt nicht auf eine vollständige Rekonstruktion der konstitutiven Regeln eines Einzelfalls ab.

### Was die ARS von Process Mining lernen kann

- **Skalierbare Entdeckungsalgorithmen**: Process Mining bietet ausgefeilte Algorithmen (z.B. Alpha-Miner, Heuristics-Miner, Inductive Miner) zur Entdeckung von Petri-Netzen aus großen Logs. Die ARS könnte diese für größere Korpora übernehmen oder anpassen, während sie die interpretative Kontrolle bewahrt.

- **Conformance Checking**: Process Mining umfasst Techniken zur Prüfung, ob ein Ereignislog mit einem gegebenen Modell konform ist. Dies könnte zur Validierung von ARS-Grammatiken an neuen Daten verwendet werden.

- **Leistungsanalyse**: Process Mining fügt Leistungsdimensionen (Zeit, Kosten, Häufigkeit) hinzu. Die ARS könnte erweitert werden, um zeitliche und ressourcenbezogene Dimensionen systematischer zu integrieren.

### Was Process Mining von der ARS lernen kann

- **Interpretative Entdeckung für kleine Logs**: Process Mining benötigt typischerweise große Logs, um zuverlässige Modelle zu produzieren. Die ARS zeigt, wie aus einem einzigen Fall (n = 1) durch interpretative Tiefe Modelle entdeckt werden können. Dies könnte für Process Mining in Bereichen mit knappen Daten wertvoll sein (z.B. medizinische Verfahren, juristische Fälle).

- **Dokumentation von Entdeckungsentscheidungen**: Process-Mining-Algorithmen treffen viele Entscheidungen (z.B. welche Pfade einbezogen werden, wie mit Rauschen umgegangen wird). Diese Entscheidungen werden selten in interpretativ zugänglicher Weise dokumentiert. Die reflexive Dokumentation der ARS könnte als Vorbild dienen.

- **Trennung von Struktur und Statistik**: Process Mining produziert oft Modelle, die strukturelle Regeln mit statistischem Rauschen vermischen. Die strikte Trennung der ARS könnte die Qualität der entdeckten Modelle verbessern, indem klar unterschieden wird, was strukturell notwendig ist und was nur empirisch häufig vorkommt.

- **XAI für Process Mining**: Die von Process Mining produzierten Modelle (z.B. spaghettiartige Petri-Netze) sind oft schwer verständlich. Die XAI-Kriterien der ARS könnten die Entwicklung verständlicherer Process-Mining-Ausgaben leiten.

# Auf dem Weg zu einem transdisziplinären Benchmark für Erklärung

Die fünf Vergleiche oben offenbaren ein gemeinsames Muster. Jeder zeitgenössische Ansatz hat technische Stärken, die die ARS verbessern könnten: Skalierbarkeit (IML, Process Mining), formale Rigorosität (Verifikation, kausale Inferenz) und systematische Kodierverfahren (Grounded Theory). Umgekehrt fehlt jedem Ansatz ein Teil der metamethodologischen Sicherungen, die die ARS bietet: interpretative Verankerung, strukturelle Entscheidbarkeit, generative Transparenz, kontrollierte Falsifikation und XAI-Validierung.

Diese Symmetrie legt nahe, dass die ARS nicht nur eine Methode unter anderen ist, sondern ein **transdisziplinärer Benchmark** dafür, was als erklärendes Modell gelten kann.

::: {#tab:benchmark}
  **Kriterium**                   **Frage**                                                             **Abwesenheit bedeutet\...**
  ------------------------------- --------------------------------------------------------------------- --------------------------------------------------------------
  Interpretative Verankerung      Sind die Symbole sinnhaft dokumentiert?                               Statistische Korrelation ohne Verstehen
  Strukturelle Entscheidbarkeit   Ist Wohlgeformtheit formal entscheidbar?                              Probabilistisches Raten statt Regelbefolgung
  Generative Transparenz          Kann das Modell Exemplare nachvollziehbar generieren?                 Simulation ohne Erklärung
  Kontrollierte Falsifikation     Können Gegenbeispiele Regeln widerlegen?                              Unfalsifizierbares post-hoc Storytelling
  XAI-Validierung                 Sind Verständlichkeit, Genauigkeit und Wissensgrenzen dokumentiert?   Technische Raffinesse ohne epistemische Rechenschaftspflicht

  : ARS als transdisziplinärer Benchmark
:::

## Die Unterscheidung von Erklärung und Simulation neu betrachtet

Die Kernunterscheidung, die aus dieser Analyse hervorgeht, ist die zwischen **Erklärung** und **Simulation**. Ein Modell *simuliert*, wenn es die statistischen Eigenschaften der Daten reproduziert. Ein Modell *erklärt*, wenn es die konstitutiven Regeln spezifiziert, die das Phänomen erzeugen.

- **Simulation** ist ausreichend für Prognose. Ein LLM, das das nächste Token genau vorhersagt, ist ein guter Simulator.

- **Erklärung** ist notwendig für Verstehen, Intervention und normative Bewertung. Eine ARS-Grammatik, die die Regeln eines Verkaufsgesprächs spezifiziert, ist eine Erklärung.

Die fünf Kriterien oben sind die Bedingungen, unter denen ein Modell als Erklärung und nicht nur als Simulation qualifiziert.

## Die Rolle des menschlichen Interpreten

Ein wiederkehrendes Thema in allen fünf Vergleichen ist die Rolle des menschlichen Interpreten. In der ARS ist der Mensch konstitutiv: Interpretation ist ein menschlicher Akt, der nicht vollständig automatisiert werden kann. In den anderen Ansätzen ist der Mensch oft extern -- entwirft Algorithmen, stellt Trainingsdaten bereit, evaluiert Ausgaben.

Dies ist keine Schwäche der ARS, sondern eine Stärke. Die ARS macht explizit, was oft implizit bleibt: dass Erklärung eine menschliche Praxis ist, keine Eigenschaft eines isoliert betrachteten Modells. Ein Modell ist erklärend *für jemanden*, der es verstehen, nutzen und dafür Rechenschaft ablegen kann.

# Fazit und Ausblick

Dieser Beitrag hat argumentiert, dass die Algorithmisch Rekursive Sequenzanalyse (ARS) nicht nur eine Methode ist, sondern eine Metamethodologie -- ein Rahmenwerk, das die Bedingungen spezifiziert, unter denen ein Modell als erklärend gelten kann. Fünf Kernprinzipien wurden identifiziert: interpretative Verankerung, strukturelle Entscheidbarkeit, generative Transparenz, kontrollierte Falsifikation und XAI-Validierung.

Der Beitrag setzte die ARS dann systematisch zu fünf zeitgenössischen Forschungsprogrammen in Beziehung: formaler Verifikation, interpretierbarem maschinellen Lernen, Grounded Theory, kausaler Inferenz und Process Mining. Für jedes wurde gezeigt, was die ARS von diesen Ansätzen lernen kann (technische Verbesserungen) und, entscheidend, was diese Ansätze von der ARS lernen können (methodologische Sicherungen, Kriterien für Erklärung).

Die metamethodologische These ist, dass die ARS-Prinzipien notwendige Bedingungen für erklärende Modelle in jeder Disziplin darstellen, die sich mit sequenziellen sozialen Prozessen befasst. Sie bieten einen transdisziplinären Benchmark für die Unterscheidung echter Erklärung von statistischer oder struktureller Beschreibung.

Für die weitere Forschung sind drei Richtungen besonders vielversprechend:

1.  **Implementierung hybrider Systeme**: Integration von ARS-Grammatiken mit Model Checkern, Rule Extractors oder Process-Mining-Algorithmen unter Wahrung der metamethodologischen Sicherungen.

2.  **Empirische Prüfung des Benchmarks**: Anwendung der fünf Kriterien auf bestehende Modelle in verschiedenen Disziplinen und Prüfung, ob sie die wahrgenommene Erklärungsqualität vorhersagen.

3.  **Erweiterung auf nicht-sequenzielle Domänen**: Obwohl die ARS für sequenzielle Daten entwickelt wurde, könnten die metamethodologischen Prinzipien auf andere Modelltypen generalisieren (z.B. Klassifikation, Clustering, Regression).

Abschließend: Die Frage ist nicht, ob ein Modell zu den Daten passt. Statistische Passung ist notwendig, aber nicht hinreichend. Die Frage ist, ob das Modell die metamethodologischen Kriterien erfüllt, die Erklärung erst möglich machen. Die ARS bietet eine konkrete, operationalisierte Antwort.

::: thebibliography
99

Baier, C., & Katoen, J.-P. (2008). *Principles of Model Checking*. MIT Press.

Glaser, B. G., & Strauss, A. L. (1967). *The Discovery of Grounded Theory: Strategies for Qualitative Research*. Aldine.

Koop, P. (1992). *Demo-Parser Chart-Parser Version 1.0*. Pascal-Quellcode.

Koop, P. (1994). *Grammatikinduktion empirisch gesicherter Verkaufsgespräche*. Scheme-Quellcode.

Koop, P. (1994). *Sequenzanalyse empirisch gesicherter Verkaufsgespräche*. Lisp-Quellcode.

Koop, P. (2023). *Qualitative Sozialforschung und Große Sprachmodelle*. Jupyter Notebook.

Koop, P. (2024/2026). *Zwischen Interpretation und Berechnung: Algorithmisch Rekursive Sequenzanalyse als Brücke zwischen qualitativer Hermeneutik und formaler Modellierung*. the-last-freedom.org.

Koop, P. (2026). *Soziale Strukturen und Prozesse: Kausale Inferenz mit Probabilistischen kontextfreien Grammatiken und Bayesschen Netzen*. the-last-freedom.org.

Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. leanpub.com.

Oevermann, U., Allert, T., Konau, E., & Krambeck, J. (1979). Die Methodologie einer objektiven Hermeneutik. In H.-G. Soeffner (Hrsg.), *Interpretative Verfahren in den Sozial- und Textwissenschaften* (S. 352-434). Metzler.

Ortigossa, E. S., Gonçalves, T., & Nonato, L. G. (2024). Explainable Artificial Intelligence (XAI)---From Theory to Methods and Applications. *IEEE Access*, 12, 80799-80846.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2. Aufl.). Cambridge University Press.

van der Aalst, W. M. P. (2016). *Process Mining: Data Science in Action* (2. Aufl.). Springer.
:::
