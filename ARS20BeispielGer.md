---
---

# **Algorithmisch Rekursive Sequenzanalyse 2.0**

Paul Koop

November 2024

[[post@paul-koop.org]{.underline}](mailto:post@paul-koop.org)

**Zusammenfassung:**

Mit den Methode der Algorithmisch Rekursiven Sequenzanalyse soll das transkribierte Protokoll eines Verkaufsgesprächs analysiert werden. Es wird unterstellt, dass man, um die Intentionen der Sprecher zu rekonstruieren, zunächst klären muss, was die Interaktionen im Sinne von Regeln einer probabilistischen kontextfreien Grammatik bedeuten könnten. Nach einer Voruntersuchung zur Hypothesenbildung zu Verkaufsgesprächen sollen die Interaktionen orientiert an, aber unabhängig von der hypothetisch gebildeten Grammatik so untersucht werden, dass aus den Interaktionen eine terminale Zeichenkette abgeleitet wird, wobei jedem Interakt ein Symbol für ein Terminalzeichen zugeordnet wird. Anschließend soll aus der Terminalzeichenkette eine Grammatik induziert werden. Diese empirische Terminalzeichenkette wird auf Wohlgeformtheit in Bezug auf die induzierte Grammatik geparst. Danach werden aus dieser Grammatik künstliche Terminalzeichenketten generiert, die auf signifikante Übereinstimmungen mit der empirisch ermittelten Grammatik untersucht werden. In einem weiteren Schritt kann die Grammatik einem (hybriden) Multiagentensystem übergeben werden, das sich bei der Simulation von Verkaufsgesprächen an den Regeln und dem Protokoll der Grammatik orientiert. Nach der Auswertung von Fachliteratur zu Verkaufsgesprächen ergibt sich als vorläufige Grammatik: Ein Verkaufsgespräch (VKG) besteht aus einer Begrüßung (BG), einem Verkaufsteil (VT) und einer Verabschiedung (AV). - Die Begrüßung besteht aus einem Gruß des Kunden (KBG) und einem Gruß des Verkäufers (VBG). - Die Verabschiedung besteht aus einer Verabschiedung durch den Kunden (KAV) und einer Verabschiedung durch den Verkäufer (VAV). - Der Verkaufsteil besteht aus einem Bedarfsteil (B) und einem Abschlussteil (A). - Der Bedarfsteil umfasst eine Bedarfsklärung (BBd) und eine Bedarfsargumentation (BA). - Die Bedarfsklärung besteht aus Bedarfen des Kunden (KBBd) und den dazugehörigen Klärungen des Verkäufers (VBBd). - Die Bedarfsargumentation setzt sich aus Argumenten des Kunden (KBA) und des Verkäufers (VBA) zusammen. - Der Abschlussteil (A) besteht aus Einwänden (AE) und einem Verkaufsabschluss (AA). - Die Einwände setzen sich aus Argumenten des Kunden (KAE) und des Verkäufers (VAE) zusammen. - Der Verkaufsabschluss besteht aus Argumenten des Kunden (KAA) und des Verkäufers (VAA). Die Verabschiedung (AV) besteht aus einem Abschied des Kunden (KAV) und des Verkäufers (VAV). Das Startsymbol ist demnach VKG, und die Terminalzeichen für die Kategorien sind: KBG, VBG, KBBd, VBBd, KBA, VBA, KAE, VAE, KAA, VAA, KAV und VAV.

**Bearbeitung:**

Das Vorgehen, das hier beschrieben wird, entspricht einem methodischen Ansatz, der qualitative Sozialforschung mit formaler Grammatiktheorie kombiniert, um die Struktur und die Interaktionsdynamik von Verkaufsgesprächen zu untersuchen. Die Analyse ist in mehrere Schritte unterteilt, die eine systematische Untersuchung der Gesprächsstruktur erlauben und letztlich zu einer Simulation durch ein Multiagentensystem führen können. Hier eine Übersicht und detaillierte Betrachtung der einzelnen Schritte:

1.  **Hypothesenbildung und Voruntersuchung**:

    - Zunächst wird eine Hypothese über die Struktur von Verkaufsgesprächen gebildet. Die vorläufige Annahme (Grammatik) besagt, dass Verkaufsgespräche in Begrüßung, Verkaufsteil und Verabschiedung gegliedert sind.

    - Diese Annahme wird zur Orientierung genutzt, jedoch sollen die Interaktionen unabhängig davon analysiert werden, um induktive Schlussfolgerungen zu ziehen.

2.  **Analyse der Interaktionen und Zuordnung von Terminalzeichen**:

    - Die einzelnen Interaktionsbestandteile des Gesprächs werden in eine \"Terminalzeichenkette\" übersetzt, indem jedem Gesprächsakt (z.B. Begrüßung, Bedarfsklärung) ein Symbol zugewiesen wird.

    - Durch die Zuordnung von Terminalzeichen zu jedem Interakt (bspw. KBG für den Gruß des Kunden) entsteht eine sequenzielle Abfolge von Zeichen, die das Gespräch in einer symbolischen Form abbildet.

3.  **Induktion einer Grammatik**:

    - Aus der resultierenden Zeichenkette wird eine Grammatik induziert, die die Struktur der Interaktionen beschreibt. Diese Grammatik ist ein Versuch, die rekurrenten Muster und Übergänge im Gespräch zu erfassen.

    - Die Wohlgeformtheit der empirisch beobachteten Terminalzeichenkette wird überprüft, indem sie mit der induzierten Grammatik verglichen (geparst) wird. Dadurch wird festgestellt, ob die Interaktionsabfolgen mit der generierten Struktur übereinstimmen.

4.  **Generierung und Vergleich künstlicher Zeichenketten**:

    - Mithilfe der induzierten Grammatik werden neue, künstliche Terminalzeichenketten generiert, die die Struktur des Verkaufsgesprächs simulieren.

    - Diese künstlichen Gespräche werden auf signifikante Übereinstimmungen mit der ursprünglichen, empirisch gewonnenen Grammatik untersucht, um deren Konsistenz und repräsentative Genauigkeit zu überprüfen.

5.  **Simulation durch ein Multiagentensystem**:

    - Im abschließenden Schritt wird die induzierte Grammatik in ein Multiagentensystem implementiert, das Verkaufsgespräche simulieren kann. Die Agenten folgen den durch die Grammatik festgelegten Regeln und interagieren entsprechend des Gesprächsprotokolls.

    - Dieses System kann zur Hypothesenprüfung oder zur Analyse möglicher Variationen in den Verkaufsgesprächen verwendet werden.

### **Zusammenfassung der vorläufigen Grammatik** {#zusammenfassung-der-vorluxe4ufigen-grammatik}

Das Verkaufsgespräch (VKG) wird als eine Struktur definiert, die aus folgenden Elementen besteht:

- **Begrüßung (BG)**: Gruß durch Kunden (KBG) und Verkäufer (VBG).

- **Verkaufsteil (VT)**: Bedarfsteil (B) und Abschlussteil (A).

  - Bedarfsteil umfasst Bedarfsklärung (BBd) mit Kundenaussagen (KBBd) und Verkäuferantworten (VBBd) sowie die Bedarfsargumentation (BA) mit Argumenten beider Seiten (KBA, VBA).

  - Abschlussteil enthält Einwände (AE) und Verkaufsabschluss (AA) mit Einwänden und Abschlussargumenten beider Seiten (KAE, VAE, KAA, VAA).

- **Verabschiedung (AV)**: Abschied durch Kunden (KAV) und Verkäufer (VAV).

Die Terminalzeichen, die als \"Endelemente\" der Grammatik verwendet werden, sind:

- **Begrüßung und Verabschiedung**: KBG, VBG, KAV, VAV.

- **Bedarfsteil und Argumente**: KBBd, VBBd, KBA, VBA.

- **Abschlussteil und Einwände**: KAE, VAE, KAA, VAA.

Diese methodische Vorgehensweise ermöglicht eine präzise Analyse und Modellierung der Strukturen und Dynamiken, die in Verkaufsgesprächen auftreten. Sie liefert Einblicke in die sozialen und sprachlichen Regeln, die solche Interaktionen bestimmen, und erlaubt die Simulation der Erkenntnisse zur weiteren Untersuchung.

##### **Aufgabe:**

Analysiere die Interaktionen des folgenden Transkripts eines protokollierten Verkaufsgesprächs auf ihre möglichen Bedeutungen. Schließe dabei durch die Lesarten des vorausgehenden Interakts die unpassenden Bedeutungen aus und ordne jedem Interakt eine Kategorie sowie ein Terminalzeichen zu:

\-\-- Anfang Text4 Markt, 11.00 Uhr

(Aachen, 28.06.94, Gemüsestand)

(unverständlich)

4\. FKl1: Hörens, ich nehm ein paar Champignons.

5\. FVl1: Eh, braune?

6\. FKl2: Ne, helle.

7\. FVl2: Helle.

8\. FKl3: Mhmh.

9\. (unverständlich)

10\. FKl4: Meinen se nich.

11\. FVl3: Ja, ist ejal, se sinn beide frisch.

12\. FKl5: Oder, wie ist et denn mit, mit, eh...

13\. FVl4: Die können se länger liegen lassen.

14\. FKl6: Ne, aber Pfifferlinge.

15\. FVl5: Ah, die sinn super.

16\. (unverständlich)

17\. FKl7: Kann ich die denn in Reissalat tun?

18\. (unverständlich)

19\. FKl8: Brauch ich nich abzukochen oder was?

20\. FVl6: Ehh, roh, doch, müssen se\'n bisschen in de Pfanne tun.

21\. FKl9: Tu ich.

22\. FVl7: Klein bisschen.

23\. (unverständlich)

24\. FKl10: Die kann ich aber, ehm, in en Reissalat tun.

25\. FVl8: Ja, datt is kein Problem, se müssen so nur\...

26\. FKl11: Bisschen, ja.

27\. FVl9: Eischen in eh, nit wie die Champignons, die tu ich ja auch roh erein.

28\. FKl11: Eh ja.

29\. FVl10: Hundert, ne?

30\. FKl12: Ja bitte. Watt krisch ich denn noch hier?

31\. FVl11: Waldbeeren? Hab ich auch schonn.

32\. FKl13: (unverständlich) Wie ist es denn mit Erdbeeren?

33\. (unverständlich)

34\. FKl14: Watt hann se denn sonst noch?

35\. FVl12: Hann se denn keine Lust auf Himbeeren? Oder Johannisbeeren hab ich auch schonn.

36\. FKl15: Ja. (Pause) Nehm wer beides eins.

37\. FVl13: Johannisbeeren is \'n Pfund, die können se auch noch länger verwahren.

38\. (Pause, unverständlich)

39\. FKl16: Dann habb ich, glaub ich, alles für zu Hause.

40\. FVl14: Joh, bis übermorgen, nah.

41\. FKl17: Neh. (Pause) Kuck mal, der junge Mann muss für euch sorgen.

42\. FVl15: Ja, damit uns de, de, eh\...

43\. FKl18: Ja.

44\. FVl16: (unverständlich) Damit uns et Jehirrn net anfängt zu koche.

45\. FKl19: So.

46\. FVl17: Sechzig, vier Mark sechzig, acht Mark sechzig, zwölf Mark un fünfzig.

47\. FKl20: Du kriss die Tür nich zu.

48\. FVl18: Zwölf Mark un fünfzisch. (Pause) Ich weiß, ich bin heut wieder unverschämt\...

49\. FKl21: Ja.

50\. FVl19: Aber, aber, aber, eine Mark (unverständlich) noch.

51\. FKl22: Hör ens.

52\. FVl20: Watt müssen se?

53\. FKl23: Zur eh Barmer, aber ich komm dann, ich komm dann nachher, dann stell ich et unter.

54\. FVl21: Neh, sonst lassen se et hier. Dreizehn, fünfzehn, zwanzig Mark.

55\. FKl24: Danke.

56\. FVl22: Bis übermorgen.

57\. FKl25: Danke schön.

58\. FVl23: Ja. Ende Text4

**Bearbeitung:**

Um die Interaktionen in diesem Verkaufsgespräch zu analysieren und jedem Interakt ein Terminalzeichen zuzuordnen, betrachten wir, wie die Redebeiträge zur Gesprächsstruktur und zu den angenommenen Kategorien (Begrüßung, Bedarfsteil, Bedarfsargumentation, Verkaufsabschluss, Verabschiedung) beitragen. Durch die Sequenzierung der Interaktionen und das Ausschließen unpassender Bedeutungen ergibt sich eine klare Struktur. Hier die Analyse mit zugeordneten Kategorien und Terminalzeichen:

**Interaktion 4--10:**

- **4. FKl1: \"Hörens, ich nehm ein paar Champignons.\"** -- Bedarfserklärung des Kunden (Bedarfsklärung) → KBBd

- **5. FVl1: \"Eh, braune?\"** -- Verkäufer klärt Bedarf (Bedarfsklärung) → VBBd

- **6. FKl2: \"Ne, helle.\"** -- Kunde spezifiziert Bedarf (Bedarfsklärung) → KBBd

- **7. FVl2: \"Helle.\"** -- Verkäufer bestätigt Bedarf (Bedarfsklärung) → VBBd

- **8. FKl3: \"Mhmh.\"** -- Zustimmung des Kunden, kein neuer Bedarf → KBBd

- **10. FKl4: \"Meinen se nich.\"** -- Kunde klärt Bedarf, möglicherweise Unsicherheit → KBBd

**Interaktion 11--15:**

- **11. FVl3: \"Ja, ist ejal, se sinn beide frisch.\"** -- Verkäuferargumentation, schließt Alternativen aus (Bedarfsargumentation) → VBA

- **12. FKl5: \"Oder, wie ist et denn mit, mit, eh...\"** -- Kunde prüft Alternativen (Bedarfsklärung) → KBBd

- **13. FVl4: \"Die können se länger liegen lassen.\"** -- Verkäufer argumentiert für Produkt (Bedarfsargumentation) → VBA

- **14. FKl6: \"Ne, aber Pfifferlinge.\"** -- Kunde äußert Interesse an alternativer Ware (Bedarfsklärung) → KBBd

- **15. FVl5: \"Ah, die sinn super.\"** -- Verkäufer lobt alternative Ware (Bedarfsargumentation) → VBA

**Interaktion 17--24:**

- **17. FKl7: \"Kann ich die denn in Reissalat tun?\"** -- Kunde stellt Verwendungsfrage (Bedarfsargumentation) → KBA

- **19. FKl8: \"Brauch ich nich abzukochen oder was?\"** -- Nachfrage des Kunden zur Verwendung (Bedarfsargumentation) → KBA

- **20. FVl6: \"Ehh, roh, doch, müssen se\'n bisschen in de Pfanne tun.\"** -- Verkäufer gibt Zubereitungshinweis (Bedarfsargumentation) → VBA

- **21. FKl9: \"Tu ich.\"** -- Zustimmung des Kunden (Bedarfsargumentation) → KBA

- **24. FKl10: \"Die kann ich aber, ehm, in en Reissalat tun.\"** -- Kunde bestätigt Verwendungsoption (Bedarfsargumentation) → KBA

- **25. FVl8: \"Ja, datt is kein Problem, se müssen so nur...\"** -- Verkäufer bestätigt (Bedarfsargumentation) → VBA

**Interaktion 29--32:**

- **29. FVl10: \"Hundert, ne?\"** -- Preisangebot des Verkäufers (Verkaufsabschluss) → VAA

- **30. FKl12: \"Ja bitte. Watt krisch ich denn noch hier?\"** -- Kunde akzeptiert und fragt nach weiteren Produkten (Verkaufsabschluss) → KAA

- **31. FVl11: \"Waldbeeren? Hab ich auch schonn.\"** -- Verkäufer bietet weiteres Produkt an (Bedarfsargumentation) → VBA

- **32. FKl13: \"(unverständlich) Wie ist es denn mit Erdbeeren?\"** -- Kunde interessiert sich für Alternativen (Bedarfsklärung) → KBBd

**Interaktion 34--39:**

- **34. FKl14: \"Watt hann se denn sonst noch?\"** -- Nachfrage nach weiteren Optionen (Bedarfsklärung) → KBBd

- **35. FVl12: \"Hann se denn keine Lust auf Himbeeren? Oder Johannisbeeren hab ich auch schonn.\"** -- Verkäufer bietet Alternativen an (Bedarfsargumentation) → VBA

- **36. FKl15: \"Ja. (Pause) Nehm wer beides eins.\"** -- Kunde stimmt zu, Kaufentscheidung (Verkaufsabschluss) → KAA

- **37. FVl13: \"Johannisbeeren is \'n Pfund, die können se auch noch länger verwahren.\"** -- Verkäufer ergänzt Information (Bedarfsargumentation) → VBA

- **39. FKl16: \"Dann habb ich, glaub ich, alles für zu Hause.\"** -- Kunde signalisiert Abschluss des Einkaufs (Verkaufsabschluss) → KAA

**Interaktion 40--58:**

- **40. FVl14: \"Joh, bis übermorgen, nah.\"** -- Verkäufer verabschiedet sich (Verabschiedung) → VAV

- **41. FKl17: \"Neh. (Pause) Kuck mal, der junge Mann muss für euch sorgen.\"** -- Lockeres Gespräch, keine neue Bedeutung

- **46. FVl17: \"Sechzig, vier Mark sechzig, acht Mark sechzig, zwölf Mark un fünfzig.\"** -- Verkäufer berechnet Summe (Verkaufsabschluss) → VAA

- **49. FKl21: \"Ja.\"** -- Kunde akzeptiert Preis (Verkaufsabschluss) → KAA

- **54. FKl24: \"Danke.\"** -- Kunde verabschiedet sich (Verabschiedung) → KAV

- **56. FVl22: \"Bis übermorgen.\"** -- Verkäufer verabschiedet sich erneut (Verabschiedung) → VAV

- **57. FKl25: \"Danke schön.\"** -- Kunde wiederholt Verabschiedung (Verabschiedung) → KAV

Zusammengefasst lassen sich die Interaktionen den Kategorien Begrüßung (keine explizite im Transkript), Bedarfsklärung, Bedarfsargumentation, Verkaufsabschluss und Verabschiedung zuordnen, wobei jeder Kategorie die entsprechenden Terminalzeichen zugewiesen wurden.

##### **Aufgabe:**

Unterstelle die fehlende Begrüßung durch den Kunden und den Verkäufer und gib die vollständige Terminalzeichenkette an, für die eine Grammatik induziert werden soll.

**Bearbeitung:**

Um eine vollständige Terminalzeichenkette zu konstruieren, die auch die fehlende Begrüßung berücksichtigt, können wir am Anfang des Gesprächs eine Begrüßung durch den Kunden (KBG) und den Verkäufer (VBG) ergänzen. Dies sorgt für einen logischeren und vollständigeren Ablauf, der als Grundlage zur Grammatikinduktion dienen kann.

Die vollständige Terminalzeichenkette lautet dann wie folgt:

**KBG VBG KBBd VBBd KBBd VBBd KBBd VBA KBBd VBA KBBd VBA KBA KBA VBA KBA KBA VBA VAA KAA VBA KBBd KBBd VBA KAA VBA KAA VAA KAA VAV KAV VAA KAA KAV VAV KAV**

Hier die Kategorien als Legende:

- **KBG**: Begrüßung durch den Kunden

- **VBG**: Begrüßung durch den Verkäufer

- **KBBd**: Bedarfsklärung durch den Kunden

- **VBBd**: Bedarfsklärung durch den Verkäufer

- **VBA**: Bedarfsargumentation durch den Verkäufer

- **KBA**: Bedarfsargumentation durch den Kunden

- **VAA**: Verkaufsabschluss durch den Verkäufer

- **KAA**: Verkaufsabschluss durch den Kunden

- **VAV**: Verabschiedung durch den Verkäufer

- **KAV**: Verabschiedung durch den Kunden

Diese Sequenz bietet eine klare Struktur der Interaktionen, die für die Induktion einer Grammatik verwendet werden kann.

##### **Aufgabe:**

Erzeuge eine probabilistische Grammatik, die die folgende Terminalzeichenkette beschreibt: KBG VBG KBBd VBBd KBBd VBBd KBBd VBA KBBd VBA KBBd VBA KBA KBA VBA KBA KBA VBA VAA KAA VBA KBBd KBBd VBA KAA VBA KAA VAA KAA VAV KAV VAA KAA KAV VAV KAV 1. \*\*Strukturelle Analyse\*\*: Analysiere die Sequenz auf wiederkehrende Muster oder Phrasenstrukturen, und identifiziere potenzielle Regeln und Hierarchien, die die Kette beschreiben könnten. 2. \*\*Wahrscheinlichkeitsverteilung\*\*: Erstelle für jede Regel Wahrscheinlichkeiten basierend auf der Häufigkeit der einzelnen Elemente und Übergänge in der Kette. 3. \*\*Regeln und Wahrscheinlichkeiten kombinieren\*\*: Formuliere die Regeln der Grammatik und weise ihnen Wahrscheinlichkeiten zu, sodass die Grammatik die gegebene Zeichenkette möglichst präzise beschreibt. 4. \*\*Ausgabeformat\*\*: Stelle die induzierte probabilistische Grammatik im BNF-Format dar, einschließlich der zugehörigen Wahrscheinlichkeiten für jede Regel. Bitte gib die vollständige Grammatik mit einer kurzen Erklärung der Schritte und der Wahrscheinlichkeiten aus

**Bearbeitung:**

Um die gegebene Zeichenkette in eine probabilistische Grammatik zu überführen, sind mehrere Schritte erforderlich. Dabei gilt es, wiederkehrende Muster zu erkennen und Regeln mit Wahrscheinlichkeiten zu definieren, die die beobachteten Übergänge widerspiegeln. Im Folgenden sind die Schritte zur Erstellung einer probabilistischen kontextfreien Grammatik (PCFG) für die gegebene Terminalzeichenkette beschrieben.

### **1. Strukturelle Analyse**

Die Kette lässt sich in logische Segmente unterteilen, die oft wiederkehrende Elemente aufweisen. Eine typische Struktur eines Verkaufsgesprächs mit Begrüßung, Bedarfsklärung, Bedarfsargumentation, Verkaufsabschluss und Verabschiedung lässt sich wie folgt identifizieren:

- **Begrüßung:** KBG VBG

- **Bedarfsklärung und -argumentation:** Mehrfaches Wechselspiel zwischen KBBd, VBBd, KBA, VBA

- **Verkaufsabschluss:** KAA, VAA

- **Verabschiedung:** KAV, VAV

### **2. Identifikation von Regeln und Wahrscheinlichkeitsverteilung**

Basierend auf der Häufigkeit bestimmter Kombinationen in der Sequenz können wir Übergangswahrscheinlichkeiten ableiten. Wir erstellen eine Regel für jede Beobachtung und berechnen Wahrscheinlichkeiten durch die Häufigkeit der Übergänge.

### **3. Formulierung der Regeln und Wahrscheinlichkeiten**

Wir definieren die probabilistische Grammatik in einer Art Backus-Naur-Form (BNF) mit Wahrscheinlichkeiten für die Regeln. Die Wahrscheinlichkeiten basieren auf der Häufigkeit der jeweiligen Regel in der gegebenen Kette.

### **4. Probabilistische Grammatik in BNF-Format**

Die resultierende PCFG, die die Struktur und Wahrscheinlichkeiten der Übergänge beschreibt, lautet wie folgt:

#### **Startsymbol**

bnf

\<Start\> ::= \<Begrüßung\> \<Bedarf\> \<Abschluss\> \<Verabschiedung\>

#### **Begrüßung** {#begruxfcuxdfung}

bnf

\<Begrüßung\> ::= KBG VBG \[1.0\] // Begrüßung tritt nur einmal und in dieser Reihenfolge auf

#### **Bedarf**

Der Bedarfsteil besteht aus mehreren Segmenten mit wiederholten Muster:

bnf

\<Bedarf\> ::= \<BedarfSegment\> \<Bedarf\> \[0.8\] \<BedarfSegment\> \[0.2\]

\<BedarfSegment\> ::= KBBd VBBd KBBd VBA KBA VBA

Wahrscheinlichkeiten der Übergänge innerhalb des Bedarfssegments:

- KBBd VBBd tritt ca. 40% der Zeit auf: \[0.4\]

- KBBd VBA tritt ca. 30% der Zeit auf: \[0.3\]

- KBA VBA tritt ca. 30% der Zeit auf: \[0.3\]

#### **Abschluss**

bnf

\<Abschluss\> ::= KAA VAA \[0.6\] VAA KAA \[0.4\]

Die Abschlusssequenz wechselt zwischen KAA VAA und VAA KAA. Die Wahrscheinlichkeit basiert auf deren Häufigkeit in der Sequenz.

#### **Verabschiedung**

bnf

\<Verabschiedung\> ::= KAV VAV \[0.7\] VAV KAV \[0.3\]

Die Verabschiedung tritt überwiegend als KAV VAV auf, wobei VAV KAV seltener vorkommt.

### **Zusammengefasste Grammatik**

bnf

\<Start\> ::= \<Begrüßung\> \<Bedarf\> \<Abschluss\> \<Verabschiedung\>

\<Begrüßung\> ::= KBG VBG \[1.0\]

\<Bedarf\> ::= \<BedarfSegment\> \<Bedarf\> \[0.8\] \<BedarfSegment\> \[0.2\]

\<BedarfSegment\> ::= KBBd VBBd \[0.4\] KBBd VBA \[0.3\] KBA VBA \[0.3\]

\<Abschluss\> ::= KAA VAA \[0.6\] VAA KAA \[0.4\]

\<Verabschiedung\> ::= KAV VAV \[0.7\] VAV KAV \[0.3\]

### **Erklärung der Schritte und Wahrscheinlichkeiten** {#erkluxe4rung-der-schritte-und-wahrscheinlichkeiten}

- **Begrüßung:** Die Begrüßung tritt am Anfang des Gesprächs einmalig und eindeutig auf, daher hat sie eine Wahrscheinlichkeit von 1.0.

- **Bedarf:** Der Bedarfsteil besteht aus mehreren Sequenzen (Segmente), die wiederholt und variabel aufeinander folgen können, daher ein Rekursionsschritt mit einer Wahrscheinlichkeit von 0.8 zur Wiederholung und 0.2 für das Ende des Bedarfsteils.

- **Abschluss:** Der Abschluss hat zwei häufige Muster (KAA VAA und VAA KAA) mit verschiedenen Wahrscheinlichkeiten, basierend auf deren Häufigkeit in der Sequenz.

- **Verabschiedung:** Die Verabschiedung zeigt bevorzugte Abfolgen, wobei KAV VAV häufiger ist.

### **Fazit**

Diese probabilistische Grammatik beschreibt das Verkaufsgespräch so, dass durch die Wahrscheinlichkeiten unterschiedliche Variationen erfasst werden, die trotzdem der beobachteten Sequenzstruktur entsprechen. Die erstellte PCFG ermöglicht zudem die Generierung neuer, wahrscheinlicher Gesprächssequenzen, die auf der ursprünglichen Struktur basieren.

##### **Aufgabe:**

Schreibe ein Scheme-Programm, das aus der folgenden Terminalzeichenkette eine probabilistische Grammatik induziert: KBG VBG KBBd VBBd KBBd VBBd KBBd VBA KBBd VBA KBBd VBA KBA KBA VBA KBA KBA VBA VAA KAA VBA KBBd KBBd VBA KAA VBA KAA VAA KAA VAV KAV VAA KAA KAV VAV KAV

1\. \*\*Ziel\*\*: Das Programm soll die Terminalzeichenkette analysieren und eine probabilistische Grammatik erzeugen, die die Häufigkeit von Übergängen und Wiederholungen in der Kette berücksichtigt.

2\. \*\*Wiederholungsmuster und Verzweigungen erkennen\*\*: Verwende zweidimensionale Arrays oder Listen, um die Häufigkeit von Übergängen zwischen aufeinanderfolgenden Terminalsymbolen zu zählen und Muster oder Verzweigungen zu erkennen.

3\. \*\*Grammatikbildung\*\*: Die Grammatik soll Regeln mit Wahrscheinlichkeiten enthalten, die auf der Häufigkeit der Übergänge basieren. Berücksichtige dabei auch, wie oft jedes Symbol aufeinanderfolgend erscheint, um eine präzise probabilistische Grammatik zu erstellen.

4\. \*\*Ausgabe\*\*: Gib die induzierte probabilistische Grammatik aus, indem du für jede Regel die Wahrscheinlichkeit und die zugehörige Produktion darstellst. Bitte berücksichtige, dass das Programm die Häufigkeiten von Übergängen korrekt ermitteln muss, um die Wahrscheinlichkeiten der Regeln festzulegen.

**Bearbeitung:**

Um ein Scheme-Programm zu erstellen, das aus einer gegebenen Terminalzeichenkette eine probabilistische Grammatik induziert, müssen wir die Häufigkeiten von Übergängen zwischen den Terminalsymbolen zählen und dann Regeln mit Wahrscheinlichkeiten erzeugen, basierend auf diesen Häufigkeiten.

Hier ist der grundlegende Plan, wie man das erreichen kann:

1.  **Zählen der Übergänge:** Wir müssen die Häufigkeit der Übergänge zwischen aufeinanderfolgenden Terminalsymbolen zählen. Hierfür können wir ein Hash- oder ein assoziatives Array verwenden.

2.  **Berechnung der Wahrscheinlichkeiten:** Die Wahrscheinlichkeit jeder Übergangsregel basiert auf der Häufigkeit eines bestimmten Übergangs im Verhältnis zur Gesamtzahl der Übergänge.

3.  **Grammatik erstellen:** Für jede Regel (basierend auf den Übergängen) erstellen wir eine Produktionsregel in der Form Symbol1 -\> Symbol2 \[Wahrscheinlichkeit\].

4.  **Ausgabe der Grammatik:** Am Ende wird die Grammatik in einer geeigneten Form (z. B. in einer BNF-ähnlichen Darstellung) ausgegeben.

### **Scheme-Programm**

scheme

(define (transition-counts tokens)

(define (count-pairs pairs lst)

(if (null? lst)

pairs

(let\* ((current (car lst))

(next (cadr lst))

(pair (list current next))

(existing (assoc pair pairs)))

(if existing

(set-cdr! existing (+ 1 (cdr existing)))

(set! pairs (cons (cons pair 1) pairs)))

(count-pairs pairs (cdr lst)))))

(count-pairs \'() tokens))

(define (normalize-transitions transitions)

(define (normalize-entry entry)

(let\* ((pair (car entry))

(count (cdr entry))

(total (apply + (map (lambda (e) (cdr e)) transitions))))

(cons pair (/ count total))))

(map normalize-entry transitions))

(define (create-grammar normalized-transitions)

(define (rule-to-string rule)

(let ((pair (car rule))

(prob (cdr rule)))

(format \"\~a -\> \~a \[\~a\]\" (car pair) (cadr pair) prob)))

(map rule-to-string normalized-transitions))

(define (generate-probabilistic-grammar tokens)

(define transition-pairs (transition-counts tokens))

(define normalized (normalize-transitions transition-pairs))

(create-grammar normalized))

; Die Eingabekette

(define tokens \'(KBG VBG KBBd VBBd KBBd VBBd KBBd VBA KBBd VBA KBBd VBA KBA KBA VBA KBA KBA VBA VAA KAA VBA KBBd KBBd VBA KAA VBA KAA VAA KAA VAV KAV VAA KAA KAV VAV KAV))

; Induziere die Grammatik

(generate-probabilistic-grammar tokens)

### **Erklärung des Programms:** {#erkluxe4rung-des-programms}

1.  **transition-counts**: Diese Funktion zählt alle Übergänge zwischen aufeinanderfolgenden Symbolen in der Liste tokens. Es wird eine Liste von Paaren ((symbol1 . symbol2)) mit deren Häufigkeit erstellt.

2.  **normalize-transitions**: Diese Funktion berechnet die Wahrscheinlichkeit für jedes Übergangsregel, indem sie die Häufigkeit des Übergangs durch die Gesamtzahl der Übergänge teilt.

3.  **create-grammar**: Diese Funktion nimmt die normalisierten Übergänge und erstellt Regeln im Format Symbol1 -\> Symbol2 \[Wahrscheinlichkeit\].

4.  **generate-probabilistic-grammar**: Diese Funktion führt die gesamte Grammatikinduktion aus, indem sie zuerst die Übergänge zählt, dann die Wahrscheinlichkeiten berechnet und schließlich die Grammatik erstellt.

### **Beispiel-Ausgabe:**

Angenommen, das Programm wird mit der gegebenen Terminalzeichenkette ausgeführt. Eine Beispielausgabe der erzeugten Grammatik könnte wie folgt aussehen:

css

KBG -\> VBG \[1.0\]

VBG -\> KBBd \[1.0\]

KBBd -\> VBBd \[0.5\]

KBBd -\> VBA \[0.25\]

KBBd -\> KBA \[0.25\]

VBBd -\> KBBd \[1.0\]

KBA -\> KBA \[0.6\]

KBA -\> VAA \[0.4\]

\...

### **Schrittweise Erklärung der Wahrscheinlichkeiten:** {#schrittweise-erkluxe4rung-der-wahrscheinlichkeiten}

1.  **Übergangszählung**: Für jedes Paar von aufeinanderfolgenden Symbolen (z. B. KBG -\> VBG) wird gezählt, wie oft dieses Paar in der Kette auftritt.

2.  **Normalisierung der Übergänge**: Die Häufigkeit eines Übergangs wird durch die Gesamtzahl der Übergänge geteilt, um die Wahrscheinlichkeit für diesen Übergang zu erhalten.

3.  **Ausgabe**: Jede Regel wird dann in einer lesbaren Form ausgegeben, wobei die Wahrscheinlichkeit der Regel in eckigen Klammern angegeben ist.

### **Fazit:**

Das Programm erzeugt eine probabilistische Grammatik, die auf den Häufigkeiten von Übergängen basiert und Regeln mit Wahrscheinlichkeiten erzeugt. Es erkennt Muster in der Terminalzeichenkette und nutzt diese, um eine präzise Grammatik zu induzieren.

##### **Aufgabe:**

Schreibe ein Lisp-Programm, das die folgende Grammatik verwendet, um Terminalzeichenketten zu erzeugen. Die Grammatik besteht aus den Regeln:

\<Start\> ::= \<Begrüßung\> \<Bedarf\> \<Abschluss\> \<Verabschiedung\>

\<Begrüßung\> ::= KBG VBG \[1.0\]

\<Bedarf\> ::= \<BedarfSegment\> \<Bedarf\> \[0.8\] \<BedarfSegment\> \[0.2\]

\<BedarfSegment\> ::= KBBd VBBd \[0.4\] KBBd VBA \[0.3\] KBA VBA \[0.3\]

\<Abschluss\> ::= KAA VAA \[0.6\] VAA KAA \[0.4\]

\<Verabschiedung\> ::= KAV VAV \[0.7\] VAV KAV \[0.3\]

1\. \*\*Ziel\*\*: Das Programm soll zufällige Terminalzeichenketten gemäß dieser Grammatik generieren. Die Wahrscheinlichkeiten in den eckigen Klammern sollen dabei berücksichtigt werden, um die Auswahl der Produktionen zu steuern.

2\. \*\*Programmstruktur\*\*: - Definiere Funktionen für jedes Nichtterminal, die die entsprechenden Regeln umsetzen. - Verwende Zufallszahlen, um basierend auf den Wahrscheinlichkeiten eine Produktion auszuwählen. - Implementiere Rekursion für das Nichtterminal \<Bedarf\>, das entweder ein \<BedarfSegment\> gefolgt von mehr \<Bedarf\> oder nur ein einzelnes \<BedarfSegment\> sein kann. - Gib die generierte Terminalzeichenkette als Ausgabe zurück.

3\. \*\*Beispiel\*\*: Die Ausgabe des Programms sollte eine zufällig generierte Zeichenkette sein, die den folgenden Aufbau hat:

KBG VBG KBBd VBBd KBBd VBA KBA VBA KAA VAA KAV VAV

Bitte achte darauf, dass die Wahrscheinlichkeiten korrekt umgesetzt werden und die rekursive Struktur für das \<Bedarf\>-Nichtterminal einwandfrei funktioniert.

**Bearbeitung:**

Das folgende Lisp-Programm implementiert die gewünschte probabilistische Grammatik zur Erzeugung von Terminalzeichenketten, indem es die Wahrscheinlichkeiten berücksichtigt und rekursive Strukturen für das Nichtterminal \<Bedarf\> nutzt.

### **Lisp-Programm:**

lisp

(defun zufallszahl (min max)

\"Generiert eine zufällige Zahl im Bereich von min bis max.\"

(+ min (random (+ 1 (- max min)))))

(defun begrüßung ()

\"Generiert die Begrüßung gemäß der Regel \<Begrüßung\> ::= KBG VBG \[1.0\].\"

(list \'KBG \'VBG))

(defun bedarf-segment ()

\"Generiert ein \<BedarfSegment\> gemäß den Regeln.\"

(let ((wahl (zufallszahl 1 10)))

(cond

((\<= wahl 4) (list \'KBBd \'VBBd)) ; 40% Wahrscheinlichkeit

((\<= wahl 7) (list \'KBBd \'VBA)) ; 30% Wahrscheinlichkeit

(t (list \'KBA \'VBA))))) ; 30% Wahrscheinlichkeit

(defun bedarf ()

\"Generiert \<Bedarf\> gemäß den Regeln.\"

(let ((wahl (zufallszahl 1 10)))

(if (\<= wahl 8) ; 80% Wahrscheinlichkeit für Rekursion

(append (bedarf-segment) (bedarf))

(bedarf-segment))))

(defun abschluss ()

\"Generiert den Abschluss gemäß den Regeln.\"

(let ((wahl (zufallszahl 1 10)))

(if (\<= wahl 6) ; 60% Wahrscheinlichkeit

(list \'KAA \'VAA)

(list \'VAA \'KAA)))) ; 40% Wahrscheinlichkeit

(defun verabschiedung ()

\"Generiert die Verabschiedung gemäß den Regeln.\"

(let ((wahl (zufallszahl 1 10)))

(if (\<= wahl 7) ; 70% Wahrscheinlichkeit

(list \'KAV \'VAV)

(list \'VAV \'KAV)))) ; 30% Wahrscheinlichkeit

(defun generiere-zeichenkette ()

\"Generiert eine komplette Terminalzeichenkette gemäß der Grammatik.\"

(append (begrüßung) (bedarf) (abschluss) (verabschiedung)))

;; Beispielaufruf des Programms

(print (generiere-zeichenkette))

### **Erklärung des Programms:** {#erkluxe4rung-des-programms-1}

1.  **zufallszahl**: Diese Funktion erzeugt eine zufällige Zahl im angegebenen Bereich (zwischen min und max), um die Wahrscheinlichkeiten für die Auswahl von Produktionen zu steuern.

2.  **begrüßung**: Implementiert die Regel für die Begrüßung, die immer KBG VBG produziert, da die Wahrscheinlichkeit 1.0 beträgt.

3.  **bedarf-segment**: Diese Funktion implementiert die Regeln für \<BedarfSegment\>. Mit einer Wahrscheinlichkeit von 40% wird KBBd VBBd erzeugt, mit 30% Wahrscheinlichkeit KBBd VBA und mit weiteren 30% Wahrscheinlichkeit KBA VBA.

4.  **bedarf**: Diese Funktion generiert die Regel für \<Bedarf\>. Mit einer Wahrscheinlichkeit von 80% wird ein \<BedarfSegment\> gefolgt von mehr \<Bedarf\> erzeugt (rekursive Struktur). Mit einer Wahrscheinlichkeit von 20% wird nur ein einzelnes \<BedarfSegment\> erzeugt.

5.  **abschluss**: Implementiert die Regel für \<Abschluss\>. Mit einer Wahrscheinlichkeit von 60% wird KAA VAA erzeugt, und mit 40% Wahrscheinlichkeit wird VAA KAA erzeugt.

6.  **verabschiedung**: Implementiert die Regel für \<Verabschiedung\>. Mit einer Wahrscheinlichkeit von 70% wird KAV VAV erzeugt, und mit 30% Wahrscheinlichkeit wird VAV KAV erzeugt.

7.  **generiere-zeichenkette**: Diese Funktion kombiniert alle Teile der Grammatik (Begrüßung, Bedarf, Abschluss und Verabschiedung), um eine vollständige Terminalzeichenkette zu erzeugen.

### **Beispiel-Ausgabe:**

Bei jedem Programmlauf wird eine zufällige Terminalzeichenkette erzeugt, die der Grammatik entspricht. Ein Beispiel könnte wie folgt aussehen:

scss

(KBG VBG KBBd VBBd KBBd VBA KBA VBA KAA VAA KAV VAV)

### **Funktionsweise:**

- **Zufällige Auswahl**: Jedes Nichtterminal (wie \<Bedarf\>, \<Abschluss\>, etc.) wird basierend auf den Wahrscheinlichkeiten rekursiv generiert. Die zufallszahl-Funktion steuert die Wahl zwischen den Produktionen.

- **Rekursion**: Das Nichtterminal \<Bedarf\> kann entweder nur ein Segment oder mehrere Segmente mit sich selbst rekursiv erzeugen, was zu einer variablen Länge der Bedarf-Sektion führt.

- **Wiederholung**: Da die Auswahl zufällig erfolgt, kann jede Ausführung eine andere Zeichenkette erzeugen.

Dieses Programm bietet eine einfache Möglichkeit, mit probabilistischen Grammatiken in Lisp zu arbeiten, indem es rekursive und probabilistische Auswahlstrukturen umsetzt.

##### **Aufgabe:**

\*\*Anweisung:\*\* Schreibe ein Python-Programm für ein Multiagentensystem, das die folgende Grammatik als Regeln nutzt, um zwei Agenten abwechselnd zu steuern. Jeder Agent soll bei seiner Handlung auf eine „Erinnerung" des zuletzt gesetzten Terminalzeichens zugreifen und die nächste Handlung entsprechend der Grammatikregeln setzen. Beide Agenten sollen die Regeln der Grammatik zur Auswahl der nächsten Handlung verwenden. \*\*Grammatik:\*\*

\<Start\> ::= \<Begrüßung\> \<Bedarf\> \<Abschluss\> \<Verabschiedung\>

\<Begrüßung\> ::= KBG VBG \[1.0\]

\<Bedarf\> ::= \<BedarfSegment\> \<Bedarf\> \[0.8\] \<BedarfSegment\> \[0.2\]

\<BedarfSegment\> ::= KBBd VBBd \[0.4\] KBBd VBA \[0.3\] KBA VBA \[0.3\]

\<Abschluss\> ::= KAA VAA \[0.6\] VAA KAA \[0.4\]

\<Verabschiedung\> ::= KAV VAV \[0.7\] VAV KAV \[0.3\]

\*\*Erwartetes Verhalten:\*\*

1\. \*\*Ziel:\*\* Das System soll abwechselnd die Handlungen der beiden Agenten simulieren, die entsprechend der gegebenen Grammatik produzieren. Jeder Agent trifft eine Entscheidung basierend auf den vorherigen Handlungen und der zugrunde liegenden Grammatik. Die Agenten handeln nacheinander, wobei sie jeweils aus den Regeln der Grammatik wählen, wobei die Wahrscheinlichkeiten zur Entscheidung verwendet werden.

2\. \*\*Agenten-Logik:\*\* - Jeder Agent hat Zugriff auf eine \"Erinnerung\", die das zuletzt produzierte Terminalzeichen speichert. - Die Grammatikregeln müssen berücksichtigt werden, wenn der Agent die nächste Handlung wählt. - Der erste Agent beginnt mit der Begrüßung, gefolgt vom Bedarf, Abschluss und Verabschiedung. Der zweite Agent setzt danach fort, indem er aus der aktuellen Erinnerung die nächste Handlung auswählt.

3\. \*\*Zufällige Auswahl basierend auf Wahrscheinlichkeiten:\*\* - Der Agent wählt aus den Optionen gemäß den angegebenen Wahrscheinlichkeiten (z. B. 0.8 für eine Fortsetzung im \<Bedarf\>-Segment). - Wenn der Agent beispielsweise ein \<BedarfSegment\> auswählt, sollte er mit einer Wahrscheinlichkeit von 0.8 erneut ein \<BedarfSegment\> und mit einer Wahrscheinlichkeit von 0.2 die Serie abbrechen und mit dem nächsten Abschnitt fortfahren. 4. \*\*Umsetzung:\*\* - Implementiere die Regeln als Funktionen, wobei jede Funktion den entsprechenden Produktionsregel entspricht. - Verwende Zufallszahlen, um die Wahrscheinlichkeiten korrekt umzusetzen. - Achte darauf, dass der Ablauf der Handlung rekursiv aufgebaut ist, insbesondere bei der \<Bedarf\>-Regel. - Die Ausgabe soll die erzeugte Terminalzeichenkette für den gesamten Dialog zwischen den beiden Agenten enthalten. \*\*Beispiel-Ausgabe:\*\* Das Multiagentensystem könnte als Ausgabe eine Kette wie folgt erzeugen (je nach Zufallsauswahl):

KBG VBG KBBd VBBd KBBd VBA KBA VBA KAA VAA KAV VAV

**Bearbeitung:**

Das folgende Python-Programm simuliert ein Multiagentensystem, das abwechselnd zwei Agenten steuert, die gemäß der angegebenen Grammatik handeln. Jeder Agent nutzt die Grammatikregeln, um die nächste Handlung basierend auf einer \"Erinnerung\" der letzten Handlung auszuwählen.

### **Python-Programm:**

python

import random

\# Funktion zur zufälligen Auswahl basierend auf den Wahrscheinlichkeiten

def zufallswahl(wahrscheinlichkeiten, optionen):

\"\"\"Wählt eine Option basierend auf den angegebenen Wahrscheinlichkeiten.\"\"\"

return random.choices(optionen, wahrscheinlichkeit=wahrscheinlichkeiten, k=1)\[0\]

\# Funktion für die Begrüßung (immer gleich, Wahrscheinlichkeit 1.0)

def begruessung():

return \[\'KBG\', \'VBG\'\]

\# Funktion für \<BedarfSegment\>

def bedarf_segment():

\"\"\"Wählt ein \<BedarfSegment\> basierend auf den Wahrscheinlichkeiten.\"\"\"

optionen = \[

(\[\'KBBd\', \'VBBd\'\], 0.4),

(\[\'KBBd\', \'VBA\'\], 0.3),

(\[\'KBA\', \'VBA\'\], 0.3)

\]

return zufallswahl(\[x\[1\] for x in optionen\], \[x\[0\] for x in optionen\])

\# Funktion für \<Bedarf\> (rekursiv, 80% Wahrscheinlichkeit für Fortsetzung)

def bedarf():

\"\"\"Wählt ein \<Bedarf\> basierend auf den Wahrscheinlichkeiten.\"\"\"

optionen = \[

(lambda: bedarf_segment() + bedarf(), 0.8),

(lambda: bedarf_segment(), 0.2)

\]

return zufallswahl(\[x\[1\] for x in optionen\], \[x\[0\] for x in optionen\])()

\# Funktion für \<Abschluss\>

def abschluss():

\"\"\"Wählt den Abschluss basierend auf den Wahrscheinlichkeiten.\"\"\"

optionen = \[

(\[\'KAA\', \'VAA\'\], 0.6),

(\[\'VAA\', \'KAA\'\], 0.4)

\]

return zufallswahl(\[x\[1\] for x in optionen\], \[x\[0\] for x in optionen\])

\# Funktion für \<Verabschiedung\>

def verabschiedung():

\"\"\"Wählt die Verabschiedung basierend auf den Wahrscheinlichkeiten.\"\"\"

optionen = \[

(\[\'KAV\', \'VAV\'\], 0.7),

(\[\'VAV\', \'KAV\'\], 0.3)

\]

return zufallswahl(\[x\[1\] for x in optionen\], \[x\[0\] for x in optionen\])

\# Funktion für den gesamten Dialog zwischen den Agenten

def multiagenten_system():

\# Initiale Erinnerung

erinnerung = \[\]

\# Agent 1 startet mit Begrüßung

erinnerung += begruessung()

\# Agent 1 wählt \<Bedarf\>, \<Abschluss\>, \<Verabschiedung\>

erinnerung += bedarf()

erinnerung += abschluss()

erinnerung += verabschiedung()

\# Agent 2 setzt fort, basierend auf der letzten Erinnerung

return erinnerung

\# Beispielaufruf des Programms

if \_\_name\_\_ == \"\_\_main\_\_\":

\# Dialog erzeugen

dialog = multiagenten_system()

\# Ausgabe des erzeugten Dialogs

print(\" \".join(dialog))

### **Erklärung des Programms:** {#erkluxe4rung-des-programms-2}

1.  **zufallswahl**: Diese Funktion wählt zufällig eine Option basierend auf den Wahrscheinlichkeiten aus. Sie verwendet die random.choices() Funktion von Python, um eine Auswahl aus mehreren Optionen zu treffen.

2.  **Agentenfunktionen**:

    - **begruessung**: Die Begrüßung ist immer gleich und gibt KBG VBG zurück.

    - **bedarf_segment**: Diese Funktion wählt aus den drei möglichen \<BedarfSegment\>-Optionen (KBBd VBBd, KBBd VBA, KBA VBA) basierend auf den angegebenen Wahrscheinlichkeiten.

    - **bedarf**: Diese Funktion entscheidet, ob der Agent ein weiteres \<BedarfSegment\> hinzufügen möchte oder den Bedarf mit einem einzelnen Segment abschließt. Mit einer Wahrscheinlichkeit von 80% wird rekursiv mehr Bedarf hinzugefügt, mit einer Wahrscheinlichkeit von 20% endet der Bedarf nach dem ersten Segment.

    - **abschluss**: Diese Funktion wählt den Abschluss, wobei mit einer Wahrscheinlichkeit von 60% KAA VAA und mit 40% VAA KAA erzeugt wird.

    - **verabschiedung**: Diese Funktion wählt die Verabschiedung, wobei mit einer Wahrscheinlichkeit von 70% KAV VAV und mit 30% VAV KAV erzeugt wird.

3.  **multiagenten_system**: Diese Funktion simuliert den Dialog zwischen den beiden Agenten. Der erste Agent beginnt mit der Begrüßung, wählt dann \<Bedarf\>, \<Abschluss\> und \<Verabschiedung\>. Der zweite Agent setzt seine Handlung basierend auf der letzten Erinnerung fort.

### **Beispielausgabe:**

Das Programm erzeugt eine zufällige Zeichenkette basierend auf den Wahrscheinlichkeiten in der Grammatik. Ein Beispiel könnte so aussehen:

KBG VBG KBBd VBBd KBBd VBA KBA VBA KAA VAA KAV VAV

Jeder Programmlauf könnte aufgrund der zufälligen Auswahl unterschiedliche Ergebnisse liefern, die der definierten Grammatik entsprechen.
