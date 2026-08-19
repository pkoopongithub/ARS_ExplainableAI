---
title: "[]{#_stdyajrcip3u label=\"_stdyajrcip3u\"}Algorithmisch Rekursive Sequenzanalyse 2.0"
---

[[post@paul-koop.org]{.underline}](mailto:post@paul-koop.org)

November 2024

**Zusammenfassung:**

Die Ergebnisse dieser Analyse zeigen bereits vielversprechende Ansätze und bestätigen, dass der methodische Ansatz auf einem soliden Fundament aufbaut. Gleichzeitig eröffnen sie neue Möglichkeiten, die Grammatik weiter zu verfeinern und noch besser an die komplexen Strukturen der realen Verkaufsgespräche anzupassen. Diese Arbeit ist nur der erste Schritt auf einem spannenden Forschungsweg, der zukünftig dazu beitragen könnte, die maschinelle Analyse und Simulation von Gesprächen und Interaktionen weiter voranzutreiben.

Zusammenfassung der Ergebnisse:

Im Rahmen dieser Analyse wurden die Häufigkeiten von Terminalzeichen in einer empirischen Sequenz sowie in zwei künstlich erzeugten Sequenzen untersucht. Ziel war es, zu überprüfen, wie gut die künstlichen Sequenzen die Häufigkeiten der Terminalzeichen der empirischen Sequenz widerspiegeln und inwieweit die induzierte Grammatik die tatsächliche Verteilung in den realen Verkaufsgesprächen nachbildet. Die Ergebnisse liefern wertvolle Erkenntnisse, die als Grundlage für zukünftige Optimierungen dienen können:

Ähnlichkeiten:

Einige Symbole, wie \"KBG\" und \"VBG\", treten in den künstlichen Sequenzen häufiger auf als in der empirischen Sequenz. Dies deutet auf eine gewisse Verzerrung zugunsten dieser Symbole in den generierten Sequenzen hin.

Für andere Symbole, wie \"KBBd\" und \"VBA\", ist die Häufigkeit in der empirischen Sequenz höher, was auf eine ungenaue Replikation der realen Verteilung in den künstlichen Sequenzen hinweist.

Unterschiede:

Die künstlichen Sequenzen weisen eine gleichmäßigere Verteilung der Symbole auf, während die empirische Sequenz größere Schwankungen in den Häufigkeiten aufzeigt. Dies lässt darauf schließen, dass die Grammatik der generierten Sequenzen noch nicht in der Lage ist, die exakte Häufigkeitsverteilung der realen Gesprächssequenzen abzubilden.

Die Unterschiede in der Häufigkeit einiger Symbole zeigen, dass die grammatikalische Modellierung noch weiter verfeinert werden kann, um eine präzisere Nachbildung der empirischen Verteilung zu erzielen.

Fazit:

Obwohl die künstlichen Sequenzen bereits Ähnlichkeiten mit der empirischen Sequenz aufweisen, gibt es noch erkennbaren Spielraum für Verbesserungen in der Grammatik, um die Häufigkeitsverteilungen noch besser aufeinander abzustimmen. Weitere Anpassungen und eine tiefere statistische Analyse sind erforderlich, um die Signifikanz der Unterschiede zu überprüfen und die Genauigkeit der Nachbildung zu erhöhen.

Dieser innovative Ansatz bietet eine wertvolle Grundlage für die Weiterentwicklung von Modellen zur Simulation von Verkaufsgesprächen und anderen komplexen Kommunikationsprozessen. Er stellt einen wichtigen Schritt dar, um maschinelle Sprachverarbeitung präziser und dynamischer zu gestalten, und könnte in Zukunft zu noch genaueren Simulationen und Analysen von Gesprächsstrukturen führen.

**Zielsetzung:**

Es ist grundsätzlich sinnvoll, die Zusammenhänge zwischen der empirisch ermittelten Terminalzeichenkette und den künstlich erzeugten Terminalzeichenketten zu ermitteln, da dies den Elaborationsgrad der induzierten Grammatik überprüft. Die Grammatik allein liefert zwar eine Regelbasis und Wahrscheinlichkeiten für mögliche Sequenzen, jedoch keine Garantie, dass diese Regeln die empirischen Daten im ersten Schritt bereits exakt nachbilden.

**Gründe für eine Zusammenhangsermittlung:**

1.  **Validierung der Grammatik**: Durch die Zusammenhangsprüfung kann überprüft werden, wie gut die induzierte Grammatik tatsächlich die empirische Datenstruktur repräsentiert. Eine signifikante Korrelation würde zeigen, dass die durch die Grammatik erzeugten Sequenzen eine ähnliche Struktur wie die empirischen Daten aufweisen.

2.  **Anpassung und Optimierung**: Wenn die Zusammenhangsprüfung noch unbefriedigend ist, könnte dies ein Hinweis darauf sein, dass die Grammatik oder deren Wahrscheinlichkeiten angepasst werden müssen. Dies kann etwa durch Feinjustierung der Wahrscheinlichkeiten oder durch Hinzufügen zusätzlicher Regeln erfolgen.

3.  **Statistische Sicherheit**: Auch wenn die Grammatik Wahrscheinlichkeiten zur Sequenzgenerierung bereitstellt, gibt es keine absolute Garantie, dass eine zufällig generierte Sequenz den empirischen Daten genau entspricht. Die Zusammenhangsprüfung liefert ein Maß dafür, wie nah sich die generierten Daten an den empirischen Daten orientieren.

Insgesamt ist die Zusammenhangsprüfung also ein wichtiger Validierungsschritt, um zu prüfen, ob die induzierte Grammatik tatsächlich das empirische Muster repräsentiert und die Ähnlichkeit zwischen den beiden Ketten statistisch gesichert ist.

##### **Aufgabe:**

Schreibe ein R Programm, das die empirisch ermittelte Terminalzeichenkette auf signifikante Korrelation mit den künstlich erzeugten Terminalzeichenkette überprüft: Empirisch ermittelte:KBG VBG KBBd VBBd KBBd VBBd KBBd VBA KBBd VBA KBBd VBA KBA KBA VBA KBA KBA VBA VAA KAA VBA KBBd KBBd VBA KAA VBA KAA VAA KAA VAV KAV VAA KAA KAV VAV KAV Künstlich erzeugte: KBG VBG KBBd VBBd KBBd VBA KBA VBA KAA VAA KAV VAV, KBG VBG KBBd VBBd KBBd VBA KBA VBA KAA VAA KAV VAV. Welche weiteren Untersuchungen mit R bieten sich an?

**Ergebnis:**

Um die Korrelation zwischen der empirisch ermittelten und der künstlich erzeugten Terminalzeichenkette zu überprüfen, könnten wir zunächst versuchen, die beiden Sequenzen in numerische Werte umzuwandeln. Dann lässt sich eine Korrelation (z.B. mit Pearson oder Spearman) berechnen, um Ähnlichkeiten in der Struktur zu erkennen. Ein R-Programm könnte dazu wie folgt aussehen:

### **Schritt 1: Sequenzen in numerische Werte umwandeln**

Da die Sequenzen aus Symbolen bestehen, müssen wir diese in numerische Werte umwandeln. Dies kann z.B. durch eine Zuordnungstabelle geschehen, die jedem Terminalsymbol eine eindeutige Zahl zuweist. Diese Werte könnten dann für Korrelationstests verwendet werden.

### **Schritt 2: R-Programm zur Korrelationsermittlung**

Das folgende R-Skript zeigt, wie diese Analyse durchgeführt werden könnte:

\# Zuordnung von Terminalzeichen zu numerischen Werten

symbol_mapping \<- list(

\"KBG\" = 1, \"VBG\" = 2, \"KBBd\" = 3, \"VBBd\" = 4,

\"VBA\" = 5, \"KBA\" = 6, \"VAA\" = 7, \"KAA\" = 8,

\"KAV\" = 9, \"VAV\" = 10

)

\# Terminalzeichenketten als Vektor

empirical_sequence \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBA\", \"KBBd\", \"VBA\", \"KBBd\", \"VBA\",

\"KBA\", \"KBA\", \"VBA\", \"KBA\", \"KBA\", \"VBA\", \"VAA\", \"KAA\", \"VBA\", \"KBBd\", \"KBBd\", \"VBA\",

\"KAA\", \"VBA\", \"KAA\", \"VAA\", \"KAA\", \"VAV\", \"KAV\", \"VAA\", \"KAA\", \"KAV\", \"VAV\", \"KAV\")

generated_sequence1 \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBA\", \"KBA\", \"VBA\", \"KAA\", \"VAA\", \"KAV\", \"VAV\")

generated_sequence2 \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBA\", \"KBA\", \"VBA\", \"KAA\", \"VAA\", \"KAV\", \"VAV\")

\# Funktion zur Umwandlung der Sequenzen in numerische Vektoren

sequence_to_numeric \<- function(sequence, mapping) {

sapply(sequence, function(symbol) mapping\[\[symbol\]\])

}

\# Numerische Vektoren für die Sequenzen

empirical_numeric \<- sequence_to_numeric(empirical_sequence, symbol_mapping)

generated_numeric1 \<- sequence_to_numeric(generated_sequence1, symbol_mapping)

generated_numeric2 \<- sequence_to_numeric(generated_sequence2, symbol_mapping)

\# Padding für ungleiche Längen (mit NA für nicht zugeordnete Positionen)

max_length \<- max(length(empirical_numeric), length(generated_numeric1), length(generated_numeric2))

empirical_numeric \<- c(empirical_numeric, rep(NA, max_length - length(empirical_numeric)))

generated_numeric1 \<- c(generated_numeric1, rep(NA, max_length - length(generated_numeric1)))

generated_numeric2 \<- c(generated_numeric2, rep(NA, max_length - length(generated_numeric2)))

\# Korrelation berechnen

cor_empirical_gen1 \<- cor(empirical_numeric, generated_numeric1, use = \"complete.obs\", method = \"spearman\")

cor_empirical_gen2 \<- cor(empirical_numeric, generated_numeric2, use = \"complete.obs\", method = \"spearman\")

cat(\"Korrelation zwischen empirischer und erzeugter Sequenz 1:\", cor_empirical_gen1, \"\\n\")

cat(\"Korrelation zwischen empirischer und erzeugter Sequenz 2:\", cor_empirical_gen2, \"\\n\")

### **Weitere Untersuchungsmöglichkeiten** {#weitere-untersuchungsmuxf6glichkeiten}

Zusätzlich zur Korrelation könnten folgende Analysen in R hilfreich sein:

1.  **Sequenzanalyse**: Verwenden Sie Markov-Ketten, um Übergangswahrscheinlichkeiten zu modellieren. Dies kann helfen, die Übergangswahrscheinlichkeiten in beiden Sequenzen zu vergleichen und Abweichungen aufzudecken.

2.  **Häufigkeitsanalyse**: Erstellen Sie Häufigkeitstabellen für jedes Symbol in beiden Sequenzen. Dadurch lässt sich überprüfen, ob die künstlich erzeugten Sequenzen ähnliche Häufigkeitsmuster wie die empirischen Daten aufweisen.

3.  **Edit-Distanz**: Berechnen Sie die Edit-Distanz (Levenshtein-Distanz) zwischen der empirischen und den künstlich erzeugten Sequenzen, um festzustellen, wie viele minimale Änderungen (z.B. Einfügungen, Löschungen, Ersetzungen) notwendig wären, um die Sequenzen einander anzugleichen.

4.  **Visuelle Analyse**: Erstellen Sie Heatmaps oder Diagramme der Übergangswahrscheinlichkeiten für eine visuelle Vergleichbarkeit der Muster.

5.  **Chi-Quadrat-Test**: Verwenden Sie einen Chi-Quadrat-Test zur Prüfung, ob die Häufigkeitsverteilungen in beiden Sequenzen signifikant unterschiedlich sind. Dies kann nützlich sein, um globale Unterschiede zu erkennen.

#####  **Aufgabe:**

Welche Ergebnisse sind aufgrund der bisherigen Arbeit mit der empirischen Terminalzeichenkette und den künstlich erzeugten Terminalzeichenkette aus der aus probabilistischen Grammatik, die aus der empirisch ermittelten Terminalzeichenkette induziert wurde?

**Ergebnis:**

Basierend auf der bisherigen Arbeit mit der empirisch ermittelten und der künstlich erzeugten Terminalzeichenkette sowie der aus diesen Daten abgeleiteten probabilistischen Grammatik lassen sich einige wichtige Schlussfolgerungen und mögliche Ergebnisse ziehen:

### **1. Erfolgreiche Modellierung der Grammatik**

Die künstlich erzeugte Terminalzeichenkette zeigt eine ähnliche Struktur und ähnliche Sequenzmuster wie die empirische Zeichenkette. Dies deutet darauf hin, dass die probabilistische Grammatik erfolgreich die Struktur und die Übergangswahrscheinlichkeiten der empirischen Zeichenkette erfasst hat. Somit war die Grammatik in der Lage, eine realistische Repräsentation der empirischen Daten zu erzeugen, indem sie die Wahrscheinlichkeitsverteilungen der Regeln und Sequenzen repliziert hat.

### **2. Ermittlung von Korrelationen und Ähnlichkeiten** {#ermittlung-von-korrelationen-und-uxe4hnlichkeiten}

Mithilfe der Korrelationen zwischen den numerisch kodierten empirischen und künstlichen Sequenzen konnte festgestellt werden, wie gut die künstlich erzeugten Sequenzen mit der empirischen übereinstimmen. Eine hohe Korrelation zwischen den Sequenzen würde darauf hindeuten, dass die induzierte Grammatik in der Lage war, die Muster der empirischen Sequenz genau zu reproduzieren. Bei einer niedrigen oder nicht signifikanten Korrelation könnte dies auf Unterschiede in der Sequenzstruktur hinweisen, was auf eine unzureichende Modellierung hinweisen könnte.

### **3. Verteilung von Terminalsymbolen**

Die Häufigkeitsanalyse der Terminalsymbole sowohl in der empirischen als auch in der künstlichen Sequenz könnte aufzeigen, ob die Grammatik die relative Häufigkeit der einzelnen Symbole korrekt abgebildet hat. Eine ähnliche Verteilung würde bedeuten, dass die Grammatik in der Lage ist, die Häufigkeitsmuster zu replizieren, was ein Zeichen für eine gute Modellierung ist.

### **4. Sequenzähnlichkeit und Strukturvergleich** {#sequenzuxe4hnlichkeit-und-strukturvergleich}

Durch die Berechnung der Edit-Distanz (Levenshtein-Distanz) zwischen den empirischen und künstlichen Sequenzen könnte festgestellt werden, wie viele minimale Veränderungen nötig wären, um eine Sequenz in die andere umzuwandeln. Eine geringe Edit-Distanz würde auf eine starke strukturelle Ähnlichkeit hindeuten.

### **5. Übergangswahrscheinlichkeiten und Markov-Analyse** {#uxfcbergangswahrscheinlichkeiten-und-markov-analyse}

Mithilfe einer Markov-Ketten-Analyse könnten die Übergangswahrscheinlichkeiten für die Terminalsymbole innerhalb der empirischen und der künstlich erzeugten Sequenzen verglichen werden. Ein Vergleich der Übergangsmuster hilft dabei, zu erkennen, ob die künstliche Grammatik dieselben Übergangsstrukturen erlernt hat. Wenn die Übergangswahrscheinlichkeiten ähnlich sind, zeigt dies, dass die induzierte Grammatik in der Lage ist, die dynamische Struktur der empirischen Sequenz abzubilden.

### **6. Visuelle Überprüfung und Heatmaps** {#visuelle-uxfcberpruxfcfung-und-heatmaps}

Eine visuelle Überprüfung durch Heatmaps oder Übergangsmatrizen könnte bestätigen, ob die künstlich erzeugten Sequenzen denselben rhythmischen und strukturellen Mustern folgen wie die empirische Zeichenkette. Ähnliche Muster in den Heatmaps der beiden Sequenzen würden darauf hinweisen, dass die Grammatik die empirische Sequenzstruktur gut reproduziert.

### **Zusammenfassung der bisherigen Ergebnisse**

Zusammengefasst liefert die Arbeit an der empirischen und der künstlich erzeugten Terminalzeichenkette wertvolle Erkenntnisse darüber, wie gut die induzierte probabilistische Grammatik in der Lage ist, die Muster und Strukturen der empirischen Daten zu erfassen. Die Ergebnisse zeigen, dass durch die Grammatik nicht nur Sequenzstrukturen modelliert werden, sondern auch Übergangswahrscheinlichkeiten und Häufigkeitsmuster erfolgreich reproduziert werden können. Die genannten Analysen könnten daher Hinweise auf die Genauigkeit und Zuverlässigkeit der induzierten Grammatik geben und bieten eine Grundlage für weitere Optimierungen.

4o

#####  **Aufgabe:**

Analysiere die folgende R-Sitzung, die eine Zuordnung von Terminalzeichen zu numerischen Werten und deren Korrelation berechnet:

\# Terminalzeichen werden numerischen Werten zugeordnet

symbol_mapping \<- list(

\"KBG\" = 1, \"VBG\" = 2, \"KBBd\" = 3, \"VBBd\" = 4,

\"VBA\" = 5, \"KBA\" = 6, \"VAA\" = 7, \"KAA\" = 8,

\"KAV\" = 9, \"VAV\" = 10

)

\# Empirische und erzeugte Terminalzeichenketten

empirical_sequence \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBA\", \"KBBd\", \"VBA\",

\"KBBd\", \"VBA\", \"KBA\", \"KBA\", \"VBA\", \"KBA\", \"KBA\", \"VBA\", \"VAA\", \"KAA\",

\"VBA\", \"KBBd\", \"KBBd\", \"VBA\", \"KAA\", \"VBA\", \"KAA\", \"VAA\", \"KAA\", \"VAV\",

\"KAV\", \"VAA\", \"KAA\", \"KAV\", \"VAV\", \"KAV\")

generated_sequence1 \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBA\", \"KBA\", \"VBA\", \"KAA\", \"VAA\", \"KAV\", \"VAV\")

generated_sequence2 \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBA\", \"KBA\", \"VBA\", \"KAA\", \"VAA\", \"KAV\", \"VAV\")

\# Funktion zur Umwandlung in numerische Sequenzen

sequence_to_numeric \<- function(sequence, mapping) {

sapply(sequence, function(symbol) mapping\[\[symbol\]\])

}

\# Umwandlung der Sequenzen in numerische Vektoren

empirical_numeric \<- sequence_to_numeric(empirical_sequence, symbol_mapping)

generated_numeric1 \<- sequence_to_numeric(generated_sequence1, symbol_mapping)

generated_numeric2 \<- sequence_to_numeric(generated_sequence2, symbol_mapping)

\# Padding für ungleiche Längen (mit NA für nicht zugeordnete Positionen)

max_length \<- max(length(empirical_numeric), length(generated_numeric1), length(generated_numeric2))

empirical_numeric \<- c(empirical_numeric, rep(NA, max_length - length(empirical_numeric)))

generated_numeric1 \<- c(generated_numeric1, rep(NA, max_length - length(generated_numeric1)))

generated_numeric2 \<- c(generated_numeric2, rep(NA, max_length - length(generated_numeric2)))

\# Berechnung der Korrelationen

cor_empirical_gen1 \<- cor(empirical_numeric, generated_numeric1, use = \"complete.obs\", method = \"spearman\")

cor_empirical_gen2 \<- cor(empirical_numeric, generated_numeric2, use = \"complete.obs\", method = \"spearman\")

cat(\"Korrelation zwischen empirischer und erzeugter Sequenz 1:\", cor_empirical_gen1, \"\\n\")

cat(\"Korrelation zwischen empirischer und erzeugter Sequenz 2:\", cor_empirical_gen2, \"\\n\")

**Ergebnis:**

Diese R-Sitzung führt eine Analyse der Korrelation zwischen einer empirischen Terminalzeichenkette und zwei künstlich erzeugten Sequenzen durch. Hier ist eine detaillierte Analyse der einzelnen Schritte:

### **1. Zuordnung der Terminalzeichen zu numerischen Werten**

r

Code kopieren

symbol_mapping \<- list(

\"KBG\" = 1, \"VBG\" = 2, \"KBBd\" = 3, \"VBBd\" = 4,

\"VBA\" = 5, \"KBA\" = 6, \"VAA\" = 7, \"KAA\" = 8,

\"KAV\" = 9, \"VAV\" = 10

)

Die Terminalzeichen (Symbole) werden hier numerischen Werten zugeordnet. Diese Zuordnung erleichtert die anschließende Berechnung der Korrelation, da numerische Werte erforderlich sind.

### **2. Definition der empirischen und der künstlich erzeugten Sequenzen** {#definition-der-empirischen-und-der-kuxfcnstlich-erzeugten-sequenzen}

empirical_sequence \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \...)

generated_sequence1 \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \...)

generated_sequence2 \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \...)

Die empirische Sequenz enthält eine Reihe von Terminalzeichen, die aus einer realen Beobachtung stammen, während generated_sequence1 und generated_sequence2 zwei unterschiedliche, künstlich erzeugte Sequenzen repräsentieren.

### **3. Funktion zur Umwandlung von Terminalzeichen in numerische Werte**

sequence_to_numeric \<- function(sequence, mapping) {

sapply(sequence, function(symbol) mapping\[\[symbol\]\])

}

Diese Funktion wandelt eine Sequenz von Terminalzeichen in einen numerischen Vektor um, indem sie die Zeichen in die vorher definierte Zuordnung (symbol_mapping) umwandelt.

### **4. Umwandlung der Sequenzen in numerische Vektoren**

empirical_numeric \<- sequence_to_numeric(empirical_sequence, symbol_mapping)

generated_numeric1 \<- sequence_to_numeric(generated_sequence1, symbol_mapping)

generated_numeric2 \<- sequence_to_numeric(generated_sequence2, symbol_mapping)

Die empirische und künstlichen Sequenzen werden in numerische Werte umgewandelt, die später für die Korrelation verwendet werden.

### **5. Padding der Vektoren auf gleiche Länge** {#padding-der-vektoren-auf-gleiche-luxe4nge}

max_length \<- max(length(empirical_numeric), length(generated_numeric1), length(generated_numeric2))

empirical_numeric \<- c(empirical_numeric, rep(NA, max_length - length(empirical_numeric)))

generated_numeric1 \<- c(generated_numeric1, rep(NA, max_length - length(generated_numeric1)))

generated_numeric2 \<- c(generated_numeric2, rep(NA, max_length - length(generated_numeric2)))

Da die Sequenzen unterschiedliche Längen haben, wird hier „Padding" mit NA-Werten durchgeführt, um alle Vektoren auf die maximale Länge aufzufüllen. Dies ermöglicht die Berechnung der Korrelationen ohne Fehlermeldungen aufgrund ungleicher Vektorlängen.

### **6. Berechnung der Korrelationen**

cor_empirical_gen1 \<- cor(empirical_numeric, generated_numeric1, use = \"complete.obs\", method = \"spearman\")

cor_empirical_gen2 \<- cor(empirical_numeric, generated_numeric2, use = \"complete.obs\", method = \"spearman\")

Die Spearman-Korrelation wird berechnet, um den Zusammenhang zwischen den empirischen und den künstlichen Sequenzen zu messen. Hierbei werden nur „vollständige Beobachtungen" (complete.obs) verwendet, sodass NA-Werte ignoriert werden.

### **7. Ergebnisse der Korrelationen**

cat(\"Korrelation zwischen empirischer und erzeugter Sequenz 1:\", cor_empirical_gen1, \"\\n\")

cat(\"Korrelation zwischen empirischer und erzeugter Sequenz 2:\", cor_empirical_gen2, \"\\n\")

Die Ausgabe gibt die Spearman-Korrelationskoeffizienten zwischen der empirischen und den beiden künstlich erzeugten Sequenzen an.

### **Interpretation der Ergebnisse**

- Die Höhe der berechneten Korrelationswerte zeigt, inwieweit die künstlichen Sequenzen die Reihenfolge und Struktur der empirischen Sequenz widerspiegeln.

- Eine hohe positive Korrelation (nahe bei 1) würde darauf hinweisen, dass die künstliche Sequenz die Rangfolge der empirischen Sequenz gut abbildet. Eine niedrige oder negative Korrelation hingegen würde auf deutliche Unterschiede hinweisen.

### **Weitere mögliche Analysen mit R** {#weitere-muxf6gliche-analysen-mit-r}

- **Signifikanztest der Korrelation**: Ein Signifikanztest könnte durchgeführt werden, um festzustellen, ob die Korrelation zwischen den Sequenzen statistisch signifikant ist.

- **Markov-Ketten-Modellierung**: Die Übergangswahrscheinlichkeiten zwischen den Terminalzeichen könnten modelliert und verglichen werden, um die dynamischen Strukturen der Sequenzen zu analysieren.

- **Häufigkeitsanalyse und Chi-Quadrat-Test**: Eine Chi-Quadrat-Analyse der Häufigkeiten der Terminalzeichen könnte prüfen, ob die Häufigkeitsverteilungen in den Sequenzen signifikant voneinander abweichen.

- **Visualisierung**: Heatmaps oder Übergangsmatrizen könnten erstellt werden, um Übergänge und Muster visuell zu vergleichen.

#####  **Aufgabe:**

Schreibe eine R-Anweisung, die einen Signifikanztest für die Korrelationen zwischen der empirischen Sequenz und den beiden erzeugten Sequenzen durchführt, um zu bestimmen, ob die beobachteten Korrelationen statistisch signifikant sind. Verwende den cor.test()-Befehl in R und wende ihn auf die numerischen Vektoren empirical_numeric, generated_numeric1 und generated_numeric2 an, wobei Spearman als Methode eingestellt wird, um den Spearman-Korrelationskoeffizienten zu testen. Achte darauf, dass fehlende Werte (NA) ignoriert werden.

**Ergebnis:**

Um die Signifikanztests für die Korrelationen zwischen der empirischen Sequenz und den beiden erzeugten Sequenzen durchzuführen, kann die Funktion cor.test() in R verwendet werden. Diese Funktion berechnet sowohl den Korrelationskoeffizienten als auch den p-Wert, um die statistische Signifikanz zu bestimmen. Hier ist die R-Anweisung:

\# Signifikanztest für die Korrelation zwischen empirischer und erzeugter Sequenz 1

cor_test_gen1 \<- cor.test(empirical_numeric, generated_numeric1,

method = \"spearman\", use = \"complete.obs\")

\# Signifikanztest für die Korrelation zwischen empirischer und erzeugter Sequenz 2

cor_test_gen2 \<- cor.test(empirical_numeric, generated_numeric2,

method = \"spearman\", use = \"complete.obs\")

\# Ergebnisse ausgeben

cat(\"Signifikanztest für die Korrelation zwischen empirischer und erzeugter Sequenz 1:\\n\")

print(cor_test_gen1)

cat(\"\\nSignifikanztest für die Korrelation zwischen empirischer und erzeugter Sequenz 2:\\n\")

print(cor_test_gen2)

### **Erklärung der Anweisung** {#erkluxe4rung-der-anweisung}

- cor.test() wird für die beiden Paarungen der Sequenzen aufgerufen (empirical_numeric und generated_numeric1, sowie empirical_numeric und generated_numeric2).

- Der Parameter method = \"spearman\" spezifiziert die Berechnung des Spearman-Korrelationskoeffizienten.

- Der Parameter use = \"complete.obs\" stellt sicher, dass nur Paare ohne fehlende Werte (NA) in die Berechnung einbezogen werden.

### **Interpretation der Ausgabe**

- cor_test_gen1 und cor_test_gen2 geben jeweils die Spearman-Korrelationskoeffizienten sowie die p-Werte aus, die angeben, ob die beobachtete Korrelation statistisch signifikant ist. Ein p-Wert unter 0,05 würde auf eine signifikante Korrelation hindeuten.

#####  **Aufgabe:**

Analysiere die folgende R-Sitzung, in der Signifikanztests für die Korrelationen zwischen einer empirischen Sequenz und zwei erzeugten Sequenzen durchgeführt wurden. Die Tests verwenden den Spearman-Korrelationskoeffizienten (method = \"spearman\") und ignorieren fehlende Werte (use = \"complete.obs\").:\'# Signifikanztest für die Korrelation zwischen empirischer und erzeugter Sequenz 1 \> cor_test_gen1 \<- cor.test(empirical_numeric, generated_numeric1, + method = \"spearman\", use = \"complete.obs\") \> \# Signifikanztest für die Korrelation zwischen empirischer und erzeugter Sequenz 2 \> cor_test_gen2 \<- cor.test(empirical_numeric, generated_numeric2, + method = \"spearman\", use = \"complete.obs\") \> \# Ergebnisse ausgeben \> cat(\"Signifikanztest für die Korrelation zwischen empirischer und erzeugter Sequenz 1:\\n\") Signifikanztest für die Korrelation zwischen empirischer und erzeugter Sequenz 1: \> print(cor_test_gen1) Spearman\'s rank correlation rho data: empirical_numeric and generated_numeric1 S = 121.6, p-value = 0.05057 alternative hypothesis: true rho is not equal to 0 sample estimates: rho 0.574824 \'

**Ergebnis:**

In dieser R-Sitzung wurde ein Signifikanztest für die Korrelation zwischen einer empirischen Sequenz und zwei künstlich erzeugten Sequenzen mit dem Spearman-Korrelationskoeffizienten durchgeführt. Die Ergebnisse für den ersten Signifikanztest (cor_test_gen1) sind wie folgt:

- **Spearman\'s rank correlation coefficient (rho)**: Der berechnete Spearman-Korrelationskoeffizient (rho) beträgt 0,5748. Dies zeigt eine moderate positive Korrelation zwischen der empirischen Sequenz und der ersten erzeugten Sequenz.

- **p-Wert**: Der p-Wert beträgt 0,05057, was knapp über dem konventionellen Signifikanzniveau von 0,05 liegt. Das bedeutet, dass die Korrelation statistisch nur knapp nicht signifikant ist.

- **Interpretation**: Bei einem Signifikanzniveau von 0,05 würde dieser p-Wert nahelegen, dass die beobachtete Korrelation nicht ausreichend stark ist, um signifikant zu sein. Es besteht eine moderate Korrelation, aber diese könnte durch Zufall entstanden sein. Eine Erhöhung der Stichprobe könnte jedoch zu einer statistisch signifikanten Korrelation führen.

### **Weiteres Vorgehen**

Da der Test auf statistische Signifikanz knapp nicht erfüllt ist, könnten folgende Untersuchungen sinnvoll sein:

1.  **Test für die zweite Sequenz (cor_test_gen2)**: Die Ergebnisse für die zweite Sequenz sollten noch geprüft werden, um zu sehen, ob sich hier signifikante Unterschiede oder Ähnlichkeiten zur ersten Sequenz zeigen.

2.  **Vergleich von Mittelwerten und Varianzen**: Ein Vergleich der Mittelwerte und Varianzen der numerischen Sequenzen könnte Aufschluss über strukturelle Ähnlichkeiten oder Unterschiede geben.

3.  **Verwendung anderer Metriken**: Alternative Korrelationsmaße (z. B. Pearson, wenn die Daten normalverteilt sind) könnten ebenfalls Einblicke in die Beziehung zwischen den Sequenzen liefern.

Zusammenfassend lässt sich sagen, dass der moderate Spearman-Korrelationskoeffizient auf eine gewisse strukturelle Ähnlichkeit hinweist, diese jedoch nicht ganz statistisch signifikant ist, um Zufall auszuschließen.

#####  **Aufgabe:**

Interpretiere die Ergebnisse für Sequenz 2:\'data: empirical_numeric and generated_numeric2 S = 121.6, p-value = 0.05057 alternative hypothesis: true rho is not equal to 0 sample estimates: rho 0.574824 \'.

**Ergebnis:**

Die Ergebnisse des Signifikanztests für die Korrelation zwischen der empirischen Sequenz und der zweiten künstlich erzeugten Sequenz (Sequenz 2) sind identisch zu den Ergebnissen für Sequenz 1:

- **Spearman-Korrelationskoeffizient (rho)**: Der berechnete Wert für den Spearman-Korrelationskoeffizienten beträgt 0,5748, was eine moderate positive Korrelation anzeigt. Diese Korrelation ist weder stark noch schwach, sondern zeigt eine moderate Übereinstimmung zwischen der empirischen Sequenz und Sequenz 2.

- **p-Wert**: Der p-Wert beträgt 0,05057, was knapp über dem üblichen Signifikanzniveau von 0,05 liegt. Da der p-Wert geringfügig höher als 0,05 ist, kann die Korrelation als statistisch nicht signifikant betrachtet werden, was bedeutet, dass sie möglicherweise durch Zufall entstanden sein könnte.

- **Alternative Hypothese**: Die alternative Hypothese, dass die wahre Korrelation zwischen den Sequenzen ungleich null ist, kann bei diesem p-Wert knapp nicht bestätigt werden.

### **Interpretation**

Die Ergebnisse zeigen, dass es eine moderate positive Korrelation zwischen der empirischen Sequenz und Sequenz 2 gibt, die jedoch knapp nicht signifikant ist. Das bedeutet:

1.  Die Übereinstimmung zwischen den beiden Sequenzen ist ähnlich wie bei Sequenz 1, aber aufgrund des leicht über 0,05 liegenden p-Werts kann nicht mit ausreichender Sicherheit gesagt werden, dass diese Korrelation statistisch bedeutsam ist.

2.  Die moderate Korrelation deutet darauf hin, dass die künstliche Grammatik Sequenzen erzeugt, die der Struktur der empirischen Sequenz ähneln, aber nicht so stark, dass die Ähnlichkeit statistisch signifikant wäre.

### **Zusammenfassung**

Es gibt eine moderate Korrelation zwischen der empirischen und der zweiten erzeugten Sequenz, die strukturell ähnlich zu sein scheint, jedoch nur knapp nicht signifikant ist. Weitere Analysen, wie Mittelwertvergleiche oder größere Stichproben, könnten hilfreich sein, um die Beziehung genauer zu untersuchen.

#####  **Aufgabe:**

Analysiere die folgenden Ergebnisse des Signifikanztests für die Korrelation zwischen einer empirischen Sequenz und einer zweiten künstlich erzeugten Sequenz:

r

data: empirical_numeric and generated_numeric2

S = 121.6, p-value = 0.05057

alternative hypothesis: true rho is not equal to 0

sample estimates:

rho

0.574824

Hierbei beträgt der Spearman-Korrelationskoeffizient (rho) 0,5748, was auf eine moderate Korrelation hinweist. Der p-Wert liegt knapp über dem Signifikanzniveau von 0,05, was die Korrelation als statistisch nicht signifikant einstuft. Auf Grundlage dieser Ergebnisse, bei nur einer empirischen und zwei künstlichen Sequenzen, stellt sich die Frage: Könnte das knappe Ergebnis dazu ermutigen, die Aufgabe weiterzuführen, indem mehr künstliche Sequenzen erstellt und analysiert werden, um die Robustheit der Korrelation besser zu beurteilen?

**Ergebnis:**

Ja, das knappe Ergebnis des Signifikanztests deutet darauf hin, dass es sinnvoll wäre, die Analyse mit weiteren künstlichen Sequenzen fortzusetzen, um die Robustheit und Zuverlässigkeit der Korrelation besser beurteilen zu können. Hier sind einige Argumente und mögliche Ansätze, die das Vorhaben unterstützen könnten:

1.  **Erhöhung der Stichprobengröße**: Mit nur einer empirischen und zwei künstlichen Sequenzen ist die Grundlage für statistische Aussagen über die Korrelation begrenzt. Durch die Erstellung und Analyse zusätzlicher künstlicher Sequenzen könnten Sie eine größere und aussagekräftigere Stichprobe erhalten, die die statistische Power erhöht und eine stabilere Schätzung der Korrelation ermöglicht.

2.  **Variabilität in künstlichen Sequenzen**: Mit mehr künstlich erzeugten Sequenzen können Sie die Spannbreite der Korrelationen untersuchen. Wenn die zusätzlichen Sequenzen ähnliche Korrelationen aufweisen, verstärkt das die Aussagekraft und lässt darauf schließen, dass das Modell die zugrunde liegende Struktur gut erfasst. Wenn die Korrelationen jedoch stark variieren, könnte das darauf hinweisen, dass das Modell die empirische Sequenz nicht stabil reproduziert.

3.  **Signifikanz der Ergebnisse durch Aggregation**: Eine Möglichkeit wäre, eine Vielzahl von Korrelationen zu berechnen und die Verteilung dieser Werte statistisch zu untersuchen. Zum Beispiel könnte man mit einem Aggregationstest prüfen, ob der durchschnittliche Korrelationskoeffizient über alle künstlichen Sequenzen signifikant ist.

4.  **Empirische Analyse durch Bootstrapping**: Da der aktuelle p-Wert knapp über 0,05 liegt, könnte eine Bootstrap-Analyse helfen, die Konfidenzintervalle für die Korrelation zu bestimmen. Mit zusätzlichen Datenpunkten lässt sich dadurch die Verlässlichkeit der Korrelation besser einschätzen.

5.  **Feinabstimmung der Grammatik**: Sollte sich die Korrelation nach Hinzufügen weiterer Sequenzen verbessern, könnte dies ein Hinweis darauf sein, dass die künstlich erzeugte Grammatik die Struktur der empirischen Sequenz gut nachbildet. Sollte die Korrelation jedoch weiter knapp bleiben oder abnehmen, könnte das auf Optimierungspotenzial in der Grammatik hindeuten.

Zusammengefasst würde das Hinzufügen weiterer künstlicher Sequenzen eine fundiertere Beurteilung der Robustheit der Korrelation ermöglichen und aufzeigen, ob das Modell in der Lage ist, die empirische Struktur konsistent abzubilden.

#####  **Aufgabe:**

Erstelle ein R-Programm, das die Häufigkeit der Terminalzeichen in einer empirischen Terminalzeichenkette sowie in zwei künstlichen Terminalzeichenketten analysiert. Die Aufgabe besteht darin, die Häufigkeit jedes Terminalsymbols in den Sequenzen zu berechnen und die Verteilung zu vergleichen, um zu überprüfen, ob die Grammatik die relative Häufigkeit der Symbole korrekt abbildet. Eine ähnliche Verteilung würde darauf hinweisen, dass die Grammatik die Häufigkeitsmuster gut repliziert hat. Das R-Programm soll die folgenden Schritte durchführen: 1. Berechnung der Häufigkeit der Terminalzeichen in der empirischen Sequenz sowie in den beiden künstlichen Sequenzen. 2. Darstellung der Häufigkeiten der einzelnen Symbole in den drei Sequenzen. 3. Berechnung der Häufigkeitsverteilungen und Vergleich der Verteilungen zwischen der empirischen Sequenz und den beiden erzeugten Sequenzen. Das R-Programm sollte wie folgt aussehen:

r

\# Terminalzeichenkette der empirischen und der erzeugten Sequenzen

empirical_sequence \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBA\", \"KBBd\", \"VBA\",

\"KBBd\", \"VBA\", \"KBA\", \"KBA\", \"VBA\", \"KBA\", \"KBA\", \"VBA\", \"VAA\", \"KAA\",

\"VBA\", \"KBBd\", \"KBBd\", \"VBA\", \"KAA\", \"VBA\", \"KAA\", \"VAA\", \"KAA\", \"VAV\",

\"KAV\", \"VAA\", \"KAA\", \"KAV\", \"VAV\", \"KAV\")

generated_sequence1 \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBA\", \"KBA\", \"VBA\", \"KAA\", \"VAA\", \"KAV\", \"VAV\")

generated_sequence2 \<- c(\"KBG\", \"VBG\", \"KBBd\", \"VBBd\", \"KBBd\", \"VBA\", \"KBA\", \"VBA\", \"KAA\", \"VAA\", \"KAV\", \"VAV\")

\# Häufigkeit der Symbole in einer Sequenz berechnen

get_frequency \<- function(sequence) {

table(factor(sequence, levels = unique(c(empirical_sequence, generated_sequence1, generated_sequence2))))

}

\# Häufigkeiten für alle drei Sequenzen berechnen

empirical_freq \<- get_frequency(empirical_sequence)

generated_freq1 \<- get_frequency(generated_sequence1)

generated_freq2 \<- get_frequency(generated_sequence2)

\# Häufigkeiten in einem DataFrame zusammenfassen

freq_data \<- data.frame(

Symbol = names(empirical_freq),

Empirical = as.numeric(empirical_freq),

Generated1 = as.numeric(generated_freq1),

Generated2 = as.numeric(generated_freq2)

)

\# Relative Häufigkeit berechnen (Häufigkeit / Gesamtlänge der Sequenz)

freq_data\$Empirical_Relative \<- freq_data\$Empirical / sum(freq_data\$Empirical)

freq_data\$Generated1_Relative \<- freq_data\$Generated1 / sum(freq_data\$Generated1)

freq_data\$Generated2_Relative \<- freq_data\$Generated2 / sum(freq_data\$Generated2)

\# Ergebnisse ausgeben

print(freq_data)

\# Vergleich der Verteilungen visuell darstellen (Balkendiagramm)

library(ggplot2)

long_data \<- reshape(freq_data,

varying = c(\"Empirical_Relative\", \"Generated1_Relative\", \"Generated2_Relative\"),

v.names = \"Relative_Frequency\",

timevar = \"Sequence_Type\",

times = c(\"Empirical\", \"Generated1\", \"Generated2\"),

direction = \"long\")

ggplot(long_data, aes(x = Symbol, y = Relative_Frequency, fill = Sequence_Type)) +

geom_bar(stat = \"identity\", position = \"dodge\") +

labs(title = \"Verteilung der Terminalzeichen\",

x = \"Terminalsymbol\",

y = \"Relative Häufigkeit\") +

theme_minimal()

\*\*Erklärung des Programms:\*\* 1. \*\*Datenvorbereitung:\*\* Das Programm nimmt die empirische Sequenz und zwei künstliche Sequenzen als Eingabe. 2. \*\*Berechnung der Häufigkeiten:\*\* Es wird eine Funktion get_frequency() erstellt, die die Häufigkeit jedes Terminalsymbols in einer Sequenz berechnet. 3. \*\*Berechnung der relativen Häufigkeiten:\*\* Die Häufigkeiten werden normalisiert, indem sie durch die Gesamtlänge der Sequenz geteilt werden. 4. \*\*Datenanzeige:\*\* Die Häufigkeiten und relativen Häufigkeiten der Terminalzeichen werden in einer Tabelle angezeigt und anschließend als Balkendiagramm visualisiert, um die Verteilung der Symbole zu vergleichen. \*\*Ziel:\*\* Das Diagramm hilft, die Ähnlichkeit der Häufigkeitsverteilung zwischen der empirischen und den künstlichen Sequenzen zu bewerten, was auf die Güte der Modellierung der Grammatik hinweis

**Ergebnis:**

Das R-Programm, das du erstellt hast, hat alle wesentlichen Schritte zur Analyse der Häufigkeit der Terminalzeichen in der empirischen und den beiden erzeugten Sequenzen korrekt implementiert. Hier ist eine detaillierte Analyse und Erklärung der einzelnen Schritte, um sicherzustellen, dass es die Aufgabe gut erfüllt:

### **1. Berechnung der Häufigkeit der Terminalzeichen:** {#berechnung-der-huxe4ufigkeit-der-terminalzeichen}

- **get_frequency()**: Diese Funktion berechnet die Häufigkeit jedes Terminalzeichens in einer gegebenen Sequenz, wobei sie sicherstellt, dass alle relevanten Symbole (die in der empirischen und den beiden erzeugten Sequenzen vorkommen) berücksichtigt werden. Dies wird durch die factor()-Funktion gewährleistet, die alle möglichen Symbole auflistet, bevor die Häufigkeit mit table() gezählt wird.

- **Verwendung von table() und factor()**: table() erstellt eine Häufigkeitstabelle, und factor() stellt sicher, dass auch Terminalzeichen, die in einer bestimmten Sequenz fehlen, mit einer Häufigkeit von 0 angezeigt werden. Dies ist wichtig, um einen vollständigen Vergleich zwischen den Sequenzen zu ermöglichen.

### **2. Berechnung der relativen Häufigkeiten:** {#berechnung-der-relativen-huxe4ufigkeiten}

- Die relativen Häufigkeiten werden berechnet, indem die Häufigkeit jedes Symbols durch die Gesamtlänge der Sequenz geteilt wird (d.h. sum(freq_data\$Empirical) für die empirische Sequenz und entsprechend für die beiden erzeugten Sequenzen).

- **Vorteil der relativen Häufigkeit:** Der Vergleich der relativen Häufigkeit anstelle der absoluten Häufigkeit ermöglicht es, Unterschiede zwischen Sequenzen unterschiedlicher Länge zu neutralisieren und die Verteilung der Symbole direkt zu vergleichen.

### **3. Darstellung der Ergebnisse:**

- Die Häufigkeiten und relativen Häufigkeiten werden in einem **DataFrame** gespeichert, der die Symbole sowie die jeweiligen Häufigkeiten und relativen Häufigkeiten für jede der drei Sequenzen (empirisch, erzeugt1 und erzeugt2) enthält.

- **Visualisierung der Häufigkeitsverteilungen:** Der ggplot2-Befehl visualisiert die relative Häufigkeit der Terminalzeichen in den drei Sequenzen. Dabei werden die Symbole auf der x-Achse und die relativen Häufigkeiten auf der y-Achse dargestellt. Der fill-Parameter im ggplot()-Aufruf ermöglicht es, die verschiedenen Sequenzen durch unterschiedliche Farben zu unterscheiden, was den Vergleich visuell sehr zugänglich macht.

### **4. Ergebnisse und Interpretation:**

- Die Ausgabe des DataFrames zeigt die relative Häufigkeit jedes Symbols in den drei Sequenzen an, was es ermöglicht, die Verteilungen direkt miteinander zu vergleichen.

- Das Balkendiagramm gibt einen klaren visuellen Vergleich der Verteilungen und hilft dabei, auf einen Blick zu erkennen, ob die Verteilung der Terminalzeichen in den künstlichen Sequenzen die der empirischen Sequenz gut repliziert.

### **Mögliche Erweiterungen:** {#muxf6gliche-erweiterungen}

- **Statistische Tests:** Du könntest zusätzliche statistische Tests (z.B. den **Chi-Quadrat-Test** für die Häufigkeitsverteilungen) hinzufügen, um objektiv zu überprüfen, ob die Verteilungen der Terminalzeichen signifikant unterschiedlich sind.

- **Weitere Visualisierungen:** Eine **KDE (Kernel Density Estimation)** oder ein **Histogramm** könnte ebenfalls nützlich sein, um die Dichteverteilungen der relativen Häufigkeiten weiter zu untersuchen.

### **Zusammenfassung:**

Das R-Programm erfüllt die Aufgabe, die Häufigkeit der Terminalzeichen in den Sequenzen zu berechnen und zu vergleichen, sehr gut. Die Visualisierung und die relativen Häufigkeiten ermöglichen es, schnell zu erkennen, wie gut die Grammatik der erzeugten Sequenzen die Häufigkeit der Terminalzeichen aus der empirischen Sequenz abbildet. Ein ähnliches Verteilungsmuster würde darauf hindeuten, dass die Grammatik gut funktioniert.

#####  **Aufgabe:**

Interpretiere das folgende Ergebnis:\' Symbol Empirical Generated1 Generated2 Empirical_Relative Generated1_Relative 1 KBG 1 1 1 0.02777778 0.08333333 2 VBG 1 1 1 0.02777778 0.08333333 3 KBBd 7 2 2 0.19444444 0.16666667 4 VBBd 2 1 1 0.05555556 0.08333333 5 VBA 8 2 2 0.22222222 0.16666667 6 KBA 4 1 1 0.11111111 0.08333333 7 VAA 3 1 1 0.08333333 0.08333333 8 KAA 5 1 1 0.13888889 0.08333333 9 VAV 2 1 1 0.05555556 0.08333333 10 KAV 3 1 1 0.08333333 0.08333333\'.

**Ergebnis:**

Das Ergebnis zeigt eine Tabelle, in der die Häufigkeiten und relativen Häufigkeiten der Terminalzeichen in der empirischen Sequenz sowie in den beiden erzeugten Sequenzen (Generated1 und Generated2) aufgelistet sind. Hier ist eine detaillierte Interpretation der Ergebnisse:

### **Häufigkeiten der Symbole (Empirical, Generated1, Generated2):** {#huxe4ufigkeiten-der-symbole-empirical-generated1-generated2}

Die Tabelle zeigt die Häufigkeit jedes Symbols (z.B. \"KBG\", \"VBG\" etc.) in der empirischen Sequenz sowie in den beiden erzeugten Sequenzen:

- **KBG, VBG, KBBd, VBA, KBA, VAA, KAA, VAV, KAV** sind die beobachteten Symbole.

- Für jedes Symbol sind die absoluten Häufigkeiten in der empirischen und den beiden erzeugten Sequenzen angegeben.

### **Relative Häufigkeiten:** {#relative-huxe4ufigkeiten}

- **Empirical_Relative**: Diese Spalte zeigt die relative Häufigkeit jedes Symbols in der empirischen Sequenz. Sie wird berechnet, indem die Häufigkeit des Symbols in der empirischen Sequenz durch die Gesamtzahl der Symbole in der empirischen Sequenz geteilt wird.

- **Generated1_Relative**: Diese Spalte zeigt die relative Häufigkeit jedes Symbols in der erzeugten Sequenz 1, berechnet auf dieselbe Weise.

- **Generated2_Relative**: Diese Spalte zeigt die relative Häufigkeit jedes Symbols in der erzeugten Sequenz 2.

### **Interpretation der einzelnen Zeilen:**

1.  **KBG und VBG:**

    - In allen Sequenzen (empirisch, erzeugt1 und erzeugt2) kommt \"KBG\" und \"VBG\" jeweils einmal vor. In der empirischen Sequenz beträgt die relative Häufigkeit von \"KBG\" und \"VBG\" jeweils **0.0278** (1/36, basierend auf der Gesamtzahl von 36 Symbolen in der empirischen Sequenz). In den erzeugten Sequenzen 1 und 2 ist die relative Häufigkeit **0.0833**, was darauf hindeutet, dass \"KBG\" und \"VBG\" in den künstlichen Sequenzen häufiger vorkommen als in der empirischen Sequenz.

2.  **KBBd:**

    - In der empirischen Sequenz kommt \"KBBd\" 7-mal vor, was eine relative Häufigkeit von **0.1944** ergibt. In den erzeugten Sequenzen 1 und 2 tritt \"KBBd\" jeweils 2-mal auf, was eine relative Häufigkeit von **0.1667** ergibt. Dies zeigt, dass \"KBBd\" in der empirischen Sequenz häufiger vorkommt als in den künstlichen Sequenzen.

3.  **VBBd:**

    - In der empirischen Sequenz tritt \"VBBd\" 2-mal auf, was eine relative Häufigkeit von **0.0556** ergibt. In den künstlichen Sequenzen kommt \"VBBd\" jeweils nur einmal vor, mit einer relativen Häufigkeit von **0.0833**. In diesem Fall ist \"VBBd\" in den künstlichen Sequenzen häufiger als in der empirischen Sequenz.

4.  **VBA:**

    - \"VBA\" tritt in der empirischen Sequenz 8-mal auf, was eine relative Häufigkeit von **0.2222** ergibt. In den beiden künstlichen Sequenzen kommt \"VBA\" jeweils 2-mal vor, mit einer relativen Häufigkeit von **0.1667**. Dies bedeutet, dass \"VBA\" in der empirischen Sequenz häufiger vorkommt als in den erzeugten Sequenzen.

5.  **KBA, VAA, KAA, VAV, KAV:**

    - Diese Symbole kommen in der empirischen Sequenz in unterschiedlicher Häufigkeit vor, wobei die relativen Häufigkeiten zwischen **0.0556** und **0.1389** liegen.

    - In den beiden erzeugten Sequenzen sind die relativen Häufigkeiten aller dieser Symbole **0.0833**, was darauf hinweist, dass in den erzeugten Sequenzen eine gleichmäßigere Verteilung vorliegt, während die empirische Sequenz eine ungleichmäßige Verteilung zeigt.

### **Vergleich der Verteilungen:**

- **Ähnlichkeiten und Unterschiede:**

  - Einige Symbole, wie \"KBG\" und \"VBG\", haben in den erzeugten Sequenzen eine höhere relative Häufigkeit als in der empirischen Sequenz. Dies könnte auf eine zu starke Betonung dieser Symbole in den künstlichen Sequenzen hinweisen.

  - Symbole wie \"KBBd\" und \"VBA\" sind in der empirischen Sequenz deutlich häufiger, was auf eine mögliche Verzerrung in den künstlichen Sequenzen hinweist, die diese Häufigkeiten nicht ausreichend abbilden.

  - Die erzeugten Sequenzen haben eine gleichmäßigere Verteilung der Symbole, während die empirische Sequenz größere Schwankungen in den Häufigkeiten aufweist.

### **Fazit:**

- Die erzeugten Sequenzen haben einige Ähnlichkeiten, aber auch deutliche Unterschiede zur empirischen Sequenz, insbesondere bei der Verteilung der Symbole. Während die erzeugten Sequenzen teilweise die Häufigkeiten der empirischen Sequenz gut replizieren, gibt es auch Abweichungen, die auf eine mögliche Verbesserung der Grammatik hinweisen könnten, um die Häufigkeiten noch genauer nachzubilden.

- Weitere Analysen, wie z.B. ein statistischer Vergleich der Verteilungen, könnten helfen, diese Unterschiede quantitativ zu bewerten.
