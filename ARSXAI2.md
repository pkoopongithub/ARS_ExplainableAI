# ARS 4.0 - Benutzerhandbuch

## Algorithmic Recursive Sequence Analysis

---

## Inhaltsverzeichnis

1. [Einleitung](#einleitung)
2. [Installation und Systemanforderungen](#installation-und-systemanforderungen)
3. [Erste Schritte](#erste-schritte)
4. [Die Benutzeroberfläche](#die-benutzeroberfläche)
5. [Dateneingabe](#dateneingabe)
6. [ARS 2.0 - Basis-Grammatik](#ars-20---basis-grammatik)
7. [ARS 3.0 - Hierarchische Grammatik mit Nonterminalen](#ars-30---hierarchische-grammatik-mit-nonterminalen)
8. [Petri-Netze](#petri-netze)
9. [Bayessche Netze](#bayessche-netze)
10. [Hybride Integration](#hybride-integration)
11. [Generierung neuer Ketten](#generierung-neuer-ketten)
12. [Menüfunktionen](#menüfunktionen)
13. [Fehlerbehandlung](#fehlerbehandlung)
14. [Häufig gestellte Fragen](#häufig-gestellte-fragen)
15. [Glossar](#glossar)

---

## Einleitung

ARS 4.0 (Algorithmic Recursive Sequence Analysis) ist ein leistungsfähiges Python-Programm mit grafischer Benutzeroberfläche zur Analyse und Modellierung von Sequenzdaten. Es wurde speziell für die qualitative Sozialforschung entwickelt und verbindet interpretative Methoden mit formalen Modellierungstechniken.

### Hauptfunktionen

- **ARS 2.0**: Berechnung von Übergangswahrscheinlichkeiten zwischen Symbolen
- **ARS 3.0**: Hierarchische Grammatikinduktion mit Nonterminalen durch Kompression von Wiederholungen
- **Petri-Netze**: Modellierung von Nebenläufigkeit, Ressourcen und Zustandsübergängen
- **Bayessche Netze**: Hidden-Markov-Modelle für latente Zustände und probabilistische Inferenz
- **Hybride Integration**: CRF, semantische Validierung mit Transformer-Embeddings, Graph-Analyse und Attention-Mechanismen

---

## Installation und Systemanforderungen

### Systemanforderungen

| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| Python-Version | 3.8 | 3.10 oder höher |
| Arbeitsspeicher | 4 GB | 8 GB oder mehr |
| Festplattenspeicher | 500 MB | 1 GB SSD |
| Betriebssystem | Windows, macOS, Linux | Windows 10/11, macOS 12+, Linux |
| Internetverbindung | Für Erstinstallation erforderlich | -

### Automatische Installation

Das Programm prüft beim Start automatisch, ob alle benötigten Python-Pakete installiert sind, und installiert fehlende Pakete bei Bedarf nach:

```bash
python ARSXAI2.py
```

Bei der ersten Ausführung werden folgende Pakete automatisch installiert:
- numpy, scipy, matplotlib
- hmmlearn, sklearn-crfsuite
- sentence-transformers, networkx
- torch, seaborn, tabulate

### Manuelle Installation

Falls Sie die Pakete manuell installieren möchten:

```bash
pip install numpy scipy matplotlib hmmlearn sklearn-crfsuite sentence-transformers networkx torch seaborn tabulate
```

### Programmstart

Starten Sie das Programm mit:

```bash
python ARSXAI2.py
```

---

## Erste Schritte

### Schnellstart-Anleitung

1. **Programm starten**: Führen Sie `python ARSXAI2.py` aus
2. **Beispieldaten laden**: Klicken Sie im linken Panel auf den Button **"Beispiel"**
3. **Daten parsen**: Klicken Sie auf **"Parsen"**
4. **Ergebnisse anzeigen**: Wählen Sie einen der Tabs aus:
   - **ARS 2.0** für Übergangswahrscheinlichkeiten
   - **ARS 3.0** für hierarchische Grammatik
   - **Generierung** für neue Ketten

### Grundlegender Workflow

1. **Eingabe**: Terminalzeichenketten im Textfeld eingeben oder Datei laden
2. **Parsen**: Daten in das interne Format konvertieren
3. **Analyse**: Gewünschte Analysemethode auswählen und ausführen
4. **Ergebnisse**: Textausgaben und Visualisierungen betrachten
5. **Generierung**: Neue Ketten mit der induzierten Grammatik erzeugen

---

## Die Benutzeroberfläche

### Hauptfenster

Das Hauptfenster ist in zwei Bereiche unterteilt:
- **Linkes Panel**: Eingabe und Steuerung
- **Rechtes Panel**: Notebook mit sechs Tabs für verschiedene Analysen

### Linkes Panel - Eingabe

| Element | Beschreibung |
|---------|--------------|
| **Trennzeichen** | Auswahl des Trennzeichens für die Symbole (Komma, Semikolon, Leerzeichen, benutzerdefiniert) |
| **Text-Eingabe** | Mehrzeiliges Textfeld für die Eingabe der Terminalzeichenketten (eine Kette pro Zeile) |
| **Datei laden** | Öffnet einen Dateidialog zum Laden einer Textdatei |
| **Parsen** | Parst die eingegebenen Daten und bereitet sie für die Analyse vor |
| **Beispiel** | Lädt die acht Beispieltranskripte |
| **Startzeichen** | Definiert das Startsymbol für die Generierung (optional) |
| **Info-Zeile** | Zeigt Anzahl der geladenen Ketten und Terminale an |

### Rechtes Panel - Tabs

| Tab | Funktion |
|-----|----------|
| **ARS 2.0** | Basis-Grammatik mit Übergangswahrscheinlichkeiten |
| **ARS 3.0** | Hierarchische Grammatik mit Nonterminalen |
| **Petri-Netze** | Modellierung mit Stellen/Transitionen-Netzen |
| **Bayessche Netze** | Hidden-Markov-Modelle für latente Zustände |
| **Hybrid** | CRF, semantische Validierung, Graph-Analyse, Attention |
| **Generierung** | Erzeugung neuer Ketten mit den induzierten Grammatiken |

### Statusleiste

Die Statusleiste am unteren Rand zeigt:
- Aktuelle Programmmeldungen
- Fortschrittsbalken bei langlaufenden Operationen

---

## Dateneingabe

### Format der Eingabedaten

Das Programm erwartet Terminalzeichenketten im folgenden Format:

- **Eine Kette pro Zeile**
- **Symbole getrennt durch ein Trennzeichen** (Komma, Semikolon, Leerzeichen oder benutzerdefiniert)

Beispiel:
```
KBG, VBG, KBBd, VBBd, KBA, VBA, KBBd, VBBd, KBA, VAA, KAA, VAV, KAV
VBG, KBBd, VBBd, VAA, KAA, VBG, KBBd, VAA, KAA
KBBd, VBBd, VAA, KAA
```

### Terminalzeichen (Beispiel)

Die Beispieltranskripte verwenden folgende Terminalzeichen:

| Symbol | Bedeutung |
|--------|-----------|
| KBG | Kunden-Gruß |
| VBG | Verkäufer-Gruß |
| KBBd | Kunden-Bedarf (konkret) |
| VBBd | Verkäufer-Nachfrage |
| KBA | Kunden-Antwort |
| VBA | Verkäufer-Reaktion |
| KAE | Kunden-Erkundigung |
| VAE | Verkäufer-Auskunft |
| KAA | Kunden-Abschluss |
| VAA | Verkäufer-Abschluss |
| KAV | Kunden-Verabschiedung |
| VAV | Verkäufer-Verabschiedung |

### Trennzeichen

Sie können zwischen verschiedenen Trennzeichen wählen:
- **Komma (,)**: Standard-Trennzeichen
- **Semikolon (;)**: Alternativ
- **Leerzeichen**: Für durch Leerzeichen getrennte Symbole
- **Benutzerdefiniert**: Beliebiges Zeichen (z.B. "|", ":", "-")

### Startzeichen

Das **Startzeichen** wird für die Generierung neuer Ketten verwendet:
- Wenn nicht angegeben, wird das erste Symbol der ersten Kette verwendet
- Muss in der Grammatik vorhanden sein (als Terminal oder Nonterminal)

---

## ARS 2.0 - Basis-Grammatik

### Übersicht

ARS 2.0 berechnet Übergangswahrscheinlichkeiten zwischen Terminalzeichen ohne hierarchische Struktur. Es handelt sich um eine Markov-Kette erster Ordnung.

### Buttons

| Button | Funktion |
|--------|----------|
| **ARS 2.0 berechnen** | Berechnet die Übergangswahrscheinlichkeiten aus den geladenen Ketten |
| **Optimierung starten** | Startet einen iterativen Optimierungsprozess (500 Iterationen) |

### Ausgabe

Die Ausgabe im Textbereich enthält:
- **Übergangswahrscheinlichkeiten**: Für jedes Startsymbol die Wahrscheinlichkeiten der Folgesymbole
- **Terminalzeichen**: Liste aller gefundenen Terminale
- **Startzeichen**: Das aktuell verwendete Startsymbol
- **Optimierte Grammatik**: Nach der Optimierung die verbesserte Grammatik
- **Beste Korrelation**: Erreichte Korrelation zwischen empirischen und generierten Daten

### Optimierung

Der Optimierungsprozess:
1. Generiert 8 künstliche Ketten mit der aktuellen Grammatik
2. Vergleicht die Häufigkeitsverteilung mit den empirischen Daten
3. Passt die Übergangswahrscheinlichkeiten iterativ an
4. Wiederholt bis zu 500 Iterationen
5. Zeigt den Fortschritt alle 50 Iterationen an

---

## ARS 3.0 - Hierarchische Grammatik mit Nonterminalen

### Übersicht

ARS 3.0 induziert eine hierarchische Grammatik durch iterative Kompression von Wiederholungen. Neue Nonterminale werden für wiederkehrende Sequenzen gebildet, und der Prozess wird wiederholt, bis idealerweise nur noch ein Startsymbol übrig bleibt.

### Methodologische Grundlagen

Der Induktionsprozess wird als **Explikation** verstanden:
- Jedes neue Nonterminal repräsentiert eine **interpretative Kategorie**
- Die Benennung expliziert die qualitative Bedeutung
- Der Prozess ist intersubjektiv nachvollziehbar

### Button

| Button | Funktion |
|--------|----------|
| **Grammatik induzieren** | Startet den iterativen Kompressionsprozess |

### Der Induktionsprozess

1. **Suche nach Wiederholungen**: Findet die beste wiederholte Sequenz in allen Ketten
   - Bewertungskriterium: (Häufigkeit × Länge) / Anzahl einzigartiger Symbole
   - Bevorzugt längere, häufigere Muster mit weniger Varianz

2. **Bildung eines Nonterminals**: Ersetzt die Sequenz durch ein neues Symbol
   - Beispiel: `KBBd → VBBd` wird zu `NT_BEDARFSKLAERUNG_KBBd_VBBd`

3. **Kompression**: Alle Vorkommen werden in allen Ketten ersetzt

4. **Wiederholung**: Der Prozess wird fortgesetzt, bis:
   - Keine weiteren Wiederholungen gefunden werden, oder
   - Alle Ketten zu einem einzigen Symbol komprimiert sind

### Startsymbol-Bestimmung

Das Startsymbol wird mit folgender Priorität bestimmt:
1. **Benutzerdefiniertes Startzeichen** (falls im Eingabefeld angegeben)
2. **Einziges verbleibendes Symbol** nach vollständiger Kompression
3. **Oberstes Nonterminal** (das niemals als Teil einer anderen Produktion vorkommt)
4. **Nonterminal mit der höchsten Hierarchieebene** (späteste Iteration)

### Ausgabe

Die Ausgabe im Textbereich enthält:
- **Terminale**: Alle ursprünglichen Symbole, die nie ersetzt wurden
- **Nonterminale**: Alle neu gebildeten Kategorien
- **Startsymbol**: Das oberste Nonterminal der Hierarchie
- **Iterationen**: Anzahl der durchgeführten Kompressionsschritte
- **Produktionsregeln**: Für jedes Nonterminal die möglichen Expansionen mit Wahrscheinlichkeiten
- **Kompressionshistorie**: Dokumentation jedes Induktionsschritts

---

## Petri-Netze

### Übersicht

Petri-Netze modellieren nebenläufige Prozesse mit:
- **Stellen** (Kreise): repräsentieren Zustände oder Ressourcen
- **Transitionen** (Rechtecke): repräsentieren Ereignisse oder Aktionen
- **Kanten**: verbinden Stellen mit Transitionen und umgekehrt
- **Marken** (Token): repräsentieren die aktuelle Belegung von Stellen

### Buttons

| Button | Funktion |
|--------|----------|
| **Einfaches Netz** | Erstellt ein einfaches Petri-Netz ohne Ressourcenmodellierung |
| **Netz mit Ressourcen** | Erstellt ein erweitertes Petri-Netz mit Ressourcen |
| **Simuliere Transkript 1** | Simuliert das erste Transkript im erstellten Petri-Netz |

### Einfaches Netz

- Stellen: `p_start`, `p_end`, `p_{sym}_ready` für jedes Symbol
- Transitionen: `t_{sym}` für jedes Symbol
- Simulation: Prüft für jedes Symbol, ob die Transition aktiviert ist

### Netz mit Ressourcen

Modellierte Ressourcen:
- **Kunde**: `p_customer_present`, `p_customer_ready`, `p_customer_paying`
- **Verkäufer**: `p_seller_ready`, `p_seller_serving`
- **Waren**: `p_goods_available`, `p_goods_selected`, `p_goods_packaged`
- **Geld**: `p_money_customer`, `p_money_register`
- **Phasen**: `p_phase_Greeting`, `p_phase_Need`, `p_phase_Consult`, `p_phase_Completion`, `p_phase_Farewell`

### Ausgabe

- **Netz-Statistik**: Anzahl der Stellen, Transitionen und Kanten
- **Ressourcen-Stellen**: Liste der Ressourcen mit initialen Token
- **Simulationsergebnis**: Für jedes Symbol, ob die Transition aktiviert war
- **Finale Markierung**: Token-Verteilung nach der Simulation
- **Visualisierung**: Graphische Darstellung des Petri-Netzes

---

## Bayessche Netze

### Übersicht

Bayessche Netze modellieren Unsicherheiten, latente Variablen und bidirektionale Inferenzen. Implementiert als Hidden-Markov-Modelle (HMM) mit 5 latenten Zuständen.

### Latente Zustände

| Zustand | Bedeutung | Typische Symbole |
|---------|-----------|------------------|
| 0 | Greeting | KBG, VBG |
| 1 | Need Determination | KBBd, VBBd |
| 2 | Consultation | KBA, VBA, KAE, VAE |
| 3 | Completion | KAA, VAA |
| 4 | Farewell | KAV, VAV |

### Buttons

| Button | Funktion |
|--------|----------|
| **HMM initialisieren** | Initialisiert ein HMM mit 5 latenten Zuständen basierend auf ARS-Daten |
| **HMM trainieren** | Trainiert das HMM mit dem Baum-Welch-Algorithmus (100 Iterationen) |
| **Dekodiere Transkript 1** | Führt eine Viterbi-Dekodierung des ersten Transkripts durch |

### Ausgabe

- **Startwahrscheinlichkeiten**: Wahrscheinlichkeit jedes Zustands zu Beginn
- **Übergangsmatrix**: Wahrscheinlichkeiten für Zustandsübergänge
- **Emissionswahrscheinlichkeiten**: Top-3 Symbole pro Zustand
- **Dekodierungsergebnis**: Für jedes Symbol im Transkript der zugeordnete Zustand

---

## Hybride Integration

### Übersicht

Die hybride Integration kombiniert verschiedene computerlinguistische Verfahren komplementär zu den interpretativen Kategorien.

### Verfahren

1. **Conditional Random Fields (CRF)**: Modelliert sequenzielle Abhängigkeiten mit Kontext
2. **Semantische Validierung**: Transformer-Embeddings für semantische Ähnlichkeit zwischen Kategorien
3. **Grammatik-Graph**: Netzwerkanalyse der Nonterminal-Hierarchie
4. **Attention-Visualisierung**: Vereinfachte Attention-Mechanismen auf Sequenzen

### Buttons

| Button | Funktion |
|--------|----------|
| **CRF trainieren** | Trainiert ein CRF-Modell auf den Sequenzdaten |
| **Semantische Validierung** | Berechnet semantische Ähnlichkeiten zwischen Kategorien |
| **Grammatik-Graph** | Erstellt einen gerichteten Graphen aus der ARS-3.0-Grammatik |
| **Attention visualisieren** | Berechnet und visualisiert vereinfachte Attention-Gewichte |

### CRF-Features

Das CRF-Modell verwendet folgende Features:
- Aktuelles Symbol, Präfixe (K/V), Suffixe (A/B/E/G/V)
- Position in der Sequenz, erste/letzte Position
- Kontext-Features (-2, -1, +1, +2)
- Bigram-Features

### Semantische Validierung

- Nutzt Sentence-Transformer-Modell `paraphrase-multilingual-MiniLM-L12-v2`
- Berechnet Intra-Kategorie-Ähnlichkeiten (Kohäsion)
- Visualisiert Ähnlichkeitsmatrix als Heatmap

### Grammatik-Graph

- Knoten: Terminale und Nonterminale
- Kanten: Ableitungsrelationen mit Gewichten
- Berechnet Zentralität der Knoten
- Visualisiert als gerichteter Graph

### Attention-Visualisierung

- Berechnet vereinfachte Attention-Gewichte basierend auf Bigram-Wahrscheinlichkeiten
- Exponentiell abfallende Gewichte für entferntere Vorgänger
- Visualisierung als Heatmap

---

## Generierung neuer Ketten

### Übersicht

Der Generierung-Tab erlaubt die Erzeugung neuer Ketten mit den induzierten Grammatiken.

### Steuerelemente

| Element | Beschreibung |
|---------|--------------|
| **Grammatik-Auswahl** | Wahl zwischen ARS 2.0 und ARS 3.0 Grammatik |
| **Anzahl** | Anzahl der zu generierenden Ketten (1-50) |
| **Generieren** | Startet die Generierung |

### ARS 2.0 Generierung

- Verwendet die optimierten oder initialen Übergangswahrscheinlichkeiten
- Beginnt beim definierten Startzeichen
- Wählt zufällig nächste Symbole basierend auf Wahrscheinlichkeiten

### ARS 3.0 Generierung

- Beginnt beim induzierten Startsymbol (oberstes Nonterminal)
- Expandiert rekursiv mit den Produktionsregeln
- Wählt Produktionen basierend auf Wahrscheinlichkeiten
- Maximale Tiefe: 20 Schritte (Schutz vor Endlosschleifen)

### Ausgabe

- Liste der generierten Ketten mit laufender Nummer
- Symbole durch "→" getrennt
- Länge der Kette in Klammern

---

## Menüfunktionen

### Datei-Menü

| Menüpunkt | Funktion |
|-----------|----------|
| **Öffnen** | Lädt eine Textdatei mit Terminalzeichenketten |
| **Beispiel laden** | Lädt die acht Beispieltranskripte |
| **Beenden** | Schließt das Programm |

### Hilfe-Menü

| Menüpunkt | Funktion |
|-----------|----------|
| **Modulstatus** | Zeigt an, welche optionalen Python-Module verfügbar sind |
| **Über** | Zeigt Informationen über das Programm |

---

## Fehlerbehandlung

### Automatische Paketinstallation

Das Programm prüft beim Start automatisch, ob alle benötigten Pakete installiert sind, und installiert fehlende Pakete bei Bedarf nach.

### Häufige Fehlermeldungen

| Fehler | Ursache | Lösung |
|--------|---------|--------|
| "Keine Daten geladen" | Es wurden noch keine Ketten eingegeben | Daten eingeben oder Beispieldaten laden |
| "networkx nicht installiert" | Fehlende Python-Bibliothek | Programm neu starten (automatische Installation) |
| "Kein HMM vorhanden" | HMM wurde nicht initialisiert | Zuerst "HMM initialisieren" klicken |
| "Kein Petri-Netz vorhanden" | Kein Netz erstellt | Zuerst "Einfaches Netz" oder "Netz mit Ressourcen" klicken |
| "Keine ARS 3.0 Grammatik" | Grammatik wurde nicht induziert | Zuerst "Grammatik induzieren" klicken |

### Thread-Sicherheit

Alle langlaufenden Operationen (Optimierung, Training, Induktion) laufen in separaten Threads, sodass die GUI reaktionsfähig bleibt. Visualisierungen werden thread-sicher im Hauptthread ausgeführt.

---

## Häufig gestellte Fragen

### Allgemein

**F: Welche Datenformate werden unterstützt?**  
A: Textdateien mit einer Terminalzeichenkette pro Zeile, Symbole durch Komma, Semikolon, Leerzeichen oder benutzerdefinierte Trennzeichen getrennt.

**F: Kann ich eigene Terminalzeichen verwenden?**  
A: Ja, beliebige Zeichenkombinationen sind möglich. Die Bedeutung wird nur in der methodologischen Reflexion dokumentiert.

### ARS 2.0

**F: Was bedeutet die Korrelation bei der Optimierung?**  
A: Die Korrelation misst die Übereinstimmung zwischen der Häufigkeitsverteilung der generierten Ketten und den empirischen Daten. Werte nahe 1 zeigen gute Übereinstimmung.

**F: Wie lange dauert die Optimierung?**  
A: Bei 500 Iterationen etwa 10-30 Sekunden, abhängig von der Anzahl der Ketten und Symbole.

### ARS 3.0

**F: Warum werden manche Nonterminale nicht weiter komprimiert?**  
A: Wenn eine Sequenz nur einmal vorkommt, wird sie nicht als Nonterminal erfasst. Die Kompression stoppt, wenn keine Wiederholungen mehr gefunden werden.

**F: Wie wird das Startsymbol bestimmt?**  
A: Priorität: 1. Benutzerdefiniert, 2. Einziges verbleibendes Symbol, 3. Oberstes Nonterminal, 4. Erstes Nonterminal.

### Petri-Netze

**F: Was bedeutet "enabled" bei der Simulation?**  
A: Eine Transition ist aktiviert (enabled), wenn in allen Vorstellen genügend Token vorhanden sind.

**F: Warum sind manche Transitionen nicht aktiviert?**  
A: Mögliche Ursachen: Fehlende Ressourcen (Token), falsche Phasen, oder die Guard-Bedingung ist nicht erfüllt.

### Bayessche Netze

**F: Was ist Viterbi-Dekodierung?**  
A: Der Viterbi-Algorithmus findet die wahrscheinlichste Sequenz von latenten Zuständen zu einer gegebenen Beobachtungssequenz.

**F: Was bedeutet der p-Wert bei der Dekodierung?**  
A: Der p-Wert ist die Wahrscheinlichkeit der gefundenen Zustandssequenz (normalisierte Log-Wahrscheinlichkeit).

### Hybride Integration

**F: Was sind CRF-Features?**  
A: Merkmale, die das CRF-Modell zum Lernen verwendet, z.B. aktuelles Symbol, Kontext, Position, Bigramme.

**F: Wie werden die semantischen Ähnlichkeiten berechnet?**  
A: Mit einem vortrainierten Sentence-Transformer-Modell, das Texte in Vektoren umwandelt und Kosinus-Ähnlichkeit berechnet.

---

## Glossar

| Begriff | Definition |
|---------|------------|
| **ARS** | Algorithmic Recursive Sequence Analysis |
| **CRF** | Conditional Random Field - probabilistisches Modell für sequenzielle Daten |
| **DBN** | Dynamic Bayesian Network - zeitliches Bayessches Netz |
| **GNN** | Graph Neural Network - neuronales Netz für Graphdaten |
| **HMM** | Hidden Markov Model - Modell mit latenten Zuständen |
| **Markierung** | Verteilung von Token in einem Petri-Netz |
| **Nonterminal** | Symbol, das weiter expandiert werden kann (interpretative Kategorie) |
| **PCFG** | Probabilistic Context-Free Grammar - probabilistische kontextfreie Grammatik |
| **Petri-Netz** | Formales Modell für nebenläufige Prozesse |
| **Stelle** | Knoten in einem Petri-Netz, repräsentiert Zustand oder Ressource |
| **Terminal** | Nicht weiter expandierbares Symbol (Sprechakt-Kategorie) |
| **Token** | Marke in einem Petri-Netz, repräsentiert Ressourcen oder Zustände |
| **Transition** | Knoten in einem Petri-Netz, repräsentiert Ereignis oder Aktion |
| **Viterbi** | Algorithmus zur Bestimmung der wahrscheinlichsten Zustandssequenz |
| **XAI** | Explainable Artificial Intelligence - erklärbare künstliche Intelligenz |

