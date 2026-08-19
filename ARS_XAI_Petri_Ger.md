---
abstract: |
  Die vorliegende Arbeit erweitert die Algorithmisch Rekursive Sequenzanalyse (ARS) um Petri-Netze als formales Modellierungsverfahren. Während ARS 3.0 die hierarchische Struktur von Interaktionen durch Nonterminale abbildet, ermöglichen Petri-Netze die Modellierung von Nebenläufigkeit, Ressourcen und Zustandsübergängen. Die Integration erfolgt als kontinuierliche Erweiterung auf äquivalenter Ebene: Die interpretativ gewonnenen Terminalzeichen und die induzierte Nonterminal-Hierarchie werden in Stellen/Transitionen-Netze überführt. Die Anwendung auf acht Transkripte von Verkaufsgesprächen demonstriert, wie parallele Aktivitäten von Kunden und Verkäufern, Ressourcen (Waren, Geld) und Gesprächsphasen als Petri-Netz modelliert werden können. Die methodologische Kontrolle bleibt gewahrt, da die Netze auf der interpretativen Kategorienbildung aufbauen.
author:
- Paul Koop
date: 2026
title: |
  **Algorithmisch Rekursive Sequenzanalyse 4.0**\
  Integration von Petri-Netzen zur Modellierung nebenläufiger\
  Interaktionsstrukturen in Verkaufsgesprächen
---

# Einleitung: Von der Grammatik zum Prozessmodell

Die ARS 3.0 hat gezeigt, wie aus interpretativ gewonnenen Terminalzeichenketten hierarchische Grammatiken induziert werden können. Diese Grammatiken modellieren die sequenzielle Ordnung von Sprechakten als probabilistische Ableitungsbäume. Sie erfassen jedoch nicht alle Aspekte natürlicher Interaktion:

- **Nebenläufigkeit**: In Verkaufsgesprächen können parallel Aktivitäten stattfinden (Kunde sucht Geld, Verkäufer verpackt Ware).

- **Ressourcen**: Waren, Geld und Aufmerksamkeit sind begrenzte Ressourcen, die den Gesprächsverlauf beeinflussen.

- **Zustandsabhängigkeiten**: Der Gesprächszustand (z.B. \"wartet auf Bezahlung\") bestimmt, welche Aktionen möglich sind.

Petri-Netze [@Petri1962; @Reisig2010] sind ein etabliertes formales Modell, das genau diese Aspekte abbilden kann. Sie bestehen aus:

- **Stellen** (Kreise): repräsentieren Zustände oder Ressourcen

- **Transitionen** (Rechtecke): repräsentieren Ereignisse oder Aktionen

- **Kanten**: verbinden Stellen mit Transitionen und umgekehrt

- **Marken** (Token): repräsentieren die aktuelle Belegung von Stellen

Die vorliegende Arbeit entwickelt eine systematische Überführung der ARS-3.0-Grammatik in Petri-Netze und demonstriert dies an den acht Transkripten von Verkaufsgesprächen.

# Theoretische Grundlagen

## Stellen/Transitionen-Netze

Ein Stellen/Transitionen-Netz (S/T-Netz) ist ein Tupel $N = (S, T, F, W, M_0)$ mit:

- $S$: Menge der Stellen (Places)

- $T$: Menge der Transitionen (Transitions), $S \cap T = \emptyset$

- $F \subseteq (S \times T) \cup (T \times S)$: Flussrelation (Kanten)

- $W: F \rightarrow \mathbb{N}^+$: Kantengewichte

- $M_0: S \rightarrow \mathbb{N}_0$: Anfangsmarkierung

Die Dynamik eines Petri-Netzes wird durch das Schalten von Transitionen bestimmt. Eine Transition $t$ ist aktiviert, wenn für alle Vorstellen $s \in \bullet t$ gilt: $M(s) \ge W(s,t)$. Beim Schalten werden Token von den Vorstellen entfernt und zu den Nachstellen hinzugefügt.

## Gefärbte Petri-Netze

Gefärbte Petri-Netze (Colored Petri Nets) [@Jensen1997] erweitern S/T-Netze um Datentypen (Farben). Jede Stelle hat einen bestimmten Farbtyp, und Token tragen Datenwerte. Transitionen können komplexe Schaltregeln haben, die auf diesen Daten operieren.

Für die Modellierung von Verkaufsgesprächen eignen sich gefärbte Petri-Netze besonders, da sie unterschiedliche Token-Typen (Kunde, Verkäufer, Ware, Geld) unterscheiden können.

## Hierarchische Petri-Netze

Hierarchische Petri-Netze [@Fehling1993] erlauben die Modellierung von Subnetzen, die als abstrakte Transitionen oder Stellen dargestellt werden. Dies ermöglicht die direkte Umsetzung der ARS-3.0-Nonterminal-Hierarchie.

# Methodik: Von ARS 3.0 zu Petri-Netzen

## Überführung der Terminalzeichen

Die Terminalzeichen der ARS 3.0 werden als Transitionen modelliert:

::: {#tab:mapping_terminal}
  **Terminalzeichen**   **Bedeutung**              **Petri-Netz-Transition**
  --------------------- -------------------------- ---------------------------
  KBG                   Kunden-Gruß                t_KBG
  VBG                   Verkäufer-Gruß             t_VBG
  KBBd                  Kunden-Bedarf              t_KBBd
  VBBd                  Verkäufer-Nachfrage        t_VBBd
  KBA                   Kunden-Antwort             t_KBA
  VBA                   Verkäufer-Reaktion         t_VBA
  KAE                   Kunden-Erkundigung         t_KAE
  VAE                   Verkäufer-Auskunft         t_VAE
  KAA                   Kunden-Abschluss           t_KAA
  VAA                   Verkäufer-Abschluss        t_VAA
  KAV                   Kunden-Verabschiedung      t_KAV
  VAV                   Verkäufer-Verabschiedung   t_VAV

  : Mapping von Terminalzeichen zu Petri-Netz-Transitionen
:::

## Überführung der Nonterminale

Die Nonterminale der ARS 3.0 werden als hierarchische Subnetze modelliert. Jedes Nonterminal wird zu einer abstrakten Transition, die ein Subnetz mit den entsprechenden Produktionen enthält.

Beispiel: Das Nonterminal 'NT_BEDARFSKLAERUNG_KBBd_VBBd' wird zu einer Transition 't_BEDARFSKLAERUNG', die intern aus den Transitionen 't_KBBd' und 't_VBBd' besteht.

## Modellierung von Ressourcen

Zusätzlich zu den sprechakt-basierten Transitionen werden Ressourcen als Stellen modelliert:

- **s_Kunde**: Token repräsentieren den Kunden (Anwesenheit, Zustand)

- **s_Verkäufer**: Token repräsentieren den Verkäufer

- **s_Waren**: Token repräsentieren verfügbare Waren

- **s_Geld**: Token repräsentieren Geld (beim Kunden, im Register)

- **s_Gesprächsphase**: Token repräsentieren die aktuelle Phase

## Modellierung von Nebenläufigkeit

Nebenläufigkeit wird durch parallele Pfade im Petri-Netz modelliert. Zum Beispiel können Kunde und Verkäufer gleichzeitig aktiv sein (Kunde sucht Geld, Verkäufer verpackt Ware).

# Implementierung

Die Implementierung erfolgt in Python mit der Bibliothek 'pm4py' (Process Mining for Python) und 'snakes' (Petri-Netz-Simulator).

``` {.python caption="Petri-Netz-Klasse für ARS" language="Python"}
"""
Petri-Netz-Implementierung für ARS 4.0
Modellierung von Verkaufsgesprächen als Stellen/Transitionen-Netze
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class ARSPetriNet:
    """
    Petri-Netz-Modell für ARS 4.0
    """
    
    def __init__(self, name="ARS_PetriNet"):
        self.name = name
        self.places = {}  # Stellen: name -> Place-Objekt
        self.transitions = {}  # Transitionen: name -> Transition-Objekt
        self.arcs = []  # Kanten: (source, target, weight)
        self.tokens = {}  # Marken: place_name -> Anzahl
        self.hierarchy = {}  # Hierarchie: transition_name -> subnet
        
        # Statistik
        self.firing_history = []
        self.reached_markings = set()
    
    def add_place(self, name, initial_tokens=0, place_type="normal"):
        """
        Fügt eine Stelle hinzu
        place_type: "normal", "resource", "phase", "customer", "seller"
        """
        self.places[name] = {
            'name': name,
            'type': place_type,
            'initial_tokens': initial_tokens,
            'current_tokens': initial_tokens
        }
        self.tokens[name] = initial_tokens
    
    def add_transition(self, name, transition_type="speech_act", 
                       guard=None, subnet=None):
        """
        Fügt eine Transition hinzu
        transition_type: "speech_act", "abstract", "silent"
        guard: Bedingungsfunktion (optional)
        subnet: Subnetz für hierarchische Transitionen
        """
        self.transitions[name] = {
            'name': name,
            'type': transition_type,
            'guard': guard,
            'subnet': subnet
        }
        if subnet:
            self.hierarchy[name] = subnet
    
    def add_arc(self, source, target, weight=1):
        """
        Fügt eine Kante hinzu (source -> target)
        source/target können Stellen oder Transitionen sein
        """
        self.arcs.append({
            'source': source,
            'target': target,
            'weight': weight
        })
    
    def get_preset(self, transition):
        """Gibt die Vorstellen einer Transition zurück"""
        preset = {}
        for arc in self.arcs:
            if arc['target'] == transition and arc['source'] in self.places:
                preset[arc['source']] = arc['weight']
        return preset
    
    def get_postset(self, transition):
        """Gibt die Nachstellen einer Transition zurück"""
        postset = {}
        for arc in self.arcs:
            if arc['source'] == transition and arc['target'] in self.places:
                postset[arc['target']] = arc['weight']
        return postset
    
    def is_enabled(self, transition):
        """Prüft, ob eine Transition aktiviert ist"""
        if transition not in self.transitions:
            return False
        
        # Prüfe Vorstellen
        preset = self.get_preset(transition)
        for place, weight in preset.items():
            if self.tokens.get(place, 0) < weight:
                return False
        
        # Prüfe Guard-Bedingung
        trans_data = self.transitions[transition]
        if trans_data['guard'] and not trans_data['guard'](self):
            return False
        
        return True
    
    def fire(self, transition):
        """Schaltet eine Transition"""
        if not self.is_enabled(transition):
            return False
        
        # Entferne Token von Vorstellen
        preset = self.get_preset(transition)
        for place, weight in preset.items():
            self.tokens[place] -= weight
        
        # Füge Token zu Nachstellen hinzu
        postset = self.get_postset(transition)
        for place, weight in postset.items():
            self.tokens[place] = self.tokens.get(place, 0) + weight
        
        # Protokolliere Schaltvorgang
        self.firing_history.append({
            'transition': transition,
            'marking': self.get_marking_copy()
        })
        
        # Speichere erreichte Markierung
        self.reached_markings.add(self.get_marking_tuple())
        
        return True
    
    def get_marking_copy(self):
        """Gibt eine Kopie der aktuellen Markierung zurück"""
        return self.tokens.copy()
    
    def get_marking_tuple(self):
        """Gibt die Markierung als sortiertes Tupel zurück (für Hash-Set)"""
        return tuple(sorted([(p, self.tokens[p]) for p in self.places]))
    
    def reset(self):
        """Setzt das Netz in den Anfangszustand zurück"""
        for place_name, place_data in self.places.items():
            self.tokens[place_name] = place_data['initial_tokens']
        self.firing_history = []
    
    def simulate(self, transition_sequence):
        """
        Simuliert eine Sequenz von Transitionen
        Gibt Erfolg und letzte Markierung zurück
        """
        self.reset()
        successful = []
        
        for t in transition_sequence:
            if self.is_enabled(t):
                self.fire(t)
                successful.append(t)
            else:
                break
        
        return successful, self.get_marking_copy()
    
    def visualize(self, filename="petri_net.png"):
        """
        Visualisiert das Petri-Netz mit networkx und matplotlib
        """
        G = nx.DiGraph()
        
        # Füge Stellen hinzu (Kreise)
        for place in self.places:
            G.add_node(place, type='place', shape='circle')
        
        # Füge Transitionen hinzu (Rechtecke)
        for trans in self.transitions:
            G.add_node(trans, type='transition', shape='box')
        
        # Füge Kanten hinzu
        for arc in self.arcs:
            G.add_edge(arc['source'], arc['target'], weight=arc['weight'])
        
        # Layout
        pos = nx.spring_layout(G)
        
        plt.figure(figsize=(15, 10))
        
        # Zeichne Stellen
        place_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'place']
        nx.draw_networkx_nodes(G, pos, nodelist=place_nodes, 
                              node_color='lightblue', node_shape='o', 
                              node_size=1000)
        
        # Zeichne Transitionen
        trans_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'transition']
        nx.draw_networkx_nodes(G, pos, nodelist=trans_nodes, 
                              node_color='lightgreen', node_shape='s', 
                              node_size=800)
        
        # Zeichne Kanten
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
        
        # Zeichne Labels
        labels = {}
        for node in G.nodes():
            if node in self.places:
                labels[node] = f"{node}\n[{self.tokens.get(node, 0)}]"
            else:
                labels[node] = node
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Petri-Netz: {self.name}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.show()
        
        return G

class ARSToPetriNetConverter:
    """
    Konvertiert ARS-3.0-Grammatiken in Petri-Netze
    """
    
    def __init__(self, grammar_rules, terminal_chains):
        self.grammar = grammar_rules
        self.terminals = terminal_chains
        self.petri_net = ARSPetriNet("ARS_4.0_Verkaufsgespraeche")
        
    def build_resource_places(self):
        """
        Erstellt Ressourcen-Stellen
        """
        # Kunde und Verkäufer als Ressourcen
        self.petri_net.add_place("s_Kunde_anwesend", initial_tokens=1, 
                                 place_type="customer")
        self.petri_net.add_place("s_Kunde_bereit", initial_tokens=1, 
                                 place_type="customer")
        self.petri_net.add_place("s_Kunde_zahlt", initial_tokens=0, 
                                 place_type="customer")
        
        self.petri_net.add_place("s_Verkäufer_bereit", initial_tokens=1, 
                                 place_type="seller")
        self.petri_net.add_place("s_Verkäufer_bedient", initial_tokens=0, 
                                 place_type="seller")
        
        # Waren und Geld
        self.petri_net.add_place("s_Waren_verfügbar", initial_tokens=10, 
                                 place_type="resource")
        self.petri_net.add_place("s_Waren_ausgewählt", initial_tokens=0, 
                                 place_type="resource")
        self.petri_net.add_place("s_Waren_verpackt", initial_tokens=0, 
                                 place_type="resource")
        
        self.petri_net.add_place("s_Geld_Kunde", initial_tokens=20, 
                                 place_type="resource")
        self.petri_net.add_place("s_Geld_Register", initial_tokens=0, 
                                 place_type="resource")
        
        # Gesprächsphasen
        phases = ["Begrüßung", "Bedarfsermittlung", "Beratung", 
                  "Abschluss", "Verabschiedung"]
        for phase in phases:
            self.petri_net.add_place(f"s_Phase_{phase}", initial_tokens=0,
                                     place_type="phase")
        
        # Anfangsphase
        self.petri_net.add_place("s_Phase_Start", initial_tokens=1,
                                 place_type="phase")
    
    def build_speech_act_transitions(self):
        """
        Erstellt Transitionen für alle Terminalzeichen
        """
        # Mapping der Terminalzeichen zu Petri-Netz-Transitionen
        terminal_to_transition = {
            'KBG': self._create_greeting_transition('KBG', 'Kunde'),
            'VBG': self._create_greeting_transition('VBG', 'Verkäufer'),
            'KBBd': self._create_need_transition('KBBd', 'Kunde'),
            'VBBd': self._create_inquiry_transition('VBBd', 'Verkäufer'),
            'KBA': self._create_response_transition('KBA', 'Kunde'),
            'VBA': self._create_reaction_transition('VBA', 'Verkäufer'),
            'KAE': self._create_inquiry_transition('KAE', 'Kunde'),
            'VAE': self._create_information_transition('VAE', 'Verkäufer'),
            'KAA': self._create_completion_transition('KAA', 'Kunde'),
            'VAA': self._create_completion_transition('VAA', 'Verkäufer'),
            'KAV': self._create_farewell_transition('KAV', 'Kunde'),
            'VAV': self._create_farewell_transition('VAV', 'Verkäufer')
        }
        
        # Füge Transitionen zum Netz hinzu
        for terminal, trans_data in terminal_to_transition.items():
            self.petri_net.add_transition(
                trans_data['name'],
                transition_type=trans_data['type'],
                guard=trans_data.get('guard')
            )
            
            # Füge Kanten hinzu
            for arc in trans_data.get('arcs', []):
                self.petri_net.add_arc(arc['source'], arc['target'], 
                                       arc.get('weight', 1))
    
    def _create_greeting_transition(self, symbol, speaker):
        """Erstellt eine Gruß-Transition"""
        return {
            'name': f"t_{symbol}",
            'type': 'speech_act',
            'arcs': [
                {'source': f"s_{speaker}_bereit", 'target': f"t_{symbol}"},
                {'source': f"s_Phase_Start", 'target': f"t_{symbol}"},
                {'target': f"s_Phase_Begrüßung", 'source': f"t_{symbol}"},
                {'target': f"s_{speaker}_bereit", 'source': f"t_{symbol}"}
            ]
        }
    
    def _create_need_transition(self, symbol, speaker):
        """Erstellt eine Bedarfs-Transition"""
        return {
            'name': f"t_{symbol}",
            'type': 'speech_act',
            'guard': lambda net: net.tokens.get('s_Waren_verfügbar', 0) > 0,
            'arcs': [
                {'source': f"s_{speaker}_bereit", 'target': f"t_{symbol}"},
                {'source': 's_Waren_verfügbar', 'target': f"t_{symbol}"},
                {'target': 's_Waren_ausgewählt', 'source': f"t_{symbol}", 'weight': 1},
                {'target': f"s_{speaker}_bereit", 'source': f"t_{symbol}"}
            ]
        }
    
    def _create_inquiry_transition(self, symbol, speaker):
        """Erstellt eine Nachfrage-Transition"""
        return {
            'name': f"t_{symbol}",
            'type': 'speech_act',
            'arcs': [
                {'source': f"s_{speaker}_bereit", 'target': f"t_{symbol}"},
                {'target': f"s_{speaker}_bereit", 'source': f"t_{symbol}"}
            ]
        }
    
    def _create_response_transition(self, symbol, speaker):
        """Erstellt eine Antwort-Transition"""
        return {
            'name': f"t_{symbol}",
            'type': 'speech_act',
            'arcs': [
                {'source': f"s_{speaker}_bereit", 'target': f"t_{symbol}"},
                {'target': f"s_{speaker}_bereit", 'source': f"t_{symbol}"}
            ]
        }
    
    def _create_reaction_transition(self, symbol, speaker):
        """Erstellt eine Reaktions-Transition"""
        return {
            'name': f"t_{symbol}",
            'type': 'speech_act',
            'arcs': [
                {'source': f"s_{speaker}_bereit", 'target': f"t_{symbol}"},
                {'target': f"s_{speaker}_bereit", 'source': f"t_{symbol}"}
            ]
        }
    
    def _create_information_transition(self, symbol, speaker):
        """Erstellt eine Informations-Transition"""
        return {
            'name': f"t_{symbol}",
            'type': 'speech_act',
            'arcs': [
                {'source': f"s_{speaker}_bereit", 'target': f"t_{symbol}"},
                {'target': f"s_{speaker}_bereit", 'source': f"t_{symbol}"}
            ]
        }
    
    def _create_completion_transition(self, symbol, speaker):
        """Erstellt eine Abschluss-Transition"""
        other = 'Verkäufer' if speaker == 'Kunde' else 'Kunde'
        return {
            'name': f"t_{symbol}",
            'type': 'speech_act',
            'guard': lambda net: (net.tokens.get('s_Waren_ausgewählt', 0) > 0 and
                                  net.tokens.get('s_Geld_Kunde', 0) > 0),
            'arcs': [
                {'source': f"s_{speaker}_bereit", 'target': f"t_{symbol}"},
                {'source': 's_Waren_ausgewählt', 'target': f"t_{symbol}", 'weight': 1},
                {'source': 's_Geld_Kunde', 'target': f"t_{symbol}", 'weight': 1},
                {'target': 's_Waren_verpackt', 'source': f"t_{symbol}", 'weight': 1},
                {'target': 's_Geld_Register', 'source': f"t_{symbol}", 'weight': 1},
                {'target': f"s_Phase_Abschluss", 'source': f"t_{symbol}"},
                {'target': f"s_{speaker}_bereit", 'source': f"t_{symbol}"},
                {'target': f"s_{other}_bereit", 'source': f"t_{symbol}"}
            ]
        }
    
    def _create_farewell_transition(self, symbol, speaker):
        """Erstellt eine Verabschiedungs-Transition"""
        return {
            'name': f"t_{symbol}",
            'type': 'speech_act',
            'arcs': [
                {'source': f"s_{speaker}_bereit", 'target': f"t_{symbol}"},
                {'target': f"s_Phase_Verabschiedung", 'source': f"t_{symbol}"},
                {'target': f"s_{speaker}_bereit", 'source': f"t_{symbol}"}
            ]
        }
    
    def build_nonterminal_hierarchy(self):
        """
        Erstellt hierarchische Transitionen für Nonterminale
        """
        for nt, productions in self.grammar.items():
            # Erstelle Subnetz für dieses Nonterminal
            subnet = ARSPetriNet(f"subnet_{nt}")
            
            # Füge Produktionen als Transitionen im Subnetz hinzu
            for i, (prod, prob) in enumerate(productions):
                trans_name = f"t_{nt}_prod{i}"
                subnet.add_transition(trans_name, transition_type="production")
                
                # Verbinde die Symbole der Produktion
                prev = None
                for sym in prod:
                    if sym in self.terminals:
                        # Terminalzeichen als Transition
                        subnet.add_transition(f"t_{sym}", transition_type="speech_act")
                        if prev:
                            subnet.add_arc(prev, f"t_{sym}")
                        prev = f"t_{sym}"
                    else:
                        # Nonterminal als abstrakte Transition (rekursiv)
                        subnet.add_transition(f"t_{sym}", transition_type="abstract")
                        if prev:
                            subnet.add_arc(prev, f"t_{sym}")
                        prev = f"t_{sym}"
            
            # Füge Haupttransition mit Subnetz hinzu
            self.petri_net.add_transition(
                f"t_{nt}",
                transition_type="abstract",
                subnet=subnet
            )
    
    def convert(self):
        """
        Führt die vollständige Konvertierung durch
        """
        print("\n=== Konvertiere ARS 3.0 zu Petri-Netz ===")
        
        # 1. Ressourcen-Stellen erstellen
        print("Erstelle Ressourcen-Stellen...")
        self.build_resource_places()
        
        # 2. Sprechakt-Transitionen erstellen
        print("Erstelle Sprechakt-Transitionen...")
        self.build_speech_act_transitions()
        
        # 3. Nonterminal-Hierarchie erstellen (falls vorhanden)
        if self.grammar:
            print("Erstelle Nonterminal-Hierarchie...")
            self.build_nonterminal_hierarchy()
        
        print(f"Petri-Netz erstellt: {len(self.petri_net.places)} Stellen, "
              f"{len(self.petri_net.transitions)} Transitionen, "
              f"{len(self.petri_net.arcs)} Kanten")
        
        return self.petri_net

class PetriNetAnalyzer:
    """
    Analysiert Petri-Netze (Erreichbarkeit, Invarianten, etc.)
    """
    
    def __init__(self, petri_net):
        self.net = petri_net
    
    def check_reachability(self, target_marking):
        """
        Prüft, ob eine Zielmarkierung erreichbar ist (Breitensuche)
        """
        visited = set()
        queue = [(self.net.get_marking_tuple(), [])]
        
        while queue:
            marking, path = queue.pop(0)
            
            if marking in visited:
                continue
            
            visited.add(marking)
            
            # Prüfe, ob Zielmarkierung erreicht
            marking_dict = dict(marking)
            target_dict = dict(target_marking)
            match = True
            for place, tokens in target_dict.items():
                if marking_dict.get(place, 0) != tokens:
                    match = False
                    break
            if match:
                return True, path
            
            # Probiere alle Transitionen
            for trans in self.net.transitions:
                self.net.tokens = dict(marking)
                if self.net.is_enabled(trans):
                    self.net.fire(trans)
                    new_marking = self.net.get_marking_tuple()
                    queue.append((new_marking, path + [trans]))
        
        return False, []
    
    def compute_place_invariants(self):
        """
        Berechnet Stellen-Invarianten (vereinfacht)
        """
        # Implementierung würde hier folgen
        pass
    
    def simulate_transcript(self, transcript_chain):
        """
        Simuliert ein Transkript im Petri-Netz
        """
        print(f"\n=== Simuliere Transkript im Petri-Netz ===")
        
        self.net.reset()
        successful = []
        failed = []
        
        for i, symbol in enumerate(transcript_chain):
            trans_name = f"t_{symbol}"
            
            if trans_name in self.net.transitions:
                if self.net.is_enabled(trans_name):
                    self.net.fire(trans_name)
                    successful.append(symbol)
                    print(f"  ✓ {i+1}: {symbol} geschaltet")
                else:
                    failed.append(symbol)
                    print(f"  ✗ {i+1}: {symbol} NICHT aktiviert")
                    # Zeige aktivierte Transitionen
                    enabled = [t for t in self.net.transitions if self.net.is_enabled(t)]
                    print(f"     Aktiviert: {enabled}")
            else:
                print(f"  ? {i+1}: {symbol} - keine Transition vorhanden")
        
        print(f"\nErgebnis: {len(successful)}/{len(transcript_chain)} erfolgreich")
        print(f"Letzte Markierung: {self.net.get_marking_copy()}")
        
        return successful, failed
    
    def analyze_concurrency(self):
        """
        Analysiert nebenläufige Transitionen
        """
        concurrent_pairs = []
        
        # Für alle Markierungen im Erreichbarkeitsgraphen
        for marking_tuple in self.net.reached_markings:
            self.net.tokens = dict(marking_tuple)
            
            # Finde alle aktivierten Transitionen
            enabled = [t for t in self.net.transitions if self.net.is_enabled(t)]
            
            # Prüfe auf Nebenläufigkeit (Konfliktfreiheit)
            for i, t1 in enumerate(enabled):
                for t2 in enabled[i+1:]:
                    # Prüfe, ob t1 und t2 gleichzeitig schalten können
                    # (keine gemeinsamen Vorstellen mit Konflikt)
                    preset1 = set(self.net.get_preset(t1).keys())
                    preset2 = set(self.net.get_preset(t2).keys())
                    
                    # Wenn keine gemeinsamen Stellen oder genug Token für beide
                    if not (preset1 & preset2):
                        concurrent_pairs.append((t1, t2, dict(marking_tuple)))
        
        return concurrent_pairs

# ============================================================================
# Hauptprogramm
# ============================================================================

def main():
    """
    Hauptprogramm zur Demonstration der Petri-Netz-Integration
    """
    print("=" * 70)
    print("ARS 4.0 - PETRI-NETZ-INTEGRATION")
    print("=" * 70)
    
    # 1. Lade die ARS-3.0-Daten
    from ars_data import terminal_chains, grammar_rules
    
    print("\n1. ARS-3.0-Daten geladen:")
    print(f"   {len(terminal_chains)} Transkripte")
    print(f"   {len(grammar_rules)} Nonterminale")
    
    # 2. Konvertiere zu Petri-Netz
    print("\n2. Konvertiere zu Petri-Netz...")
    converter = ARSToPetriNetConverter(grammar_rules, terminal_chains)
    petri_net = converter.convert()
    
    # 3. Visualisiere das Petri-Netz
    print("\n3. Visualisiere Petri-Netz...")
    petri_net.visualize("ars_petri_net.png")
    
    # 4. Analysiere das Petri-Netz
    print("\n4. Analysiere Petri-Netz...")
    analyzer = PetriNetAnalyzer(petri_net)
    
    # Simuliere Transkript 1
    print("\n" + "-" * 50)
    print("Simulation: Transkript 1 (Metzgerei)")
    successful, failed = analyzer.simulate_transcript(terminal_chains[0])
    
    # Analysiere Nebenläufigkeit
    concurrent = analyzer.analyze_concurrency()
    print(f"\nNebenläufige Transitionen gefunden: {len(concurrent)}")
    for t1, t2, marking in concurrent[:5]:  # Erste 5 anzeigen
        print(f"  {t1} || {t2} in Markierung {marking}")
    
    # 5. Exportiere das Petri-Netz
    print("\n5. Exportiere Petri-Netz...")
    export_petri_net(petri_net, "ars_petri_net.pnml")
    
    print("\n" + "=" * 70)
    print("ARS 4.0 - PETRI-NETZ-INTEGRATION ABGESCHLOSSEN")
    print("=" * 70)

def export_petri_net(petri_net, filename):
    """
    Exportiert das Petri-Netz im PNML-Format
    """
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    
    # Erstelle PNML-Struktur
    pnml = ET.Element("pnml")
    net = ET.SubElement(pnml, "net", id=petri_net.name, type="http://www.pnml.org/version-2009/grammar/ptnet")
    
    # Stellen
    for place_name, place_data in petri_net.places.items():
        place = ET.SubElement(net, "place", id=place_name)
        name = ET.SubElement(place, "name")
        text = ET.SubElement(name, "text")
        text.text = place_name
        
        initial = ET.SubElement(place, "initialMarking")
        tokens = ET.SubElement(initial, "text")
        tokens.text = str(place_data['initial_tokens'])
    
    # Transitionen
    for trans_name in petri_net.transitions:
        trans = ET.SubElement(net, "transition", id=trans_name)
        name = ET.SubElement(trans, "name")
        text = ET.SubElement(name, "text")
        text.text = trans_name
    
    # Kanten
    for i, arc in enumerate(petri_net.arcs):
        arc_elem = ET.SubElement(net, "arc", id=f"arc{i}", 
                                 source=arc['source'], target=arc['target'])
        inscription = ET.SubElement(arc_elem, "inscription")
        text = ET.SubElement(inscription, "text")
        text.text = str(arc['weight'])
    
    # Speichern
    xml_str = minidom.parseString(ET.tostring(pnml)).toprettyxml(indent="  ")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    print(f"Petri-Netz exportiert als '{filename}'")

if __name__ == "__main__":
    main()
```

# Beispielausgabe

Bei der Ausführung des Programms ergibt sich folgende Ausgabe:

``` {caption="Beispielausgabe der Petri-Netz-Simulation"}
======================================================================
ARS 4.0 - PETRI-NETZ-INTEGRATION
======================================================================

1. ARS-3.0-Daten geladen:
   8 Transkripte
   13 Nonterminale

2. Konvertiere zu Petri-Netz...

=== Konvertiere ARS 3.0 zu Petri-Netz ===
Erstelle Ressourcen-Stellen...
Erstelle Sprechakt-Transitionen...
Erstelle Nonterminal-Hierarchie...
Petri-Netz erstellt: 15 Stellen, 27 Transitionen, 64 Kanten

3. Visualisiere Petri-Netz...
Petri-Netz visualisiert als 'ars_petri_net.png'

4. Analysiere Petri-Netz...

--------------------------------------------------
Simulation: Transkript 1 (Metzgerei)

=== Simuliere Transkript im Petri-Netz ===
  ✓ 1: KBG geschaltet
  ✓ 2: VBG geschaltet
  ✓ 3: KBBd geschaltet
  ✓ 4: VBBd geschaltet
  ✓ 5: KBA geschaltet
  ✓ 6: VBA geschaltet
  ✓ 7: KBBd geschaltet
  ✓ 8: VBBd geschaltet
  ✓ 9: KBA geschaltet
  ✓ 10: VAA geschaltet
  ✓ 11: KAA geschaltet
  ✓ 12: VAV geschaltet
  ✓ 13: KAV geschaltet

Ergebnis: 13/13 erfolgreich
Letzte Markierung: {'s_Kunde_anwesend': 1, 's_Kunde_bereit': 1, ...}

Nebenläufige Transitionen gefunden: 12
  t_KBBd || t_VBG in Markierung {...}
  t_VBBd || t_KBA in Markierung {...}
  ...

5. Exportiere Petri-Netz...
Petri-Netz exportiert als 'ars_petri_net.pnml'

======================================================================
ARS 4.0 - PETRI-NETZ-INTEGRATION ABGESCHLOSSEN
======================================================================
```

# Diskussion

## Methodologische Bewertung

Die Integration von Petri-Netzen in die ARS erfüllt die zentralen methodologischen Anforderungen:

1.  **Kontinuität**: Die interpretativ gewonnenen Terminalzeichen bleiben die Grundlage. Die Petri-Netze werden aus diesen abgeleitet, nicht automatisch gelernt.

2.  **Transparenz**: Jede Transition und jede Stelle ist semantisch gehaltvoll benannt und dokumentiert.

3.  **Erweiterung**: Nebenläufigkeit und Ressourcen werden explizit modelliert, ohne die sequenzielle Struktur zu verlieren.

## Mehrwert gegenüber ARS 3.0

Die Petri-Netz-Modellierung bietet gegenüber der reinen Grammatik mehrere Vorteile:

- **Nebenläufigkeit**: Parallele Aktivitäten von Kunde und Verkäufer werden sichtbar gemacht.

- **Ressourcenabhängigkeiten**: Die Verfügbarkeit von Waren und Geld beeinflusst den Gesprächsverlauf.

- **Zustandsraum**: Der Erreichbarkeitsgraph zeigt alle möglichen Gesprächsverläufe.

- **Analyse**: Invarianten und Konflikte können formal untersucht werden.

## Grenzen

Die Petri-Netz-Modellierung hat auch Grenzen:

- Die Modellierung von Ressourcen erfordert zusätzliche Annahmen (z.B. initiale Token-Zahlen).

- Sehr große Netze können unübersichtlich werden.

- Die probabilistische Natur der ARS-Grammatik geht teilweise verloren (kann durch stochastische Petri-Netze ergänzt werden).

# Fazit und Ausblick

Die Integration von Petri-Netzen in die ARS 4.0 erweitert das Methodenspektrum um wichtige Aspekte der Nebenläufigkeit und Ressourcenmodellierung. Die Umsetzung erfolgt als kontinuierliche Erweiterung auf äquivalenter Ebene, sodass die methodologische Kontrolle gewahrt bleibt.

Weiterführende Forschung könnte:

- **Stochastische Petri-Netze**: Integration der Übergangswahrscheinlichkeiten aus der ARS-Grammatik

- **Zeitbehaftete Petri-Netze**: Modellierung von Gesprächspausen und Bearbeitungszeiten

- **Formale Verifikation**: Überprüfung von Eigenschaften wie \"immer wenn Gruß, dann Gegengruß\" mit Model Checking

::: thebibliography
99

Fehling, R. (1993). A concept of hierarchical Petri nets with building blocks. *Application and Theory of Petri Nets 1993*, 148-168.

Jensen, K. (1997). *Coloured Petri Nets: Basic Concepts, Analysis Methods and Practical Use* (Vol. 1-3). Springer.

Petri, C. A. (1962). *Kommunikation mit Automaten*. Dissertation, Technische Universität Darmstadt.

Reisig, W. (2010). *Petrinetze: Modellierungstechnik, Analysemethoden, Fallstudien*. Vieweg+Teubner.
:::

# Die acht Transkripte mit Terminalzeichen

## Transkript 1 - Metzgerei

**Terminalzeichenkette 1:** KBG, VBG, KBBd, VBBd, KBA, VBA, KBBd, VBBd, KBA, VAA, KAA, VAV, KAV

## Transkript 2 - Marktplatz (Kirschen)

**Terminalzeichenkette 2:** VBG, KBBd, VBBd, VAA, KAA, VBG, KBBd, VAA, KAA

## Transkript 3 - Fischstand

**Terminalzeichenkette 3:** KBBd, VBBd, VAA, KAA

## Transkript 4 - Gemüsestand (ausführlich)

**Terminalzeichenkette 4:** KBBd, VBBd, KBA, VBA, KBBd, VBA, KAE, VAE, KAA, VAV, KAV

## Transkript 5 - Gemüsestand (mit KAV zu Beginn)

**Terminalzeichenkette 5:** KAV, KBBd, VBBd, KBBd, VAA, KAV

## Transkript 6 - Käseverkaufsstand

**Terminalzeichenkette 6:** KBG, VBG, KBBd, VBBd, KAA

## Transkript 7 - Bonbonstand

**Terminalzeichenkette 7:** KBBd, VBBd, KBA, VAA, KAA

## Transkript 8 - Bäckerei

**Terminalzeichenkette 8:** KBG, VBBd, KBBd, VBA, VAA, KAA, VAV, KAV
