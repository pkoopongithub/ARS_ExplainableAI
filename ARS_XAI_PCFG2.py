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
