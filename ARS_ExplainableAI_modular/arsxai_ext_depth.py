#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arsxai_ext_depth.py - Depth-Bounded PCFG Erweiterung für ARSXAI
================================================================

Erweitert GrammarInducer um Tiefenbeschränkung.

Verwendung:
    from arsxai_ext_depth import DepthBoundedGrammarInducer
    
    inducer = DepthBoundedGrammarInducer(max_depth=5)
    inducer.train(chains)

Version: 10.0 (Depth-Bounded Erweiterung)
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from collections import Counter, defaultdict
from .ARSXAI import GrammarInducer, MethodologicalReflection, XAIModel


class DepthBoundedGrammarInducer(GrammarInducer):
    """
    Erweitert GrammarInducer um Tiefenbeschränkung.
    
    Attribute:
        max_depth: Maximale Hierarchietiefe (default 5)
        depth_map: nonterminal -> Tiefe
    """
    
    def __init__(self, max_depth: int = 5):
        super().__init__()
        self.max_depth = max_depth
        self.depth_map: Dict[str, int] = {}  # nonterminal -> Tiefe
        self.skipped_patterns: List[Dict] = []
    
    def train(self, chains: List[List[str]], max_iterations: int = 20) -> List[List[str]]:
        """Induziert Grammatik mit Tiefenbeschränkung"""
        self.chains = [list(chain) for chain in chains]
        
        # Sammle alle Terminale
        all_symbols = set()
        for chain in chains:
            for symbol in chain:
                all_symbols.add(symbol)
        self.terminals = all_symbols
        
        current_chains = [list(chain) for chain in chains]
        iteration = 0
        rule_counter = 1
        
        self.rules = {}
        self.nonterminals = set()
        self.symbol_to_nonterminals = defaultdict(set)
        self.compression_history = []
        self.hierarchy_levels = {}
        self.depth_map = {}
        self.skipped_patterns = []
        
        while iteration < max_iterations:
            best_seq = self._find_best_repetition(current_chains)
            
            if best_seq is None:
                break
            
            # Prüfe Tiefenbeschränkung
            depth = self._estimate_depth(best_seq)
            if depth > self.max_depth:
                self._mark_as_skipped(best_seq, depth)
                continue
            
            new_nonterminal = self._generate_nonterminal_name(best_seq, depth)
            base_name = new_nonterminal
            while new_nonterminal in self.nonterminals:
                new_nonterminal = f"{base_name}_{rule_counter}"
                rule_counter += 1
            
            rationale = self._generate_rationale(best_seq)
            self.reflection.log_interpretation(best_seq, new_nonterminal, rationale)
            
            self.rules[new_nonterminal] = [(list(best_seq), 1.0)]
            self.nonterminals.add(new_nonterminal)
            self.hierarchy_levels[new_nonterminal] = iteration
            self.depth_map[new_nonterminal] = depth
            
            # Aktualisiere symbol_to_nonterminals
            for symbol in best_seq:
                self.symbol_to_nonterminals[symbol].add(new_nonterminal)
            
            # Zähle Vorkommen
            occurrences = 0
            for chain in current_chains:
                for i in range(len(chain) - len(best_seq) + 1):
                    if tuple(chain[i:i+len(best_seq)]) == best_seq:
                        occurrences += 1
            
            self.compression_history.append({
                'iteration': iteration,
                'sequence': best_seq,
                'new_symbol': new_nonterminal,
                'occurrences': occurrences,
                'depth': depth
            })
            
            current_chains = self._compress_sequences(current_chains, best_seq, new_nonterminal)
            iteration += 1
            self.iteration_count = iteration
            
            if self._all_chains_identical(current_chains):
                if current_chains and current_chains[0]:
                    unique_symbol = current_chains[0][0]
                    if unique_symbol in self.rules:
                        self.start_symbol = unique_symbol
                    else:
                        self.start_symbol = self._find_top_level_nonterminal()
                    break
        
        if self.start_symbol is None:
            self.start_symbol = self._find_top_level_nonterminal()
        
        # Terminale aktualisieren
        all_symbols = set()
        for chain in self.chains:
            for sym in chain:
                all_symbols.add(sym)
        self.terminals = all_symbols - self.nonterminals
        
        # Berechne Wahrscheinlichkeiten
        self._calculate_probabilities()
        
        self.induction_done = True
        self.trained = True
        self.confidence = self._calculate_confidence()
        
        return current_chains
    
    def _generate_nonterminal_name(self, sequence: Tuple, depth: int) -> str:
        """Generiert Namen mit Tiefeninformation"""
        first = sequence[0] if sequence else "X"
        last = sequence[-1] if sequence else "X"
        return f"P_{first}_{last}_{len(sequence)}_d{depth}"
    
    def _estimate_depth(self, sequence: Tuple) -> int:
        """Schätzt benötigte Tiefe für eine Sequenz"""
        max_depth = 0
        for sym in sequence:
            if sym in self.depth_map:
                max_depth = max(max_depth, self.depth_map[sym] + 1)
        return max_depth
    
    def _mark_as_skipped(self, sequence: Tuple, depth: int):
        """Markiert ein Muster als übersprungen (zu tief)"""
        self.skipped_patterns.append({
            'sequence': sequence,
            'depth': depth
        })
    
    def get_depth_statistics(self) -> str:
        """Gibt Tiefenstatistik aus"""
        lines = []
        lines.append("📊 **TIEFENSTATISTIK**")
        lines.append("=" * 60)
        
        if not self.depth_map:
            lines.append("Keine Tiefeninformationen verfügbar.")
            return "\n".join(lines)
        
        # Verteilung der Tiefen
        depth_counts = Counter(self.depth_map.values())
        lines.append("\nTiefenverteilung:")
        for depth in sorted(depth_counts.keys()):
            count = depth_counts[depth]
            percentage = (count / len(self.depth_map)) * 100
            lines.append(f"  Tiefe {depth}: {count} Nonterminale ({percentage:.1f}%)")
        
        # Nonterminale nach Tiefe
        lines.append("\nNonterminale nach Tiefe:")
        for depth in sorted(set(self.depth_map.values())):
            nts = [nt for nt, d in self.depth_map.items() if d == depth]
            lines.append(f"  Tiefe {depth}: {', '.join(nts[:5])}" + 
                        (f" ... und {len(nts)-5} weitere" if len(nts) > 5 else ""))
        
        # Übersprungene Muster
        if self.skipped_patterns:
            lines.append(f"\n⚠️ Übersprungene Muster (Tiefe > {self.max_depth}):")
            for pattern in self.skipped_patterns[:5]:
                seq_str = ' → '.join(pattern['sequence'])
                lines.append(f"  • {seq_str} (Tiefe {pattern['depth']})")
        
        return "\n".join(lines)