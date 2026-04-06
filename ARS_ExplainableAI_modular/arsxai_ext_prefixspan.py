#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arsxai_ext_prefixspan.py - PrefixSpan Erweiterung für ARSXAI
=============================================================

Ermöglicht Mustererkennung für große Korpora mit PrefixSpan.

Verwendung:
    from arsxai_ext_prefixspan import PrefixSpanGrammarInducer
    
    inducer = PrefixSpanGrammarInducer(min_support=2)
    inducer.train(large_chains)  # >1000 Ketten

Version: 10.0 (PrefixSpan-Erweiterung)
"""

from typing import List, Dict, Tuple, Optional, Any
from .ARSXAI import GrammarInducer

# Prüfe Verfügbarkeit
PREFIXSPAN_AVAILABLE = False

try:
    from prefixspan import PrefixSpan
    PREFIXSPAN_AVAILABLE = True
except ImportError:
    pass


class PrefixSpanGrammarInducer(GrammarInducer):
    """
    Erweitert GrammarInducer um PrefixSpan für große Datenmengen.
    """
    
    def __init__(self, min_support: int = 2):
        super().__init__()
        self.min_support = min_support
        self.prefixspan_available = PREFIXSPAN_AVAILABLE
    
    def train(self, chains: List[List[str]], max_iterations: int = 20) -> List[List[str]]:
        """Induziert Grammatik mit PrefixSpan-Unterstützung"""
        if not self.prefixspan_available:
            print("⚠️ PrefixSpan nicht verfügbar. Falle auf Standard zurück.")
            return super().train(chains, max_iterations)
        
        # Für große Daten: PrefixSpan verwenden
        if len(chains) > 1000:
            return self._train_with_prefixspan(chains, max_iterations)
        else:
            return super().train(chains, max_iterations)
    
    def _train_with_prefixspan(self, chains: List[List[str]], max_iterations: int) -> List[List[str]]:
        """Trainiert mit PrefixSpan für große Daten"""
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
        
        while iteration < max_iterations:
            best_seq = self._find_with_prefixspan(current_chains)
            
            if best_seq is None:
                break
            
            new_nonterminal = self._generate_nonterminal_name(best_seq)
            base_name = new_nonterminal
            while new_nonterminal in self.nonterminals:
                new_nonterminal = f"{base_name}_{rule_counter}"
                rule_counter += 1
            
            rationale = self._generate_rationale(best_seq)
            self.reflection.log_interpretation(best_seq, new_nonterminal, rationale)
            
            self.rules[new_nonterminal] = [(list(best_seq), 1.0)]
            self.nonterminals.add(new_nonterminal)
            self.hierarchy_levels[new_nonterminal] = iteration
            
            for symbol in best_seq:
                self.symbol_to_nonterminals[symbol].add(new_nonterminal)
            
            occurrences = 0
            for chain in current_chains:
                for i in range(len(chain) - len(best_seq) + 1):
                    if tuple(chain[i:i+len(best_seq)]) == best_seq:
                        occurrences += 1
            
            self.compression_history.append({
                'iteration': iteration,
                'sequence': best_seq,
                'new_symbol': new_nonterminal,
                'occurrences': occurrences
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
        
        all_symbols = set()
        for chain in self.chains:
            for sym in chain:
                all_symbols.add(sym)
        self.terminals = all_symbols - self.nonterminals
        
        self._calculate_probabilities()
        self.induction_done = True
        self.trained = True
        self.confidence = self._calculate_confidence()
        
        return current_chains
    
    def _find_with_prefixspan(self, chains: List[List[str]]) -> Optional[Tuple]:
        """Findet Muster mit PrefixSpan"""
        if not self.prefixspan_available:
            return None
        
        try:
            ps = PrefixSpan(chains)
            patterns = ps.frequent(self.min_support)
            
            valid_patterns = [(seq, support) for seq, support in patterns if len(seq) >= 2]
            
            if not valid_patterns:
                return None
            
            # Bestes Muster nach Länge * Support
            best = max(valid_patterns, key=lambda x: len(x[0]) * x[1])
            return tuple(best[0])
                
        except Exception as e:
            print(f"PrefixSpan-Fehler: {e}")
            return None