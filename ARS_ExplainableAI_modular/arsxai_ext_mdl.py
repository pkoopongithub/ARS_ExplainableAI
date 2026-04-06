#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arsxai_ext_mdl.py - MDL-Optimierung für ARSXAI
================================================

Implementiert Minimum Description Length Prinzip für Grammatikbewertung.

Verwendung:
    from arsxai_ext_mdl import MDLOptimizer
    
    optimizer = MDLOptimizer()
    ratio = optimizer.calculate_compression_ratio(chains, grammar)

Version: 10.0 (MDL-Erweiterung)
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from .ARSXAI import GrammarInducer


class MDLOptimizer:
    """
    Implementiert Minimum Description Length Prinzip.
    Bewertet Grammatiken nach Kompressionsrate.
    """
    
    def __init__(self):
        self.compression_history: List[float] = []
    
    def calculate_compression_ratio(self, original_chains: List[List[str]], grammar: GrammarInducer) -> float:
        """
        Berechnet Kompressionsrate: 1 - (komprimierte_Länge / originale_Länge)
        
        Returns:
            float: Kompressionsrate (0-1), höher ist besser
        """
        if not grammar.rules:
            return 0.0
        
        # Originale Länge (Anzahl der Terminalsymbole)
        original_length = sum(len(chain) for chain in original_chains)
        
        # Komprimierte Länge (Nonterminale zählen als 1)
        compressed_length = 0
        for chain in original_chains:
            compressed = self._compress_chain(chain, grammar)
            compressed_length += len(compressed)
        
        ratio = 1 - (compressed_length / original_length) if original_length > 0 else 0
        self.compression_history.append(ratio)
        return round(ratio, 3)
    
    def _compress_chain(self, chain: List[str], grammar: GrammarInducer, max_iter: int = 10) -> List[str]:
        """Wendet Grammatikregeln iterativ an, um Kette zu komprimieren"""
        current = list(chain)
        for _ in range(max_iter):
            changed = False
            # Suche nach anwendbaren Regeln (rückwärts, längste zuerst)
            for nt, productions in sorted(grammar.rules.items(), 
                                         key=lambda x: -len(x[1][0][0] if x[1] else 0)):
                for prod, _ in productions:
                    prod_len = len(prod)
                    i = 0
                    while i <= len(current) - prod_len:
                        if current[i:i+prod_len] == prod:
                            current[i:i+prod_len] = [nt]
                            changed = True
                            break
                        i += 1
            if not changed:
                break
        return current
    
    def compare_grammars(self, grammar1: GrammarInducer, grammar2: GrammarInducer, 
                         chains: List[List[str]]) -> Dict:
        """
        Vergleicht zwei Grammatiken nach MDL-Prinzip.
        
        Returns:
            dict: Vergleichsergebnisse mit Scores
        """
        ratio1 = self.calculate_compression_ratio(chains, grammar1)
        ratio2 = self.calculate_compression_ratio(chains, grammar2)
        
        # Grammatikkomplexität (Anzahl der Regeln)
        complexity1 = len(grammar1.rules) if hasattr(grammar1, 'rules') else 0
        complexity2 = len(grammar2.rules) if hasattr(grammar2, 'rules') else 0
        
        # MDL-Score: Kompression - Komplexitätsstrafe
        mdl1 = ratio1 - (complexity1 * 0.01)
        mdl2 = ratio2 - (complexity2 * 0.01)
        
        return {
            'grammar1': {
                'compression_ratio': ratio1,
                'complexity': complexity1,
                'mdl_score': mdl1
            },
            'grammar2': {
                'compression_ratio': ratio2,
                'complexity': complexity2,
                'mdl_score': mdl2
            },
            'better': 'grammar1' if mdl1 > mdl2 else 'grammar2' if mdl2 > mdl1 else 'equal'
        }
    
    def optimal_cutoff(self, compression_gains: List[float]) -> int:
        """
        Findet natürliche Grenze für Iterationen (Elbow-Methode).
        
        Returns:
            int: Optimale Anzahl von Iterationen
        """
        if len(compression_gains) < 3:
            return len(compression_gains)
        
        gains = np.array(compression_gains)
        if len(gains) < 2:
            return len(gains)
        
        diffs = np.diff(gains)
        if len(diffs) < 1:
            return len(gains)
        
        threshold = np.mean(diffs) * 0.5
        for i, diff in enumerate(diffs):
            if diff < threshold:
                return i + 1
        
        return len(gains)
    
    def get_statistics_string(self) -> str:
        """Gibt Statistik der Kompression als String zurück"""
        if not self.compression_history:
            return "Keine Kompressionsdaten vorhanden."
        
        lines = []
        lines.append("📊 **MDL-Kompressionsstatistik**")
        lines.append("=" * 50)
        
        for i, ratio in enumerate(self.compression_history):
            lines.append(f"Iteration {i+1}: {ratio:.1%} Kompression")
        
        optimal = self.optimal_cutoff(self.compression_history)
        lines.append(f"\n✅ Optimale Iterationen: {optimal}")
        
        return "\n".join(lines)


class MDLGrammarInducer(GrammarInducer):
    """
    Erweitert GrammarInducer um MDL-Optimierung für Musterauswahl.
    """
    
    def __init__(self):
        super().__init__()
        self.mdl_optimizer = MDLOptimizer()
        self.mdl_scores: Dict[str, float] = {}
        self.compression_gains: List[float] = []
    
    def _mdl_gain(self, sequence: Tuple, count: int) -> float:
        """
        Berechnet MDL-Ersparnis (Kompression).
        
        Returns:
            float: Ersparnis (positiv = lohnend)
        """
        original_cost = len(sequence) * count
        compressed_cost = count + len(sequence)
        gain = original_cost - compressed_cost
        return gain / (original_cost + 1)
    
    def _find_best_repetition(self, chains: List[List[str]], min_length: int = 2, max_length: int = 5) -> Tuple:
        """MDL-optimierte Musterauswahl"""
        sequence_counter = Counter()
        
        for chain in chains:
            max_len = min(max_length, len(chain))
            for length in range(min_length, max_len + 1):
                for i in range(len(chain) - length + 1):
                    seq = tuple(chain[i:i+length])
                    sequence_counter[seq] += 1
        
        repeated = {seq: count for seq, count in sequence_counter.items() if count >= 2}
        if not repeated:
            return None
        
        # MDL-basierte Bewertung
        best_score = -float('inf')
        best_seq = None
        
        for seq, count in repeated.items():
            gain = self._mdl_gain(seq, count)
            if gain > best_score:
                best_score = gain
                best_seq = seq
        
        return best_seq
    
    def get_mdl_statistics(self) -> str:
        """Gibt MDL-Statistik aus"""
        return self.mdl_optimizer.get_statistics_string()
    
    def compare_with_standard(self, standard_grammar: GrammarInducer, 
                              chains: List[List[str]]) -> Dict:
        """Vergleicht diese Grammatik mit der Standard-Grammatik"""
        return self.mdl_optimizer.compare_grammars(self, standard_grammar, chains)