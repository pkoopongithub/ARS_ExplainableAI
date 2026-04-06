#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arsxai_ext_seminfo.py - SemInfo-Maximierung für ARSXAI
=======================================================

Generiert semantische Namen für Nonterminale mit Sentence-Transformers.

Verwendung:
    from arsxai_ext_seminfo import SemInfoMaximizer, SemInfoGrammarInducer
    
    inducer = SemInfoGrammarInducer(model_name='paraphrase-multilingual-MiniLM-L12-v2')
    inducer.train(chains)

Version: 10.0 (SemInfo-Erweiterung)
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from .ARSXAI import GrammarInducer

# Prüfe Verfügbarkeit
SEMINFO_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SEMINFO_AVAILABLE = True
except ImportError:
    pass


class SemInfoMaximizer:
    """
    Maximiert semantische Information der Nonterminale.
    Benötigt sentence-transformers.
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.model = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self.semantic_cache: Dict[Tuple, str] = {}
        
        if SEMINFO_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"✓ SemInfo-Modell geladen: {model_name}")
            except Exception as e:
                print(f"✗ SemInfo-Modell konnte nicht geladen werden: {e}")
    
    def compute_embeddings(self, symbols: List[str]) -> Dict[str, np.ndarray]:
        """
        Erstellt Embeddings für eine Liste von Symbolen.
        
        Returns:
            dict: Symbol -> Embedding-Vektor
        """
        if self.model is None:
            return {}
        
        to_compute = [s for s in symbols if s not in self.embeddings]
        
        if to_compute:
            try:
                new_embeddings = self.model.encode(to_compute)
                for sym, emb in zip(to_compute, new_embeddings):
                    self.embeddings[sym] = emb
            except Exception as e:
                print(f"Fehler bei Embedding-Berechnung: {e}")
        
        return self.embeddings
    
    def semantic_coherence(self, sequence: List[str]) -> float:
        """
        Misst semantische Kohärenz einer Sequenz.
        
        Returns:
            float: Kohärenz-Score (0-1), höher = zusammenhängender
        """
        if self.model is None or len(sequence) < 2:
            return 0.5
        
        self.compute_embeddings(sequence)
        
        similarities = []
        for i in range(len(sequence) - 1):
            sym1 = sequence[i]
            sym2 = sequence[i + 1]
            
            if sym1 in self.embeddings and sym2 in self.embeddings:
                emb1 = self.embeddings[sym1]
                emb2 = self.embeddings[sym2]
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                similarities.append(max(0, sim))
        
        if not similarities:
            return 0.5
        
        return float(np.mean(similarities))
    
    def suggest_name(self, sequence: List[str]) -> Optional[str]:
        """
        Generiert einen semantischen Namen für eine Sequenz.
        """
        if self.model is None or len(sequence) < 2:
            return None
        
        cache_key = tuple(sequence)
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
        
        self.compute_embeddings(sequence)
        valid_embs = [self.embeddings[s] for s in sequence if s in self.embeddings]
        
        if not valid_embs:
            return None
        
        coherence = self.semantic_coherence(sequence)
        
        if coherence > 0.8:
            name = "KOHÄRENT"
        elif coherence > 0.6:
            name = "ZUSAMMENHÄNGEND"
        elif coherence > 0.4:
            name = "GEMISCHT"
        else:
            name = "DIVERGENT"
        
        result = f"{name}_{len(sequence)}"
        self.semantic_cache[cache_key] = result
        return result
    
    def get_status_string(self) -> str:
        """Gibt Status des SemInfo-Maximizers zurück"""
        lines = []
        lines.append("🧠 **SemInfo-Maximizer Status**")
        lines.append("=" * 50)
        lines.append(f"Modell: {self.model_name}")
        lines.append(f"Verfügbar: {'✓' if self.model else '✗'}")
        lines.append(f"Gecachte Embeddings: {len(self.embeddings)}")
        lines.append(f"Semantische Namen im Cache: {len(self.semantic_cache)}")
        return "\n".join(lines)


class SemInfoGrammarInducer(GrammarInducer):
    """
    Erweitert GrammarInducer um semantische Namen.
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        super().__init__()
        self.seminfo = SemInfoMaximizer(model_name)
    
    def _generate_nonterminal_name(self, sequence: Tuple) -> str:
        """Generiert Namen mit semantischer Information"""
        semantic = self.seminfo.suggest_name(list(sequence))
        if semantic:
            return f"S_{semantic}"
        return super()._generate_nonterminal_name(sequence)
    
    def get_seminfo_status(self) -> str:
        """Gibt Status des SemInfo-Maximizers zurück"""
        return self.seminfo.get_status_string()