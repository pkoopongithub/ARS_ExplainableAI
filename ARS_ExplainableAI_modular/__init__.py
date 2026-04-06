"""
ARSXAI - Algorithmic Recursive Sequence Analysis with Explainable AI
====================================================================

Modulare Architektur:
- ARSXAI.py: Vollständige Referenzimplementierung (lesbar, linear)
- arsxai_ext_*.py: Optionale Erweiterungen

Version: 10.0 (Modular Synthese)
"""

from .ARSXAI import (
    # Basis-Klassen
    XAIModel,
    GrammarInducer,
    MethodologicalReflection,
    NaturalLanguageExplainer,
    
    # GUI
    ARSXAIBaseGUI,
    
    # Hilfsklassen
    DataValidator,
    DerivationVisualizer,
    MultiFormatExporter,
    XAIModelManager,
    PlotThread,
    
    # Weitere Modelle
    ARS20,
    ARSHiddenMarkovModel,
    ARSCRFModel,
    ChainGenerator,
    ARSPetriNet,
    
    # Konstanten
    MODULE_STATUS,
    GRAPHVIZ_AVAILABLE
)

# Optionale Erweiterungen (nur wenn importiert)
__all__ = [
    'XAIModel',
    'GrammarInducer', 
    'ARSXAIBaseGUI',
    'MODULE_STATUS',
    'GRAPHVIZ_AVAILABLE'
]