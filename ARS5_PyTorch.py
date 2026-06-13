"""
ARS 5.0 - PyTorch Implementation
Empirical Grammar of Market Conversations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# ============================================================================
# 1. SYMBOLIC COMPONENT: Grammar Definition
# ============================================================================

class ARSGrammar:
    """Symbolic grammar component - the structural knowledge."""
    
    # Terminal symbols
    SYMBOLS = ['KBG', 'VBG', 'KBBd', 'VBBd', 'KBA', 'VBA', 
               'KAE', 'VAE', 'KAA', 'VAA', 'KAV', 'VAV']
    
    SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOLS)}
    IDX_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}
    
    def __init__(self):
        # Initial transition probabilities (will be learned)
        # Start with uniform or empirical estimates
        self.transition_probs = torch.ones(len(self.SYMBOLS), len(self.SYMBOLS)) / len(self.SYMBOLS)
        
        # Constitutive rules (hard constraints)
        self.constitutive_rules = self._init_constitutive_rules()
        
    def _init_constitutive_rules(self) -> Dict[Tuple[int, int], bool]:
        """Initialize hard constraints that cannot be violated."""
        rules = {}
        
        # A greeting must be reciprocated (unless skipped)
        rules[(self.SYMBOL_TO_IDX['KBG'], self.SYMBOL_TO_IDX['VBG'])] = True
        rules[(self.SYMBOL_TO_IDX['VBG'], self.SYMBOL_TO_IDX['KBBd'])] = True
        
        # Customer inquiry must be answered
        rules[(self.SYMBOL_TO_IDX['KAE'], self.SYMBOL_TO_IDX['VAE'])] = True
        
        # Farewells are reciprocal
        rules[(self.SYMBOL_TO_IDX['KAV'], self.SYMBOL_TO_IDX['VAV'])] = True
        rules[(self.SYMBOL_TO_IDX['VAV'], self.SYMBOL_TO_IDX['KAV'])] = True
        
        return rules
    
    def is_valid_transition(self, from_idx: int, to_idx: int) -> bool:
        """Check if a transition violates a constitutive rule."""
        if (from_idx, to_idx) in self.constitutive_rules:
            return self.constitutive_rules[(from_idx, to_idx)]
        return True
    
    def update_probabilities(self, counts: torch.Tensor):
        """Update symbolic probabilities based on observed counts."""
        # Renormalize each row
        row_sums = counts.sum(dim=1, keepdim=True)
        self.transition_probs = counts / (row_sums + 1e-10)
        # Add small epsilon for unseen transitions
        self.transition_probs = (self.transition_probs + 1e-6) / (1e-6 * len(self.SYMBOLS) + 1)
    
    def get_prob(self, from_idx: int, to_idx: int) -> float:
        """Get symbolic probability of a transition."""
        return self.transition_probs[from_idx, to_idx].item()


# ============================================================================
# 2. NEURAL COMPONENT: Transition Network
# ============================================================================

class ARSNeuralTransitionNetwork(nn.Module):
    """
    Neural network for learning transition probabilities.
    System 1: Fast, pattern-based, sub-symbolic.
    """
    
    def __init__(self, n_symbols: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.n_symbols = n_symbols
        
        # Architecture
        self.fc1 = nn.Linear(n_symbols, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, n_symbols)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: One-hot encoded current symbol (batch_size, n_symbols)
            
        Returns:
            Probability distribution over next symbols (batch_size, n_symbols)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)
    
    def predict_next(self, symbol_idx: int) -> np.ndarray:
        """Predict probability distribution for next symbol."""
        x = torch.zeros(1, self.n_symbols)
        x[0, symbol_idx] = 1.0
        with torch.no_grad():
            probs = self.forward(x)
        return probs.numpy()[0]


# ============================================================================
# 3. HYBRID NEURO-SYMBOLIC SYSTEM
# ============================================================================

class ARSNeuroSymbolicSystem:
    """
    ARS 5.0: Dual-dynamics architecture.
    - Neural component: learns probabilities from data (statistical plasticity)
    - Symbolic component: maintains structural rules (structural stability)
    """
    
    def __init__(self, learning_rate: float = 0.001):
        self.neural_network = ARSNeuralTransitionNetwork()
        self.symbolic_grammar = ARSGrammar()
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=learning_rate)
        
        # Training tracking
        self.counts = torch.zeros(len(ARSGrammar.SYMBOLS), len(ARSGrammar.SYMBOLS))
        self.loss_history = []
        
    def train_on_transition(self, from_sym: str, to_sym: str):
        """
        Train on a single observed transition.
        
        This implements the dual update loop:
        1. Update symbolic counts
        2. Update neural network via backpropagation
        3. Enforce constitutive rules
        """
        from_idx = ARSGrammar.SYMBOL_TO_IDX[from_sym]
        to_idx = ARSGrammar.SYMBOL_TO_IDX[to_sym]
        
        # ===== Symbolic update (fast, counting-based) =====
        if self.symbolic_grammar.is_valid_transition(from_idx, to_idx):
            self.counts[from_idx, to_idx] += 1
            self.symbolic_grammar.update_probabilities(self.counts)
        
        # ===== Neural update (slow, gradient-based) =====
        # Prepare input
        x = torch.zeros(1, len(ARSGrammar.SYMBOLS))
        x[0, from_idx] = 1.0
        
        # Target distribution (from symbolic component as teacher)
        target = self.symbolic_grammar.transition_probs[from_idx].clone()
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.neural_network(x)
        
        # Loss: KL divergence between neural prediction and symbolic probabilities
        # This aligns the neural network with the symbolic component
        loss = F.kl_div(output.log(), target.unsqueeze(0), reduction='batchmean')
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def train_on_corpus(self, corpus: List[List[str]], epochs: int = 10):
        """Train on the entire corpus (8 transcripts)."""
        print(f"Training on {len(corpus)} transcripts for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_transitions = 0
            
            for chain in corpus:
                for i in range(len(chain) - 1):
                    loss = self.train_on_transition(chain[i], chain[i + 1])
                    epoch_loss += loss
                    n_transitions += 1
            
            avg_loss = epoch_loss / n_transitions
            print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.6f}")
    
    def predict_next(self, from_sym: str) -> Dict[str, float]:
        """Predict next symbol distribution."""
        from_idx = ARSGrammar.SYMBOL_TO_IDX[from_sym]
        
        # Neural prediction
        neural_probs = self.neural_network.predict_next(from_idx)
        
        # Symbolic probability
        symbolic_probs = self.symbolic_grammar.transition_probs[from_idx].numpy()
        
        # Combined prediction (weighted average)
        combined = 0.5 * neural_probs + 0.5 * symbolic_probs
        
        return {ARSGrammar.IDX_TO_SYMBOL[i]: combined[i] for i in range(len(ARSGrammar.SYMBOLS))}
    
    def generate_sequence(self, max_len: int = 20, start_sym: str = 'KBG') -> List[str]:
        """Generate a well-formed sequence."""
        sequence = [start_sym]
        
        for _ in range(max_len - 1):
            probs = self.predict_next(sequence[-1])
            # Filter invalid transitions (constitutive rules)
            valid_symbols = []
            valid_probs = []
            for sym, prob in probs.items():
                to_idx = ARSGrammar.SYMBOL_TO_IDX[sym]
                from_idx = ARSGrammar.SYMBOL_TO_IDX[sequence[-1]]
                if self.symbolic_grammar.is_valid_transition(from_idx, to_idx):
                    valid_symbols.append(sym)
                    valid_probs.append(prob)
            
            if not valid_symbols:
                break
            
            # Normalize and sample
            valid_probs = np.array(valid_probs) / np.sum(valid_probs)
            next_sym = np.random.choice(valid_symbols, p=valid_probs)
            sequence.append(next_sym)
            
            if next_sym in ['KAV', 'VAV']:  # End of conversation
                break
        
        return sequence
    
    def explain_transition(self, from_sym: str, to_sym: str) -> Dict:
        """Explain why a transition is valid/invalid."""
        from_idx = ARSGrammar.SYMBOL_TO_IDX[from_sym]
        to_idx = ARSGrammar.SYMBOL_TO_IDX[to_sym]
        
        return {
            'transition': f"{from_sym} → {to_sym}",
            'valid_by_constitutive_rule': self.symbolic_grammar.is_valid_transition(from_idx, to_idx),
            'neural_probability': self.neural_network.predict_next(from_idx)[to_idx],
            'symbolic_probability': self.symbolic_grammar.get_prob(from_idx, to_idx),
            'count': self.counts[from_idx, to_idx].item(),
            'explanation': self._generate_explanation(from_sym, to_sym)
        }
    
    def _generate_explanation(self, from_sym: str, to_sym: str) -> str:
        """Generate human-readable explanation."""
        explanations = {
            ('KBG', 'VBG'): "Customer greeting is normally followed by seller greeting.",
            ('KBG', 'VBBd'): "Customer greeting can be followed directly by seller inquiry (skip).",
            ('VBA', 'KBBd'): "Seller reaction leads to additional customer need (upselling).",
            ('KAE', 'VAE'): "Customer inquiry must be answered by seller information (constitutive).",
            ('KAV', 'VAV'): "Farewells are always reciprocated (constitutive).",
            ('KAA', 'VBG'): "Conversation can restart after completion (new customer)."
        }
        return explanations.get((from_sym, to_sym), "Standard transition in sales conversation.")


# ============================================================================
# 4. EMPIRICAL CORPUS DATA
# ============================================================================

# The eight transcripts as terminal symbol chains
EMPIRICAL_CHAINS = [
    # T1: Butcher shop
    ['KBG', 'VBG', 'KBBd', 'VBBd', 'KBA', 'VBA', 'KBBd', 'VBBd', 'KBA', 'VAA', 'KAA', 'VAV', 'KAV'],
    # T2: Cherry stall
    ['VBG', 'KBBd', 'VBBd', 'VAA', 'KAA', 'VBG', 'KBBd', 'VAA', 'KAA'],
    # T3: Fish stall
    ['KBBd', 'VBBd', 'VAA', 'KAA'],
    # T4: Vegetable stall
    ['KBBd', 'VBBd', 'KBA', 'VBA', 'KBBd', 'VBA', 'KAE', 'VAE', 'KAA', 'VAV', 'KAV'],
    # T5: Vegetable stall (new customer)
    ['KAV', 'KBBd', 'VBBd', 'KBBd', 'VAA', 'KAV'],
    # T6: Cheese stall
    ['KBG', 'VBG', 'KBBd', 'VBBd', 'KAA'],
    # T7: Candy stall
    ['KBBd', 'VBBd', 'KBA', 'VAA', 'KAA'],
    # T8: Bakery
    ['KBG', 'VBBd', 'KBBd', 'VBA', 'VAA', 'KAA', 'VAV', 'KAV']
]


# ============================================================================
# 5. DEMONSTRATION
# ============================================================================

def main():
    print("=" * 70)
    print("ARS 5.0 - PyTorch Implementation")
    print("The Empirical Grammar of Market Conversations")
    print("=" * 70)
    
    # Create the neuro-symbolic system
    system = ARSNeuroSymbolicSystem()
    
    # Train on the corpus
    print("\n--- Training ---")
    system.train_on_corpus(EMPIRICAL_CHAINS, epochs=20)
    
    # Show learned transition probabilities
    print("\n--- Learned Transition Probabilities (sample) ---")
    for from_sym in ['KBG', 'KBBd', 'VBA', 'KAA']:
        probs = system.predict_next(from_sym)
        top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n{from_sym} → {', '.join([f'{s}: {p:.3f}' for s, p in top_probs])}")
    
    # Generate example sequences
    print("\n--- Generated Sequences ---")
    for i in range(5):
        seq = system.generate_sequence(max_len=15)
        print(f"Seq {i+1}: {' → '.join(seq)}")
    
    # Explanation examples
    print("\n--- Explanations ---")
    test_transitions = [('KBG', 'VBG'), ('KBG', 'VBBd'), ('VBA', 'KBBd'), ('KAE', 'VAE')]
    for from_sym, to_sym in test_transitions:
        explanation = system.explain_transition(from_sym, to_sym)
        print(f"\n{explanation['transition']}")
        print(f"  {explanation['explanation']}")
        print(f"  Probability: {explanation['neural_probability']:.3f}")
    
    return system


if __name__ == "__main__":
    system = main()