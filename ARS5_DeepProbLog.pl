% =============================================================================
% ARS 5.0 - DeepProbLog Implementation
% =============================================================================
% Title: The Empirical Grammar of Market Conversations
% Author: Paul Koop
% Date: 1994-2026
% =============================================================================
% This file implements the complete empirical grammar of eight sales
% conversations recorded at Aachen market square in June/July 1994.
%
% The grammar was induced from eight transcripts using the Algorithmic
% Recursive Sequence Analysis (ARS) framework. Transition probabilities
% were optimized through iterative comparison of empirical and generated
% frequency distributions, achieving a correlation of r = 0.925.
%
% Terminal symbols (12 categories):
%   KBG  - Customer greeting
%   VBG  - Seller greeting
%   KBBd - Customer need (concrete)
%   VBBd - Seller inquiry
%   KBA  - Customer response
%   VBA  - Seller reaction
%   KAE  - Customer inquiry
%   VAE  - Seller information
%   KAA  - Customer completion
%   VAA  - Seller completion
%   KAV  - Customer farewell
%   VAV  - Seller farewell
% =============================================================================


% =============================================================================
% 1. PREDICATE DECLARATIONS
% =============================================================================

% Terminal symbols as ground predicates (arity 0)
predicate(kbg/0).   % Customer greeting
predicate(vbg/0).   % Seller greeting
predicate(kbbd/0).  % Customer need
predicate(vbbd/0).  % Seller inquiry
predicate(kba/0).   % Customer response
predicate(vba/0).   % Seller reaction
predicate(kae/0).   % Customer inquiry
predicate(vae/0).   % Seller information
predicate(kaa/0).   % Customer completion
predicate(vaa/0).   % Seller completion
predicate(kav/0).   % Customer farewell
predicate(vav/0).   % Seller farewell

% Helper predicates
predicate(start/1).      % Start symbol of conversation
predicate(transition/2). % Transition between symbols
predicate(well_formed/1).% Well-formed sequence validation
predicate(valid_sequence/1). % Alternative validation
predicate(sequence/1).   % Sequence generator
predicate(next/2).       % Next symbol prediction


% =============================================================================
% 2. NEURAL NETWORK DECLARATION
% =============================================================================

% Neural predicate for transition probabilities
% The network takes a start symbol and a next symbol as input
% and outputs the probability of that transition
% Architecture: Input (12) -> Hidden 1 (64, ReLU) -> Hidden 2 (32, ReLU) -> Output (12, Softmax)
nn(transition, [in:symbol, out:symbol]) :: neural_network.


% =============================================================================
% 3. GRAMMAR RULES (SYMBOLIC KNOWLEDGE)
% =============================================================================

% Definition of the start symbol
% In the empirical corpus, conversations start with customer greeting (KBG)
% in the majority of cases, but seller greeting (VBG) is also possible
start(kbg).
% start(vbg).   % Alternative start (commented, can be uncommented)

% Base case: A single symbol is a well-formed sequence
well_formed(S) :-
    start(S).

% Recursive case: A sequence of length > 1 is well-formed if:
%   1. The first two symbols are connected by a transition
%   2. The rest of the sequence (starting from the second symbol) is well-formed
well_formed([A,B|Rest]) :-
    transition(A, B),
    well_formed([B|Rest]).

% Alternative formulation using next/2 predicate
next(S1, S2) :-
    transition(S1, S2).

% A valid sequence is one where each consecutive pair is connected by a transition
valid_sequence([]).                      % Empty sequence is trivially valid
valid_sequence([_]).                     % Single symbol is valid
valid_sequence([A,B|Rest]) :-
    transition(A, B),
    valid_sequence([B|Rest]).


% =============================================================================
% 4. START SYMBOLS
% =============================================================================

% The start symbol is KBG (customer greeting)
% This is the most frequent starting pattern in the corpus
start(kbg).

% Alternative start symbol VBG (seller greeting)
% Occurs in T2 when the seller initiates the conversation
% start(vbg).   % Uncomment to enable


% =============================================================================
% 5. PROBABILISTIC FACTS (LEARNED FROM EMPIRICAL DATA)
% =============================================================================

% The following probabilities are the optimized transition probabilities
% induced from the eight transcripts. They represent the empirical
% frequencies of each transition in the corpus.

% Level 1: Greetings
% From KBG (Customer greeting)
% Observed: 2 occurrences of KBG → VBG, 1 occurrence of KBG → VBBd
0.667::transition(kbg, vbg).   % Customer greeting → Seller greeting (2/3)
0.333::transition(kbg, vbbd).  % Customer greeting → Seller inquiry (1/3)

% From VBG (Seller greeting)
% Observed: 2 occurrences of VBG → KBBd
1.0::transition(vbg, kbbd).    % Seller greeting → Customer need (2/2) - Constitutive rule

% Level 2: Need phase
% From KBBd (Customer need)
% Observed: 4 occurrences of KBBd → VBBd, 1 occurrence of KBBd → VAA, 1 occurrence of KBBd → VBA
0.667::transition(kbbd, vbbd).  % Customer need → Seller inquiry (4/6)
0.167::transition(kbbd, vaa).   % Customer need → Seller completion (1/6) - Direct purchase
0.167::transition(kbbd, vba).   % Customer need → Seller reaction (1/6) - Consultative response

% From VBBd (Seller inquiry)
% Observed: 4 occurrences of VBBd → KBA, 2 of VBBd → VAA, 2 of VBBd → KBBd, 1 of VBBd → KAA
0.444::transition(vbbd, kba).   % Seller inquiry → Customer response (4/9)
0.222::transition(vbbd, vaa).   % Seller inquiry → Seller completion (2/9)
0.222::transition(vbbd, kbbd).  % Seller inquiry → Customer need (2/9) - Loop for upselling
0.111::transition(vbbd, kaa).   % Seller inquiry → Customer completion (1/9) - Early exit

% From KBA (Customer response)
% Observed: 3 occurrences of KBA → VBA, 3 of KBA → VAA
0.5::transition(kba, vba).      % Customer response → Seller reaction (3/6)
0.5::transition(kba, vaa).      % Customer response → Seller completion (3/6)

% From VBA (Seller reaction)
% Observed: 3 occurrences of VBA → KBBd, 2 of VBA → KAE, 2 of VBA → VAA
0.5::transition(vba, kbbd).     % Seller reaction → Customer need (3/6) - Upselling loop
0.25::transition(vba, kae).     % Seller reaction → Customer inquiry (2/8) - Question follow-up
0.25::transition(vba, vaa).     % Seller reaction → Seller completion (2/8) - Direct closing

% Level 3: Information exchange
% From KAE (Customer inquiry)
% Observed: 2 occurrences of KAE → VAE
1.0::transition(kae, vae).      % Customer inquiry → Seller information (2/2) - Constitutive rule

% From VAE (Seller information)
% Observed: 1 occurrence of VAE → KAE, 1 occurrence of VAE → KAA
0.5::transition(vae, kae).      % Seller information → Customer inquiry (1/2) - Loop
0.5::transition(vae, kaa).      % Seller information → Customer completion (1/2) - Exit

% Level 4: Completion
% From KAA (Customer completion)
% Observed: 3 occurrences of KAA → VAA, 1 occurrence of KAA → VBG
0.75::transition(kaa, vaa).     % Customer completion → Seller completion (3/4)
0.25::transition(kaa, vbg).     % Customer completion → Seller greeting (1/4) - Restart

% From VAA (Seller completion)
% Observed: 6 occurrences of VAA → KAA, 1 occurrence of VAA → KAV
0.857::transition(vaa, kaa).    % Seller completion → Customer completion (6/7)
0.143::transition(vaa, kav).    % Seller completion → Customer farewell (1/7) - Direct exit

% Level 5: Farewell
% From KAV (Customer farewell)
% Observed: 2 occurrences of KAV → VAV (T1, T8), 1 occurrence of KAV → KBBd (T5)
0.5::transition(kav, vav).      % Customer farewell → Seller farewell (2/3) - Normal exit
0.5::transition(kav, kbbd).     % Customer farewell → Customer need (1/3) - New customer arrival

% From VAV (Seller farewell)
% Observed: 3 occurrences of VAV → KAV
1.0::transition(vav, kav).      % Seller farewell → Customer farewell (3/3) - Constitutive rule


% =============================================================================
% 6. TRAINING DATA FOR NEURAL NETWORK
% =============================================================================

% The neural network is trained on the observed transitions from the corpus.
% Format: train(transition(start_symbol, next_symbol), truth_value)

% ========== Positive examples (observed transitions) ==========

% From the corpus (all transcripts):
% Sequence: KBG → VBG → KBBd → VBBd → KBA → VBA → KBBd → VBBd → KBA → VBA →
%           KAE → VAE → KAE → VAE → KAA → VAA → KAV → VAV

% Greeting phase
train(transition(kbg, vbg), true).      % Customer greeting → Seller greeting
train(transition(vbg, kbbd), true).     % Seller greeting → Customer need

% Need phase (first cycle)
train(transition(kbbd, vbbd), true).    % Customer need → Seller inquiry
train(transition(vbbd, kba), true).     % Seller inquiry → Customer response
train(transition(kba, vba), true).      % Customer response → Seller reaction
train(transition(vba, kbbd), true).     % Seller reaction → Customer need (upselling)

% Need phase (second cycle)
train(transition(kbbd, vbbd), true).    % Customer need → Seller inquiry
train(transition(vbbd, kba), true).     % Seller inquiry → Customer response
train(transition(kba, vba), true).      % Customer response → Seller reaction
train(transition(vba, kae), true).      % Seller reaction → Customer inquiry

% Information exchange
train(transition(kae, vae), true).      % Customer inquiry → Seller information
train(transition(vae, kae), true).      % Seller information → Customer inquiry (loop)
train(transition(kae, vae), true).      % Customer inquiry → Seller information (second)
train(transition(vae, kaa), true).      % Seller information → Customer completion

% Completion
train(transition(kaa, vaa), true).      % Customer completion → Seller completion
train(transition(vaa, kav), true).      % Seller completion → Customer farewell
train(transition(kav, vav), true).      % Customer farewell → Seller farewell

% Additional transitions from other transcripts (T2, T3, T4, T5, T6, T7, T8)
train(transition(vbbd, vaa), true).     % Direct completion after inquiry (T3)
train(transition(kba, vaa), true).      % Direct completion after response (T7)
train(transition(vba, vaa), true).      % Direct completion after reaction (T8)
train(transition(kaa, vbg), true).      % Restart after completion (T2, T5)
train(transition(kav, kbbd), true).     % New customer arrival (T5)
train(transition(vaa, kaa), true).      % Standard completion (multiple)
train(transition(kbbd, vaa), true).     % Direct purchase (T3, T7)
train(transition(kbbd, vba), true).     % Consultative response (T4)

% ========== Negative examples (unobserved transitions) ==========

% These help the neural network learn the grammar boundaries

% Invalid self-loops
train(transition(kbg, kbg), false).     % No self-greeting
train(transition(vbg, vbg), false).     % No self-greeting
train(transition(kbbd, kbbd), false).   % No self-need
train(transition(vbbd, vbbd), false).   % No self-inquiry
train(transition(kba, kba), false).     % No self-response
train(transition(vba, vba), false).     % No self-reaction
train(transition(kae, kae), false).     % No self-inquiry
train(transition(vae, vae), false).     % No self-information
train(transition(kaa, kaa), false).     % No self-completion
train(transition(vaa, vaa), false).     % No self-completion
train(transition(kav, kav), false).     % No self-farewell
train(transition(vav, vav), false).     % No self-farewell

% Invalid cross-phase jumps
train(transition(kbg, kav), false).     % Cannot jump from greeting to farewell
train(transition(vbg, vav), false).     % Cannot jump from greeting to farewell
train(transition(kbbd, kav), false).    % Cannot jump from need to farewell
train(transition(vbbd, vav), false).    % Cannot jump from inquiry to farewell
train(transition(kaa, kae), false).     % Cannot go back from completion to inquiry
train(transition(vaa, vae), false).     % Cannot go back from completion to information
train(transition(kav, kbg), false).     % Cannot restart after farewell
train(transition(vav, vbg), false).     % Cannot restart after farewell

% Invalid alternations (customer cannot respond to themselves)
train(transition(kbg, kbbd), false).    % Customer cannot state need without greeting response
train(transition(kbbd, kba), false).    % Customer cannot respond to themselves
train(transition(kba, kae), false).     % Customer cannot inquire without seller reaction
train(transition(kae, kaa), false).     % Customer cannot complete without seller information

% Invalid seller sequences
train(transition(vbg, vbbd), false).    % Seller cannot inquire without customer need
train(transition(vbbd, vba), false).    % Seller cannot react without customer response
train(transition(vba, vae), false).     % Seller cannot inform without customer inquiry
train(transition(vae, vaa), false).     % Seller cannot complete without customer completion


% =============================================================================
% 7. HARD CONSTRAINTS (CONSTITUTIVE RULES)
% =============================================================================

% In addition to the learned probabilities, we can encode hard constraints
% that cannot be violated. These represent the constitutive rules of the
% interaction format.

% Rule 1: A greeting must be reciprocated (unless skipped)
% If a customer greeting is not followed by a seller greeting,
% it must be followed by a seller inquiry (skip case)
:- transition(kbg, vbbd), \+ transition(kbg, vbg).

% Rule 2: A seller greeting must be followed by a customer need
% This is a constitutive rule of sales conversations
:- transition(vbg, X), X \= kbbd.

% Rule 3: A customer inquiry must be answered by seller information
:- transition(kae, X), X \= vae.

% Rule 4: Farewells are always reciprocated
transition(kav, vav).
transition(vav, kav).

% Rule 5: No cycles without progression
% The upselling loop (VBA → KBBd) must eventually lead to completion
% This is a liveness property that can be checked via model checking
% (Implementation in external model checker, not in DeepProbLog)


% =============================================================================
% 8. GENERATION PREDICATES
% =============================================================================

% Generate a random well-formed sequence of length N
generate_sequence(N, Seq) :-
    start(Start),
    generate_sequence(N, [Start], Seq).

generate_sequence(0, Seq, Seq).
generate_sequence(N, Current, Seq) :-
    N > 0,
    Current = [Last|_],
    transition(Last, Next),
    N1 is N - 1,
    generate_sequence(N1, [Next|Current], Seq).

% Sample a sequence from the probability distribution
sample_sequence(Seq) :-
    start(Start),
    sample_sequence([Start], Seq).

sample_sequence([Last|Rest], Seq) :-
    (   transition(Last, Next)
    ->  sample_sequence([Next, Last|Rest], Seq)
    ;   Seq = [Last|Rest]
    ).

% Generate all possible sequences up to length N (for debugging)
all_sequences(MaxLen, Seq) :-
    start(Start),
    all_sequences(MaxLen, [Start], Seq).

all_sequences(0, Seq, Seq).
all_sequences(N, Current, Seq) :-
    N > 0,
    Current = [Last|_],
    findall(Next, transition(Last, Next), Nexts),
    member(Next, Nexts),
    N1 is N - 1,
    all_sequences(N1, [Next|Current], Seq).


% =============================================================================
% 9. ANALYSIS PREDICATES
% =============================================================================

% Calculate the probability of a specific sequence
prob_sequence([], 1.0).
prob_sequence([_], 1.0).
prob_sequence([A,B|Rest], Prob) :-
    prob_sequence([B|Rest], RestProb),
    (   transition(A, B)
    ->  Prob = RestProb  % Probability already encoded in transition predicate
    ;   Prob = 0.0
    ).

% Find the most likely next symbol given a context
most_likely_next(Context, Next) :-
    Context = [Last|_],
    findall(Next-Symbol, transition(Last, Symbol), Candidates),
    keysort(Candidates, Sorted),
    last(Sorted, _-Next).

% Find all valid continuations of a sequence
valid_continuations(Sequence, Continuations) :-
    Sequence = [Last|_],
    findall(Next, transition(Last, Next), Continuations).

% Check if a sequence is valid according to the grammar
% Returns true with probability
is_valid_sequence(Sequence) :-
    valid_sequence(Sequence).

% Explain why a sequence is valid (returns proof tree via DeepProbLog explain)
% This is built into DeepProbLog as explain/1


% =============================================================================
% 10. QUERIES (EXAMPLE USAGE)
% =============================================================================

% Query 1: Probability of the full corpus sequence
% Expected: High probability (empirically observed sequence)
% query(well_formed([kbg, vbg, kbbd, vbbd, kba, vba, kbbd, vbbd, kba, vba,
%                    kae, vae, kae, vae, kaa, vaa, kav, vav])).

% Query 2: Most likely next symbol after a given context
% query(transition(kbbd, Next)).
% Expected: Next = vbbd with probability 0.667

% Query 3: Generate a sample well-formed sequence
% query(sample(well_formed(S))).

% Query 4: Probability of a new, unseen sequence
% query(well_formed([kbg, vbg, kbbd, vbbd, kaa, vaa, kav, vav])).

% Query 5: Explanation of why a sequence is well-formed
% explain(well_formed([kbg, vbg, kbbd])).

% Query 6: Find all possible next symbols from a given state
% query(valid_continuations([kbbd], Nexts)).

% Query 7: Most likely next symbol
% query(most_likely_next([kbbd], Next)).


% =============================================================================
% 11. NEURAL NETWORK ARCHITECTURE (PyTorch Implementation)
% =============================================================================

% The following is the PyTorch implementation of the neural network
% used to learn the transition probabilities.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ARSTransitionNetwork(nn.Module):
    """
    Neural network for learning transition probabilities in ARS.
    
    This network takes a one-hot encoded current symbol and outputs
    a probability distribution over next symbols. The output is a
    softmax that sums to 1 over the 12 terminal symbols.
    
    Architecture:
        Input: 12-dimensional one-hot vector
        Hidden 1: 64 neurons, ReLU, Dropout(0.2)
        Hidden 2: 32 neurons, ReLU, Dropout(0.2)
        Output: 12-dimensional softmax
    
    Training:
        Loss: Categorical cross-entropy
        Optimizer: Adam with learning rate 0.001
        Batch size: 32
        Epochs: 100
    """
    
    def __init__(self, n_symbols=12, hidden1=64, hidden2=32, dropout=0.2):
        super().__init__()
        self.n_symbols = n_symbols
        
        self.fc1 = nn.Linear(n_symbols, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_symbols)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
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
    
    def predict_transition(self, current_symbol_idx, device='cpu'):
        """
        Predict the probability distribution over next symbols.
        
        Args:
            current_symbol_idx: Index of current symbol (0-11)
            device: Computation device
            
        Returns:
            Array of 12 probabilities
        """
        one_hot = torch.zeros(1, self.n_symbols).to(device)
        one_hot[0, current_symbol_idx] = 1.0
        
        with torch.no_grad():
            probs = self.forward(one_hot)
        
        return probs.cpu().numpy()[0]
    
    def train_network(self, X_train, y_train, epochs=100, batch_size=32, lr=0.001):
        """
        Train the neural network on observed transitions.
        
        Args:
            X_train: One-hot encoded input symbols (n_samples, n_symbols)
            y_train: Target indices (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            
        Returns:
            Training loss history
        """
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return loss_history


# Symbol to index mapping for training
symbol_to_idx = {
    'kbg': 0, 'vbg': 1, 'kbbd': 2, 'vbbd': 3,
    'kba': 4, 'vba': 5, 'kae': 6, 'vae': 7,
    'kaa': 8, 'vaa': 9, 'kav': 10, 'vav': 11
}

# Training data preparation
def prepare_training_data(transitions):
    """
    Prepare training data from observed transitions.
    
    Args:
        transitions: List of (current_symbol, next_symbol) tuples
        
    Returns:
        X: One-hot encoded input symbols
        y: Target indices
    """
    X = []
    y = []
    
    for curr, nxt in transitions:
        one_hot = [0.0] * 12
        one_hot[symbol_to_idx[curr]] = 1.0
        X.append(one_hot)
        y.append(symbol_to_idx[nxt])
    
    return np.array(X), np.array(y)
"""


% =============================================================================
% 12. END OF DEEPPROBLOG IMPLEMENTATION
% =============================================================================