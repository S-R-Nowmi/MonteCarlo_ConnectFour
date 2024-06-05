import numpy as np
import random
import math
import sys

# Node class for the UCT algorithm and PMCGS
class Node:
    def __init__(self, board=None, move=None, parent=None, player=1):
        if board is not None:
            self.board = np.copy(board)  # For UCT
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.plays = 0
        self.player = player

    def add_child(self, child):
        self.children.append(child)

    def update(self, result):
        self.plays += 1
        self.wins += result
     

    def uct_select_child(self):
        best_value = float('-inf')
        best_nodes = []
        for child in self.children:
            if child.plays == 0:
                uct_value = float('inf')
            else:
                uct_value = child.wins / child.plays + math.sqrt(2 * math.log(self.plays) / child.plays)
            if uct_value > best_value:
                best_value = uct_value
                best_nodes = [child]
            elif uct_value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes) if best_nodes else None
    
    def __repr__(self):
        return f"Node(Move: {self.move}, Wins: {self.wins}, Plays: {self.plays}, Player: {'Red' if self.player == 1 else 'Yellow'})"


# Utility functions
def read_game_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        algorithm = lines[0].strip()
        current_player = 1 if lines[1].strip() == 'R' else 2
        board = np.array([[0 if cell == 'O' else 1 if cell == 'R' else 2 for cell in line.strip()] for line in lines[2:8]])
    return algorithm, current_player, board


def is_valid_location(board, col):
    return board[0, col] == 0

def get_next_open_row(board, col):
    for r in range(5, -1, -1):
        if board[r, col] == 0:
            return r

def drop_piece(board, row, col, piece):
    board[row, col] = piece

def check_win(board, piece):
    # Horizontal, vertical, and positively/negatively sloped diagonal checks
    for c in range(7-3):
        for r in range(6):
            if all(board[r, c+i] == piece for i in range(4)):
                return True
    for c in range(7):
        for r in range(6-3):
            if all(board[r+i, c] == piece for i in range(4)):
                return True
    for c in range(7-3):
        for r in range(6-3):
            if all(board[r+i, c+i] == piece for i in range(4)):
                return True
    for c in range(7-3):
        for r in range(3, 6):
            if all(board[r-i, c+i] == piece for i in range(4)):
                return True
    return False


def undo_move(board, row, col):
    board[row, col] = 0



# UR Algorithm
def select_random_move(board, current_player, verbosity):
    legal_moves = [c for c in range(7) if is_valid_location(board, c)]
    if not legal_moves:
        return None
    move = random.choice(legal_moves)
    row = get_next_open_row(board, move)
    drop_piece(board, row, move, current_player)
    if check_win(board, current_player):
        if verbosity.lower() != "none":
            print(f"Winning move found at column: {move + 1}")
    undo_move(board, row, move)
    return move



# UCT Algorithm (expand, simulate, backpropagate, best_move, mcts, run_mcts)


def expand(node):
    valid_moves = [col for col in range(7) if is_valid_location(node.board, col)]
    for move in valid_moves:
        temp_board = np.copy(node.board)
        row = get_next_open_row(temp_board, move)
        drop_piece(temp_board, row, move, node.player)
        next_player = 3 - node.player
        child_node = Node(temp_board, move, node, next_player)
        node.add_child(child_node)

def simulate(node):
    temp_board = np.copy(node.board)
    current_player = node.player
    while True:
        valid_moves = [col for col in range(7) if is_valid_location(temp_board, col)]
        if not valid_moves:
            return 0  # Draw condition
        move = random.choice(valid_moves)
        row = get_next_open_row(temp_board, move)
        drop_piece(temp_board, row, move, current_player)
        if check_win(temp_board, current_player):
            return 1 if current_player == node.player else -1
        current_player = 3 - current_player


def backpropagate(node, result):
    while node is not None:
        node.update(result)
        node = node.parent
        result = -result  # Swap result for opponent's perspective



def best_move(root):
    best_ratio = max((child.wins / child.plays) for child in root.children if child.plays > 0)
    best_children = [child for child in root.children if child.plays > 0 and child.wins / child.plays == best_ratio]
    selected_child = random.choice(best_children)
    return selected_child.move

def mcts(root, iterations, verbose=False):
    for _ in range(iterations):
        node = root
        while node.children:
            node = node.uct_select_child()
            if verbose:
                # Print node selection details for each iteration
                print(f"Move selected: {node.move + 1}, wi: {node.wins}, ni: {node.plays}")
        if node.plays > 0 and not check_win(node.board, node.player):
            expand(node)
            if verbose:
                print("NODE ADDED")
        if node.children:
            node = random.choice(node.children)
        result = simulate(node)
        backpropagate(node, result)
        if verbose and result != 0:
            # Print terminal node value after simulation
            print(f"TERMINAL NODE VALUE: {result}")


def run_mcts(board, current_player, simulations=100, verbosity='none'):
    root = Node(board, player=current_player)
    expand(root)

    mcts(root, simulations, verbose=verbosity == 'verbose')

    if verbosity == 'verbose':
        print(f"wi: {root.wins}, ni: {root.plays}")
        for i, child in enumerate(root.children, start=1):
            if child.plays > 0:  # This check will prevent ZeroDivisionError
                print(f"V{i}: {child.wins/child.plays:.2f} (Wins: {child.wins}, Plays: {child.plays})")
            else:
                print(f"V{i}: N/A (Wins: {child.wins}, Plays: {child.plays})")

    move = best_move(root)
    
    if verbosity != 'none':
        print(f"FINAL Move selected: {move + 1}")



# PMCGS Algorithm
# Node class for PMCGS

# Node class for PMCGS
class Nodep:
    def __init__(self, board=None, move=None, parent=None, player=1):
        self.board = np.copy(board) if board is not None else None
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.plays = 0
        self.player = player

    def add_child(self, child):
        self.children.append(child)

    def update(self, result):
        self.plays += 1
        self.wins += result

def pmcgs_expand(node):
    valid_moves = [col for col in range(7) if is_valid_location(node.board, col)]
    for move in valid_moves:
        temp_board = np.copy(node.board)
        row = get_next_open_row(temp_board, move)
        drop_piece(temp_board, row, move, node.player)
        child_node = Nodep(temp_board, move, node, 3 - node.player)
        node.add_child(child_node)

def pmcgs_simulate(node, verbosity):
    temp_board = np.copy(node.board)
    current_player = node.player
    move_sequence = []

    while True:
        valid_moves = [col for col in range(7) if is_valid_location(temp_board, col)]
        if not valid_moves:  # Draw condition
            return 0, move_sequence

        move = random.choice(valid_moves)
        move_sequence.append(move)
        row = get_next_open_row(temp_board, move)
        drop_piece(temp_board, row, move, current_player)

        if check_win(temp_board, current_player):
            return 1 if current_player == node.player else -1, move_sequence

        current_player = 3 - current_player  # Switch player




def run_pmcgs(board, current_player, simulations, verbosity="brief"):
    root = Nodep(board=board, player=current_player)
    pmcgs_expand(root)  # Initial expansion based on possible moves
    
    move_performance = [None if is_valid_location(board, col) else "Null" for col in range(7)]

    for _ in range(simulations):
        node = root
        path = [node]
        
        while node.children:
            prev_wins = node.wins
            prev_plays = node.plays

            chosen_child = random.choice(node.children)
            path.append(chosen_child)

            if verbosity == "verbose":
                print(f"wi: {prev_wins}\tni: {prev_plays}")
                print(f"Move selected: {chosen_child.move + 1}")
                
            if not chosen_child.children:
                pmcgs_expand(chosen_child)
                if verbosity == "verbose" and chosen_child.children:
                    print("NODE ADDED")

            result, _ = pmcgs_simulate(chosen_child, verbosity=False)  # Simulate from the chosen child
            chosen_child.update(result)  # Update the chosen child node with the simulation result

            if verbosity == "verbose":
                print(f"TERMINAL NODE VALUE: {result}")
                print("Updated values:")
            
            for node in path:
                node.update(result)
                if verbosity == "verbose":
                    print(f"wi: {node.wins}\tni: {node.plays}")
                result = -result  # Invert the result for the opponent
            
    for index, child in enumerate(root.children):
        if child.plays > 0:
            move_performance[index] = child.wins / child.plays
        else:
            move_performance[index] = "Null"  # Ensure a value is set even if no simulations were run for this move
    
    best_move_index = move_performance.index(max(filter(lambda x: x != "Null", move_performance), key=float))
    best_move = best_move_index + 1
    
    if verbosity == "verbose":
        for i, performance in enumerate(move_performance, start=1):
            print(f"Column {i}: {performance}")
    
    print(f"FINAL Move selected: {best_move}")






def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <file_path> <verbosity> <simulations>")
        sys.exit(1)

    file_path, verbosity, simulations = sys.argv[1], sys.argv[2].lower(), int(sys.argv[3])
    algorithm, current_player, board = read_game_file(file_path)

    if algorithm == "UR":
        move = select_random_move(board, current_player, verbosity)
        if move is not None:
            if verbosity != "none":
                print(f"FINAL Move selected: {move + 1}")
    elif algorithm == "UCT":
        run_mcts(board, current_player, simulations, verbosity)
    elif algorithm == "PMCGS":
        """
        # Ensure verbosity is correctly handled for PMCGS
        if verbosity.lower() not in ["verbose", "brief", "none"]:
            print("Invalid verbosity level. Please choose between 'Verbose', 'Brief', or 'None'.")
        else:
            run_pmcgs(board, current_player, simulations, verbosity)
            """
        run_pmcgs(board, current_player, simulations, verbosity)
    else:
        print(f"Algorithm '{algorithm}' is not implemented.")

if __name__ == "__main__":
    main()
