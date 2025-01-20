import random
from typing import Any, Dict, List, Optional

import numpy as np

from .inference import (
    generate_feedback,
    generate_improved_version,
    generate_initial_answer,
    generate_rating,
)


class Node:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS).

    Each node corresponds to a specific state in the search tree, maintaining information
    about its children, parent, visits, value, and associated answer.

    Attributes:
        original_prompt (str): The original input prompt for the node.
        answer (str): The answer associated with this node.
        parent (Optional[Node]): The parent node of this node.
        max_children (int): The maximum number of children this node can have.
        exploration_weight (float): The weight used to balance exploration and exploitation.
        children (List[Node]): The list of child nodes.
        visits (int): The number of times this node has been visited.
        value (float): The accumulated value of this node.
        level (int): The depth of the node in the tree, with the root node at level 0.
    """

    def __init__(
        self,
        original_prompt: str,
        answer: str,
        max_children: int,
        exploration_weight: float,
        level: int = 0,
        parent: Optional["Node"] = None,
    ) -> None:
        self.original_prompt: str = original_prompt
        self.answer: str = answer
        self.parent: Optional["Node"] = parent
        self.max_children: int = max_children
        self.exploration_weight: float = exploration_weight

        self.children: List["Node"] = []
        self.visits: int = 1
        self.value: float = 0.0  # accumulated value

        self.level: int = 0 if parent is None else parent.level + 1

    def is_fully_expanded(self) -> bool:
        """
        Checks if the node is fully expanded, meaning it has the maximum number of children.

        Returns:
            bool: True if the node is fully expanded, False otherwise.
        """
        return len(self.children) >= self.max_children

    def best_child(self) -> "Node":
        """
        Selects the best child of the current node based on the UCT.

        Returns:
            Node: The child node with the highest computed weight.
        """
        choices_weights: List[float] = []
        for child in self.children:
            if child.visits == 0:
                weight: float = float("inf")  # explore unvisited nodes first
            else:
                exploitation_term: float = child.value / child.visits
                exploration_term: float = self.exploration_weight * np.sqrt(
                    np.log(self.visits) / child.visits
                )
                weight = exploitation_term + exploration_term

            choices_weights.append(weight)

        return self.children[np.argmax(choices_weights)]

    def most_visited_child(self) -> "Node":
        """
        Selects the child node with the highest number of visits.

        Returns:
            Node: The most visited child node.
        """
        return max(self.children, key=lambda child: child.visits)

    def add_child(self, child_node: "Node") -> None:
        """
        Adds a child node to the current node.

        Args:
            child_node (Node): The child node to add.
        """
        self.children.append(child_node)


class MCTS:
    """
    Implements the Monte Carlo Tree Search (MCTS) algorithm.

    This class orchestrates the search process by selecting, expanding, simulating,
    and backpropagating through the tree to find the best answer.

    Attributes:
        original_prompt (str): The input prompt used for the search process.
        iterations (int): The number of iterations to perform during the search.
        max_children (int): The maximum number of children allowed for each node.
        request_settings (Dict[str, Any]): Configuration settings for the underlying model.
        verbose (bool): Whether to print progress information during the search.
        exploration_weight (float): The weight used to balance exploration and exploitation.
        initial_answer (str): The initial answer generated from the input prompt.
        root (Node): The root node of the search tree.
    """

    def __init__(
        self,
        original_prompt: str,
        request_settings: Dict[str, Any],
        iterations: int,
        max_children: int,
        verbose: bool,
        exploration_weight: float,
    ) -> None:
        self.original_prompt: str = original_prompt
        self.iterations: int = iterations
        self.max_children: int = max_children
        self.request_settings: Dict[str, Any] = request_settings
        self.verbose: bool = verbose
        self.exploration_weight: float = exploration_weight

        self.initial_answer: str = generate_initial_answer(original_prompt, request_settings)
        self.root: Node = Node(
            original_prompt=self.original_prompt,
            answer=self.initial_answer,
            max_children=self.max_children,
            parent=None,
            level=0,
            exploration_weight=self.exploration_weight,
        )

    def search(self) -> str:
        """
        Executes the MCTS search process and returns the best answer.

        Returns:
            str: The best answer found during the search.
        """
        for i in range(self.iterations):
            self.print_to_terminal(f"Iteration {i + 1}/{self.iterations}")

            node: Node = self.select(self.root)
            self.print_to_terminal(f"Selected node from level: {node.level}")

            if not node.is_fully_expanded():
                self.print_to_terminal(f"Expand node at level: {node.level}")
                node = self.expand(node)

            reward: float = self.simulate(node)
            self.print_to_terminal(f"Simulated reward: {reward}")
            self.backpropagate(node, reward)

        best_node: Node = self.root.most_visited_child()
        self.print_to_terminal(
            f"Best node has {best_node.visits} visits and is at level {best_node.level}"
        )
        return best_node.answer

    def select(self, node: Node) -> Node:
        """
        Selects a node to expand or simulate based on the exploration and exploitation weights.

        Args:
            node (Node): The root node from which to begin the selection.

        Returns:
            Node: The selected node.
        """
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def expand(self, node: Node) -> Node:
        """
        Expands the current node by creating new child nodes and assigning answers to them.

        Args:
            node (Node): The node to expand.

        Returns:
            Node: A randomly selected child node after expansion.
        """
        for j in range(self.max_children - len(node.children)):  # expand max_children nodes
            # create and add child node
            child_node: Node = Node(
                original_prompt=self.original_prompt,
                answer=node.answer,  # will be later updated with the model
                parent=node,
                max_children=self.max_children,
                level=node.level + 1,
                exploration_weight=self.exploration_weight,
            )
            node.add_child(child_node)

            feedback: str = generate_feedback(
                prompt=self.original_prompt,
                answer=child_node.answer,
                request_settings=self.request_settings,
            )

            improved_version: str = generate_improved_version(
                prompt=self.original_prompt,
                answer=child_node.answer,
                feedback=feedback,
                request_settings=self.request_settings,
            )

            child_node.answer = improved_version

        selected_node: Node = random.choice(node.children)
        return selected_node

    def backpropagate(self, node: Optional[Node], reward: float) -> None:
        """
        Backpropagates the reward through the tree from a leaf node to the root.

        Args:
            node (Optional[Node]): The node from which to start backpropagation.
            reward (float): The reward value to propagate.
        """
        while node is not None:
            node.visits += 1
            node.value += reward

            node = node.parent

    def simulate(self, node: Node) -> float:
        """
        Simulates the process of generating a rating for the current node's answer.

        Args:
            node (Node): The node whose answer is being rated.

        Returns:
            float: The rating score for the node's answer.
        """
        rating: float = generate_rating(
            prompt=self.original_prompt,
            answer=node.answer,
            request_settings=self.request_settings,
        )
        return rating

    def print_to_terminal(self, msg: str) -> None:
        """
        Prints a message to the terminal if verbose mode is enabled.

        Args:
            msg (str): The message to print.
        """
        if self.verbose:
            print(msg)
