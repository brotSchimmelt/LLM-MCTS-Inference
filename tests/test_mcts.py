from unittest.mock import patch

import pytest

from llm_mcts_inference.mcts import MCTS, Node


@pytest.fixture
def mock_node():
    """Fixture to create a mock Node for testing."""
    return Node(
        original_prompt="Test prompt",
        answer="Test answer",
        max_children=3,
        exploration_weight=1.0,
    )


def test_node_initialization(mock_node):
    """Test the initialization of the Node class."""
    assert mock_node.original_prompt == "Test prompt"
    assert mock_node.answer == "Test answer"
    assert mock_node.max_children == 3
    assert mock_node.exploration_weight == 1.0
    assert mock_node.parent is None
    assert mock_node.level == 0
    assert mock_node.visits == 1
    assert mock_node.value == 0.0
    assert len(mock_node.children) == 0


def test_node_is_fully_expanded(mock_node):
    """Test the is_fully_expanded method."""
    assert not mock_node.is_fully_expanded()
    for i in range(3):  # Add max_children nodes
        mock_node.add_child(Node("Prompt", f"Answer {i}", 3, 1.0))
    assert mock_node.is_fully_expanded()


def test_node_add_child(mock_node):
    """Test the add_child method."""
    child_node = Node("Child prompt", "Child answer", 3, 1.0)
    mock_node.add_child(child_node)
    assert len(mock_node.children) == 1
    assert mock_node.children[0] is child_node


def test_node_best_child():
    """Test the best_child method."""
    parent_node = Node("Prompt", "Answer", 3, 1.0)
    for i in range(3):
        child_node = Node(
            parent_node.original_prompt,
            parent_node.answer + f" {i}",
            parent_node.max_children,
            parent_node.exploration_weight,
        )
        child_node.visits = 1
        child_node.value = (i + 1) * 10
        parent_node.add_child(child_node)
    best_child = parent_node.best_child()
    assert best_child.value == 30


@pytest.fixture
def mock_mcts():
    """Fixture to create a mock MCTS instance for testing."""
    with (
        patch("llm_mcts_inference.inference.generate_initial_answer", return_value="Mock response"),
        patch("llm_mcts_inference.inference.get_model_response", return_value="Mock response"),
    ):
        return MCTS(
            original_prompt="Test prompt",
            model_settings={
                "base_url": "mock_base_url",
                "api_key": "mock_api_key",
                "model_name": "mock_model",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "seed": 42,
            },
            iterations=10,
            max_children=3,
            verbose=False,
            exploration_weight=1.0,
        )


def test_mcts_initialization(mock_mcts):
    """Test the initialization of the MCTS class."""
    assert mock_mcts.original_prompt == "Test prompt"
    assert mock_mcts.iterations == 10
    assert mock_mcts.max_children == 3
    assert mock_mcts.exploration_weight == 1.0
    assert mock_mcts.verbose is False
    assert mock_mcts.root.answer == "Mock response"
    assert mock_mcts.root.level == 0
