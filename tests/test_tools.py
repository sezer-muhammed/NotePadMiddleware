from unittest.mock import MagicMock

from langgraph.types import Command
from langchain_middleware_notepad.tools import (
    notepad_append,
    notepad_replace,
    notepad_is_exist,
    notepad_read,
    notepad_clear,
)

class MockRuntime:
    def __init__(self, state):
        self.state = state

def test_notepad_append():
    # Case 1: Empty start
    runtime = MockRuntime({"notepad": ""})
    cmd = notepad_append.invoke({"text": "Hello", "runtime": runtime})
    assert isinstance(cmd, Command)
    assert cmd.update["notepad"] == "Hello"

    # Case 2: Existing content
    runtime = MockRuntime({"notepad": "Hello"})
    cmd = notepad_append.invoke({"text": "World", "runtime": runtime})
    assert cmd.update["notepad"] == "Hello\nWorld"

def test_notepad_replace():
    runtime = MockRuntime({"notepad": "Old"})
    cmd = notepad_replace.invoke({"text": "New", "runtime": runtime})
    assert cmd.update["notepad"] == "New"

def test_notepad_is_exist():
    # Case 1: Empty
    runtime = MockRuntime({"notepad": ""})
    assert notepad_is_exist.invoke({"runtime": runtime}) == "false"
    
    # Case 2: Whitespace only (should be considered empty per spec "strip()")
    runtime = MockRuntime({"notepad": "   "})
    assert notepad_is_exist.invoke({"runtime": runtime}) == "false"

    # Case 3: Exists
    runtime = MockRuntime({"notepad": "Exists"})
    assert notepad_is_exist.invoke({"runtime": runtime}) == "true"

def test_notepad_read():
    # Case 1: Empty
    runtime = MockRuntime({"notepad": ""})
    assert notepad_read.invoke({"runtime": runtime}) == "<empty>"

    # Case 2: Content
    runtime = MockRuntime({"notepad": "Content"})
    assert notepad_read.invoke({"runtime": runtime}) == "Content"

def test_notepad_clear():
    runtime = MockRuntime({"notepad": "Something"})
    cmd = notepad_clear.invoke({"runtime": runtime})
    assert cmd.update["notepad"] == ""
