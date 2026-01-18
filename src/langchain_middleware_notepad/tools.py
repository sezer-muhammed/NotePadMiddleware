from typing import TYPE_CHECKING, Literal

from langchain_core.tools import tool
from langgraph.types import Command

if TYPE_CHECKING:
    from langgraph.prebuilt import InjectedToolRuntime
    from typing_extensions import Annotated
    
    ToolRuntime = Annotated[InjectedToolRuntime, "The tool runtime"]
else:
    # Runtime mock/placeholder for runtime type checking if langgraph not installed in dev env
    ToolRuntime = object


@tool
def notepad_append(text: str, runtime: ToolRuntime) -> Command:
    """
    Append text to the existing note in the notepad.
    
    Args:
        text: The text to append.
        runtime: The tool runtime context.
    """
    current_note = runtime.state.get("notepad", "")
    if not current_note:
        new_note = text
    else:
        new_note = f"{current_note}\n{text}"
    
    return Command(update={"notepad": new_note})  # type: ignore[arg-type]


@tool
def notepad_replace(text: str, runtime: ToolRuntime) -> Command:
    """
    Replace the entire note in the notepad with new text.
    
    Args:
        text: The new text to replace the existing note.
        runtime: The tool runtime context.
    """
    return Command(update={"notepad": text})  # type: ignore[arg-type]


@tool
def notepad_is_exist(runtime: ToolRuntime) -> Literal["true", "false"]:
    """
    Check if the note currently exists and is non-empty.
    
    Args:
        runtime: The tool runtime context.
        
    Returns:
        "true" if the note exists and is not empty, "false" otherwise.
    """
    current_note = runtime.state.get("notepad", "")
    exists = bool(current_note.strip())
    return "true" if exists else "false"


@tool
def notepad_read(runtime: ToolRuntime) -> str:
    """
    Read the full content of the note.
    
    Args:
        runtime: The tool runtime context.
        
    Returns:
        The content of the note, or "<empty>" if there is no note.
    """
    current_note = runtime.state.get("notepad", "")
    if not current_note:
        return "<empty>"
    return current_note


@tool
def notepad_clear(runtime: ToolRuntime) -> Command:
    """
    Clear the note, setting it to an empty string.
    
    Args:
        runtime: The tool runtime context.
    """
    return Command(update={"notepad": ""})  # type: ignore[arg-type]
