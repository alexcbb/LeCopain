# src/features/guess_who/schema.py
from pydantic import BaseModel, Field
from typing import List

class AskRequest(BaseModel):
    """Request model for asking the AI a question."""
    question: str = Field(..., description="The yes/no question asked by the user.")
    secret_animal: str = Field(..., description="The secret animal the AI is 'thinking' of.")
    # Optional: Could add current_list if filtering logic moves here
    # current_list: List[str] = Field(..., description="The current list of possible animals.")

class AskResponse(BaseModel):
    """Response model for the AI's answer."""
    # Changed description from French 'oui'/'non' to English 'yes'/'no'
    answer: str = Field(..., description="The AI's answer ('yes' or 'no').")
    error: str | None = None

class SelectAnimalResponse(BaseModel):
    """Response model for selecting a random animal."""
    selected_animal: str = Field(..., description="The animal randomly selected by the AI.")
    error: str | None = None

class AnimalListResponse(BaseModel):
    """Response model for retrieving the list of all animals."""
    animals: List[str] = Field(..., description="The complete list of animal names.")
    error: str | None = Field(None, description="Optional error message.")

# Optional: Schema for filtering based on answer (if you add that route)
class FilterRequest(BaseModel):
    question: str
    answer: str # "yes" or "no"
    current_list: List[str]

class FilterResponse(BaseModel):
    kept_animals: List[str]
    reasoning: str
    error: str | None = None

class GenerateQuestionRequest(BaseModel):
    """Request model for the AI to generate a question."""
    current_list: List[str] = Field(..., description="The AI's current list of possible animals for the user.")
    previous_questions: List[str] = Field(default_factory=list, description="Questions already asked by the AI.")

class GenerateQuestionResponse(BaseModel):
    """Response model for the AI's generated question."""
    question: str = Field(..., description="The question generated by the AI.")
    error: str | None = Field(None, description="Optional error message.")