# src/features/guess_who/router.py
import logging
from fastapi import APIRouter, HTTPException, Body

# Import schemas and services for this feature
# Schema descriptions were already translated
from .schema import AnimalListResponse, AskRequest, AskResponse, FilterRequest, FilterResponse, GenerateQuestionRequest, GenerateQuestionResponse, SelectAnimalResponse
# Service function names remain the same
from .services import ALL_CHARACTERS, filter_list, generate_ai_question, select_random_animal, answer_question #, filter_list (if added)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/select_animal",
    response_model=SelectAnimalResponse,
    # Translated summary and description
    summary="Select AI's Secret Animal",
    description="Randomly selects an animal for the AI to 'think' of.",
)
async def http_select_animal():
    """Endpoint for the AI to choose its secret animal."""
    try:
        animal = await select_random_animal()
        return SelectAnimalResponse(selected_animal=animal)
    except HTTPException as e:
        # Re-raise HTTP exceptions from the service layer
        raise e
    except Exception as e:
        # Translated log message
        logger.exception("Unexpected error during animal selection.")
        # Return a structured error using the schema
        # Translated error message
        return SelectAnimalResponse(
            selected_animal="",
            error=f"An unexpected server error occurred: {type(e).__name__}"
        )


@router.post(
    "/ask",
    response_model=AskResponse,
    # Translated summary and description
    summary="Ask the AI a Question",
    description="Sends a yes/no question to the AI about its secret animal.",
)
async def http_ask_question(
    request_data: AskRequest = Body(...) # Use Body to receive JSON payload
):
    """
    Endpoint for the user to ask a question.
    Requires the question and the AI's current secret animal in the request body.
    """
    # Translated log message
    logger.info(f"Received question for animal '{request_data.secret_animal}': '{request_data.question}'")
    try:
        # Expecting "yes" or "no" from the translated service now
        ai_answer = await answer_question(
            question=request_data.question,
            secret_animal=request_data.secret_animal
        )
        return AskResponse(answer=ai_answer)

    except HTTPException as e:
        # Re-raise HTTP exceptions (like 400 for invalid animal, 50x for LLM issues)
        raise e
    except Exception as e:
        # Translated log message
        logger.exception(f"Unexpected error processing question for animal '{request_data.secret_animal}'.")
        # Return a structured error using the schema
        # Translated error message
        return AskResponse(
            answer="",
            error=f"An unexpected server error occurred while processing the question: {type(e).__name__}"
        )


@router.get(
    "/animals", # Ou un autre chemin comme /characters ou /all_animals
    response_model=AnimalListResponse,
    summary="Get All Animal Names", # Traduit pour la cohérence
    description="Retrieves the complete list of animal names available in the game.", # Traduit
)
async def http_get_all_animals():
    """
    Endpoint pour obtenir la liste de tous les animaux possibles.
    (Endpoint to get the list of all possible animals.)
    """
    # Traduit le message de log
    logger.info("Request received for the list of all animals.")
    try:
        # Retourne directement la liste importée dans le modèle de réponse
        return AnimalListResponse(animals=ALL_CHARACTERS)
    except Exception as e:
        # Traduit le message de log d'erreur
        logger.exception("Unexpected error retrieving animal list.")
        # Retourne une erreur structurée en utilisant le schéma
        # Traduit le message d'erreur
        return AnimalListResponse(
            animals=[], # Retourne une liste vide en cas d'erreur
            error=f"An unexpected server error occurred: {type(e).__name__}")

# Optional: Route for filtering (if you add it)
@router.post(
    "/filter",
    response_model=FilterResponse,
    # Translated summary and description
    summary="Filter Animal List",
    description="Uses the LLM to filter the animal list based on a question and answer.",
)
async def http_filter_list(request_data: FilterRequest = Body(...)):
    # Translated log message
    logger.info(f"Filtering list based on Q:'{request_data.question}', A:'{request_data.answer}'")
    try:
        kept_animals, reasoning = await filter_list(
            question=request_data.question,
            answer=request_data.answer, # Expects "yes" or "no"
            current_list=request_data.current_list
        )
        return FilterResponse(kept_animals=kept_animals, reasoning=reasoning)
    except HTTPException as e:
        raise e
    except Exception as e:
        # Translated log message
        logger.exception("Unexpected error during list filtering.")
        # Return a structured error using the schema
        # Translated error message
        return FilterResponse(
            kept_animals=[],
            error=f"An unexpected server error occurred during filtering: {type(e).__name__}"
        )
    

@router.post(
    "/generate_question",
    response_model=GenerateQuestionResponse,
    summary="Generate AI Question",
    description="Generates a yes/no question for the AI to ask based on its current list of possibilities.",
)
async def http_generate_question(request_data: GenerateQuestionRequest = Body(...)):
    """
    Endpoint for the AI to generate its next question.
    Requires the AI's current list of possible user animals.
    """
    logger.info(f"Request received to generate AI question from list: {request_data.current_list}")
    try:
        question = await generate_ai_question(current_list=request_data.current_list,  previous_questions=request_data.previous_questions)
        return GenerateQuestionResponse(question=question)

    except HTTPException as e:
        # Re-raise HTTP exceptions (like 400 for empty list, 50x for LLM issues)
        raise e
    except Exception as e:
        logger.exception("Unexpected error during AI question generation.")
        # Return a structured error using the schema
        return GenerateQuestionResponse(
            question="",
            error=f"An unexpected server error occurred during question generation: {type(e).__name__}"
        )