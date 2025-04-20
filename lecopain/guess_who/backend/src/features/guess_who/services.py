# src/features/guess_who/services.py
import asyncio
import random
import os
import logging
import ast
from typing import List, Tuple
import json
from fastapi import HTTPException
import random

# --- Mistral Client Setup ---
# Make sure to install the library: pip install mistralai
from .control_atomic import robot_move_grid
from mistralai import Mistral

from pydantic import BaseModel

class Response(BaseModel):
    resonning: str
    question: str

logger = logging.getLogger(__name__)

# Load API key from environment variable - REVERTED HARDCODED KEY
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") # <-- Use environment variable
MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest") # Or choose another model

if not MISTRAL_API_KEY:
    # Translated log message
    logger.error("CRITICAL: MISTRAL_API_KEY environment variable not set.")
    # Or raise an exception during startup if preferred
    mistral_client = None
else:
    try:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        # Translated log message
        logger.info(f"Mistral client initialized successfully for model '{MODEL_NAME}'.")
    except Exception as e:
        # Translated log message
        logger.exception(f"Failed to initialize Mistral client: {e}")
        mistral_client = None

# --- Animal Data (Consider moving to core/constants.py if used elsewhere) ---
# Animal names remain in French as they are identifiers from the original game setup
ANIMAL_COORDS = {
    "Grenouille": [0, 0], "Chien": [0, 1], "Chat": [0, 2], "Vache": [0, 3],
    "Lion": [0, 4], "Girafe": [0, 5], "Singe": [0, 6], "Pieuvre": [0, 7],
    "Poisson": [1, 0], "Pingouin": [1, 1], "Rouge-Gorge": [1, 2], "Elephant": [1, 3],
    "Chenille": [1, 4], "Requin-Tigre": [1, 5], "Corbeau": [1, 6], "Ours Polaire": [1, 7],
    "Araignée": [2, 0], "Mouche": [2, 1], "Chouette": [2, 2], "Escargot": [2, 3],
    "Serpent": [2, 4], "Rat": [2, 5], "Mouton": [2, 6], "Crocodile": [2, 7]
}
ALL_CHARACTERS = list(ANIMAL_COORDS.keys())

# --- Helper Functions (Adapted from your script) ---

def _extract_list(text: str) -> list[str]:
    """Attempts to extract the first Python list literal from a string."""
    try:
        # Find the first line starting with '['
        list_line = next(line for line in text.splitlines() if line.strip().startswith("["))
        # Evaluate the line as a Python literal
        evaluated = ast.literal_eval(list_line.strip())
        if isinstance(evaluated, list):
            return [str(item) for item in evaluated] # Ensure items are strings
        else:
            # Translated log message
            logger.warning(f"Extracted literal is not a list: {evaluated}")
            return []
    except StopIteration:
        # Translated log message
        logger.warning(f"Could not find line starting with '[' in text: {text}")
        return []
    except (SyntaxError, ValueError, TypeError) as e:
        # Translated log message
        logger.error(f"Error extracting list using ast.literal_eval: {e} from text: {text}")
        return []
    except Exception as e:
        # Translated log message
        logger.error(f"Unexpected error extracting list: {e} from text: {text}")
        return []

async def _llm_queryV2(prompt: str) -> str:
    """Sends a prompt to the Mistral API using the client."""
    if not mistral_client:
        # Translated log message
        logger.error("Mistral client is not available.")
        # Translated detail message
        raise HTTPException(status_code=503, detail="LLM service is unavailable.")

    try:
        # Mistral client's chat method might be synchronous.
        # Run it in a thread pool executor to avoid blocking the async event loop.
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,  # Use default executor
            lambda: mistral_client.chat.parse(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                random_seed=random.randint(0, 2**32-1),  # Random seed for reproducibility
                response_format= Response,
                top_p=0.9,
            )
        )
        print("couuuuuuuuccccooouuuuuuuuuu")
        logger.info(f"##########{response}######")  # Debugging line to see the raw response
        # Check if response is valid and has choices
        if response and response.choices:
            content = response.choices[0].message.parsed.question.strip()
            # Translated log message
            logger.debug(f"LLM Query successful. Prompt: '{prompt[:50]}...', Response: '{content[:50]}...'")
            return content
        else:
            # Translated log message
            logger.error(f"Invalid response received from Mistral API: {response}")
            # Translated detail message
            raise HTTPException(status_code=502, detail="Invalid response from LLM service.")

    except Exception as e:
        # Translated log message
        logger.exception(f"Error querying Mistral API: {e}")
        # Re-raise HTTPException if it came from the client, otherwise wrap
        if isinstance(e, HTTPException):
            raise e
        else:
            # Translated detail message
            raise HTTPException(status_code=500, detail=f"Error communicating with LLM: {type(e).__name__}")


async def _llm_query(prompt: str) -> str:
    """Sends a prompt to the Mistral API using the client."""
    if not mistral_client:
        # Translated log message
        logger.error("Mistral client is not available.")
        # Translated detail message
        raise HTTPException(status_code=503, detail="LLM service is unavailable.")

    try:
        # Mistral client's chat method might be synchronous.
        # Run it in a thread pool executor to avoid blocking the async event loop.
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,  # Use default executor
            lambda: mistral_client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                random_seed=random.randint(0, 2**32-1),  # Random seed for reproducibility
            )
        )
        print("couuuuuuuuccccooouuuuuuuuuu")
        logger.info(f"##########{response}######")  # Debugging line to see the raw response
        # Check if response is valid and has choices
        if response and response.choices:
            content = response.choices[0].message.content.strip()
            # Translated log message
            logger.debug(f"LLM Query successful. Prompt: '{prompt[:50]}...', Response: '{content[:50]}...'")
            return content
        else:
            # Translated log message
            logger.error(f"Invalid response received from Mistral API: {response}")
            # Translated detail message
            raise HTTPException(status_code=502, detail="Invalid response from LLM service.")

    except Exception as e:
        # Translated log message
        logger.exception(f"Error querying Mistral API: {e}")
        # Re-raise HTTPException if it came from the client, otherwise wrap
        if isinstance(e, HTTPException):
            raise e
        else:
            # Translated detail message
            raise HTTPException(status_code=500, detail=f"Error communicating with LLM: {type(e).__name__}")


# --- Service Functions ---

async def select_random_animal() -> str:
    """Selects a random animal from the list."""
    if not ALL_CHARACTERS:
         # Translated log message
         logger.warning("Animal list is empty.")
         # Translated detail message
         raise HTTPException(status_code=500, detail="Animal list is not configured.")
    selected = random.choice(ALL_CHARACTERS)
    # Translated log message
    logger.info(f"Randomly selected animal: {selected}")
    return selected

async def answer_question(question: str, secret_animal: str) -> str:
    """
    Uses the LLM to answer a yes/no question based on the secret animal.
    """
    if secret_animal not in ANIMAL_COORDS:
        # Translated log message
        logger.error(f"Invalid secret animal provided: {secret_animal}")
        # Translated detail message
        raise HTTPException(status_code=400, detail=f"Invalid secret animal: {secret_animal}")

    # Translated prompt
    prompt = f"""
You are playing the game "Guess Who?". You are secretly the character '{secret_animal}'.
A player asks you the following question: "{question}"
Answer only and literally with "yes" or "no", without any other punctuation or sentences.
"""
    try:
        raw_response = await _llm_query(prompt)
        # Clean the response to be strictly "yes" or "no"
        # Keep logic targeting French oui/non unless LLM response guarantees English
        cleaned_response = raw_response.lower().strip().rstrip('.?!')
        # Adjust based on expected LLM output language (now prompted in English)
        if "yes" in cleaned_response:
             answer = "yes"
        elif "no" in cleaned_response:
             answer = "no"
        else:
             # Translated log message - adapted for yes/no
            logger.warning(f"LLM response was not clearly 'yes' or 'no': '{raw_response}'. Defaulting based on presence of 'yes'.")
            answer = "yes" if "yes" in cleaned_response else "no" # Simple fallback

        # Translated log message - adapted for yes/no
        logger.info(f"Question: '{question}' for animal '{secret_animal}'. Answer: '{answer}' (Raw: '{raw_response}')")
        return answer

    except HTTPException as e:
        # Propagate HTTP exceptions from _llm_query
        raise e
    except Exception as e:
        # Translated log message
        logger.exception(f"Unexpected error in answer_question for animal '{secret_animal}': {e}")
        # Translated detail message
        raise HTTPException(status_code=500, detail="Failed to get answer from LLM.")

# Optional: Service function for filtering (if you add the route)
async def filter_list(question: str, answer: str, current_list: List[str]) -> Tuple[List[str], str]:
    """
    Uses the LLM to filter the list based on the question and answer,
    expecting a JSON response.
    """
    # Updated filter_prompt asking for JSON
    filter_prompt = f"""
You are playing the game "Guess Who?". Your opponent asked: "{question}"
The answer was: "{answer}" (only "yes" or "no").
Here is the list of possible characters remaining: {current_list}

Based ONLY on the question and the answer, determine the updated list of characters to keep and provide a brief explanation for the removals.

Respond ONLY with a valid JSON object containing two keys:
1.  `kept_characters`: A JSON array (list) of strings representing the characters to keep.
2.  `reasoning`: A single string explaining briefly why the other characters were removed.

Do not include any text before or after the JSON object.

Example JSON Response:
```json
{{
  "kept_characters": ["A","B", "C"],
  "reasoning": ""
}}
```
    """
    try:
        raw_response = await _llm_query(filter_prompt)
        print(f"##########{raw_response}######")  # Debugging line to see the raw response
        # Attempt to parse the JSON response
        try:
            # Find the JSON part in case the LLM still adds extra text (optional robustness)
            # More robust: Find the first '{' and last '}'
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = raw_response[json_start:json_end+1]
            else:
                # Assume the whole response should be JSON if markers aren't clear
                json_str = raw_response.strip()

            parsed_json = json.loads(json_str)

            # Extract data, providing defaults if keys are missing
            kept_animals = parsed_json.get('kept_characters', [])
            reasoning = parsed_json.get('reasoning', 'No reasoning provided in JSON.')


            # Validate extracted data types (optional but recommended)
            if not isinstance(kept_animals, list) or not all(isinstance(item, str) for item in kept_animals):
                raise ValueError("'kept_characters' should be a list of strings.")
            if not isinstance(reasoning, str):
                raise ValueError("'reasoning' should be a string.")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON. Raw: '{raw_response}'")
            # Translated detail message
            raise HTTPException(status_code=500, detail="LLM response was not valid JSON.")
        except ValueError as ve: # Catch type validation errors
            logger.error(f"Invalid JSON structure or types from LLM: {ve}. Raw: '{raw_response}'")
            # Translated detail message
            raise HTTPException(status_code=500, detail=f"LLM response JSON structure/type error: {ve}")
        except Exception as e: # Catch unexpected parsing issues
            logger.exception(f"Error processing LLM JSON response: {e}. Raw: '{raw_response}'")
            # Translated detail message
            raise HTTPException(status_code=500, detail="Failed to process LLM JSON response.")


        # Basic validation: Ensure kept animals are a subset of the original list
        valid_kept_animals = [animal for animal in kept_animals if animal in current_list]

        if len(valid_kept_animals) != len(kept_animals):
            invalid_suggestions = [animal for animal in kept_animals if animal not in current_list]
            # Translated log message
            logger.warning(f"LLM filter suggested animals not in the original list ({invalid_suggestions}). Raw JSON: '{json_str}', Original: {current_list}, Kept valid: {valid_kept_animals}")
            # Note: We only keep the valid ones, correcting the LLM's mistake silently for the user.

        removed_animals = [animal for animal in current_list if animal not in valid_kept_animals]
        for animal in removed_animals:
            print(f"Kept animal: {animal}")  # Debugging line to see kept animals
            if animal in ANIMAL_COORDS.keys():
                print(f"Animal {animal} found in ANIMAL_COORDS.")
                coord = ANIMAL_COORDS[animal]
                print(f"Coordinates for {animal}: {coord}")
                robot_move_grid(coord[0], coord[1])  # Assuming robot_move_grid is defined elsewhere
        # Translated log message
        logger.info(f"Filtered list based on Q:'{question}', A:'{answer}'. Kept: {valid_kept_animals}. Reasoning: '{reasoning}'")

        return valid_kept_animals, reasoning

    except HTTPException as e:
        # Re-raise HTTPExceptions directly
        raise e
    except Exception as e:
        # Translated log message
        logger.exception(f"Unexpected error in filter_list during LLM call or processing: {e}")
        # Translated detail message
        raise HTTPException(status_code=500, detail="Failed to filter list using LLM.")
        

async def generate_ai_question(current_list: List[str], previous_questions: List[str]) -> str:
    """
    Uses the LLM to generate a discriminating question based on the current list of possible animals.
    """
    if not current_list:
        logger.warning("AI asked to generate question for an empty list.")
        # Handle this case - maybe raise an error or return a default message
        # Depending on game logic, this might indicate the AI should guess or an error occurred.
        raise HTTPException(status_code=400, detail="Cannot generate question from an empty list.")

    if len(current_list) == 1:
         # If only one animal left, AI should guess, not ask.
         # This logic might be better handled in the frontend, but we can prevent unnecessary LLM calls.
         # Or return a specific message indicating a guess is needed.
         # For now, let's still generate a question, though it might be trivial.
         logger.info(f"Generating question for single remaining animal: {current_list[0]}")


    previous_block = (
        f"You have already asked these questions: {previous_questions}.\n"
        "Avoid repeating them exactly.\n"
        if previous_questions else ""
    )

    #prompt = f"""
    #You are playing the game "Guess Who?". You need to guess your opponent's secret animal.
    #Your current list of possible animals for the opponent is: {current_list}.
    #{previous_block}
    #Generate a single, effective yes/no question that will help you eliminate the most possibilities from this list.
    #Focus on common distinguishing features.
    #Avoid questions that have already been asked or that are specific to only one animal unless few options remain.
    #Respond ONLY with the question itself, and nothing else.
    #"""
    prompt = f"""
    You need to guess your opponent's secret animal.
    Your current list of possible animals for the opponent is: {current_list}.
    {previous_block}
    Generate a single, effective yes/no question that will eliminate the minimum number of animals from this list.
    Avoid questions that have already been asked.
    Respond ONLY with the question itself, and nothing else.
    Please think about five questions before deciding the final question in english.
    You must response in json format with two keys:
    1. Reasonning: A string explaining briefly why the other characters were removed.
    2. Question: The final question.
    Do not include any text before or after the JSON object.
    """
    try:
        # Use the existing LLM query helper
        generated_question = await _llm_queryV2(prompt)

        # Basic cleaning (remove potential quotes or extra phrases if LLM doesn't follow instructions perfectly)
        cleaned_question = generated_question.strip().strip('"')
        logger.info(f"\n\n\n{prompt}\n\n\n")

        logger.info(f"Generated AI question for list {current_list}: '{cleaned_question}' (Raw: '{generated_question}')")
        return cleaned_question
    except HTTPException as e:
        # Propagate HTTP exceptions from _llm_query
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error generating AI question for list {current_list}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate question via LLM.")
