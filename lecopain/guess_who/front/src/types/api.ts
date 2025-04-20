import { ALL_CHARACTERS } from '../services/apiService'; // Example import path

export interface SelectAnimalResponse {
  selected_animal: string;
  error?: string | null;
}

export interface AskRequest {
  question: string;
  secret_animal: string;
}

export interface AskResponse {
  answer: "yes" | "no" | ""; // Allow empty on error
  error?: string | null;
}

export interface AnimalListResponse {
  animals: typeof ALL_CHARACTERS; // Use the actual list type if possible
  error?: string | null;
}

export interface FilterRequest {
  question: string;
  answer: "yes" | "no";
  current_list: string[];
}

export interface FilterResponse {
  kept_animals: string[];
  reasoning?: string | null; 
  error?: string | null;
}

export interface GenerateQuestionRequest {
    current_list: string[];
    previous_questions: string[];
}

export interface GenerateQuestionResponse {
    question: string;
    error?: string | null;
}

export interface TranscriptionResponse {
    transcription?: string;
    error?: string;
}
