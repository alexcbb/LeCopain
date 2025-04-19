// src/services/apiService.ts
import {
    AskRequest, AskResponse, SelectAnimalResponse, AnimalListResponse,
    FilterRequest, FilterResponse, GenerateQuestionRequest, GenerateQuestionResponse,
    TranscriptionResponse
  } from '../types/api';
  
  const API_BASE_URL = "http://localhost:8000/api/guess_who"; // Or your full base URL
  
  // --- Helper for Fetch ---
  async function fetchApi<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          ...(options.headers || {}),
        },
      });
  
      if (!response.ok) {
        let errorBody: any = { detail: `HTTP error! Status: ${response.status}` };
        try {
          // Try to parse specific FastAPI error detail
          errorBody = await response.json();
        } catch (e) { /* Ignore if body isn't JSON */ }
        throw new Error(errorBody.detail || `Request failed with status ${response.status}`);
      }
  
      // Handle cases where backend might return 204 No Content or non-JSON success
      if (response.status === 204 || response.headers.get('content-length') === '0') {
          return {} as T; // Return an empty object or handle appropriately
      }
  
      return await response.json() as T;
  
    } catch (error) {
      console.error(`API call to ${endpoint} failed:`, error);
      // Re-throw a consistent error format or handle as needed
      throw error instanceof Error ? error : new Error('An unknown API error occurred');
    }
  }
  
  // --- API Functions ---
  
  // Example constant - replace with actual list if needed elsewhere, otherwise remove
  export const ALL_CHARACTERS = ["Grenouille", "Chien", "Chat", "Vache", "Lion", "Girafe", "Singe", "Pieuvre", "Poisson", "Pingouin", "Rouge-Gorge", "Elephant", "Chenille", "Requin-Tigre", "Corbeau", "Ours Polaire", "Araignée", "Mouche", "Chouette", "Escargot", "Serpent", "Rat", "Mouton", "Crocodile"];
  
  
  export const apiSelectAnimal = (): Promise<SelectAnimalResponse> => {
    return fetchApi<SelectAnimalResponse>('/select_animal', { method: 'POST' });
  };
  
  export const apiGetAllAnimals = (): Promise<AnimalListResponse> => {
    // Ensure the backend returns the expected format, including the "animals" key
    return fetchApi<AnimalListResponse>('/animals', { method: 'GET' });
  };
  
  export const apiAsk = (data: AskRequest): Promise<AskResponse> => {
    return fetchApi<AskResponse>('/ask', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  };
  
  export const apiFilter = (data: FilterRequest): Promise<FilterResponse> => {
    return fetchApi<FilterResponse>('/filter', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  };
  
  export const apiGenerateQuestion = (data: GenerateQuestionRequest): Promise<GenerateQuestionResponse> => {
    return fetchApi<GenerateQuestionResponse>('/generate_question', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  };
  
  // --- Transcription API Call (if needed separately) ---
  // Assuming the transcription endpoint expects FormData
  export const apiTranscribe = async (audioBlob: Blob): Promise<TranscriptionResponse> => {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm'); // Backend expects 'file'
  
      try {
          const response = await fetch("http://localhost:8000/api/stt/transcribe", { // Use full URL here
              method: 'POST',
              body: formData,
              // No 'Content-Type' header needed for FormData, browser sets it with boundary
          });
  
          if (!response.ok) {
              let errorBody: any = { detail: `HTTP error! Status: ${response.status}` };
              try { errorBody = await response.json(); } catch (e) {}
              throw new Error(errorBody.detail || `Transcription request failed with status ${response.status}`);
          }
  
          if (response.status === 204 || response.headers.get('content-length') === '0') {
              return {}; // Or { transcription: "" } if appropriate
          }
  
          return await response.json() as TranscriptionResponse;
  
      } catch (error) {
          console.error(`API call to /transcribe failed:`, error);
          throw error instanceof Error ? error : new Error('An unknown transcription API error occurred');
      }
  };