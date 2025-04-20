// src/App.tsx
import React, { useState, useEffect, useCallback, useRef } from "react";
import OldTvScreen from "./components/OldTvScreen";
import { GlassesStyle, glassesMap } from "./constants/glasses";
import { MouthState, mouthsMap } from "./constants/mouths";
import { PupilState, pupilsMap } from "./constants/pupils";
import { useAnimatedEyes } from "./hooks/useAnimatedEyes";
import { useAnimatedMouth } from "./hooks/useAnimatedMouth";
import { useRealtimeTranscription } from "./hooks/useRealtimeTranscription";
// --- Game Logic Imports ---
import {
  apiSelectAnimal,
  apiGetAllAnimals,
  apiAsk,
  apiFilter,
  apiGenerateQuestion,
} from "./services/apiService";
import { AnimalListResponse } from "./types/api"; // Only import needed types here

// --- Constants ---
const TRANSCRIPTION_KEY = " ";
const REACTION_DURATION_MS = 3000;
const availableGlassesStyles: GlassesStyle[] = Object.keys(
  glassesMap
) as GlassesStyle[];
const availablePupilStates: PupilState[] = Object.keys(
  pupilsMap
) as PupilState[];
const availableMouthStates: MouthState[] = Object.keys(
  mouthsMap
) as MouthState[];

// --- Game State Enum ---
enum GamePhase {
  LOADING, // Initial loading of animals
  STARTING, // AI selecting animal
  USER_TURN_ASK, // User needs to ask a question
  USER_RECORDING_QUESTION, // User is holding space to ask
  AI_PROCESSING_QUESTION, // Waiting for AI answer
  AI_ANSWERING, // AI is about to speak its answer
  AI_TURN_ASK, // AI needs to generate a question
  AI_GENERATING_QUESTION, // Waiting for AI question
  AI_SPEAKING_QUESTION, // AI is about to speak its question
  USER_TURN_ANSWER, // User needs to answer yes/no
  USER_RECORDING_ANSWER, // User is holding space to answer
  AI_PROCESSING_ANSWER, // Waiting for AI to filter list
  AI_FILTERING, // AI is filtering its list
  AI_MAKING_GUESS, // AI thinks it knows the answer
  GAME_OVER_AI_WINS,
  GAME_OVER_USER_WINS, // TBD: Add user guess logic
  ERROR,
}

function App() {
  // --- Existing Hooks ---
  const [glassesStyle, setGlassesStyle] = useState<GlassesStyle>("neutral");
  const [aiQuestionHistory, setAiQuestionHistory] = useState<string[]>([]);

  const {
    pupilState,
    eyeOffsetX,
    eyeOffsetY,
    currentPupilAscii,
    setPupilStateBase,
    moveEyes,
    resetEyePosition,
    isBlinking,
    previousPupilState,
    isAutonomous,
    toggleAutonomousMode,
  } = useAnimatedEyes("center");
  const {
    mouthState,
    baseMouthState,
    setMouthStateBase,
    toggleTalking,
    isTalkingContinuously,
  } = useAnimatedMouth({ initialMouthState: "neutral" });
  const {
    isRecording,
    statusMessage: sttStatusMessage, // Rename to avoid conflict
    transcriptionText: currentTranscription, // Rename to avoid conflict
    startRecording,
    stopRecordingAndSend,
  } = useRealtimeTranscription();

  // --- Game State ---
  const [gamePhase, setGamePhase] = useState<GamePhase>(GamePhase.LOADING);
  const [aiSecretAnimal, setAiSecretAnimal] = useState<string | null>(null);
  const [aiPossibleAnimals, setAiPossibleAnimals] = useState<string[]>([]); // AI's list of user possibilities
  const [lastUserQuestion, setLastUserQuestion] = useState<string | null>(null);
  const [lastAiAnswer, setLastAiAnswer] = useState<"yes" | "no" | null>(null);
  const [lastAiQuestion, setLastAiQuestion] = useState<string | null>(null);
  const [lastUserAnswer, setLastUserAnswer] = useState<"yes" | "no" | null>(
    null
  );
  const [messageToSpeak, setMessageToSpeak] = useState<string>(""); // Text for TTS
  const [displayMessage, setDisplayMessage] =
    useState<string>("Loading game..."); // Message shown below face
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // --- Refs ---
  const keydownTriggeredRef = useRef(false);
  const isRecordingRef = useRef(isRecording);
  const isTalkingContinuouslyRef = useRef(isTalkingContinuously);
  const reactionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const transitionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  // Ref to store the finalized transcription *after* recording stops
  const finalTranscriptionRef = useRef<string | null>(null);

  // --- Update Refs ---
  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);
  useEffect(() => {
    isTalkingContinuouslyRef.current = isTalkingContinuously;
  }, [isTalkingContinuously]);

  useEffect(() => {
    // Cette fonction de nettoyage s'exécute quand le composant se démonte
    return () => {
      if (reactionTimeoutRef.current) {
        console.log("Cleanup: Clearing reaction timeout.");
        clearTimeout(reactionTimeoutRef.current);
      }

      if (transitionTimeoutRef.current) {
        console.log("Cleanup: Clearing transition timeout.");
        clearTimeout(transitionTimeoutRef.current);
      }
    };
  }, []);

  // --- Text-To-Speech Effect ---
  // This effect watches `messageToSpeak` and triggers the TTS/mouth animation
  useEffect(() => {
    const stopMouthIfNeeded = () => {
      if (isTalkingContinuouslyRef.current) {
        console.log(
          "App TTS: Stopping mouth animation (speech ended/cancelled/error)."
        );
        toggleTalking(); // Stable function from useAnimatedMouth
      }
    };

    if (messageToSpeak && messageToSpeak.trim() !== "") {
      console.log(`App TTS: Preparing to speak: "${messageToSpeak}"`);
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel(); // Cancel previous speech
        const utterance = new SpeechSynthesisUtterance(messageToSpeak);
        utterance.lang = "en-US"; // Or choose based on language needed

        utterance.onstart = () => {
          if (!isTalkingContinuouslyRef.current) {
            console.log(
              "App TTS: Speech started - Activating mouth animation."
            );
            toggleTalking();
          }
        };
        utterance.onend = () => {
          console.log("App TTS: Speech ended - Deactivating mouth animation.");
          stopMouthIfNeeded();
          // Clear the message so this effect doesn't re-trigger accidentally
          //setMessageToSpeak(""); // NO! This causes re-renders. Let the game logic clear it if needed.
        };
        utterance.onerror = (event) => {
          if (event.error === "interrupted") {
            console.warn("TTS: Utterance interrupted, likely expected.");
            return; // Ne change pas l'état du jeu ni l'affichage
          }
          console.error("App TTS: SpeechSynthesisUtterance Error:", event);
          stopMouthIfNeeded();
          setErrorMessage("Text-to-speech error.");
          setGamePhase(GamePhase.ERROR);
        };
        window.speechSynthesis.speak(utterance);
      } else {
        console.error("Text-to-speech not supported.");
        setErrorMessage("Text-to-speech not supported by browser.");
        setGamePhase(GamePhase.ERROR);
      }
    } else {
      // If messageToSpeak is cleared, ensure mouth stops
      if (window.speechSynthesis?.speaking) {
        window.speechSynthesis.cancel();
      }
      stopMouthIfNeeded();
    }

    // Cleanup function
    return () => {
      if ("speechSynthesis" in window && window.speechSynthesis.speaking) {
        console.log(
          "App TTS Cleanup: Cancelling speech on effect re-run/unmount."
        );
        window.speechSynthesis.cancel(); // This should trigger onend/onerror which handles mouth animation
      }
    };
    // IMPORTANT: Only depend on messageToSpeak. toggleTalking is stable.
  }, [messageToSpeak, toggleTalking]);

  // --- Game Logic Effects ---

  // Effect 1: Initialization (Loading -> Starting)
  useEffect(() => {
    if (gamePhase === GamePhase.LOADING) {
      console.log("Game Phase: LOADING");
      setDisplayMessage("Loading animal list...");
      const initializeGame = async () => {
        try {
          // 1. Fetch all animals
          const animalsResponse: AnimalListResponse = await apiGetAllAnimals();
          if (
            animalsResponse.error ||
            !animalsResponse.animals ||
            animalsResponse.animals.length === 0
          ) {
            throw new Error(
              animalsResponse.error ||
                "Failed to load animal list or list is empty."
            );
          }
          const animals = animalsResponse.animals;
          console.log("Animals loaded:", animals);
          setAiPossibleAnimals([...animals]); // Initialize AI's possibilities

          // 2. AI Selects Animal
          setDisplayMessage("AI is choosing its secret animal...");
          setGamePhase(GamePhase.STARTING); // Move to next phase
        } catch (error) {
          console.error("Initialization failed:", error);
          setErrorMessage(
            `Initialization failed: ${
              error instanceof Error ? error.message : String(error)
            }`
          );
          setGamePhase(GamePhase.ERROR);
        }
      };
      initializeGame();
    }
  }, [gamePhase]); // Runs only when gamePhase is LOADING

  // Effect 2: AI Selects Animal (Starting -> User_Turn_Ask)
  useEffect(() => {
    if (gamePhase === GamePhase.STARTING) {
      console.log("Game Phase: STARTING");
      const selectAnimal = async () => {
        try {
          const response = await apiSelectAnimal();
          if (response.error || !response.selected_animal) {
            throw new Error(response.error || "AI failed to select an animal.");
          }
          console.log("AI selected animal:", response.selected_animal);
          setAiSecretAnimal(response.selected_animal);

          // Start the game
          const startMessage =
            "I have chosen my secret animal. Your turn to ask a question. Hold Space to speak.";
          setDisplayMessage(startMessage);
          setMessageToSpeak(startMessage); // Trigger TTS
          setGamePhase(GamePhase.USER_TURN_ASK); // Ready for user
        } catch (error) {
          console.error("AI animal selection failed:", error);
          setErrorMessage(
            `AI animal selection failed: ${
              error instanceof Error ? error.message : String(error)
            }`
          );
          setGamePhase(GamePhase.ERROR);
        }
      };
      selectAnimal();
    }
  }, [gamePhase]); // Runs only when gamePhase is STARTING

  // Effect 3: Process User's Question (AI_Processing_Question -> AI_Answering)
  // This effect runs *after* the user has finished recording and the transcription is finalized.
  useEffect(() => {
    if (
      gamePhase === GamePhase.AI_PROCESSING_QUESTION &&
      finalTranscriptionRef.current
    ) {
      console.log("Game Phase: AI_PROCESSING_QUESTION");
      const userQuestion = finalTranscriptionRef.current;
      finalTranscriptionRef.current = null; // Consume the transcription

      if (
        !userQuestion ||
        userQuestion.trim() === "" ||
        userQuestion.includes("[Silence")
      ) {
        console.log("User question was empty or silent.");
        const retryMsg =
          "I didn't hear a question. Please ask again. Hold Space to speak.";
        setDisplayMessage(retryMsg);
        setMessageToSpeak(retryMsg);
        setGamePhase(GamePhase.USER_TURN_ASK);
        return;
      }

      if (!aiSecretAnimal) {
        console.error("AI secret animal is not set!");
        setErrorMessage("Critical Error: AI secret animal missing.");
        setGamePhase(GamePhase.ERROR);
        return;
      }

      setDisplayMessage(`You asked: "${userQuestion}"\nAI is thinking...`);
      setLastUserQuestion(userQuestion); // Store for potential filtering later

      const askAi = async () => {
        try {
          const response = await apiAsk({
            question: userQuestion,
            secret_animal: aiSecretAnimal,
          });
          if (response.error || !response.answer) {
            throw new Error(
              response.error || "AI failed to answer the question."
            );
          }
          console.log(`AI answer for "${userQuestion}": ${response.answer}`);
          setLastAiAnswer(response.answer); // Store AI answer

          // Prepare to speak the answer
          setGamePhase(GamePhase.AI_ANSWERING);
        } catch (error) {
          console.error("Error asking AI:", error);
          setErrorMessage(
            `Error getting AI answer: ${
              error instanceof Error ? error.message : String(error)
            }`
          );
          setGamePhase(GamePhase.ERROR); // Go to error state
          // Optionally, allow user to retry? setGamePhase(GamePhase.USER_TURN_ASK);
        }
      };
      askAi();
    }
  }, [gamePhase, aiSecretAnimal]); // Runs when phase changes to AI_PROCESSING_QUESTION and transcription is ready

  // Effect 4: AI Speaks its Answer (AI_Answering -> AI_Turn_Ask)
  useEffect(() => {
    if (gamePhase === GamePhase.AI_ANSWERING && lastAiAnswer) {
      console.log("Game Phase: AI_ANSWERING");
      const answer = lastAiAnswer;
      setLastAiAnswer(null); // Consume the answer
  
      setDisplayMessage(`AI answers: "${answer}"`);
      setMessageToSpeak(answer); // Speak "yes" or "no"
  
      // 👇 Supprime cette ligne (c'est elle qui fait la transition trop tôt)
      // setGamePhase(GamePhase.AI_TURN_ASK);
  
      if (transitionTimeoutRef.current) {
        clearTimeout(transitionTimeoutRef.current);
      }
  
      transitionTimeoutRef.current = setTimeout(() => {
        console.log("Effect 4: 2-second timeout finished. Transitioning to AI_TURN_ASK.");
        setGamePhase(GamePhase.AI_TURN_ASK);
        transitionTimeoutRef.current = null;
      }, 2000);
    }
  }, [gamePhase, lastAiAnswer]);

  // Effect 5: AI Generates Question (AI_Turn_Ask -> AI_Generating_Question -> AI_Speaking_Question)
  useEffect(() => {
    if (gamePhase === GamePhase.AI_TURN_ASK) {
      console.log("Game Phase: AI_TURN_ASK");
      setDisplayMessage("AI is thinking of a question...");
      setGamePhase(GamePhase.AI_GENERATING_QUESTION); // Indicate processing

      const generateQuestion = async () => {
        if (aiPossibleAnimals.length === 0) {
          // This shouldn't happen with correct filtering, indicates an error state
          console.error("AI has no possible animals left to ask about.");
          setErrorMessage("AI Error: No possible animals remaining.");
          setMessageToSpeak(
            "Something went wrong, I have no possibilities left."
          );
          setGamePhase(GamePhase.ERROR);
          return;
        }

        if (aiPossibleAnimals.length === 1) {
          // AI should make a guess!
          console.log("AI believes it knows the answer:", aiPossibleAnimals[0]);
          setGamePhase(GamePhase.AI_MAKING_GUESS);
          return;
        }

        try {
          const response = await apiGenerateQuestion({
            current_list: aiPossibleAnimals,
            previous_questions: aiQuestionHistory,
          });
          if (response.error || !response.question) {
            throw new Error(
              response.error || "AI failed to generate a question."
            );
          }
          console.log("AI generated question:", response.question);
          setLastAiQuestion(response.question); // Store the question
          setGamePhase(GamePhase.AI_SPEAKING_QUESTION); // Ready to speak
        } catch (error) {
          console.error("Error generating AI question:", error);
          setErrorMessage(
            `Error generating AI question: ${
              error instanceof Error ? error.message : String(error)
            }`
          );
          setGamePhase(GamePhase.ERROR);
        }
      };
      // Add a small delay for realism?
      setTimeout(generateQuestion, 500); // e.g., 500ms delay
    }
  }, [gamePhase, aiPossibleAnimals]);

  // Effect 6: AI Speaks its Question (AI_Speaking_Question -> User_Turn_Answer)
  useEffect(() => {
    if (gamePhase === GamePhase.AI_SPEAKING_QUESTION && lastAiQuestion) {
      console.log("Game Phase: AI_SPEAKING_QUESTION");
      const question = lastAiQuestion;
      // Don't consume lastAiQuestion here, need it for filtering later
      setAiQuestionHistory((prev) => [...prev, question]);

      const fullMessage = `AI asks: ${question}\nAnswer 'yes' or 'no'. Hold Space to speak.`;
      setDisplayMessage(fullMessage);
      setMessageToSpeak(question); // Speak the question

      // Transition to user needing to answer AFTER starting speech
      setGamePhase(GamePhase.USER_TURN_ANSWER);
    }
  }, [gamePhase, lastAiQuestion]);

  // Effect 7: Process User's Answer (AI_Processing_Answer -> AI_Filtering)
  useEffect(() => {
    if (
      gamePhase === GamePhase.AI_PROCESSING_ANSWER &&
      finalTranscriptionRef.current
    ) {
      console.log("Game Phase: AI_PROCESSING_ANSWER");
      const userAnswerRaw = finalTranscriptionRef.current;
      finalTranscriptionRef.current = null; // Consume

      if (
        !userAnswerRaw ||
        userAnswerRaw.trim() === "" ||
        userAnswerRaw.includes("[Silence")
      ) {
        console.log("User answer was empty or silent.");
        const retryMsg = `I didn't hear your answer to: "${lastAiQuestion}". Please say 'yes' or 'no'. Hold Space.`;
        setDisplayMessage(retryMsg);
        console.log(`%c[DEBUG] setMessageToSpeak appelé avec: "${retryMsg}" depuis [Effect 7 - Réponse Invalide]`, 'color: orange; font-weight: bold;');
        setMessageToSpeak(retryMsg);
        setGamePhase(GamePhase.USER_TURN_ANSWER); // Go back to let user answer again
        return;
      }

      // Basic Yes/No check (can be improved)
      const userAnswerClean = userAnswerRaw
        .toLowerCase()
        .trim()
        .replace(/[.!?]/g, "");
      let userAnswer: "yes" | "no" | null = null;
      if (
        userAnswerClean === "yes" ||
        userAnswerClean === "yeah" ||
        userAnswerClean === "yep"
      ) {
        userAnswer = "yes";
      } else if (
        userAnswerClean === "no" ||
        userAnswerClean === "nope" ||
        userAnswerClean === "nah"
      ) {
        userAnswer = "no";
      }

      if (!userAnswer) {
        console.log(`User answer unclear: "${userAnswerRaw}"`);
        const retryMsg = `I didn't understand if that was 'yes' or 'no' to: "${lastAiQuestion}". Please try again. Hold Space.`;
        setDisplayMessage(retryMsg);
        setMessageToSpeak(retryMsg);
        setGamePhase(GamePhase.USER_TURN_ANSWER); // Go back
        return;
      }

      if (!lastAiQuestion) {
        console.error("AI last question is not set for filtering!");
        setErrorMessage("Critical Error: AI question missing for filtering.");
        setGamePhase(GamePhase.ERROR);
        return;
      }

      setDisplayMessage(
        `You answered: "${userAnswer}" to "${lastAiQuestion}"\nAI is updating its list...`
      );
      setLastUserAnswer(userAnswer); // Store the processed answer

      if (userAnswer === "no") {
        console.log("Triggering 'no' reaction animation...");
        // Annule un éventuel timer de réaction précédent
        if (reactionTimeoutRef.current) {
          clearTimeout(reactionTimeoutRef.current);
        }

        // Définit l'état visuel de la réaction (lunettes + pupilles)
        // Vérifie si les styles existent bien dans vos constantes
        if (glassesMap.surpise && pupilsMap.big) {
          setGlassesStyle("surpise"); // Change les lunettes
          setPupilStateBase("big"); // Change l'état de base des pupilles
        } else {
          console.warn(
            "Reaction styles 'surprise' or 'big' not found in constants!"
          );
        }

        // Définit un timer pour annuler la réaction après REACTION_DURATION_MS
        reactionTimeoutRef.current = setTimeout(() => {
          console.log("Reverting from 'no' reaction animation...");
          setGlassesStyle("neutral"); // Remet les lunettes neutres
          setPupilStateBase("center"); // Remet les pupilles au centre
          reactionTimeoutRef.current = null; // Efface la référence du timer terminé
        }, REACTION_DURATION_MS);
      }

      // Prepare for filtering phase
      setGamePhase(GamePhase.AI_FILTERING);
    }
  }, [gamePhase, lastAiQuestion]); // Runs when AI_PROCESSING_ANSWER and transcription ready

  // Effect 8: AI Filters its List (AI_Filtering -> User_Turn_Ask or Game Over)
  useEffect(() => {
    if (
      gamePhase === GamePhase.AI_FILTERING &&
      lastAiQuestion &&
      lastUserAnswer
    ) {
      console.log("Game Phase: AI_FILTERING");
      const question = lastAiQuestion;
      const answer = lastUserAnswer;
      // Consume answer now, question was maybe already consumed or needed if filtering fails? Reset it after call.
      setLastUserAnswer(null);

      const filterAiList = async () => {
        try {
          const response = await apiFilter({
            question: question,
            answer: answer,
            current_list: aiPossibleAnimals,
          });
          if (response.error || !response.kept_animals) {
            throw new Error(response.error || "AI failed to filter its list.");
          }

          const kept = response.kept_animals;
          const reasoning = response.reasoning; 
          console.log("AI filtered list, kept:", kept);
          setAiPossibleAnimals(kept); // Update AI's possibilities

          // Consume question after successful filter
          setLastAiQuestion(null);

          const reasoningOrFallback = reasoning || "Okay, I've updated my list based on your answer.";

          // Check game state after filtering
          if (kept.length === 1) {
            // AI thinks it knows!
            setDisplayMessage(`${reasoningOrFallback}\nNow I think I know...`); // Show reasoning on screen
            setMessageToSpeak(reasoningOrFallback);
            setGamePhase(GamePhase.AI_MAKING_GUESS);
          } else if (kept.length === 0) {
            // AI has no possibilities left - opponent likely made a mistake or LLM failed
            console.error("AI filtered list to zero possibilities.");
            setErrorMessage(
              "AI Error: No possibilities left based on answers."
            );
            setMessageToSpeak(
              "Hmm, based on your answers, no animal fits! Let's restart?"
            ); // Or declare error
            // Maybe offer restart? For now, error state.
            setGamePhase(GamePhase.ERROR);
          } else {
            // Game continues, user's turn
            const nextTurnMsg = `${reasoningOrFallback} Your turn to ask. Hold Space to speak.`;
            setDisplayMessage(nextTurnMsg);
            setMessageToSpeak(nextTurnMsg); // Speak the combined message
            setGamePhase(GamePhase.USER_TURN_ASK);
          }
        } catch (error) {
          console.error("Error filtering AI list:", error);
          setErrorMessage(
            `Error during AI filtering: ${
              error instanceof Error ? error.message : String(error)
            }`
          );
          setGamePhase(GamePhase.ERROR);
          // Maybe reset lastUserAnswer so user can try again? Or just error out.
          // setLastUserAnswer(answer); // Restore if retry needed
        }
      };
      filterAiList();
    }
  }, [gamePhase, lastAiQuestion, lastUserAnswer, aiPossibleAnimals]);

  // Effect 9: AI Makes a Guess (AI_Making_Guess -> Game_Over_AI_WINS)
  useEffect(() => {
    if (gamePhase === GamePhase.AI_MAKING_GUESS) {
      console.log("Game Phase: AI_MAKING_GUESS");
      if (aiPossibleAnimals.length === 1) {
        const guess = aiPossibleAnimals[0];
        const guessMessage = `I think your animal is... ${guess}!`;
        setDisplayMessage(`${guessMessage}\nGame Over!`);
        setMessageToSpeak(guessMessage);
        setGamePhase(GamePhase.GAME_OVER_AI_WINS);
      } else {
        // Should not happen if logic is correct
        console.error("AI entered guessing phase without exactly one animal.");
        setErrorMessage("Error: AI guess attempted with incorrect list size.");
        setGamePhase(GamePhase.ERROR);
      }
    }
  }, [gamePhase, aiPossibleAnimals]);

  // --- Keyboard Input Handler ---
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Allow recording only in specific user turn phases
      const canRecord =
        gamePhase === GamePhase.USER_TURN_ASK ||
        gamePhase === GamePhase.USER_TURN_ANSWER;

      if (
        event.key === TRANSCRIPTION_KEY &&
        canRecord &&
        !keydownTriggeredRef.current &&
        !isRecordingRef.current
      ) {
        keydownTriggeredRef.current = true;
        console.log("App: Spacebar Down - Starting recording...");
        finalTranscriptionRef.current = null; // Clear previous final transcription
        // Update phase to reflect recording state
        if (gamePhase === GamePhase.USER_TURN_ASK)
          setGamePhase(GamePhase.USER_RECORDING_QUESTION);
        if (gamePhase === GamePhase.USER_TURN_ANSWER)
          setGamePhase(GamePhase.USER_RECORDING_ANSWER);
        startRecording();
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === TRANSCRIPTION_KEY) {
        // Allow stopping only if we were in a recording phase
        const wasRecordingPhase =
          gamePhase === GamePhase.USER_RECORDING_QUESTION ||
          gamePhase === GamePhase.USER_RECORDING_ANSWER;

        keydownTriggeredRef.current = false;
        if (wasRecordingPhase && isRecordingRef.current) {
          console.log("App: Spacebar Up - Stopping recording and sending...");
          // Stop recording. The result will be handled by the `currentTranscription` effect below.
          stopRecordingAndSend();
          // Status message is handled internally by the hook
        } else {
          console.log(
            "App: Spacebar Up - Not in a valid recording state to stop."
          );
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      if (isRecordingRef.current) {
        console.log("App: Unmounting while recording - Stopping recording...");
        stopRecordingAndSend();
      }
    };
    // Add gamePhase to dependencies to re-evaluate `canRecord`
  }, [startRecording, stopRecordingAndSend, gamePhase]);

  // Effect 10: Capture Final Transcription after recording stops
  // This effect watches the transcription coming *from the hook*
  useEffect(() => {
    // Only process if we just stopped recording (isRecording is false, but was true)
    // and we are in a phase *expecting* a transcription result.
    const previousPhase =
      gamePhase === GamePhase.USER_RECORDING_QUESTION
        ? GamePhase.AI_PROCESSING_QUESTION
        : gamePhase === GamePhase.USER_RECORDING_ANSWER
        ? GamePhase.AI_PROCESSING_ANSWER
        : null;

    if (!isRecording && previousPhase && currentTranscription) {
      console.log("App: Final transcription captured:", currentTranscription);
      finalTranscriptionRef.current = currentTranscription; // Store it
      // Now transition to the processing phase, which will use the ref's value
      setGamePhase(previousPhase);
    }
    // Depend on isRecording and the raw currentTranscription from the hook
  }, [isRecording, currentTranscription, gamePhase]);

  // --- Helper Function for Button Class ---
  const getButtonClass = (/* ... unchanged ... */): string => {
    const baseClasses =
      "px-3 py-1 rounded text-xs transition-colors text-white min-w-[60px] text-center";
    const activeSwitchClasses = "bg-cyan-600 font-semibold hover:bg-cyan-500";
    const activeClasses = "bg-emerald-600 font-semibold";
    const inactiveClasses = "bg-zinc-600 hover:bg-zinc-500";
    const disabledClasses =
      "bg-zinc-700 text-zinc-500 opacity-50 cursor-not-allowed";

    // Simplified example, needs full implementation from your original code
    return `${baseClasses} ${inactiveClasses}`; // Placeholder
  };

  // --- Render ---
  return (
    <div className="relative min-h-screen bg-black flex flex-col">
      {/* --- TV Screen --- */}
      <div className="flex-grow flex items-center justify-center">
        <OldTvScreen
          glassesStyle={glassesStyle}
          pupilAscii={currentPupilAscii}
          mouthState={mouthState}
          eyeOffsetX={eyeOffsetX}
          eyeOffsetY={eyeOffsetY}
        />
      </div>
      {/* --- Display Area (Below Face) --- */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 w-full max-w-4xl p-4 bg-black/80 rounded-lg shadow-lg backdrop-blur-sm">
        {/* Game Status Message */}
        <p className="text-center text-lg text-emerald-300 font-mono whitespace-pre-wrap mb-2">
          {displayMessage || "..."}
        </p>
        {/* Transcription / STT Status */}
        <p className="text-center text-sm text-yellow-300 font-mono h-4">
          {/* Show STT status only when relevant */}
          {gamePhase === GamePhase.USER_RECORDING_QUESTION ||
          gamePhase === GamePhase.USER_RECORDING_ANSWER ||
          gamePhase === GamePhase.AI_PROCESSING_QUESTION || // Show transcription here too
          gamePhase === GamePhase.AI_PROCESSING_ANSWER
            ? currentTranscription || sttStatusMessage
            : ""}
          {/* Add indicator for when user should speak */}
          {(gamePhase === GamePhase.USER_TURN_ASK ||
            gamePhase === GamePhase.USER_TURN_ANSWER) &&
          !isRecording
            ? "Hold [SPACE] to speak"
            : ""}
          {isRecording ? "Recording..." : ""}
        </p>
        {/* Error Message */}
        {errorMessage && (
          <p className="text-center text-sm text-red-400 font-mono mt-1">
            Error: {errorMessage}
          </p>
        )}
        {/* Game Over Controls */}
        {(gamePhase === GamePhase.GAME_OVER_AI_WINS ||
          gamePhase === GamePhase.GAME_OVER_USER_WINS ||
          gamePhase === GamePhase.ERROR) && (
          <div className="text-center mt-4">
            <button
              onClick={() => {
                // Reset all game states to initial values
                setGamePhase(GamePhase.LOADING);
                setAiSecretAnimal(null);
                setAiPossibleAnimals([]);
                setLastUserQuestion(null);
                setLastAiAnswer(null);
                setLastAiQuestion(null);
                setLastUserAnswer(null);
                setMessageToSpeak("");
                setDisplayMessage("Loading game...");
                setErrorMessage(null);
                setAiQuestionHistory([]);
                finalTranscriptionRef.current = null;
                // Reset visual elements maybe?
                setGlassesStyle("neutral");
                setPupilStateBase("center");
                resetEyePosition();
                setMouthStateBase("neutral");
                if (isTalkingContinuouslyRef.current) toggleTalking(); // Ensure mouth stops
              }}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded font-semibold"
            >
              Play Again?
            </button>
          </div>
        )}
      </div>
      {/* --- Original Control Panel (Top Right) --- */}
      {/* Keep this if you still need manual controls for face parts */}
      {/* Hide it during the game? Add a condition: `{!aiSecretAnimal && ...}` maybe? */}
      <div className="absolute top-0 right-0 z-10 flex flex-wrap justify-center gap-x-8 gap-y-4 p-4 bg-zinc-800/90 rounded-lg shadow-lg max-w-3xl backdrop-blur-sm">
        {/* --- Transcription Section (Displays Status Only) --- */}
        <div className="flex flex-col gap-2 items-center w-full sm:w-auto">
          <span className="text-white font-bold text-sm mb-1 uppercase tracking-wider">
            Input Status
          </span>
          <div // Changed from button to div
            className={`w-full px-4 py-2 rounded text-white font-semibold transition-colors text-sm ${
              isRecording
                ? "bg-red-600" // Style if recording
                : gamePhase === GamePhase.USER_TURN_ASK ||
                  gamePhase === GamePhase.USER_TURN_ANSWER
                ? "bg-green-600" // Style if ready for user input
                : "bg-gray-500" // Style otherwise (e.g., AI turn)
            } cursor-default`}
            aria-label={
              isRecording
                ? "Recording user input..."
                : gamePhase === GamePhase.USER_TURN_ASK ||
                  gamePhase === GamePhase.USER_TURN_ANSWER
                ? "Ready for user input (Hold Space)"
                : "Waiting for AI or processing..."
            }
            aria-live="polite"
          >
            {isRecording
              ? "Recording..."
              : gamePhase === GamePhase.USER_TURN_ASK ||
                gamePhase === GamePhase.USER_TURN_ANSWER
              ? "Ready"
              : "AI Turn"}
          </div>
          <p className="text-zinc-400 text-[10px] mt-1 h-4 text-center">
            {sttStatusMessage}
          </p>
        </div>

        {/* --- Other Controls (Pupils, Mouth, Glasses, Eyes) --- */}
        {/* Keep the rest of your original control sections here */}
        {/* ... (Pupils, Mouth, Glasses, Position Yeux sections) ... */}
      </div>{" "}
      {/* End Control Panel */}
    </div>
  );
}

export default App;
