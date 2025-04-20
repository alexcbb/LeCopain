// src/hooks/useRealtimeTranscription.ts (MODIFIED FOR HTTP POST)
import { useState, useRef, useEffect, useCallback } from "react";

// --- Constantes ---
// L'URL du WebSocket n'est plus nécessaire
const API_ENDPOINT = "http://localhost:8000/api/stt/transcribe"; // URL du nouvel endpoint POST
// TIMESLICE_MS peut rester pour collecter les chunks périodiquement
const TIMESLICE_MS = 500;

// --- Interface pour les valeurs retournées par le hook (mise à jour) ---
interface UseRealtimeTranscriptionReturn {
    isRecording: boolean; // Renommé pour plus de clarté
    statusMessage: string;
    transcriptionText: string;
    // translationText: string; // On le retire si le backend ne le fournit plus
    startRecording: () => void; // Renommé
    stopRecordingAndSend: () => void; // Renommé et logique changée
}

export function useRealtimeTranscription(): UseRealtimeTranscriptionReturn {
    // --- États Internes ---
    const [isRecording, setIsRecording] = useState(false);
    const [statusMessage, setStatusMessage] = useState("Prêt à enregistrer");
    const [transcriptionText, setTranscriptionText] = useState("");
    // const [translationText, setTranslationText] = useState(""); // Retiré

    // --- Références ---
    // socketRef n'est plus nécessaire
    // const socketRef = useRef<WebSocket | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioStreamRef = useRef<MediaStream | null>(null);
    const audioChunksRef = useRef<Blob[]>([]); // Pour stocker les morceaux audio
    const isBusyRef = useRef(false); // Pour éviter les clics rapides

    // --- Arrêt de l'enregistrement et Envoi ---
    // Cette fonction arrête seulement l'enregistreur. L'envoi se fera dans onstop.
    const stopRecordingAndSend = useCallback(() => {
        if (isBusyRef.current || !mediaRecorderRef.current || mediaRecorderRef.current.state !== "recording") {
            console.warn("useRealtimeTranscription: Stop request ignored, not recording or busy.");
            return;
        }
        isBusyRef.current = true;
        console.log(">>> useRealtimeTranscription: stopRecordingAndSend CALLED <<<");

        setStatusMessage("Traitement audio...");
        setIsRecording(false); // Mise à jour immédiate
        // --- AJOUTER CE LOG ---
        console.log(">>> useRealtimeTranscription: setIsRecording(false) CALLED. isRecording should be false now.");
        // --------------------

        try {
            mediaRecorderRef.current.stop();
             console.log("useRealtimeTranscription: MediaRecorder stop() called.");
        } catch (e) {
            console.error("Error calling MediaRecorder.stop():", e);
            setStatusMessage("Erreur arrêt enregistrement.");
            if (audioStreamRef.current) {
                audioStreamRef.current.getTracks().forEach((track) => track.stop());
                audioStreamRef.current = null;
            }
            mediaRecorderRef.current = null;
            audioChunksRef.current = [];
            isBusyRef.current = false;
        }

    }, []);


    // --- Démarrage de l'Enregistrement ---
    const startRecording = useCallback(async () => {
        if (isRecording || isBusyRef.current) {
            console.warn("useRealtimeTranscription: Start request ignored, already recording or busy.");
            return;
        }
        isBusyRef.current = true;
        console.log("useRealtimeTranscription: Attempting to start recording...");
        setStatusMessage("Préparation...");

        // Nettoyage préventif
        if (mediaRecorderRef.current || audioStreamRef.current) {
            console.warn("useRealtimeTranscription: Previous resources detected. Cleaning up before start...");
            const recorder = mediaRecorderRef.current;
            const stream = audioStreamRef.current;
            if (recorder && recorder.state === "recording") try { recorder.stop(); } catch (e) { }
            if (stream) stream.getTracks().forEach(t => t.stop());
            mediaRecorderRef.current = null;
            audioStreamRef.current = null;
        }
        audioChunksRef.current = []; // Vider les chunks précédents
        setTranscriptionText(""); // Reset transcription on start
        // setTranslationText(""); // Reset si utilisé

        setStatusMessage("Demande accès micro...");

        try {
            // 1. Accès Microphone
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioStreamRef.current = stream;
            setStatusMessage("Micro OK. Démarrage enregistrement...");

            // 2. Créer MediaRecorder
            const options = { mimeType: "audio/webm;codecs=opus" }; // ou autre type supporté
            let actualMimeType = options.mimeType;
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                console.warn(`mimeType ${options.mimeType} non supporté, essai par défaut.`);
                actualMimeType = "";
            }
            const recorder = new MediaRecorder(stream, actualMimeType ? options : undefined);
            mediaRecorderRef.current = recorder;

            // 3. Setup Listeners pour MediaRecorder
            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                    // console.log(`Chunk recorded: ${event.data.size} bytes`); // DEBUG
                }
            };

            recorder.onerror = (event) => {
                console.error("useRealtimeTranscription: MediaRecorder error:", event);
                setStatusMessage("Erreur enregistrement.");
                // Tenter un nettoyage
                stopRecordingAndSend(); // Appelle la fonction qui gère l'arrêt (et le nettoyage si erreur)
            };

            // --- C'EST ICI QUE L'ENVOI SE FAIT MAINTENANT ---
            recorder.onstop = async () => {
                 console.log("useRealtimeTranscription: MediaRecorder stopped (onstop event). Processing chunks...");
                // Étape cruciale : créer le Blob final et envoyer la requête
                if (audioChunksRef.current.length === 0) {
                    console.warn("No audio chunks recorded.");
                    setStatusMessage("Aucune donnée audio enregistrée.");
                     // Libérer micro ici car on n'attend pas de fetch
                     if (audioStreamRef.current) {
                         audioStreamRef.current.getTracks().forEach((track) => track.stop());
                         audioStreamRef.current = null;
                     }
                    isBusyRef.current = false; // Débloquer
                    return;
                }

                 // Utiliser le mimeType réel si disponible, sinon laisser le navigateur deviner
                const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorderRef.current?.mimeType || 'audio/webm' });
                audioChunksRef.current = []; // Vider pour la prochaine fois

                const formData = new FormData();
                formData.append('file', audioBlob, 'recording.webm'); // Le nom de fichier est arbitraire

                setStatusMessage("Envoi & Transcription...");

                try {
                    const response = await fetch(API_ENDPOINT, {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) {
                        // Essayer de lire le corps de l'erreur s'il y en a un
                        let errorBody = `HTTP error! status: ${response.status}`;
                        try {
                             const errorData = await response.json();
                             errorBody = errorData.detail || errorBody; // FastAPI met souvent l'erreur dans 'detail'
                        } catch (e) { /* Ignorer si le corps n'est pas JSON */ }
                        throw new Error(errorBody);
                    }

                    const result = await response.json();
                    console.log("Transcription result:", result);

                    if (result.transcription !== undefined) {
                         setTranscriptionText(result.transcription || "[Silence détecté]"); // Afficher quelque chose si vide
                         setStatusMessage("Transcription terminée.");
                    } else if (result.error) {
                         setTranscriptionText(`[Erreur: ${result.error}]`);
                         setStatusMessage("Erreur de transcription.");
                    } else {
                        setTranscriptionText("[Réponse invalide]");
                        setStatusMessage("Erreur: Réponse serveur invalide.");
                    }
                    // Gérer translation si présente :
                    // if (result.translation) setTranslationText(result.translation);

                } catch (error) {
                    console.error("Error during transcription request:", error);
                    setStatusMessage(`Erreur: ${(error as Error).message}`);
                    setTranscriptionText(""); // Clear text on error
                } finally {
                     // Libérer le micro APRÈS la fin de la requête (succès ou échec)
                     if (audioStreamRef.current) {
                         audioStreamRef.current.getTracks().forEach((track) => track.stop());
                         audioStreamRef.current = null;
                          console.log("Microphone stream stopped after transcription attempt.");
                     }
                     mediaRecorderRef.current = null; // Nettoyer la ref recorder
                     isBusyRef.current = false; // Débloquer le hook
                     console.log(">>> useRealtimeTranscription: Request finished, busy lock released <<<");
                }
            }; // Fin de onstop

            // 4. Démarrer l'enregistrement (collecte les chunks)
            recorder.start(TIMESLICE_MS);
            setIsRecording(true);
            setStatusMessage("Enregistrement en cours...");
            isBusyRef.current = false;
            // Ne pas débloquer isBusyRef ici, l'enregistrement est en cours

        } catch (error) {
            console.error("useRealtimeTranscription: Failed getUserMedia or MediaRecorder setup:", error);
            if ((error as Error).name === 'NotAllowedError') {
                setStatusMessage("Erreur: Accès micro refusé.");
            } else if ((error as Error).name === 'NotFoundError') {
                setStatusMessage("Erreur: Aucun micro détecté.");
            } else {
                setStatusMessage("Erreur démarrage: " + (error as Error).message);
            }
            // Nettoyer si getUserMedia échoue
            if (audioStreamRef.current) {
                audioStreamRef.current.getTracks().forEach((track) => track.stop());
                audioStreamRef.current = null;
            }
             mediaRecorderRef.current = null;
             isBusyRef.current = false; // Débloquer après erreur
        } finally {
             // Débloquer le 'busy' seulement si le démarrage a échoué avant l'enregistrement
             if (!isRecording && isBusyRef.current) {
                  // isBusyRef.current = false; // Déjà fait dans le catch
             }
             // Si on est en train d'enregistrer, isBusy restera true jusqu'à l'arrêt.
        }
    }, []); // stopRecordingAndSend est stable


    // --- Effet de Nettoyage Global ---
    useEffect(() => {
        return () => {
            console.log("useRealtimeTranscription: Cleanup on unmount.");
             // S'assurer que tout est arrêté
             const recorder = mediaRecorderRef.current;
             const stream = audioStreamRef.current;
             if (recorder && recorder.state === "recording") try { recorder.stop(); } catch(e){}
             if (stream) stream.getTracks().forEach(t => t.stop());
             mediaRecorderRef.current = null;
             audioStreamRef.current = null;
             audioChunksRef.current = [];
             isBusyRef.current = false; // S'assurer que c'est débloqué
        };
    }, []); // Seulement au montage/démontage

    // --- Retourner les valeurs et fonctions ---
    return {
        isRecording,
        // isTalking: isRecording, // Garder l'alias si l'animation bouche doit suivre
        statusMessage,
        transcriptionText,
        // translationText, // Retiré
        startRecording,
        stopRecordingAndSend,
    };
}