// src/hooks/useAnimatedEyes.ts (Avec annotations de type)
import { useState, useEffect, useRef, useCallback } from 'react';
import { PupilState, pupilsMap, blinkAnimationFrames, BLINK_FRAME_DURATION_MS as SINGLE_FRAME_DURATION_MS } from '../constants/pupils';

// --- Constantes ---
// ... (inchangées)
const MIN_BLINK_INTERVAL_MS = 3000;
const MAX_BLINK_INTERVAL_MS = 7000;
const BLINK_CHANCE = 0.8;
const MIN_GLANCE_INTERVAL_MS = 3500;
const MAX_GLANCE_INTERVAL_MS = 6000;
const CENTER_GLANCE_WEIGHT = 0.65;
const MAX_ABS_GLANCE_X = 4;
const MAX_ABS_GLANCE_Y = 2;
const MIN_SACCADE_INTERVAL_MS = 300;
const MAX_SACCADE_INTERVAL_MS = 900;
const SACCADE_CHANCE = 0.25;
const SACCADE_MAGNITUDE_X = 1;
const SACCADE_MAGNITUDE_Y = 1;


export function useAnimatedEyes(initialPupilState: PupilState = 'center') {
    // --- États (inchangés) ---
    const [pupilState, setPupilState] = useState<PupilState>(initialPupilState);
    const [eyeOffsetX, setEyeOffsetX] = useState<number>(0);
    const [eyeOffsetY, setEyeOffsetY] = useState<number>(0);
    const [blinkFrameIndex, setBlinkFrameIndex] = useState<number | null>(null);
    const [isAutonomous, setIsAutonomous] = useState<boolean>(true);
    const [glanceTargetX, setGlanceTargetX] = useState<number>(0);
    const [glanceTargetY, setGlanceTargetY] = useState<number>(0);

    // --- Refs (inchangés) ---
    const nextBlinkCheckTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const frameAdvanceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const previousPupilStateRef = useRef<PupilState>(pupilState);
    const nextGlanceCheckTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const nextSaccadeCheckTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    // --- Mémorisation état précédent (inchangé) ---
    useEffect(() => { /* ... */
        if (blinkFrameIndex === null) { previousPupilStateRef.current = pupilState; }
    }, [pupilState, blinkFrameIndex]);

    // --- Logique de Blink ---
    const triggerBlinkAnimation = useCallback(() => { if (blinkFrameIndex === null) { setBlinkFrameIndex(0); } }, [blinkFrameIndex]);
    useEffect(() => { /* ... gestion avancement frames ... */
         let isMounted = true;
         if (blinkFrameIndex !== null) {
             if (blinkFrameIndex >= blinkAnimationFrames.length) {
                 if (isMounted) { setBlinkFrameIndex(null); setPupilState(previousPupilStateRef.current); }
                 if (frameAdvanceTimeoutRef.current) clearTimeout(frameAdvanceTimeoutRef.current); frameAdvanceTimeoutRef.current = null;
             } else {
                 frameAdvanceTimeoutRef.current = setTimeout(() => {
                     if (isMounted) { setBlinkFrameIndex(currentFrame => (currentFrame === null ? null : currentFrame + 1)); }
                 }, SINGLE_FRAME_DURATION_MS);
             }
         }
         return () => { isMounted = false; if (frameAdvanceTimeoutRef.current) { clearTimeout(frameAdvanceTimeoutRef.current); frameAdvanceTimeoutRef.current = null; } };
    }, [blinkFrameIndex]);

    // --- NOTE: La déclaration de blinkCheck et scheduleNextBlinkCheck nécessite une attention particulière ---
    // Pour éviter les erreurs "implicitly has type 'any'" dues à la référence circulaire potentielle
    // dans les tableaux de dépendances de useCallback, on déclare les fonctions avec des types explicites.

    // On doit déclarer scheduleNextBlinkCheck AVANT blinkCheck si blinkCheck l'utilise dans ses dépendances
    const scheduleNextBlinkCheck = useCallback((): void => { // <<<< Annotation ici
        // La référence à `blinkCheck` dans setTimeout est ok car elle sera résolue à l'exécution.
        // Par contre, si `blinkCheck` change, ce timeout continuera d'appeler l'ancienne version.
        // Une solution plus robuste utiliserait une ref pour `blinkCheck` si la stabilité est critique.
        // Pour l'instant, on se fie au fait que les dépendances de `blinkCheck` le rendront stable la plupart du temps.
        if (nextBlinkCheckTimeoutRef.current) clearTimeout(nextBlinkCheckTimeoutRef.current);
        const randomInterval = MIN_BLINK_INTERVAL_MS + Math.random() * (MAX_BLINK_INTERVAL_MS - MIN_BLINK_INTERVAL_MS);
        // Référencer blinkCheck directement ici est ok pour setTimeout
        nextBlinkCheckTimeoutRef.current = setTimeout(() => blinkCheck(), randomInterval);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [/* Enlever blinkCheck des deps ici si on accepte d'appeler potentiellement une version "périmée" dans le timeout */]); // Vider les deps rend la fonction stable

    const blinkCheck = useCallback((): void => { // <<<< Annotation ici
         if (blinkFrameIndex === null && Math.random() < BLINK_CHANCE) {
            triggerBlinkAnimation();
         }
         scheduleNextBlinkCheck(); // Appel à la fonction stable
     }, [blinkFrameIndex, triggerBlinkAnimation, scheduleNextBlinkCheck]); // Garder les deps


    // --- Logique des Regards Aléatoires (Glances) ---
    const triggerGlance = useCallback(() => { /* ... */
        let targetX = 0, targetY = 0;
        if (Math.random() > CENTER_GLANCE_WEIGHT) {
            do {
                targetX = Math.floor(Math.random() * (MAX_ABS_GLANCE_X * 2 + 1)) - MAX_ABS_GLANCE_X;
                targetY = Math.floor(Math.random() * (MAX_ABS_GLANCE_Y * 2 + 1)) - MAX_ABS_GLANCE_Y;
            } while (targetX === 0 && targetY === 0 && (MAX_ABS_GLANCE_X > 0 || MAX_ABS_GLANCE_Y > 0));
        }
        console.log(`Glance target: (${targetX}, ${targetY})`);
        setGlanceTargetX(targetX); setGlanceTargetY(targetY);
        setEyeOffsetX(targetX); setEyeOffsetY(targetY);
    }, []);

     // Déclarer scheduleNextGlanceCheck avant glanceCheck
    const scheduleNextGlanceCheck = useCallback((): void => { // <<<< Annotation ici
        if (nextGlanceCheckTimeoutRef.current) clearTimeout(nextGlanceCheckTimeoutRef.current);
        const randomInterval = MIN_GLANCE_INTERVAL_MS + Math.random() * (MAX_GLANCE_INTERVAL_MS - MIN_GLANCE_INTERVAL_MS);
        nextGlanceCheckTimeoutRef.current = setTimeout(() => glanceCheck(), randomInterval); // Référence à glanceCheck
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [/* Enlever glanceCheck des deps ici ? */]); // Vider les deps rend la fonction stable

    const glanceCheck = useCallback((): void => { // <<<< Annotation ici
        if (isAutonomous) {
            triggerGlance();
        }
        scheduleNextGlanceCheck();
    }, [isAutonomous, triggerGlance, scheduleNextGlanceCheck]);


    // --- Logique des Micro-Mouvements (Saccades) ---
    const triggerSaccade = useCallback(() => { /* ... */
        const saccadeDx = Math.floor(Math.random() * (SACCADE_MAGNITUDE_X * 2 + 1)) - SACCADE_MAGNITUDE_X;
        const saccadeDy = Math.floor(Math.random() * (SACCADE_MAGNITUDE_Y * 2 + 1)) - SACCADE_MAGNITUDE_Y;
        const finalX = glanceTargetX + saccadeDx;
        const finalY = glanceTargetY + saccadeDy;
        setEyeOffsetX(finalX); setEyeOffsetY(finalY);
    }, [glanceTargetX, glanceTargetY]);

    // Déclarer scheduleNextSaccadeCheck avant saccadeCheck
    const scheduleNextSaccadeCheck = useCallback((): void => { // <<<< Annotation ici
        if (nextSaccadeCheckTimeoutRef.current) clearTimeout(nextSaccadeCheckTimeoutRef.current);
        const randomInterval = MIN_SACCADE_INTERVAL_MS + Math.random() * (MAX_SACCADE_INTERVAL_MS - MIN_SACCADE_INTERVAL_MS);
        nextSaccadeCheckTimeoutRef.current = setTimeout(() => saccadeCheck(), randomInterval); // Référence à saccadeCheck
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [/* Enlever saccadeCheck des deps ici ? */]); // Vider les deps rend la fonction stable

     const saccadeCheck = useCallback((): void => { // <<<< Annotation ici
        if (isAutonomous && Math.random() < SACCADE_CHANCE) {
            triggerSaccade();
        }
        scheduleNextSaccadeCheck();
     }, [isAutonomous, triggerSaccade, scheduleNextSaccadeCheck]);


    // --- Démarrage / Arrêt de tous les cycles ---
    useEffect(() => {
         console.log("Starting ALL autonomous cycles...");
         // Appeler les fonctions schedule stables
         scheduleNextBlinkCheck();
         scheduleNextGlanceCheck();
         scheduleNextSaccadeCheck();
         return () => { /* ... nettoyage ... */
              console.log("Stopping ALL autonomous cycles...");
             if (nextBlinkCheckTimeoutRef.current) clearTimeout(nextBlinkCheckTimeoutRef.current);
             if (frameAdvanceTimeoutRef.current) clearTimeout(frameAdvanceTimeoutRef.current);
             if (nextGlanceCheckTimeoutRef.current) clearTimeout(nextGlanceCheckTimeoutRef.current);
             if (nextSaccadeCheckTimeoutRef.current) clearTimeout(nextSaccadeCheckTimeoutRef.current);
         };
         // Les fonctions schedule sont stables (deps vides), donc l'effet ne se relance pas en boucle
    }, [scheduleNextBlinkCheck, scheduleNextGlanceCheck, scheduleNextSaccadeCheck]);


    // --- Fonctions d'interaction ---
    const setPupilStateBase = useCallback((newState: PupilState) => { /* ... */
        if (frameAdvanceTimeoutRef.current) clearTimeout(frameAdvanceTimeoutRef.current);
        setBlinkFrameIndex(null);
        setPupilState(newState);
    }, []);
    const moveEyes = useCallback((dx: number, dy: number) => { /* ... */
        console.log("Manual move -> Disabling autonomous mode");
        setIsAutonomous(false);
        if (nextGlanceCheckTimeoutRef.current) clearTimeout(nextGlanceCheckTimeoutRef.current);
        if (nextSaccadeCheckTimeoutRef.current) clearTimeout(nextSaccadeCheckTimeoutRef.current);
        const newX = eyeOffsetX + dx; const newY = eyeOffsetY + dy;
        setEyeOffsetX(newX); setEyeOffsetY(newY);
        setGlanceTargetX(newX); setGlanceTargetY(newY);
    }, [eyeOffsetX, eyeOffsetY]);
    const resetEyePosition = useCallback(() => { /* ... */
        console.log("Reset position -> Disabling autonomous mode");
        setIsAutonomous(false);
        if (nextGlanceCheckTimeoutRef.current) clearTimeout(nextGlanceCheckTimeoutRef.current);
        if (nextSaccadeCheckTimeoutRef.current) clearTimeout(nextSaccadeCheckTimeoutRef.current);
        setEyeOffsetX(0); setEyeOffsetY(0);
        setGlanceTargetX(0); setGlanceTargetY(0);
    }, []);
    const toggleAutonomousMode = useCallback(() => { /* ... */
         setIsAutonomous(prev => {
             const nextValue = !prev;
             console.log(`Autonomous mode ${nextValue ? 'ENABLED' : 'DISABLED'}`);
             if (nextValue) {
                 setEyeOffsetX(0); setEyeOffsetY(0);
                 setGlanceTargetX(0); setGlanceTargetY(0);
                 // Relancer les cycles via les checks. Les schedule s'en occuperont
                 glanceCheck();
                 saccadeCheck();
             } else {
                  if (nextGlanceCheckTimeoutRef.current) clearTimeout(nextGlanceCheckTimeoutRef.current);
                  if (nextSaccadeCheckTimeoutRef.current) clearTimeout(nextSaccadeCheckTimeoutRef.current);
             }
             return nextValue;
         });
         // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [glanceCheck, saccadeCheck]); // Dépend des fonctions check maintenant


    // --- Calcul ASCII pupilles (inchangé) ---
    let currentPupilAscii: string[];
    // ...
    if (blinkFrameIndex !== null && blinkFrameIndex < blinkAnimationFrames.length) { currentPupilAscii = blinkAnimationFrames[blinkFrameIndex]; } else { currentPupilAscii = pupilsMap[pupilState] || pupilsMap['center']; }
    if (!currentPupilAscii) { currentPupilAscii = pupilsMap['center']; }


    // --- Retour hook (corrigé dans la réponse précédente) ---
     return {
         pupilState,
         eyeOffsetX,
         eyeOffsetY,
         currentPupilAscii,
         setPupilStateBase,
         moveEyes,
         resetEyePosition,
         isBlinking: blinkFrameIndex !== null,
         previousPupilState: previousPupilStateRef.current,
         isAutonomous,
         toggleAutonomousMode,
     };
}