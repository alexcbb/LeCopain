// src/hooks/useAnimatedMouth.ts
import { useState, useEffect, useRef, useCallback } from 'react';
import { MouthState, mouthsMap } from '../constants/mouths';

// --- Constantes ---
const TALKING_FRAME_DURATION_MS = 150; // Durée de chaque forme de bouche en ms
const TALKING_SEQUENCE: MouthState[] = ['talking1', 'talking2', 'talking1']; // Séquence simple

// Interface pour les props du hook
interface UseAnimatedMouthProps {
    initialMouthState?: MouthState;
}

export function useAnimatedMouth({ initialMouthState = 'neutral' }: UseAnimatedMouthProps) {
    // --- États Internes ---
    const [baseMouthState, setBaseMouthState] = useState<MouthState>(initialMouthState);
    const [animatedMouthState, setAnimatedMouthState] = useState<MouthState>(initialMouthState);
    const [talkingFrameIndex, setTalkingFrameIndex] = useState<number>(-1); // Index dans TALKING_SEQUENCE
    // NOUVEAU: État pour savoir si le mode "parler en continu" est activé
    const [isTalkingContinuously, setIsTalkingContinuously] = useState<boolean>(false);
    // RENOMMÉ: Indique si une frame d'animation est activement planifiée
    const [isAnimatingFrame, setIsAnimatingFrame] = useState<boolean>(false);

    // --- Référence pour le Timeout ---
    const animationTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    // NOUVEAU: Référence pour suivre l'état isTalkingContinuously dans les callbacks de timeout
    const isTalkingContinuouslyRef = useRef<boolean>(isTalkingContinuously);
    useEffect(() => {
        isTalkingContinuouslyRef.current = isTalkingContinuously;
    }, [isTalkingContinuously]);


    // --- Fonction pour Arrêter l'Animation (interne) ---
    // Arrête le timeout et réinitialise l'état d'affichage à l'état de base
    const stopAnimationCycle = useCallback(() => {
        if (animationTimeoutRef.current) {
            clearTimeout(animationTimeoutRef.current);
            animationTimeoutRef.current = null;
        }
        setIsAnimatingFrame(false);
        setTalkingFrameIndex(-1);
        // Important : On revient à l'état de BASE, pas nécessairement 'neutral'
        setAnimatedMouthState(baseMouthState);
    }, [baseMouthState]); // Dépend de baseMouthState pour savoir où revenir


    // --- Fonction pour Avancer à la Prochaine Frame (modifiée pour boucler) ---
    const advanceTalkingFrame = useCallback(() => {
        // Utilise la ref pour obtenir la valeur la plus récente dans le callback
        if (!isTalkingContinuouslyRef.current) {
             // Si le mode continu a été désactivé pendant le timeout, on arrête tout
            stopAnimationCycle();
            return;
        }

        setTalkingFrameIndex(prevIndex => {
            let nextIndex = prevIndex + 1;

            // Boucle la séquence si on atteint la fin
            if (nextIndex >= TALKING_SEQUENCE.length) {
                nextIndex = 0; // Recommence la séquence
            }

            // Met à jour la bouche affichée
            setAnimatedMouthState(TALKING_SEQUENCE[nextIndex]);

            // Planifie la frame suivante UNIQUEMENT si on est toujours en mode continu
            if (isTalkingContinuouslyRef.current) {
                 animationTimeoutRef.current = setTimeout(advanceTalkingFrame, TALKING_FRAME_DURATION_MS);
            } else {
                // Si entretemps isTalkingContinuously est devenu false, on arrête après cette frame
                stopAnimationCycle(); // Arrête proprement
                return -1; // Indique la fin pour l'état talkingFrameIndex
            }

            return nextIndex; // Met à jour l'état talkingFrameIndex
        });
    }, [stopAnimationCycle]); // Dépend de stopAnimationCycle


    // --- Fonction pour Basculer le Mode Parler Continu (remplace startTalking) ---
    const toggleTalking = useCallback(() => {
        setIsTalkingContinuously(prev => {
            const nextIsTalking = !prev;
            if (nextIsTalking) {
                // --- Activation ---
                console.log("Starting continuous talk sequence...");
                // Si une animation était déjà en cours (peu probable mais possible), l'arrêter proprement
                stopAnimationCycle();
                // Démarrer la première frame
                setIsAnimatingFrame(true);
                setTalkingFrameIndex(0); // Commence à la première frame
                setAnimatedMouthState(TALKING_SEQUENCE[0]);
                // Planifier la suivante
                animationTimeoutRef.current = setTimeout(advanceTalkingFrame, TALKING_FRAME_DURATION_MS);
            } else {
                // --- Désactivation ---
                console.log("Stopping continuous talk sequence...");
                // L'arrêt effectif se fait dans advanceTalkingFrame ou stopAnimationCycle
                // On se contente de mettre le flag à false, le cycle en cours s'arrêtera
                 // stopAnimationCycle(); // Appel redondant, advanceTalkingFrame le fera
            }
            return nextIsTalking;
        });
    }, [advanceTalkingFrame, stopAnimationCycle]); // Dépend des fonctions d'animation


    // --- Fonction pour Définir l'État de Base (appelée par les boutons) ---
    // Doit maintenant aussi désactiver le mode parler continu
    const setMouthStateBase = useCallback((newState: MouthState) => {
        console.log("Setting base mouth state, stopping continuous talk.");
        // 1. Désactiver le mode continu
        setIsTalkingContinuously(false);
        // 2. Arrêter tout cycle d'animation en cours et revenir à l'état NOUVELLEMENT défini
        //    Il faut passer newState à stopAnimationCycle ou le mettre à jour avant
        setBaseMouthState(newState); // Mettre à jour l'état de base d'abord
        // Arrêter l'animation et afficher directement le nouvel état de base
        if (animationTimeoutRef.current) {
            clearTimeout(animationTimeoutRef.current);
            animationTimeoutRef.current = null;
        }
        setIsAnimatingFrame(false);
        setTalkingFrameIndex(-1);
        setAnimatedMouthState(newState); // Afficher directement le nouvel état

    }, []); // Pas de dépendance directe à stopAnimationCycle ici car on redéfinit la logique d'arrêt


    // --- Effet de Nettoyage ---
    useEffect(() => {
        return () => {
            if (animationTimeoutRef.current) {
                clearTimeout(animationTimeoutRef.current);
            }
        };
    }, []);


    // --- Valeurs Retournées par le Hook ---
    return {
        mouthState: animatedMouthState, // L'état de la bouche à afficher
        baseMouthState,                // L'état de base sélectionné
        setMouthStateBase,             // Pour les boutons de sélection de base
        toggleTalking,                 // La fonction pour le switch "Parler"
        isTalkingContinuously,         // Pour savoir si le mode parler est actif (pour l'état du switch et désactiver les boutons)
    };
}