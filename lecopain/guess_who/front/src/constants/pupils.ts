// src/constants/pupils.ts

// Définir les positions possibles pour les pupilles
export type PupilState = 'center' |'big';

// Mapper chaque position à son dessin ASCII
// DOIT AVOIR LES MÊMES DIMENSIONS que glassesMap.neutral
export const pupilsMap: Record<PupilState, string[]> = {
  center: [ // Pupilles au centre
    "","","",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                            ", // Ligne 8
    "                               ████                                     ████                              ", // Ligne 9
    "                             ████████                                 ████████                            ", // Ligne 10
    "                             ████████                                 ████████                            ", // Ligne 11
    "                               ████                                     ████                              ", // Ligne 12
    "                                                                                               ", // Ligne 13
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "","",""
  ],
  big: [ // Pupilles au centre
    "","","",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                               ████                                     ████                ", // Ligne 8
    "                             ████████                                 ████████                            ", // Ligne 9
    "                           ████████████                             ████████████                          ", // Ligne 10
    "                           ████████████                             ████████████                          ", // Ligne 11
    "                             ████████                                 ████████                            ", // Ligne 12
    "                               ████                                     ████                   ", // Ligne 13
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "","",""
  ]
  // Ajoutez d'autres positions ici...
};

// --- Animation de Blink ---

// Durée de chaque frame de l'animation en ms
export const BLINK_FRAME_DURATION_MS = 30; // Ajustez pour vitesse désirée

// Séquence des frames pour l'animation de blink
// Chaque frame DOIT avoir les mêmes dimensions que glassesMap.neutral
export const blinkAnimationFrames: string[][] = [
  // Frame 0: Yeux mi-fermés (exemple)
  [ // Pupilles au centre
    "","","",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                            ", // Ligne 8
    "                               ████                                     ████                              ", // Ligne 9
    "                             ████████                                 ████████                            ", // Ligne 10
    "                             ████████                                 ████████                            ", // Ligne 11
    "                               ████                                     ████                              ", // Ligne 12
    "                                                                                               ", // Ligne 13
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "","","",
  ],
  [ // Pupilles au centre
    "","","",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                            ", // Ligne 8
    "                                                                                                  ", // Ligne 9
    "                             ████████                                 ████████                            ", // Ligne 10
    "                             ████████                                 ████████                            ", // Ligne 11
    "                               ████                                     ████                              ", // Ligne 12
    "                                                                                               ", // Ligne 13
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "","","",
  ],
  [ // Pupilles au centre
    "","","",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                            ", // Ligne 8
    "                                                                                                  ", // Ligne 9
    "                                                                                       ", // Ligne 10
    "                             ████████                                 ████████                            ", // Ligne 11
    "                               ████                                     ████                              ", // Ligne 12
    "                                                                                               ", // Ligne 13
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "","","",
  ],

  [ // Pupilles au centre
    "","","",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                            ", // Ligne 8
    "                                                                                                  ", // Ligne 9
    "                                                                                       ", // Ligne 10
    "                             ██    ██                                 ██    ██                            ", // Ligne 11
    "                               ████                                     ████                              ", // Ligne 12
    "                                                                                               ", // Ligne 13
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "                                                                                                                        ",
    "","","",
  ],
  
];
