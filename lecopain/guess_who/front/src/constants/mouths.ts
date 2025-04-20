// src/constants/mouths.ts

// Add new states like 'talking1', 'talking2'
export type MouthState = 'neutral' | 'talking1' | 'talking2'; // Adaptez

export const mouthsMap: Record<MouthState, string[]> = {
  neutral: [
    `                                                                                                                         `,
    `                                                   ███████████                                                           `,
    `                                                  █████████████                                                          `,
  ],
  // Example talking state 1 (slightly open)
  talking1: [
    ` `,
    `                                                   ███     ███                                                           `,
    `                                                  █████████████                                                          `,
  ],
  // Example talking state 2 (more open)
  talking2: [
    `                                                   ███     ███                                                           `,
    `                                                  ███       ███                                                          `,
    `                                                   █████████                                                             `,
  ],
  // Add more states if desired...
};
