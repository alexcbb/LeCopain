// src/components/AsciiFace.tsx (NETTOYÉ)
import React from 'react';

import { GlassesStyle, glassesMap } from '../constants/glasses';
import { MouthState, mouthsMap } from '../constants/mouths';
// Importer les utilitaires
import {
    applyHorizontalShift,
    applyVerticalShiftToPupils,
    padLinesToWidth
} from '../utils/asciiUtils';

// Interface (inchangée)
interface AsciiFaceProps {
    glassesStyle: GlassesStyle;
    pupilLines: string[];
    mouthState: MouthState;
    eyeOffsetX: number;
    eyeOffsetY: number;
}

// Constante (peut rester ici ou être déplacée/importée)
const MAX_ABS_OFFSET_X = 20;

// =======================================================================
// FONCTION D'ASSEMBLAGE (Utilise les imports)
// =======================================================================
function assembleAsciiFace(
    glassesStyle: GlassesStyle,
    pupilLines: string[],
    mouthState: MouthState,
    eyeOffsetX: number,
    eyeOffsetY: number
): string {
    // 1. Blocs de base et dimensions
    const glassesLines = glassesMap[glassesStyle] || [];
    const mouthLinesBase = mouthsMap[mouthState] || [];

    const originalEyeHeight = glassesLines.length;
    const originalEyeWidth = glassesLines[0]?.length || 0;
    const originalMouthHeight = mouthLinesBase.length;
    const originalMouthWidth = mouthLinesBase[0]?.length || 0;

    if (originalEyeHeight === 0 || originalMouthHeight === 0 || pupilLines.length !== originalEyeHeight) {
       console.error("Blocs ASCII vides ou dimensions incompatibles.");
       return "Error";
    }

    const limitedOffsetX = Math.max(-MAX_ABS_OFFSET_X, Math.min(MAX_ABS_OFFSET_X, eyeOffsetX));

    // 2. Décaler les PUPILLES (X)
    const horizontallyShiftedPupils = applyHorizontalShift(pupilLines, limitedOffsetX); // Utilise l'import
    const pupilWidthAfterXShift = horizontallyShiftedPupils.reduce((max, line) => Math.max(max, line.length), 0);

    // 3. Décaler les PUPILLES (Y)
    const fullyShiftedPupilLines = applyVerticalShiftToPupils( // Utilise l'import
        horizontallyShiftedPupils,
        eyeOffsetY,
        originalEyeHeight,
        pupilWidthAfterXShift
    );

    // 4. Superposition
    const combinedEyeLines = glassesLines.map((glassLine, lineIndex) => {
        const pupilLine = fullyShiftedPupilLines[lineIndex] || '';
        let combinedLine = '';
        const lineLengthForIteration = Math.max(originalEyeWidth, pupilLine.length);
        for (let charIndex = 0; charIndex < lineLengthForIteration; charIndex++) {
            const glassChar = glassLine[charIndex] || ' ';
            const pupilChar = pupilLine[charIndex] || ' ';
            combinedLine += (pupilChar !== ' ') ? pupilChar : glassChar;
        }
        return combinedLine;
    });
    const finalEyeWidth = combinedEyeLines.reduce((max, line) => Math.max(max, line.length), 0);

    // 5. Largeur Stable
    const maxPossibleEyeWidth = originalEyeWidth + MAX_ABS_OFFSET_X;
    const stableFinalWidth = Math.max(maxPossibleEyeWidth, originalMouthWidth, finalEyeWidth);

    // 6. Hauteur Stable
    const finalHeight = originalEyeHeight + originalMouthHeight;

    // 7. Créer Canvas
    const canvas: string[] = Array(finalHeight).fill(' '.repeat(stableFinalWidth));

    // 8. Dessiner Yeux
    const paddedEyeLines = padLinesToWidth(combinedEyeLines, stableFinalWidth); // Utilise l'import
    paddedEyeLines.forEach((line, index) => {
        if (index >= 0 && index < canvas.length) {
             canvas[index] = line + canvas[index].substring(line.length);
        }
    });

    // 9. Dessiner Bouche
    const paddedMouthLines = padLinesToWidth(mouthLinesBase, stableFinalWidth); // Utilise l'import
    paddedMouthLines.forEach((line, index) => {
        const yCanvas = originalEyeHeight + index;
        if (yCanvas >= 0 && yCanvas < canvas.length) {
            canvas[yCanvas] = line + canvas[yCanvas].substring(line.length);
        }
    });

    // 10. Retourner Canvas
    return canvas.join('\n');
}


// =======================================================================
// COMPOSANT REACT (inchangé)
// =======================================================================
const AsciiFace: React.FC<AsciiFaceProps> = ({
    glassesStyle,
    pupilLines,
    mouthState,
    eyeOffsetX,
    eyeOffsetY
}) => {
  const finalAsciiArt = assembleAsciiFace(
      glassesStyle,
      pupilLines,
      mouthState,
      eyeOffsetX,
      eyeOffsetY
  );
  return <>{finalAsciiArt}</>;
};

export default AsciiFace;