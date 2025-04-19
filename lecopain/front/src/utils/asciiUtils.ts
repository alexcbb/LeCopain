// src/utils/asciiUtils.ts

/** Applique un décalage horizontal à des lignes ASCII */
export function applyHorizontalShift(lines: string[], offsetX: number): string[] {
    if (offsetX === 0 || !lines || lines.length === 0) return lines;
    if (offsetX > 0) {
        const padding = ' '.repeat(offsetX);
        return lines.map(line => padding + line);
    } else {
        const absOffset = Math.abs(offsetX);
        return lines.map(line => line.substring(absOffset));
    }
}

/**
 * Applique un décalage vertical aux lignes de pupilles en ajoutant/supprimant
 * des lignes VIDES, tout en conservant la hauteur totale du bloc.
 */
export function applyVerticalShiftToPupils(
    lines: string[],
    offsetY: number,
    originalHeight: number, // Hauteur attendue (celle des lunettes)
    lineWidth: number // Largeur pour créer des lignes vides
): string[] {
     if (offsetY === 0 || !lines || !lines.length || originalHeight <= 0) return lines;

     // S'assurer que lineWidth est au moins 0
     const safeLineWidth = Math.max(0, lineWidth);
     const currentLines = [...lines];
     const blankLine = ' '.repeat(safeLineWidth);

     if (offsetY > 0) { // Descendre
         const linesToRemove = Math.min(offsetY, currentLines.length);
         currentLines.splice(0, linesToRemove);
         // Ne pas ajouter plus que la hauteur originale
         const linesToAdd = Math.min(offsetY, originalHeight - currentLines.length);
         for (let i = 0; i < linesToAdd; i++) {
             currentLines.push(blankLine);
         }
     } else { // Monter
         const absOffset = Math.abs(offsetY);
         const linesToRemove = Math.min(absOffset, currentLines.length);
         currentLines.splice(currentLines.length - linesToRemove, linesToRemove);
         // Ne pas ajouter plus que la hauteur originale
         const linesToAdd = Math.min(absOffset, originalHeight - currentLines.length);
         for (let i = 0; i < linesToAdd; i++) {
             currentLines.unshift(blankLine);
         }
     }

     // Ajustement final pour garantir la hauteur exacte
     if (currentLines.length < originalHeight) {
         const paddingNeeded = originalHeight - currentLines.length;
         if (offsetY >= 0) { // Padding en bas si on descendait/stable
             for (let i = 0; i < paddingNeeded; i++) currentLines.push(blankLine);
         } else { // Padding en haut si on montait
             for (let i = 0; i < paddingNeeded; i++) currentLines.unshift(blankLine);
         }
     } else if (currentLines.length > originalHeight) {
         currentLines.splice(originalHeight); // Tronquer si trop long
     }

     return currentLines;
}


/** Padde les lignes à une largeur donnée */
export function padLinesToWidth(lines: string[], width: number): string[] {
    if (!lines) return [];
    const safeWidth = Math.max(0, width);
    return lines.map(line => {
        const currentLength = line?.length || 0; // Gérer ligne potentiellement undefined/null
        return (line || '') + ' '.repeat(Math.max(0, safeWidth - currentLength));
    });
}