// src/components/OldTvScreen.tsx (MODIFIÉ pour recevoir pupilAscii)

import React from "react";
import AsciiFace from './AsciiFace';
import { GlassesStyle } from '../constants/glasses';
// import { PupilState } from '../constants/pupils'; // Plus besoin de PupilState ici
import { MouthState } from '../constants/mouths';

// Interface MISE À JOUR
interface OldTvScreenProps {
  glassesStyle: GlassesStyle;
  pupilAscii: string[]; // << Changé ici
  mouthState: MouthState;
  eyeOffsetX: number;
  eyeOffsetY: number;
}

const OldTvScreen: React.FC<OldTvScreenProps> = ({
  glassesStyle,
  pupilAscii, // << Changé ici
  mouthState,
  eyeOffsetX,
  eyeOffsetY
}) => {

  return (
    <div className="screen-container relative w-screen h-screen overflow-hidden rounded-[15px] bg-crt-bg shadow-inset-screen">
      <pre
        className="
          text-crt-green font-mono text-[11px] leading-tight tracking-normal
          absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
          w-max text-left
          [text-shadow:0_0_5px_#33ff33,0_0_10px_rgba(51,255,51,0.5)]
        "
      >
        <AsciiFace
          glassesStyle={glassesStyle}
          // Passer le tableau ASCII directement
          pupilLines={pupilAscii} // << Nom de prop changé pour AsciiFace
          mouthState={mouthState}
          eyeOffsetX={eyeOffsetX}
          eyeOffsetY={eyeOffsetY}
        />
      </pre>
    </div>
  );
};

export default OldTvScreen;