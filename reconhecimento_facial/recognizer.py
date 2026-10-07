from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

from reconhecimento_facial.settings import (
    AUTHORIZED_DIR,
    DETECTOR_BACKEND,
    DISTANCE_METRIC,
    MODEL_NAME,
)


@dataclass(frozen=True, slots=True)
class Recognition:
    x: int
    y: int
    width: int
    height: int
    authorized: bool
    identity: str | None = None
    distance: float | None = None


def _get_identity(identity_path: str) -> str:
    return Path(identity_path).parent.name


def _recognize(
    image_source: str | np.ndarray,
    source_name: str,
    verbose: bool = False,
) -> list[Recognition]:
    try:
        detected_faces = DeepFace.extract_faces(
            img_path=image_source,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
    except ValueError:
        if verbose:
            print(f"  Nenhum rosto detectado em {source_name}")

        return []

    matches_by_face = DeepFace.find(
        img_path=image_source,
        db_path=str(AUTHORIZED_DIR),
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        distance_metric=DISTANCE_METRIC,
        enforce_detection=True,
        align=True,
        silent=True,
    )

    recognitions: list[Recognition] = []

    for face_index, detected_face in enumerate(detected_faces):
        area = detected_face["facial_area"]

        x = int(area["x"])
        y = int(area["y"])
        width = int(area["w"])
        height = int(area["h"])

        if face_index >= len(matches_by_face):
            match_dataframe = None
        else:
            match_dataframe = matches_by_face[face_index]

        if match_dataframe is None or match_dataframe.empty:
            recognitions.append(
                Recognition(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    authorized=False,
                )
            )
            continue

        best_match = (
            match_dataframe
            .sort_values("distance")
            .iloc[0]
        )

        distance = float(best_match["distance"])
        threshold = float(best_match["threshold"])
        identity = _get_identity(best_match["identity"])
        authorized = distance <= threshold

        if verbose:
            print(
                f"  Rosto {face_index + 1}: "
                f"candidato={identity}, "
                f"distância={distance:.4f}, "
                f"limite={threshold:.4f}, "
                f"autorizado={authorized}"
            )

        recognitions.append(
            Recognition(
                x=x,
                y=y,
                width=width,
                height=height,
                authorized=authorized,
                identity=identity if authorized else None,
                distance=distance,
            )
        )

    return recognitions


def recognize_image(
    image_path: Path,
) -> tuple[np.ndarray, list[Recognition]]:
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Não foi possível abrir: {image_path}")

    recognitions = _recognize(
        image_source=str(image_path),
        source_name=image_path.name,
        verbose=True,
    )

    return image, recognitions


def recognize_frame(
    frame: np.ndarray,
) -> list[Recognition]:
    return _recognize(
        image_source=frame,
        source_name="câmera",
        verbose=False,
    )