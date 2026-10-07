import cv2
import numpy as np

from reconhecimento_facial.recognizer import Recognition

GREEN = (0, 255, 0)
RED = (0, 0, 255)


def _draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    text_y = y - 10 if y > 30 else y + 25

    cv2.putText(
        image,
        text,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def annotate_image(
    image: np.ndarray,
    recognitions: list[Recognition],
) -> np.ndarray:
    output = image.copy()

    if not recognitions:
        height, width = output.shape[:2]

        cv2.rectangle(
            output,
            (5, 5),
            (width - 5, height - 5),
            RED,
            3,
        )

        _draw_label(
            output,
            "ACESSO NEGADO",
            15,
            35,
            RED,
        )

        return output

    for recognition in recognitions:
        color = GREEN if recognition.authorized else RED
        text = (
            "ACESSO LIBERADO"
            if recognition.authorized
            else "ACESSO NEGADO"
        )

        cv2.rectangle(
            output,
            (recognition.x, recognition.y),
            (
                recognition.x + recognition.width,
                recognition.y + recognition.height,
            ),
            color,
            3,
        )

        _draw_label(
            output,
            text,
            recognition.x,
            recognition.y,
            color,
        )

    return output