import cv2

from reconhecimento_facial.annotator import annotate_image
from reconhecimento_facial.recognizer import (
    Recognition,
    recognize_frame,
)
from reconhecimento_facial.settings import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    PROCESS_EVERY_N_FRAMES,
)


def open_camera() -> cv2.VideoCapture:
    camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not camera.isOpened():
        raise RuntimeError(
            "Não foi possível acessar a câmera. "
            "Verifique as permissões do macOS."
        )

    return camera


def main() -> None:
    camera = open_camera()

    frame_number = 0
    last_recognitions: list[Recognition] = []

    print("Câmera iniciada. Pressione Q para encerrar.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Não foi possível capturar o frame.")
                break

            # Comportamento semelhante a um espelho.
            frame = cv2.flip(frame, 1)

            frame = cv2.resize(
                frame,
                (CAMERA_WIDTH, CAMERA_HEIGHT),
            )

            should_process = (
                frame_number % PROCESS_EVERY_N_FRAMES == 0
            )

            if should_process:
                try:
                    last_recognitions = recognize_frame(frame)
                except Exception as error:
                    print(f"Erro no reconhecimento: {error}")
                    last_recognitions = []

            output = annotate_image(
                frame,
                last_recognitions,
            )

            cv2.putText(
                output,
                "Pressione Q para sair",
                (10, CAMERA_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Sistema de reconhecimento facial",
                output,
            )

            frame_number += 1

            pressed_key = cv2.waitKey(1) & 0xFF

            if pressed_key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()