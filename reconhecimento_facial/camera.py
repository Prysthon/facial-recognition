from queue import Empty, Full, Queue
from threading import Event, Thread
import time

import cv2
import numpy as np

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

RecognitionResult = tuple[
    float,
    list[Recognition],
    str | None,
]


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


def replace_queue_item(
    target_queue: Queue,
    item: object,
) -> None:
    try:
        target_queue.get_nowait()
    except Empty:
        pass

    try:
        target_queue.put_nowait(item)
    except Full:
        pass


def recognition_worker(
    frame_queue: Queue[np.ndarray],
    result_queue: Queue[RecognitionResult],
    stop_event: Event,
) -> None:
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue

        try:
            recognitions = recognize_frame(frame)

            result: RecognitionResult = (
                time.monotonic(),
                recognitions,
                None,
            )
        except Exception as error:
            result = (
                time.monotonic(),
                [],
                str(error),
            )

        replace_queue_item(result_queue, result)


def draw_status(
    frame: np.ndarray,
    text: str,
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    camera = open_camera()

    frame_queue: Queue[np.ndarray] = Queue(maxsize=1)
    result_queue: Queue[RecognitionResult] = Queue(maxsize=1)
    stop_event = Event()

    worker = Thread(
        target=recognition_worker,
        args=(frame_queue, result_queue, stop_event),
        daemon=True,
    )

    worker.start()

    frame_number = 0
    last_result: RecognitionResult | None = None

    print("Câmera iniciada. Pressione Q para encerrar.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Não foi possível capturar o frame.")
                break

            frame = cv2.flip(frame, 1)

            frame = cv2.resize(
                frame,
                (CAMERA_WIDTH, CAMERA_HEIGHT),
            )

            should_submit = (
                frame_number % PROCESS_EVERY_N_FRAMES == 0
            )

            if should_submit:
                replace_queue_item(
                    frame_queue,
                    frame.copy(),
                )

            try:
                while True:
                    last_result = result_queue.get_nowait()
            except Empty:
                pass

            if last_result is None:
                output = frame.copy()

                draw_status(
                    output,
                    "ANALISANDO...",
                    (0, 255, 255),
                )
            else:
                result_time, recognitions, error = last_result

                result_age = time.monotonic() - result_time

                if error is not None:
                    output = frame.copy()

                    draw_status(
                        output,
                        "ERRO NO RECONHECIMENTO",
                        (0, 0, 255),
                    )

                    print(f"Erro no reconhecimento: {error}")

                elif result_age > 3:
                    output = frame.copy()

                    draw_status(
                        output,
                        "ATUALIZANDO...",
                        (0, 255, 255),
                    )

                else:
                    output = annotate_image(
                        frame,
                        recognitions,
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
        stop_event.set()
        camera.release()
        cv2.destroyAllWindows()
        worker.join(timeout=1)


if __name__ == "__main__":
    main()