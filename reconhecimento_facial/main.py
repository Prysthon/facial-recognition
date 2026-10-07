from pathlib import Path

import cv2

from reconhecimento_facial.annotator import annotate_image
from reconhecimento_facial.recognizer import recognize_image
from reconhecimento_facial.settings import (
    AUTHORIZED_DIR,
    RESULTS_DIR,
    SUPPORTED_EXTENSIONS,
    TEST_DIR,
)


def list_images(
    directory: Path,
    recursive: bool = False,
) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")

    return sorted(
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def validate_directories() -> list[Path]:
    authorized_images = list_images(
        AUTHORIZED_DIR,
        recursive=True,
    )

    if not authorized_images:
        raise FileNotFoundError(
            f"Nenhuma foto autorizada encontrada em {AUTHORIZED_DIR}"
        )

    test_images = list_images(TEST_DIR)

    if not test_images:
        raise FileNotFoundError(
            f"Nenhuma foto de teste encontrada em {TEST_DIR}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    return test_images


def process_image(image_path: Path) -> None:
    image, recognitions = recognize_image(image_path)
    annotated_image = annotate_image(image, recognitions)

    output_path = (
        RESULTS_DIR
        / f"{image_path.stem}_resultado{image_path.suffix.lower()}"
    )

    saved = cv2.imwrite(
        str(output_path),
        annotated_image,
    )

    if not saved:
        raise OSError(
            f"Não foi possível salvar o resultado em {output_path}"
        )

    statuses = [
        "liberado" if item.authorized else "negado"
        for item in recognitions
    ]

    status_text = ", ".join(statuses) if statuses else "negado"

    print(
        f"{image_path.name}: acesso {status_text} "
        f"-> {output_path}"
    )


def main() -> None:
    test_images = validate_directories()

    print(f"Processando {len(test_images)} imagem(ns)...")

    for image_path in test_images:
        try:
            process_image(image_path)
        except Exception as error:
            print(f"Erro ao processar {image_path.name}: {error}")


if __name__ == "__main__":
    main()