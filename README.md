# Sistema de Reconhecimento Facial

Sistema de controle de acesso por reconhecimento facial, com processamento de imagens
e execução em tempo real pela câmera. Utiliza DeepFace para reconhecimento e OpenCV
para captura e visualização.

A organização foi baseada no Cookiecutter Data Science (CCDS), adaptada ao código
implementado no projeto.

## Instalação

Requisitos: Python 3.12 e `uv` instalados.

Na raiz do projeto, instale as dependências:

```sh
uv sync --locked
```

## Preparação das imagens

Crie as pastas de entrada e coloque as fotos nesta organização:

```text
data/
├── raw/
│   ├── authorized/
│   │   ├── pessoa_1/
│   │   │   ├── foto_1.jpg
│   │   │   └── foto_2.jpg
│   │   └── pessoa_2/
│   │       └── foto_1.png
│   └── test/
│       ├── imagem_1.jpg
│       └── imagem_2.png
└── processed/
    └── results/             # Criada ao processar imagens
```

- `authorized/`: fotos das pessoas autorizadas, em uma subpasta por pessoa.
  O nome da subpasta identifica a pessoa reconhecida.
- `test/`: imagens a processar, colocadas diretamente nessa pasta.
- `results/`: imagens de saída com as marcações de acesso liberado ou negado.

São aceitos arquivos `.jpg`, `.jpeg` e `.png`. A busca de fotos autorizadas é
recursiva; a busca de imagens de teste não inclui subpastas.

A pasta `data/` é ignorada pelo Git; as imagens devem ser fornecidas localmente.

## Execução

Execute os comandos na raiz do projeto.

### Processar imagens

```sh
uv run python -m reconhecimento_facial.main
```

É necessário ter fotos em `data/raw/authorized/` e imagens em `data/raw/test/`.
Os resultados são salvos em `data/processed/results/`, com o sufixo `_resultado`
no nome de cada imagem.

### Usar a câmera

```sh
uv run python -m reconhecimento_facial.camera
```

Prepare as fotos das pessoas autorizadas e permita o acesso à câmera no sistema
operacional. Pressione **Q** na janela de vídeo para encerrar.

## Configuração

Os parâmetros da aplicação ficam em `reconhecimento_facial/settings.py`:

| Parâmetro | Valor atual | Finalidade |
| --- | --- | --- |
| `MODEL_NAME` | `Facenet512` | Modelo de reconhecimento |
| `DETECTOR_BACKEND` | `yunet` | Detector de rostos |
| `DISTANCE_METRIC` | `cosine` | Métrica de comparação |
| `CAMERA_INDEX` | `0` | Índice da câmera |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640` / `480` | Resolução usada na captura e exibição |
| `PROCESS_EVERY_N_FRAMES` | `10` | Intervalo de envio de frames para reconhecimento |

Esse arquivo também define os diretórios de entrada e saída e as extensões aceitas.

## Organização do projeto

```text
├── Makefile                     # Instalação, ambiente, limpeza e verificações
├── README.md
├── pyproject.toml               # Metadados, dependências e configuração do Ruff
├── uv.lock                      # Versões das dependências
├── data/                        # Dados locais; ignorada pelo Git
├── docs/                        # Reservada para documentação
├── models/                      # Reservada para artefatos de modelos
├── notebooks/                   # Reservada para exploração em notebooks
├── references/                  # Reservada para materiais de referência
├── reports/                     # Reservada para relatórios
│   └── figures/                 # Reservada para figuras dos relatórios
└── reconhecimento_facial/
    ├── __init__.py              # Inicialização do pacote e carregamento de config
    ├── config.py                # Caminhos gerais, carregamento de .env e logging
    ├── settings.py              # Parâmetros do reconhecimento e da câmera
    ├── recognizer.py            # Detecção, comparação e resultados do reconhecimento
    ├── annotator.py             # Desenho das caixas e dos textos nas imagens
    ├── main.py                  # Processamento das imagens de teste
    └── camera.py                # Captura e reconhecimento pela câmera
```

As pastas reservadas estão vazias, com arquivos `.gitkeep` para mantê-las no Git.

## Desenvolvimento

Para verificar a formatação e executar a análise estática com Ruff:

```sh
uv run make lint
```

Para aplicar as correções automáticas e formatar o código:

```sh
uv run make format
```

Os demais comandos disponíveis estão no `Makefile`.
