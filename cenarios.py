from dataclasses import dataclass
from pathlib import Path


CENARIOS_DIR = Path("cenarios")
VIDEO_PADRAO = "video.mp4"
IMAGEM_BASE_PADRAO = "base.png"
COORDENADAS_PADRAO = "vagas_coordenadas.pkl"
EXTENSOES_VIDEO = (".mp4", ".avi", ".mov", ".mkv")


@dataclass
class Cenario:
    nome: str
    pasta: Path
    video: Path
    imagem_base: Path
    coordenadas: Path


def encontrar_video(pasta):
    video_padrao = pasta / VIDEO_PADRAO
    if video_padrao.exists():
        return video_padrao

    videos = [
        arquivo
        for arquivo in sorted(pasta.iterdir())
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VIDEO
    ]
    return videos[0] if videos else video_padrao


def listar_cenarios():
    CENARIOS_DIR.mkdir(exist_ok=True)
    cenarios = []

    for pasta in sorted(CENARIOS_DIR.iterdir()):
        if not pasta.is_dir():
            continue

        video = encontrar_video(pasta)
        imagem_base = pasta / IMAGEM_BASE_PADRAO
        coordenadas = pasta / COORDENADAS_PADRAO

        if video.exists() or imagem_base.exists() or coordenadas.exists():
            cenarios.append(
                Cenario(
                    nome=pasta.name,
                    pasta=pasta,
                    video=video,
                    imagem_base=imagem_base,
                    coordenadas=coordenadas,
                )
            )

    return cenarios


def escolher_cenario(acao):
    cenarios = listar_cenarios()

    if not cenarios:
        raise FileNotFoundError(
            "Nenhum cenario encontrado. Crie uma pasta em 'cenarios/' com um video .mp4, .avi, .mov ou .mkv."
        )

    print(f"\nEscolha o estacionamento para {acao}:\n")
    for idx, cenario in enumerate(cenarios, start=1):
        status = []
        status.append("video" if cenario.video.exists() else "sem video")
        status.append("base" if cenario.imagem_base.exists() else "sem base")
        status.append("coords" if cenario.coordenadas.exists() else "sem coords")
        print(f"{idx} - {cenario.nome} ({', '.join(status)})")

    while True:
        escolha = input("\nDigite o numero do cenario: ").strip()
        if escolha.isdigit():
            indice = int(escolha)
            if 1 <= indice <= len(cenarios):
                cenario = cenarios[indice - 1]
                print(f"Cenario selecionado: {cenario.nome}\n")
                return cenario

        print("Opcao invalida. Tente novamente.")
