# Projeto de Visao Computacional - Estacionamento

Prototipo academico para detectar vagas livres e ocupadas em um estacionamento usando Python, OpenCV e YOLO.

## Rodando com Docker

Construa a imagem:

```powershell
docker compose build
```

Se o Docker reclamar que a imagem antiga `estacionamento-vc:latest` ja existe, ela e de uma configuracao anterior. A configuracao atual usa `trabalho-vc-estacionamento:dev`.

Para subir o container executando a deteccao:

```powershell
docker compose up
```

Tambem e possivel rodar comandos especificos dentro da imagem:

```powershell
docker compose run --rm estacionamento-vc python selector.py
docker compose run --rm estacionamento-vc python main.py
```

O fluxo esperado e:

1. Rodar `selector.py`.
2. Marcar 4 pontos para cada vaga na imagem.
3. Clicar com o botao direito para salvar `vagas_coordenadas.pkl`.
4. Rodar `main.py`.

### Janela grafica no Windows com Docker

`selector.py` e `main.py` usam janelas do OpenCV (`cv2.imshow`). Para abrir essas janelas a partir do container no Windows, deixe um X Server rodando no host.

Configuracao recomendada com VcXsrv:

1. Instale o VcXsrv.
2. Abra o `XLaunch`.
3. Selecione `Multiple windows`.
4. Selecione `Start no client`.
5. Marque `Disable access control`.
6. Finalize e mantenha o VcXsrv aberto.

O `docker-compose.yml` ja esta configurado com:

```yaml
DISPLAY: host.docker.internal:0.0
QT_X11_NO_MITSHM: "1"
```

Depois disso, rode normalmente:

```powershell
docker compose run --rm estacionamento-vc python selector.py
docker compose run --rm estacionamento-vc python main.py
```

Se a janela nao abrir, verifique se o VcXsrv esta rodando e se o firewall do Windows permitiu acesso para ele.

## Rodando localmente sem Docker

Instale as dependencias:

```powershell
pip install -r requirements.txt
```

Marque as vagas:

```powershell
python selector.py
```

Execute a deteccao:

```powershell
python main.py
```
