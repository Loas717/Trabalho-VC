# Projeto de Visao Computacional - Estacionamento

Prototipo academico para detectar vagas livres e ocupadas em um estacionamento usando Python, OpenCV e YOLO.

## Rodando localmente no Windows

Use Python 3.11. O Python 3.14 pode falhar com PyTorch/Ultralytics porque as dependencias ainda podem nao ter wheels compativeis.

Prepare o ambiente:

```powershell
.\setup_windows.ps1
```

Ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Marque as vagas:

```powershell
python selector.py
```

Na janela do seletor:

- se `vagas_coordenadas.pkl` existir, as vagas salvas ja aparecem marcadas;
- clique em 4 pontos para adicionar uma nova vaga;
- arraste os pontos amarelos para editar uma vaga existente;
- pressione `d` para apagar a vaga sob o mouse;
- pressione `u` para desfazer o ultimo ponto ou a ultima vaga;
- clique com o botao direito ou pressione `s` para salvar;
- pressione `q` para sair.

Execute a deteccao:

```powershell
python main.py
```

Por padrao, a deteccao usa `yolo11s.pt`, que e mais pesado que `yolo11n.pt`, mas tende a detectar melhor os veiculos. Na primeira execucao, o modelo pode ser baixado automaticamente.

O resultado tambem usa suavizacao temporal: uma vaga so muda de estado depois de aparecer como livre/ocupada em varios frames, reduzindo piscadas e erros momentaneos.

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

Alternativa mais direta pelo PowerShell:

```powershell
& "C:\Program Files\VcXsrv\vcxsrv.exe" :0 -multiwindow -clipboard -wgl -ac
```

Se o Windows Firewall perguntar, permita o acesso do VcXsrv em redes privadas.

O `docker-compose.yml` ja esta configurado com:

```yaml
DISPLAY: host.docker.internal:0
QT_X11_NO_MITSHM: "1"
LIBGL_ALWAYS_INDIRECT: "1"
```

Depois disso, rode normalmente:

```powershell
docker compose run --rm estacionamento-vc python selector.py
docker compose run --rm estacionamento-vc python main.py
```

Se a janela nao abrir, verifique se o VcXsrv esta rodando e se o firewall do Windows permitiu acesso para ele.

Para diagnosticar a conexao com o X Server:

```powershell
docker compose run --rm estacionamento-vc python -c "import os; print(os.environ.get('DISPLAY'))"
```

O resultado esperado e `host.docker.internal:0`.
