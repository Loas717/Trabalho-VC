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

O programa vai perguntar qual estacionamento/cenario voce quer editar.

Na janela do seletor:

- se `vagas_coordenadas.pkl` existir no cenario escolhido, as vagas salvas ja aparecem marcadas;
- se `base.png` nao existir, ela sera criada automaticamente a partir do primeiro frame do video do cenario;
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

O programa tambem vai perguntar qual estacionamento/cenario voce quer analisar.

Tambem e possivel informar o cenario direto pelo terminal:

```powershell
python main.py --cenario estacionamento_video
```

Por padrao, a deteccao usa `yolo11s.pt`, que e mais pesado que `yolo11n.pt`, mas tende a detectar melhor os veiculos. Na primeira execucao, o modelo pode ser baixado automaticamente.

O resultado tambem usa suavizacao temporal: uma vaga so muda de estado depois de aparecer como livre/ocupada em varios frames, reduzindo piscadas e erros momentaneos.

As janelas do seletor e da deteccao abrem sempre em `1280x720`. A imagem/video e encaixado nesse tamanho com bordas pretas quando necessario, e as coordenadas continuam sendo salvas no tamanho original.

Para videos grandes, como 4K, a inferencia do YOLO e feita em largura reduzida, definida por `largura_inferencia`, e depois convertida de volta para a escala original. Isso melhora bastante a velocidade sem mudar as coordenadas das vagas.

## Como a decisao de ocupacao funciona

O `main.py` combina tres evidencias:

- YOLO detecta veiculos e associa cada veiculo a melhor vaga, evitando que a mesma caixa marque varias vagas vizinhas.
- Um classificador de recortes, quando existir em `models/parking_occupancy_yolo11n_cls.pt`, decide se a vaga parece `empty` ou `occupied`.
- Um analisador de linhas procura marcacoes brancas/amarelas da vaga no recorte normalizado. Se muitas linhas aparecem e nao ha veiculo forte na vaga, isso ajuda a considerar a vaga livre.

Cada vaga e recortada com transformacao de perspectiva para um quadrado padrao. Isso deixa os recortes mais parecidos com datasets como PKLot e CNRPark, que classificam vagas a partir de patches individuais.

## Configuracao por cenario

Cada cenario usa seu proprio arquivo de configuracao:

```text
cenarios/<nome>/config.example.json
```

Ao rodar `main.py`, se esse arquivo ainda nao existir na pasta do cenario, ele sera criado automaticamente com os valores iniciais. Depois disso, a deteccao consome os parametros desse JSON especifico do cenario.

Principais parametros:

```json
{
  "limiar_sobreposicao": 0.45,
  "limiar_sobreposicao_sem_ponto_base": 0.75,
  "limiar_linhas_livre": 0.18,
  "limiar_linhas_livre_forte": 0.65,
  "confianca_ocupada": 0.96,
  "confianca_ocupada_sem_yolo": 0.995,
  "confianca_livre": 0.90,
  "confianca_livre_fraca": 0.50,
  "liberar_sem_evidencia_de_veiculo": true,
  "permitir_ocupada_sem_yolo": false
}
```

Significado:

- `limiar_sobreposicao`: porcentagem minima da area da vaga coberta pela caixa do veiculo para marcar como ocupada quando a base do veiculo tambem cai na vaga. `0.45` significa 45% da vaga. Aumentar reduz falsos ocupados em vagas vizinhas, mas pode perder carros pequenos/parciais.
- `limiar_sobreposicao_sem_ponto_base`: limite mais alto usado quando a caixa cobre a vaga, mas a base do veiculo nao caiu dentro dela. Isso reduz falso ocupado causado por caixa grande de carro vizinho.
- `limiar_linhas_livre`: escore minimo de linhas visiveis para ajudar a marcar a vaga como livre. `0.18` significa que ha sinal suficiente de pintura/linha no recorte. Aumentar exige linhas mais claras; diminuir facilita marcar como livre, mas pode errar com carros brancos ou reflexos.
- `limiar_linhas_livre_forte`: quando passa desse valor e a base do carro nao esta na vaga, a evidencia de vaga livre vence o classificador. Ajuda em vagas vazias com linhas muito visiveis.
- `confianca_ocupada`: confianca minima do classificador para aceitar `occupied` quando tambem existe evidencia do YOLO. `0.96` exige 96% de confianca.
- `confianca_ocupada_sem_yolo`: confianca minima para aceitar `occupied` sem apoio do YOLO, usada apenas se `permitir_ocupada_sem_yolo` for `true`.
- `confianca_livre`: confianca minima do classificador para aceitar `empty`. `0.90` exige 90% de confianca. Aumentar evita falso livre; diminuir libera mais vagas.
- `confianca_livre_fraca`: confianca minima para usar `empty` como sinal fraco de ausencia de veiculo quando o YOLO tambem nao ve carro na vaga.
- `liberar_sem_evidencia_de_veiculo`: quando `true`, uma vaga ocupada pode voltar para livre se nao houver evidencia atual de veiculo nela. Isso evita o estado ficar preso como ocupado depois que o carro sai.
- `permitir_ocupada_sem_yolo`: quando `false`, o classificador sozinho nao marca vaga como ocupada. Isso e util quando o modelo ainda esta confundindo piso, placa ou linhas com carro.

Outros parametros uteis:

- `limiar_sobreposicao_associacao`: valor menor usado apenas para associar um veiculo a uma vaga quando o ponto inferior da caixa cai dentro dela.
- `janela_suavizacao` e `min_ocupacoes_na_janela`: controlam quantos frames recentes precisam concordar para mudar o estado final.
- `detectar_a_cada_n_frames`: reduz custo processando YOLO a cada N frames.
- `tamanho_recorte_vaga`: tamanho do patch normalizado usado pelo classificador e pela analise de linhas.

## Modo debug

Para salvar frames anotados, recortes e um CSV com as evidencias:

```powershell
python main.py --cenario estacionamento_video --salvar-debug --debug-a-cada 30
```

Saidas:

```text
debug/<cenario>/
  vagas_debug.csv
  frames/
  recortes/
```

Como interpretar `vagas_debug.csv`:

- `estado_final`: resultado exibido apos suavizacao temporal.
- `ocupada_agora`: decisao instantanea daquele frame antes da suavizacao.
- `origem`: `C` = classificador, `Y` = YOLO, `L` = linhas da vaga, `-` = manteve estado anterior.
- `classe_classificador` e `confianca_classificador`: saida do modelo de recorte.
- `sobreposicao_yolo`: quanto da vaga foi coberta pela caixa do veiculo associado.
- `ponto_inferior_na_vaga`: indica se a base da caixa do veiculo caiu dentro da vaga.
- `escore_linhas`: forca das linhas detectadas no recorte; valores maiores indicam mais marcacao visivel.

Para testar sem abrir janela e parar rapido:

```powershell
python main.py --cenario estacionamento_video --sem-janela --limite-frames 120 --salvar-debug
```

## Treinando classificador de vagas

Se existir um dataset no formato:

```text
dataset/
  empty/
  occupied/
```

Voce pode gerar recortes do seu proprio video para rotular:

```powershell
python exportar_recortes_vagas.py --cenario estacionamento_video --intervalo 30
```

Os recortes serao salvos em:

```text
dataset_recortes/<cenario>/sem_rotulo/
```

Revise visualmente esses arquivos e mova copias para:

```text
dataset/
  empty/
  occupied/
```

Use frames variados: vagas no sol, sombra, carros claros, carros escuros, vagas perto de arvores, vagas parcialmente ocluidas e exemplos dos erros que aparecerem no debug.

prepare o dataset de treino/validacao:

```powershell
python preparar_dataset_classificacao.py
```

Treine o classificador leve com YOLO11 nano:

```powershell
python treinar_classificador_vagas.py
```

O modelo final sera salvo em:

```text
models/parking_occupancy_yolo11n_cls.pt
```

Quando esse arquivo existir, o `main.py` usa duas evidencias para decidir se uma vaga esta ocupada:

- deteccao de veiculo com YOLO;
- classificacao do recorte da vaga como `occupied`.
- linhas visiveis da vaga como apoio para casos incertos.

## O que checar depois de ajustar

Depois de qualquer mudanca de limiar ou treino do classificador:

1. Rode `main.py --salvar-debug` em um trecho curto.
2. Abra alguns frames em `debug/<cenario>/frames`.
3. Confira no CSV se os erros vieram de `C`, `Y` ou `L`.
4. Se muitas vagas vazias viram ocupadas por caixa de carro vizinha, aumente `limiar_sobreposicao`.
5. Se carros pequenos ou distantes nao ocupam a vaga, reduza um pouco `limiar_sobreposicao` ou `limiar_sobreposicao_associacao`.
6. Se vagas vazias em sombra nao sao reconhecidas como livres, reduza levemente `limiar_linhas_livre` ou inclua mais exemplos no classificador.
7. Se carros brancos viram vaga livre por causa de linhas/reflexos, aumente `limiar_linhas_livre` e priorize exemplos desse tipo no treino.
8. Se o classificador esta muito indeciso, acrescente recortes do proprio cenario e treine novamente.

## Estrutura dos cenarios

Cada estacionamento deve ficar em uma pasta propria dentro de `cenarios/`:

```text
cenarios/
  estacionamento_video/
    video.mp4
    base.png
    vagas_coordenadas.pkl

  shopping_a/
    video.mp4
    base.png
    vagas_coordenadas.pkl

  rua_b/
    14360955_3840_2160_30fps.mp4
```

Arquivos esperados:

- video do cenario: preferencialmente `video.mp4`, mas o sistema tambem reconhece automaticamente arquivos `.mp4`, `.avi`, `.mov` e `.mkv` dentro da pasta.
- `base.png`: imagem usada pelo seletor para marcar as vagas. Se nao existir, o `selector.py` cria a partir do primeiro frame do video.
- `vagas_coordenadas.pkl`: coordenadas das vagas daquele cenario. Se nao existir, o `selector.py` cria quando voce salvar.

Para adicionar um novo estacionamento, crie uma pasta dentro de `cenarios/` e coloque o video dentro dela. Se houver mais de um video na mesma pasta, o sistema usa `video.mp4` primeiro; se ele nao existir, usa o primeiro video encontrado em ordem alfabetica. Depois rode:

```powershell
python selector.py
```

Escolha o novo cenario no menu, marque as vagas e salve.

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
