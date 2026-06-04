# Projeto de Visao Computacional - Vagas de Estacionamento

Prototipo academico para detectar vagas livres e ocupadas em uma camera fixa com vista superior do estacionamento.

A deteccao principal usa subtracao de fundo: cada vaga marcada e comparada com a mesma regiao em `base.png`, que deve representar a vaga vazia. O metodo nao classifica "carro"; ele mede quanto o recorte da vaga mudou em relacao ao fundo.

## Fluxo recomendado com Kaggle

O dataset recomendado para demonstracao e o Parking Lot Detection Counter:

```text
https://www.kaggle.com/datasets/iasadpanwhar/parking-lot-detection-counter
```

Ele contem video em top view e uma mascara binaria das vagas. O projeto usa essa mascara apenas para gerar as coordenadas das vagas; a ocupacao continua sendo decidida por preprocessamento/subtracao, sem classificar carros.

Baixe o ZIP do Kaggle e extraia de forma que exista esta pasta:

```text
datasets/parking-lot-detection-counter/parking/
```

Arquivos esperados dentro dela:

```text
parking_1920_1080.mp4
mask_1920_1080.png
```

Importe o dataset para um cenario local:

```powershell
python importar_kaggle.py --sobrescrever
```

Isso cria:

```text
cenarios/kaggle_parking/
  video.mp4
  base.png
  mask.png
  vagas_coordenadas.pkl
  config.json
```

Rode a deteccao:

```powershell
python main.py --cenario kaggle_parking
```

Para testar sem abrir janela:

```powershell
python main.py --cenario kaggle_parking --sem-janela --limite-frames 120 --salvar-debug
```

## Por que essa abordagem

Para o problema do trabalho, o requisito e saber se uma vaga esta ocupada. Com camera fixa e vista top view, isso pode ser feito com custo baixo:

- nao baixa modelo pesado;
- nao roda YOLO nem rede neural no `main.py`;
- processa apenas os recortes das vagas, nao a imagem inteira;
- usa `absdiff`, limiarizacao e morfologia leve do OpenCV;
- aplica suavizacao temporal para reduzir piscadas.

## Rodando localmente no Windows

Use Python 3.11.

Instale as dependencias:

```powershell
pip install -r requirements.txt
```

Marque as vagas:

```powershell
python selector.py
```

O programa pergunta qual estacionamento/cenario editar.

Na janela do seletor:

- se `vagas_coordenadas.pkl` existir no cenario, as vagas salvas ja aparecem marcadas;
- se `base.png` nao existir, ela sera criada a partir do primeiro frame de `video.mp4`;
- clique em 4 pontos para adicionar uma vaga;
- clique dentro de uma vaga ja marcada para seleciona-la;
- arraste os pontos amarelos para ajustar uma vaga existente;
- pressione `c` para copiar a vaga selecionada para o centro do mouse;
- use as setas ou `i`, `j`, `k`, `l` para mover a vaga selecionada;
- use `I`, `J`, `K`, `L` para mover mais rapido;
- pressione `d` para apagar a vaga selecionada ou a vaga sob o mouse;
- pressione `u` para desfazer o ultimo ponto ou a ultima vaga;
- clique com o botao direito ou pressione `s` para salvar;
- pressione `q` para sair.

Execute a deteccao:

```powershell
python main.py
```

Tambem e possivel escolher o cenario direto:

```powershell
python main.py --cenario estacionamento_video
```

Para testar sem abrir janela:

```powershell
python main.py --cenario estacionamento_video --sem-janela --limite-frames 120
```

## Arquivos do cenario

Cada estacionamento fica em uma pasta dentro de `cenarios/`:

```text
cenarios/
  estacionamento_video/
    video.mp4
    base.png
    vagas_coordenadas.pkl
    config.json
```

Arquivos esperados:

- `video.mp4`: video analisado.
- `base.png`: imagem do estacionamento vazio ou a melhor aproximacao do fundo.
- `vagas_coordenadas.pkl`: coordenadas das vagas marcadas no seletor.
- `config.json`: opcional; sobrescreve os limiares padrao.

Para melhor resultado, use uma `base.png` sem carros nas vagas monitoradas. Se o primeiro frame do video ja tiver carros, substitua `base.png` por uma imagem limpa do mesmo enquadramento e resolucao.

## Gerando ou testando uma base limpa

Se voce tiver apenas o video, gere uma tentativa de base por mediana de frames:

```powershell
python gerar_base.py --cenario estacionamento_video --amostras 90
```

Isso cria:

```text
cenarios/estacionamento_video/base_gerada.png
```

Se a imagem ficar boa, aplique como `base.png`:

```powershell
python gerar_base.py --cenario estacionamento_video --amostras 90 --aplicar
```

O script cria `base_backup.png` antes de substituir. A mediana funciona melhor quando carros se movem ao longo do video. Se os mesmos carros ficam parados o tempo todo, eles continuam aparecendo na base.

Para testar uma imagem limpa externa, por exemplo uma criada no ChatGPT, coloque o arquivo no cenario e rode o seletor usando essa base:

```powershell
python selector.py --cenario estacionamento_video --base cenarios/estacionamento_video/base_chatgpt.png
```

Marque ou ajuste as vagas nessa imagem e salve. Depois rode a deteccao usando a mesma base:

```powershell
python main.py --cenario estacionamento_video --base cenarios/estacionamento_video/base_chatgpt.png
```

A imagem externa precisa ter o mesmo enquadramento e perspectiva do video. Se ela tiver sido redesenhada com linhas, sombras ou proporcoes diferentes, a subtracao pode gerar falsos positivos.

Se a imagem externa estiver em outra resolucao ou levemente deslocada, tente alinhar primeiro:

```powershell
python alinhar_base.py --cenario estacionamento_video --base-externa cenarios/estacionamento_video/base_chatgpt.png
```

Isso gera:

```text
cenarios/estacionamento_video/base_alinhada.png
```

Use a imagem alinhada no seletor e na deteccao:

```powershell
python selector.py --cenario estacionamento_video --base cenarios/estacionamento_video/base_alinhada.png
python main.py --cenario estacionamento_video --base cenarios/estacionamento_video/base_alinhada.png
```

Se o alinhamento ficar bom, aplique como `base.png`:

```powershell
python alinhar_base.py --cenario estacionamento_video --base-externa cenarios/estacionamento_video/base_chatgpt.png --aplicar
```

## Como a decisao funciona

Para cada vaga:

1. O poligono marcado e transformado em um recorte quadrado padrao.
2. O recorte atual e comparado com o recorte equivalente de `base.png`.
3. O codigo converte para cinza, aplica blur, compensa diferenca media de iluminacao e calcula `absdiff`.
4. Pixels acima de `limiar_pixels_diferentes` entram na mascara de mudanca.
5. A vaga fica ocupada quando a proporcao de pixels alterados e a diferenca media passam dos limiares de ocupacao.
6. A vaga volta para livre quando os valores caem abaixo dos limiares de liberacao.
7. Se os valores ficam no meio, o estado anterior e mantido para evitar oscilacao.
8. A decisao final usa uma janela temporal para evitar alternancia brusca.

## Configuracao

Exemplo de `cenarios/<nome>/config.json`:

```json
{
  "limiar_pixels_diferentes": 28,
  "limiar_ocupacao": 0.18,
  "limiar_livre": 0.10,
  "limiar_diferenca_media": 0.035,
  "limiar_diferenca_media_livre": 0.020,
  "normalizar_iluminacao": true,
  "blur_kernel": 5,
  "morfologia_kernel": 3,
  "morfologia_iteracoes": 1,
  "janela_suavizacao": 5,
  "min_ocupacoes_na_janela": 4,
  "analisar_a_cada_n_frames": 2,
  "pular_frames": true,
  "tamanho_recorte_vaga": 160
}
```

Principais parametros:

- `limiar_pixels_diferentes`: diferenca minima de intensidade para um pixel contar como alterado.
- `limiar_ocupacao`: porcentagem minima do recorte que precisa mudar para marcar ocupada.
- `limiar_livre`: porcentagem abaixo da qual a vaga volta a ser livre.
- `limiar_diferenca_media`: diferenca media minima do recorte inteiro.
- `limiar_diferenca_media_livre`: diferenca media abaixo da qual a vaga volta a ser livre.
- `normalizar_iluminacao`: compensa mudanca global de brilho entre base e frame atual.
- `janela_suavizacao` e `min_ocupacoes_na_janela`: controlam estabilidade temporal.
- `analisar_a_cada_n_frames`: reduz custo reaproveitando a ultima analise por alguns frames.
- `tamanho_recorte_vaga`: tamanho do patch normalizado usado na comparacao.

## Modo debug

Para salvar frames anotados, recortes, mascaras de mudanca, diffs e CSV:

```powershell
python main.py --cenario estacionamento_video --sem-janela --limite-frames 120 --salvar-debug
```

Saidas:

```text
debug/<cenario>/
  vagas_debug.csv
  frames/
  recortes/
  mascaras/
  diffs/
```

Como calibrar:

- se vagas vazias aparecem ocupadas, aumente `limiar_ocupacao`, `limiar_livre` ou `limiar_pixels_diferentes`;
- se carros pequenos nao ocupam a vaga, reduza `limiar_ocupacao`;
- se a vaga fica presa como ocupada depois que o carro sai, aumente `limiar_livre` ou `limiar_diferenca_media_livre`;
- se sombra ou sol variando gera falso positivo, mantenha `normalizar_iluminacao` como `true` e aumente um pouco `limiar_diferenca_media`;
- se a mascara fica cheia de ruido, aumente `morfologia_kernel` para `5`;
- se o estado pisca, aumente `janela_suavizacao` e `min_ocupacoes_na_janela`.

## Docker

Construa a imagem:

```powershell
docker compose build
```

Execute:

```powershell
docker compose run --rm estacionamento-vc python main.py --cenario estacionamento_video
```

Para rodar com janela grafica no Windows, deixe um X Server como VcXsrv aberto. O `docker-compose.yml` ja define `DISPLAY=host.docker.internal:0`.
