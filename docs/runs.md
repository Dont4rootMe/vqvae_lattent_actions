# Журнал запусков

`$BASE` = `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization`, репозиторий на кластере
`$BASE/vqvae_lattent_actions`, данные из `action_chunks` (манифест `r0_v2.1-6885099`, sha 05d18b2b), общий eval-набор
`$BASE/runs/eval_set/eval_set-05d18b2b-val500.npz` (44 642 чанка). Трекинг: Comet, workspace `dont4rootme`,
проект `hier-action-tokenizer`; ключ лежит вне репозитория (`/mnt/.../afedorov/.comet.config`).

Точки отсчёта на том же eval-наборе (см. `tokenizer_arms/docs/runs.md`):

| Токенайзер | токенов/чанк | бит/чанк | RMSE | L1 |
|---|---|---|---|---|
| FAST+ pretrained | 59.4 | ~650 | 0.0194 | 0.0146 |
| BEAST nb6_deg2, только активные dims | 149 | ~1190 | 0.0300 | 0.0091 |
| ActionCodec, 10 токенов | 10 | 110 | 0.0806 | 0.0459 |
| OAT, 10 регистров | 10 | 100 | 0.1028 | 0.0624 |

| Имя | Где | Что | Каталог | Статус |
|---|---|---|---|---|
| `lerobot-research-r09-smoke` | IB tmux, GPU 0,1 | pytest в боевом окружении (34 passed, 1 skipped) + 200 шагов обучения, Comet online | `$BASE/runs/hier_smoke` | exit=0 12:06 MSK; 15.8M параметров, 10 токенов × 2048 кодов = 110 бит/чанк; использование кодбука 1–2 кода (warmup 2000 не пройден за 200 шагов) |
| `lerobot-research-r09-usage` | IB tmux, GPU 7 | диагностика: 1500 шагов, warmup 50, batch 128, eval каждые 250 — раскрывается ли кодбук | `$BASE/runs/hier_codeusage` | запущен 12:08 MSK |
