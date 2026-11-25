# H9 — Self-loop transitions и no-response branches в MCTS

**Статус:** PROPOSED — сначала требуется code audit  
**Ветка:** `h9/self-loop-transitions`

> H9 продолжает новые линии H1–H4. Перед реализацией проверить, не
> поддерживает ли текущий planner эквивалентные переходы уже сейчас.

## 1. Исследовательский вопрос и гипотеза

После показа айтема без положительной реакции preference state не
обязательно должно обновляться так же, как после клика.

**H9.** Явное моделирование no-response/skip branch, где preference
component может перейти в себя, улучшает истинный (J(\pi)) относительно
transition model, которая не различает positive response и no-response.

\[ \theta\_{t+1}=\theta_t \quad \text{при no-response} \]

либо, при наличии time drift,

\[ \theta\_{t+1}=F\_{\Delta t}(\theta_t) \quad\text{без response-driven
update}. \]

**H0.** При корректно заданных reward и response probabilities self-loop
не меняет (J(\pi)), либо его эффект полностью объясняется изменением
reward/termination semantics, а не transition model.

## 2. Что именно является self-loop

Self-loop относится к **preference component** состояния, а не
обязательно ко всему MDP state.

После показа без положительной реакции:

- preference vector может остаться прежним;
- время/шаг увеличивается;
- exposure/consumed set обновляется;
- cooldown/repeat constraints могут измениться;
- reward и termination process продолжают работать.

Поэтому полный state может измениться, даже если
(\theta\_{t+1}=\theta_t). В MCTS это новая вершина на следующей глубине
с тем же preference component, а не бесконечная cyclic pointer на
текущий node.

Разделять:

1.  **Positive response:** клик/взаимодействие; применяется positive
    update.
2.  **Observed no-response:** айтем был показан, attribution window
    завершилось без положительного отклика; отдельная ветвь.
3.  **Unobserved/censored:** неизвестно, был ли айтем показан или
    завершилось ли окно; не считать автоматически no-response.

Отсутствие записи в логах нельзя автоматически трактовать как skip.

## 3. Обязательный code audit

В H1–H4 report описано, что после fold-in принятого айтема он
добавляется в `consumed`, а consumed items исключаются из дальнейшего
поиска. Этого недостаточно, чтобы утверждать, что no-response transition
отсутствует.

Найти фактический путь:

- где action становится shown/exposed;
- где семплируется response;
- где выполняется fold-in;
- при каких исходах обновляется consumed set;
- как строится stochastic branch;
- может ли branch сохранять preference state;
- как учитываются time, termination и повторный показ.

**Gate H9-A:** если эквивалентный no-response self-loop уже реализован и
корректно тестируется, не добавлять дубликат. Тогда проверить его
поведение/абляцию либо закрыть H9 как already implemented.

## 4. Формализация

Пусть:

\[ s_t=(b_t,\theta_t,E_t,\tau_t), \]

где (b_t) — belief, (\theta_t) — preference component в simulator, (E_t)
— exposure/consumed/cooldown state, (\tau_t) — время/шаг.

После action (a_t) семплируется исход:

\[ y_t\sim p(y\mid s_t,a_t). \]

- При (y_t=1): response-conditioned update.
- При (y_t=0) и confirmed exposure: no-response branch;
  (\theta\_{t+1}=\theta_t) либо только time drift.
- При censored/unobserved: отдельная missingness ветвь, только если
  среда её моделирует.

При любом подтверждённом показе обновляются exposure/time components.
Повторное включение того же айтема в candidate pool — отдельный
sub-experiment с cooldown, а не обязательная часть self-loop гипотезы.

## 5. Arms

| Arm | Семантика                                                   |
|-----|-------------------------------------------------------------|
| S0  | текущая реализация после audit                              |
| S1  | explicit no-response branch, identity preference transition |
| S2  | no-response + time-only drift                               |
| S3  | no-response + cooldown-based re-exposure — exploratory      |
| S4  | oracle transition semantics                                 |

Reward, response probability, termination, horizon и candidate
generation должны быть одинаковыми. Если S1 меняет доступность действий
из-за exposure/consumed logic, это нужно контролировать отдельно.

## 6. Эксперимент и метрики

1.  Unit-level audit на positive, observed no-response и censored
    исходах.
2.  Synthetic environment с контролируемой no-response probability.
3.  Stationary preference regime и режим с time drift.
4.  Closed-loop policies в true environment с одинаковыми seeds/users.
5.  Cooldown/re-exposure — отдельная абляция.

**Primary:** true (J(\pi)), paired (\Delta J), oracle regret.  
**Secondary:** доля response/no-response branches, показы до positive
response, repeated exposure rate, число уникальных items, termination
rate, response calibration, tree size, latency.

Нужно исключить бесплатное увеличение горизонта: finite horizon,
exposure cost, time и termination должны соблюдаться.

## 7. Критерии фальсификации

H9 поддерживается, если:

1.  audit подтверждает, что baseline не моделирует эквивалентную
    no-response branch;
2.  S1/S2 улучшает true (J(\pi)) с paired CI95, не включающим ноль, в
    заранее выбранном no-response regime;
3.  эффект сохраняется при одинаковых reward, exposure cost, termination
    и budget;
4.  нет искусственного выигрыша из-за бесплатного времени или
    бесконечных циклов.

H9 отклоняется, если baseline уже эквивалентен S1, корректный self-loop
не улучшает true (J), либо видимый выигрыш вызван ошибкой
reward/termination. Если эффект есть только с cooldown re-exposure, это
отдельный split result.

## 8. Предлагаемые артефакты

- `experiments/h9/audit_transition_semantics.py`
- `experiments/h9/run_self_loop_ablation.py`
- `tests/test_h9_no_response_transition.py`
- `outputs/h9/transition_audit.json`
- `outputs/h9/planning_results.json`
- `outputs/h9/REPORT.md`

## 9. Готовый промпт для coding agent

``` text
Проверь и реализуй H9 в planning_over_MF_MDP по docs/H9_SELF_LOOP_TRANSITIONS_DESIGN.md.

Сначала только аудит:
1. Проверь git status, ветку, transition code и MCTS expansion.
2. Прочитай docs/H1_H4_HYPOTHESES_REPORT_RU.md.
3. Проследи action через positive response, no-response и missing/censored outcome.
4. Найди, может ли preference component остаться прежним после confirmed no-response.
5. Если эквивалент уже есть — не добавляй дубликат; подготовь тест/абляцию и сообщи, что гипотеза уже частично реализована.

Если функциональности нет, реализуй:
- explicit positive-response branch;
- explicit confirmed-no-response branch;
- censored branch только если реально моделируется;
- no-response identity transition для preference component;
- корректное обновление времени, exposure/consumed state, cooldown и termination;
- finite-horizon tree unrolling: новый node на следующей глубине, без бесконечной cyclic pointer.

Ограничения:
- unexposed item не является negative;
- повторный показ не обязателен для self-loop; cooldown/re-exposure — отдельный arm;
- не меняй reward и transition одновременно без controls;
- не допускай бесплатного увеличения горизонта.

Тесты:
1. Positive response вызывает ожидаемый update.
2. Confirmed no-response оставляет preference component прежним в identity mode.
3. При time drift меняется только заданная часть state.
4. Exposure/time/consumed state обновляется корректно.
5. Censored event не становится negative.
6. Finite horizon завершает rollout.
7. Seed воспроизводим.
8. Нет бесконечного expansion.

Эксперимент:
- current baseline;
- explicit self-loop;
- self-loop + time drift;
- cooldown re-exposure как отдельная абляция;
- oracle transition.
Сравни true J(pi), paired CI95 и oracle regret при одинаковых users/seeds/reward/candidates/horizon/budget.

Сохраняй audit и результаты в outputs/h9/. Если audit показывает, что поведение уже реализовано, не создавай новую реализацию. Запусти tests, создай outputs/h9/REPORT.md с точной семантикой и verdict. Не коммить/пушить без разрешения.
```
