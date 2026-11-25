# Отчёт по гипотезам H5–H9

Дата: 2026-11-25.
Продолжение `H1_H4_HYPOTHESES_REPORT_RU.md`; нумерация линий та же (новые H1–H9,
не фазовые H1–H3 из Phase 5). Дизайн-документы: `docs/H5_…DESIGN.md` …
`docs/H9_…DESIGN.md`; план пакета: `docs/README_H5_H10_RESEARCH_PLAN_RU.md`.

| Линия | Суть | Статус |
|---|---|---|
| H5 | Bayesian MF: posterior uncertainty факторов для планирования | 🔴 REJECTED (калибровка провалена на Stage A; sampling значимо вредит на Stage C) |
| H6 | Response-aware recency/dual-channel fold-in user embedding | 🟡 COMPLETE: SPLIT — recency REJECTED, tuned dual-channel SUPPORTED (conditional, эффект малый) |
| H7 | SASRec-style трансформер как action-conditioned world model | 🔴 REJECTED (лучшая transition-модель, но true J значимо хуже даже oracle) |
| H8 | Logistic MF: Bernoulli likelihood внутри факторизации | 🔴 REJECTED на обоих уровнях (predictive и decision) |
| H9 | Self-loop / no-response ветви в MCTS | ⚪ PROPOSED — только дизайн-док; аудит и эксперименты не запускались |

---

## H5 — Bayesian MF / uncertainty-aware planning

**Гипотеза.** Текущая MF-линия использует точечные оценки латентных факторов; при
короткой истории и редких айтемах они ненадёжны, но планировщик не знает, насколько.
H5 утверждает: вероятностная оценка неопределённости user- и item-факторов (posterior
uncertainty **обученных** iALS-факторов — parameter/epistemic uncertainty, в отличие от
per-user belief-state из H1) повышает истинный долгосрочный return:
`ΔJ = J(π_Bayes) − J(π_point) > 0`, особенно в sparse/cold-start режимах. Подгипотезы:
H5a — posterior covariance корректно отражает неопределённость факторов; H5b — передача
uncertainty в планировщик улучшает J(π); H5c — эффект концентрируется у sparse users и
tail items.

**Что сделано.**
- Модель: conditional-Gaussian / mean-field posterior в фиксированной точке iALS
  (`src/mf_mdp/bayes/posterior.py`, документированные likelihood/priors/аппроксимация);
  предиктивная дисперсия — mean-field формула
  `Var[s] = νᵀΣ_u ν + μᵀΛ_i μ + tr(Σ_u Λ_i)`; Thompson-style coherent sampling — один
  Gibbs sweep (U|V̂, затем V|U) **на rollout**, не на шаг; bootstrap/ensemble-контроль.
- Три стадии с заранее зафиксированными gate'ами:
  - **Stage A** — synthetic posterior validation (истинные факторы известны): 8 ячеек
    {probit, logistic} × длина истории {5, 50} × exposure {uniform, zipf}; coverage
    50%/90% интервалов, z-статистики, Procrustes-aligned recovery.
  - **Stage B** — predictive holdout на ML-1M (LLN, конфиг воспроизведения; 30 175 пар).
  - **Stage C** — closed-loop Plan-1 в Env-C, армы B0 (point iALS), B1 (posterior mean,
    equality-контроль), B2 (posterior sampling), B3 (bootstrap ×8), B4 (oracle);
    8 seeds × 300 eval-пользователей, парные CI95; dense- и sparse-режимы.
- 10 тестов зелёные. Отчёт: `planning_over_MF_MDP/outputs/h5/REPORT.md`.

**Результат — REJECTED по обоим критериям.**
- **Stage A: gate калибровки провален во всех 8 ячейках (H5a — REJECTED).**
  Under-coverage *растёт* с объёмом данных: cov90 падает 0.79 (L=5) → **0.15** (L=50).
  При L=50 доминирует смещение, а не ширина: mean z = **+2.8…+3.9σ** — WLS-объектив
  iALS с confidence-weighted {0,1} таргетами подгоняет величину, насыщающуюся к
  таргетам, а не истинное скалярное произведение: предсказания на held-out позитивах
  систематически завышены (+0.59) при сжимающейся предиктивной sd. Постериор,
  центрированный на неправильной величине, не может покрывать детерминированное
  смещение misspecification — никакая ковариация это не чинит.
- **Stage B (ML-1M):** posterior-дисперсия — реальный сигнал для proper scoring
  (NLL рушится 573 → 2.18), но интервалы всё равно under-cover (0.68 при номинале
  0.90); дисперсия наибольшая у sparse users и tail items (как и должна быть), но
  некалибрована и потому непригодна как сигнал.
- **Stage C: H5b — REJECTED.** Posterior sampling не улучшает true J нигде: в dense
  режиме он значимо **вредит** (ΔJ = **−0.0054**, CI95 [−0.0105, −0.0003]), в sparse —
  нуль (+0.0018 [−0.0045, +0.0081]). Oracle regret растёт с uncertainty (0.060 → 0.065).
  Equality-контроль B1 ≡ B0 выполняется точно — контраст чисто механизм uncertainty.
- **H5c — не подтверждена:** sampling-арм рекомендует *больше* tail-айтемов
  (tail share 0.563 vs 0.505), но зарабатывает *меньше* — дополнительная tail-экспозиция
  шумовая, а не информационная.
- **Bootstrap-контроль воспроизводит направление вреда** (−0.0065 [−0.0132, +0.0006]):
  эффект объясняется чистой ценой дисперсии моделей — байесовского преимущества нет
  даже в зачатке. Цена: ~85× latency против детерминированного планировщика.

**Вывод по H5:** детерминированный iALS-постериор уже содержит всё, что планировщик
может эксплуатировать; дополнительная parameter uncertainty добавляет только шум
ранжирования. Машинерия (`bayes/posterior.py`) протестирована и переиспользуема, но
линия закрыта.

---

## H6 — Dynamic Positive/Negative User Embeddings (response-aware ALS fold-in)

**Гипотеза.** Обычный positive-only ALS fold-in накапливает только положительные
взаимодействия. H6 утверждает: update rule user embedding, который отдельно учитывает
положительные и **наблюдаемые** отрицательные реакции и уменьшает вес устаревшей
истории (recency decay), повышает истинный J(π) относительно текущего positive-only
fold-in в средах с preference drift / assimilation / satiation; выигрыш должен расти с
силой drift. **Критическое правило:** отсутствие взаимодействия — не negative;
`y=0` допустимо только при подтверждённом показе с завершённым окном наблюдения.
Линия меняет *только* online update rule: item factors, reward, response model,
candidate pool и planner заморожены.

**Что сделано.**
- Армы: U0 (текущий positive-only fold-in), U1 (+fixed recency decay), U2 (signed
  fold-in по exposures), U3 (раздельные positive/negative каналы,
  `u_eff = normalize(u₀ + α₊u⁺ − α₋u⁻)`), U4 (U3 + tuned decay/α₋), U5 (oracle).
- 72/72 теста зелёные (17 новых инвариантов, включая online≡batch и
  «unexposed ≠ negative»).
- **Локальный пилот** (2026-11-21): Env-C, H=12, режимы stationary / drift_medium /
  drift_strong, Plan-1 (`post`) заморожен, iALS item factors заморожены внутри ячейки,
  seeds=8, n_eval=300, парные CI95.
- **Серверный этап** (adrianna, 232 ячейки ~6 мин): E1 — drift-градиент, seeds=32,
  n_eval=1000; E2 — декомпозиция динамики (assimilation_only / satiation_only /
  reversion_only), seeds=24; E2b — честный reversion (θ₀=e₁-anchor); E3 — горизонт
  H=24; E4 — **KuaiRand-1K** (реальные exposure-логи, ~3.5M тестовых событий, replay
  по сессиям, time-split).
- Отчёт: `planning_over_MF_MDP_h6/outputs/h6/REPORT.md` + JSON-артефакты.

**Результат — SPLIT (по компонентам разные вердикты).**
- **Recency (U1, fixed decay) — REJECTED.** Значимо вредна при среднем drift
  (−0.0034 [−0.005, −0.002]) и деградирует с ростом горизонта (−0.0017 при H=12 →
  −0.0051 при H=24): фиксированное забывание выбрасывает ещё валидные positives.
- **Signed fold-in (U2) — SPLIT/weak:** только +0.0055 [+0.002, +0.009] при strong
  drift, в остальных режимах нуль или минус.
- **Dual-channel fixed (U3) — SPLIT:** +0.0139/+0.0209 при stationary и +0.0133 при
  strong drift (значимо), но **значимо вредит** при чистой ассимиляции
  (**−0.0397**, 7/24 позитивных сида) и anchor-reversion (−0.0183): skip-канал
  дублирует push-away динамику среды.
- **Dual-channel tuned (U4) — SUPPORTED (conditional):** единственный арм, значимо
  положительный во **всех 6 валидных режимах** серверного этапа (stationary
  +0.0247, drift_medium +0.0147, drift_strong +0.0179, H=24 strong +0.0242; при
  чистой ассимиляции ≈0). Предсказание H6 «выигрыш растёт с силой drift»
  подтверждено монотонностью (+0.014 → +0.029 локально при росте drift).
- **Критическое правило подтверждено с обратной стороны:** violation-контроль
  (1 unexposed-айтем/шаг как y=0 negative) даёт **−0.0253 [−0.037, −0.014]** —
  трактовка unexposed как negatives активно вредна.
- **Механизм — decision-level, не предиктивный:** улучшения log-loss не
  сопровождают выигрыш в J и наоборот (S6).
- **Потолок:** cos(u_t, θ_t) ≈ 0 для *всех* армов — представление bound; oracle gap
  0.46–0.88 J, эффекты ≤ +0.03 живут на марже (~6% gap у лучшей ячейки).
- **E4 (KuaiRand, реальные логи):** fixed-конфиги не переносятся — U3 ранжирует хуже
  (medrank 2322 vs 1878 у U0) при паритетном предиктиве; tuned-U4 на логах честно не
  настроить (нет true J).
- Побочная находка-исправление: Env-C нормализует θ на сферу каждый шаг ⇒ при θ₀=0
  любой reversion λ — no-op (подтверждено ячейками); добавлен честный anchor-режим.

**Вывод по H6:** общий вердикт **SPLIT**. Fixed recency не внедрять; negative-канал
ценен только с полной confidence и при сильном drift; tuned dual-channel (decay, α₋
подобраны на валидации) даёт малый, но реальный и устойчивый прирост — первый
значимо положительный компонентный эффект во всей линейке H1–H8. Обязательное условие:
negatives только по подтверждённым показам.

---

## H7 — SASRec-style трансформер как action-conditioned world model

**Гипотеза.** В H3 SASRec был генератором кандидатов; H7 проверяет другую роль —
**модель динамики среды**: совместное распределение реакции и следующего состояния
предпочтений `p_φ(y_t, Δθ_t | H_t, θ_t, a_t)`. Утверждение: такой world model снижает
multi-step transition error и повышает true J(π) model-based планировщика относительно
identity/static MF transition и простых transition-базлайнов. Явно не повторяет H2
(GRU как state — REJECTED) и H3 (proposer — COMPLETE).

**Что сделано.**
- World model (`src/mf_mdp/worldmodel/`, `src/mf_mdp/planning/worldmodel_planner.py`):
  causal event encoder (item, response, time-gap), response-голова `p(y|H,a,θ̂)` и
  диагонально-Гауссова transition-голова `Δθ` с `θ̂' = normalize(θ̂ + Δθ)`; входной θ̂ —
  сэмпл из MF-белифа N(u, P). Обучение teacher forcing; таргеты — наблюдаемый y и
  истинный Δθ (simulator-supervised, только train; планировщик θ не получает —
  проверено API-тестами).
- Армы (фиксированы reward / root-пул top-8 MF cosine / горизонт / termination /
  бюджет): W0 — MF fold-in (≡ H2 mf+stochastic); W1 — **oracle** (аналитический push
  среды + Байес-усреднённая истинная response-вероятность, квадратура Гаусса–Эрмита
  48 узлов); W2 — MLP; W3 — GRU; W4 — SASRec-style torch-трансформер (2 pre-LN блока,
  causal MHSA).
- **E7-A** (transition recovery): 8 ячеек Env-C (α⁺∈{0.1,0.35,0.6}, σ_obs∈{0.1,1.0},
  H∈{4,10}, momentum=0.3) × 3 сида; one-step/multi-step Δθ error, calibration,
  free-run.
- **E7-B** (closed-loop planning в истинной среде): 3 ячейки × 5 seeds × 200 eval;
  J(π), paired CI95, oracle regret, model exploitation gap.
- Диагностика механизма: `sampling_diagnostic.json` (n_theta_samples ∈ {1, 8, 32}).
- 13 новых тестов; полный сьют 104/105 (единственный красный — пре-существующий
  `test_h1.py::test_greedy_differs_from_oracle_in_envP` с базы H1, не регрессия H7).
- Отчёт: `planning_over_MF_MDP_h7/outputs/h7/REPORT.md`.

**Результат — REJECTED.**
- **E7-A однозначно положителен:** W4 — лучшая transition-модель во всех 8 ячейках
  (rollout cos_err на h=5: 0.053 vs 0.259 у W0; на α⁺=0.6: 0.082 vs 0.487); разрыв
  растёт с силой динамики и горизонтом (H=10, h=10: 0.093 vs 0.533); калибровка σ
  приемлемая (coverage95 0.91–0.94). MLP близко, GRU хуже. Но response-голова
  learned-моделей уступает MF logistic в каждой ячейке (согласуется с H2).
- **E7-B отрицателен со значимостью:** в nontrivial-динаmics ячейках plain MF fold-in
  (W0) **значимо лучше всех world-model армов** — base: w4−w0 = **−0.068**
  [−0.089, −0.045]; α⁺=0.6: −0.038 [−0.057, −0.018]; momentum: +0.008 ns.
- **Центральный факт: oracle W1 хуже W0** (regret −0.058 [−0.078, −0.037]) — *точная*
  модель динамики с оптимальной response-вероятностью проигрывает эпистемическому
  fold-in.
- **Архитектура на J не влияет:** w4−w2 и w4−w3 CI покрывают 0 везде, несмотря на
  разрыв точности переходов в E7-A — transition accuracy вообще не биндит J.
- **Model exploitation gap мал и отрицателен** (−0.014…−0.070) — это не история про
  эксплуатацию дыр модели; модели честные.
- **Диагностика механизма:** гипотеза «θ̂-sampling noise сглаживает value-ландшафт»
  **опровергнута** (n=1 делает армы ещё хуже: −0.113/−0.131; n=32 ≡ n=8). Механизм
  семантический: весь планировочный стек (utility, hazard, root-пул) MF-native, и
  fold-in, обновляющий u в той же геометрии, сохраняет внутреннюю согласованность
  value-функции; физически faithful θ-rollout эту согласованность ломает. Планировщик
  максимизирует внутренний utility — выигрывает модель, лучше с ним согласованная,
  а не более точная.

**Вывод по H7:** ровно пред-регистрированный паттерн фальсификации «one-step
prediction лучше, но true J не меняется/ухудшается». Специфический вклад H7 в общую
картину: точность модели переходов не является узким местом — даже exact-oracle
проигрывает оптимистичному эпистемическому переходу; узкое место — согласованность
планировочного стека с MF-геометрией (belief/reward-семантика), выше по стеку.

---

## H8 — Logistic Matrix Factorization для implicit feedback

**Гипотеза.** Bernoulli/logistic likelihood, встроенный прямо в user-item
факторизацию (`s_ui = μ + b_u + b_i + p_uᵀq_i`) и обученный на корректно размеченных
exposure-response наблюдениях, даёт более качественные вероятности/ранжирование, чем
двухстадийный пайплайн iALS (weighted squared loss) + response-LR, и улучшает true
J(π) при фиксированных candidate pool и budget. Отличие от H4: там кросс-признаки на
**фиксированных** iALS-факторах (REJECTED); здесь меняется сам объектив факторизации.

**Что сделано.**
- **E8-A аудит observation model** (`outputs/h8/data_audit.json`): KuaiRand-1K —
  каждая строка подтверждённый показ, валидные негативы, есть randomized-exposure
  файл (click 0.379 в policy-логе vs 0.174 в random — selection effect логирующей
  политики реален) ⇒ exposure-aware Bernoulli корректен; ML-1M — полей экспозиции
  нет ⇒ только PU/sampled-negative режим, ranking-only, без calibration-утверждений.
- Армы: `ials_lr` (текущий пайплайн, H4 m0), `ials_lr_h` (+ исторические признаки,
  сильнейший базлайн H4 m0h), `lmf` (exposure-aware Logistic MF, основная гипотеза),
  `lmf_prop` (+ clipped inverse-propensity веса), `lmf_pu` (PU-режим). Единый
  MAP/GD fold-in протокол для decision-time представления пользователя, item-сторона
  заморожена; HP на валидации.
- **Predictive:** KuaiRand-1K (31 684 train / 1 801 test / 205 randomized;
  воспроизводимость протокола побитовая — m0 AUC 0.5621, m0h 0.7474 совпали с H4);
  ML-1M PU (test n=29 294; m0/m0h побитово = H4). **Decision:** Env-C, 5 seeds,
  протокол H4 §4 (depth-2, L=8, shared hazard).
- 10 тестов зелёные. Отчёт: `planning_over_MF_MDP_h8/outputs/h8/REPORT.md`.

**Результат — REJECTED на обоих уровнях.**
- **KuaiRand (основной тестбед):** LMF значимо хуже сильнейшего текущего пайплайна:
  `lmf − ials_lr_h` ΔLL CI95 **[+0.133, +0.172]**, ΔAUC −0.04; калибровка ~2.6× хуже
  (ECE 0.106 vs 0.041). Propensity-взвешивание не спасает ([+0.138, +0.178]).
  Cold-start преимущества нет — отставание нарастает к позициям 5–7 сессии.
- **Randomized-подвыборка (n=205, недомощена):** паритет (lmf 0.817 vs m0h 0.815) —
  то есть часть преимущества m0h на policy-логах — доступ к регулярностям логирующей
  политики, а не качество отклик-модели; приписать LMF превосходство нельзя.
- **ML-1M (PU-only):** lmf_pu проигрывает даже m0 (AUC 0.474 vs 0.623) и вдвое
  проигрывает ранжирование следующего позитива (R@10 **0.039 vs 0.070** — при том что
  0.070 = точный масштаб воспроизведения бумаги, протокол корректен).
- **Decision-level (Env-C):** J не двигается — lmf −0.010 [−0.066, +0.034],
  lmf_pu +0.008 [−0.007, +0.033], оба ns; sanity-контраст `ials_lr − optimistic` =
  +0.050 [+0.008, +0.093] воспроизводит сигнал H2/H4, протокол валиден.
- Повторяется паттерн H4 «предиктив и планирование разъезжаются»: lmf_pu — худшая
  holdout-калибровка (ECE 0.229) при точечно наивысшем J среди обученных армов.
- Побочно: PU-абляция на Env-C показывает цену игнора exposure-структуры
  (AUC 0.522 vs 0.731) — косвенное подтверждение правила «unobserved ≠ negative».

**Вывод по H8:** сессионные исторические признаки поверх iALS-постериора
информационно ёмче совместной Bernoulli-факторизации на доступных объёмах; замена
объектива факторизации не даёт ни предиктивного, ни decision-выигрыша.

---

## H9 — Self-loop transitions и no-response branches в MCTS

**Гипотеза.** После показа айтема без положительной реакции preference-состояние не
обязано обновляться как после клика. H9 утверждает: явное моделирование
no-response/skip-ветви, где preference-компонент переходит в себя
(`θ_{t+1} = θ_t` при no-response, либо только time-drift), улучшает истинный J(π)
относительно transition model, не различающей positive response и no-response.
Self-loop относится к preference-компоненту, а не ко всему state: время, exposure set,
cooldown и termination обновляются в любом случае. Те же критические правила:
unexposed ≠ negative; censored ≠ no-response; finite horizon без бесплатного
увеличения горизонта.

**Статус: PROPOSED — не проверялась.** На ветке `h9/self-loop-transitions` (worktree
`planning_over_MF_MDP_h9`) существует только дизайн-документ
(`docs/H9_SELF_LOOP_TRANSITIONS_DESIGN.md`, untracked); обязательный **code audit
(Gate H9-A)** не выполнялся, кода/тестов/outputs нет.

**Что запланировано (по дизайн-доку):**
1. Аудит фактического пути action → response sampling → fold-in → consumed set:
   может ли preference-компонент остаться прежним после confirmed no-response уже
   сейчас. Если эквивалентный self-loop реализован — не дублировать, закрыть как
   already implemented.
2. Если нет — армы: S0 (текущая семантика), S1 (explicit no-response identity),
   S2 (no-response + time-only drift), S3 (cooldown re-exposure, exploratory),
   S4 (oracle). Метрики: true J(π), paired CI95, oracle regret; контроль, что выигрыш
   не объясняется ошибкой reward/termination.

---

## Общая картина после H5–H9

- **Три REJECTED (H5, H7, H8), один SPLIT (H6), одна неначатая линия (H9).** Паттерн
  H1–H4 — «апгрейды компонентов не двигают true J в MF-flavored режиме» — подтверждён
  ещё на трёх независимых компонентах: parameter uncertainty (H5), likelihood/objective
  факторизации (H8), transition model (H7).
- **H6 — первый значимо положительный компонентный эффект** всей линейки: tuned
  dual-channel fold-in (U4) положителен во всех 6 валидных режимах серверного этапа,
  с монотонным ростом по силе drift. Но эффект мал (≤ +0.03 J, ~6% oracle gap),
  conditional (требует tuned decay/α₋ и доверенных exposure-лейблов) и не переносится
  на реальные логи в fixed-конфиге (E4 KuaiRand).
- **H7 даёт самый сильный структурный факт пакета:** точность transition-модели не
  биндит J вообще — *exact-oracle* переход проигрывает оптимистичному эпистемическому
  fold-in. Планировочный стек MF-native, и выигрывает модель, согласованная с его
  внутренним utility, а не с истинной динамикой. Это перенаправляет поиск узкого
  места с «точности модели мира» на согласованность belief/reward/utility-семантики.
- **Кросс-линейные подтверждения методологии:** «unexposed ≠ negative» подтверждено
  с обратной стороны дважды (H6 violation-контроль −0.025; H8 PU-абляция AUC
  0.522 vs 0.731); разрыв «predictive ≠ decision» повторился в H5 (Stage B vs C),
  H7 (E7-A vs E7-B) и H8 (lmf_pu калибровка vs J).
- **Следующие шаги по плану** (`README_H5_H10_RESEARCH_PLAN_RU.md`): H9 audit
  (Gate H9-A — проверить, не реализован ли no-response self-loop уже сейчас), затем
  H10-A (GRPO в истинном simulator с фиксированными MF state и candidate pool).

**Артефакты линий:** `outputs/h5/REPORT.md` (основной репо),
`planning_over_MF_MDP_h6/outputs/h6/`, `planning_over_MF_MDP_h7/outputs/h7/`,
`planning_over_MF_MDP_h8/outputs/h8/`; тесты `test_h5_bayesian_mf.py` (10),
H6-сьют (72/72), `test_h7_world_model.py` (13), `test_h8_logistic_mf.py` (10).
