# INSTALL_LOCAL.md — Installation et exploitation locale (paper trading)

> Usage **strictement personnel** de [LLM-Auto-Trader](https://github.com/Erfaniaa/LLM-Auto-Trader),
> en **paper trading** (aucun ordre réel). Ce document décrit l'installation sur la
> machine de Davyd (KDE neon / Ubuntu 24.04), la configuration choisie et le mode
> d'emploi. **Ceci n'est pas un conseil en investissement.**

Dépôt : fork personnel `github.com/davyd26/LLM-Auto-Trader` (upstream :
`Erfaniaa/LLM-Auto-Trader`).

---

## 1. Prérequis

| Outil | Version vérifiée |
|---|---|
| git | 2.43+ |
| python3 | 3.13.9 (≥ 3.11 requis par le projet) |
| pip | 25.x |
| gh (GitHub CLI) | 2.102.0 (installé localement dans `~/.local/bin`) |

Le projet tourne sur **Kraken spot en EUR**, sans clé d'exchange (inutile en paper mode).

---

## 2. Récupération du dépôt

```bash
# Fork + clone (crée ~/Perso/LLM-Auto-Trader avec les remotes origin=fork, upstream=amont)
gh repo fork Erfaniaa/LLM-Auto-Trader --clone
```

Si `gh` n'est pas installé :

```bash
# Binaire officiel, sans sudo :
curl -fsSL -o /tmp/gh.tar.gz \
  https://github.com/cli/cli/releases/download/v2.102.0/gh_2.102.0_linux_amd64.tar.gz
tar xzf /tmp/gh.tar.gz -C /tmp && cp /tmp/gh_2.102.0_linux_amd64/bin/gh ~/.local/bin/
gh auth login          # GitHub.com -> HTTPS -> navigateur
```

---

## 3. Installation

```bash
cd ~/Perso/LLM-Auto-Trader
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

# Extras demandés :
pip install -e ".[local]"     # provider OpenAI-compatible (utilise httpx, déjà en core)
pip install -e ".[dev]"       # pytest, ruff, mypy, coverage

# Extras nécessaires à l'exécution réelle (importés par le bot) :
pip install -e ".[all]"       # yfinance (macro), python-telegram-bot, SDK providers, etc.
```

> `.[local]` seul ne suffit pas pour lancer le bot : `src/data/macro.py` importe
> `yfinance` (extra `finance-extra`). `.[all]` couvre l'ensemble des besoins runtime.

---

## 4. Configuration `.env`

```bash
cp .env.example .env
```

`.env` est **déjà dans `.gitignore`** (vérifié) — ne jamais le committer.

### Variables effectivement utilisées

| Variable | Valeur locale | Rôle |
|---|---|---|
| `BOT_MODE` | `paper` | Simulation ; **jamais** `live` |
| `EXCHANGE_NAME` | `kraken` | Exchange européen de référence |
| `EXCHANGE_API_KEY` / `_SECRET` | *(vides)* | Inutiles en paper (pas de clé exchange) |
| `EXCHANGE_MARKET_TYPE` | `spot` | Pas de perpétuels ni de levier |
| `EXCHANGE_QUOTE_CCY` | `EUR` | Devise de cotation |
| `EXCHANGE_MAKER_FEE` | `0.0025` | Frais maker réels (spot Kraken standard 0,25 %) |
| `EXCHANGE_TAKER_FEE` | `0.0040` | Frais taker réels (spot Kraken standard 0,40 %) |
| `EXCHANGE_MAINT_MARGIN` | `0.0` | Sans objet en spot |
| `DATA_EXCHANGES` | `kraken` | Données cross-exchange (on reste en EUR) |
| `LLM_PROVIDER` | `openai-compat` | Passerelle OpenCode Go (endpoint OpenAI-compatible) |
| `LLM_MODEL` | `deepseek-v4.1-flash` | Modèle réellement servi par OpenCode Go |
| `LLM_API_KEY` | `oc_sk_…` | Clé du compte opencode (voir avertissement §8) |
| `LLM_BASE_URL` | `https://opencode.ai/inference/go/v1` | Endpoint OpenCode Go |
| `LLM_EXTRA_HEADERS` | `{"x-opencode-session":"ses_llmautotrader0001", "User-Agent":"llm-auto-trader/0.1"}` | Header de session **obligatoire** pour OpenCode Go |
| `LLM_EXTENDED_THINKING` / `_LONG_CONTEXT` | `false` | Options Anthropic, inutiles ici |
| `LLM_INPUT_PRICE_PER_M` | `0.30` | Tarif DeepSeek V4.1 Flash (peak, $/M) |
| `LLM_OUTPUT_PRICE_PER_M` | `1.20` | Idem sortie |
| `LLM_CACHED_INPUT_PRICE_PER_M` | `0.006` | Idem cache |
| `LLM_DAILY_BUDGET_USD` | `1.00` | Coupe-circuit budget journalier |
| `LLM_MONTHLY_BUDGET_USD` | `15.00` | Coupe-circuit budget mensuel |
| `REBALANCE_CADENCE` | `daily` | Affichée ; le scheduler réel reste adaptatif (5–60 min) |
| `DB_PATH` | `data/trading_bot.db` | Base SQLite de décisions/trades |
| `LOG_LEVEL` / `LOG_DIR` | `INFO` / `logs` | Journalisation |

> **Kraken n'a pas de profil intégré** dans `src/exchange/profile.py` (le registre
> fournit binance, bybit, okx, mexc, gate, kucoinfutures, bitget). On force donc le
> profil via les overrides `EXCHANGE_*` ci-dessus ; à défaut, le fallback générique
> serait 2 bps maker / 5 bps taker et une cotation USDT.

---

## 5. Lancer le paper trading

```bash
source .venv/bin/activate
python -m scripts.paper_trade
```

Le bot :

1. charge les marchés Kraken (lecture **publique**, sans clé) ;
2. résout les symboles EUR (`BTC/EUR`, `ETH/EUR`, `TRX/EUR`, `PAXG/EUR`) ;
3. collecte le snapshot de marché et interroge le LLM (JSON structuré) ;
4. passe la décision par la couche de risque, puis écrit dans `data/trading_bot.db`.

Arrêt propre : `Ctrl-C` (SIGINT).

Le scheduler est **adaptatif** (min 5 min, max 60 min) : un lancement effectue
immédiatement un pas, puis respecte l'intervalle demandé par le LLM.

---

## 6. Lancer les tests

```bash
source .venv/bin/activate
pytest -xvs
# ou rapidement :
pytest -q
```

Résultat local : **624 passed** (la suite est déterministe : CCXT mocké, LLM pré-enregistré).
Les tests Anthropic ne s'exécutent que si l'extra `anthropic` est installé (c'est le
cas avec `.[all]`).

---

## 7. Backtest CPCV (Combinatorial Purged Cross-Validation)

Pipeline en trois étapes, piloté par un identifiant de run (`--run-id`).

```bash
source .venv/bin/activate
RUN=low-freq-aggressive-v1
CACHE=~/.cache/llm-auto-trader/ohlcv.db

# 1) Construire les snapshots hebdomadaires (manifest + un JSON par observation)
python scripts/prepare_snapshots.py \
  --window 2024-01-01:2025-12-31 \
  --universe BTC/USDT,ETH/USDT,PAXG/USDT \
  --cache-db "$CACHE" \
  --run-id "$RUN"

# 2) Obtenir les décisions du LLM pour chaque snapshot
#    mode print    : imprime les instructions (dispatch manuel via sous-agents)
#    mode headless : exécute `claude -p` par snapshot (chemin automatisé)
python scripts/dispatch_decisions.py \
  --run-id "$RUN" --runs-dir runs \
  --decisions-db "runs/$RUN/decisions.db" \
  --mode headless --model opus

# 3) Simuler les chemins CPCV et agréger
python scripts/run_cpcv.py \
  --run-id "$RUN" --runs-dir runs \
  --cache-db "$CACHE" \
  --decisions-db "runs/$RUN/decisions.db" \
  --cpcv-n 8 --cpcv-k 2 --purge 2 --embargo 2 \
  --decisions-source cached
```

- `--decisions-source cached` réutilise les décisions mises en cache ; le mode
  baseline peut générer des décisions de repli à la volée.
- Sorties : `runs/<run_id>/` (manifest, `decisions.db`, résumé JSON, courbes CSV).

---

## 8. Fichiers modifiés dans ce fork

| Fichier | Changement | Motif |
|---|---|---|
| `src/llm/providers/openai_compat_provider.py` | Fusion de headers HTTP optionnels via `LLM_EXTRA_HEADERS` (JSON) | OpenCode Go exige `x-opencode-session` |
| `src/utils/logger.py` | `import logging.handlers` explicite | `logging.handlers.RotatingFileHandler` non résolu sur Python 3.13 |
| `src/bot.py` | `initialize()` utilise `resolve_symbols_for_exchange(...)` avec la devise de cotation | Permet les paires **spot EUR** Kraken (la fonction existait mais n'était pas branchée) |

Aucune modification de l'historique git ; le fork reste public.

---

## 9. Limitations et points à décider

1. **Compte paper non financé.** En paper mode sans clé d'exchange, `fetch_balance`
   échoue et l'équité reste à `0 €` → toutes les décisions sont `hold`
   (« No free equity »). Pour des décisions d'achat/vente, il faudra soit un
   **solde paper simulé** (petite évolution à ajouter), soit une clé exchange en
   lecture seule (à éviter en paper).
2. **Usage OpenCode Go.** La passerelle est « conçue pour OpenCode et autres agents
   de codage », avec « trafic surveillé pour abus ». Utiliser le bot de trading via
   cette clé **sort du cadre prévu** et peut exposer le compte. Alternative : clé
   DeepSeek directe (`api.deepseek.com/v1`, modèle `deepseek-chat`) ou LLM local
   (`llama.cpp` sur `http://127.0.0.1:8081/v1`).
3. **Aucune clé d'exchange** n'a été demandée ni stockée (paper mode).
4. **Frais Kraken** renseignés au palier standard (0,25 % / 0,40 %) — à ajuster
   selon le palier réel de volume.
5. Les avertissements non bloquants suivants sont normaux sur Kraken spot :
   `fetchFundingRate`/`fetchOpenInterest` non supportés, erreurs yfinance macro,
   `fetch_balance` sans clé.

---

## 10. Rappels de sécurité

- Ne jamais passer `BOT_MODE=live` sans décision explicite et réfléchie.
- Ne jamais committer `.env` ni de clé API (vérifié : `.env` est ignoré).
- `data/trading_bot.db` et `logs/` sont locaux et ignorés par git.
