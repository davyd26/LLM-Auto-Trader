# GUIDE_UTILISATION.md — Se servir du bot (paper trading)

Guide simple et concret. Le bot demande à un LLM **quoi faire** ; une couche de
risque déterministe **décide si c'est permis**, puis enregistre la décision.
Ici tout tourne en **paper** (simulation), sur **Kraken en EUR**, sans clé d'exchange.

> Détails d'installation : voir `INSTALL_LOCAL.md`.

---

## 0. En 30 secondes

```bash
cd ~/Perso/LLM-Auto-Trader
source .venv/bin/activate
python -m scripts.paper_trade      # démarre ; Ctrl-C pour arrêter
```

C'est tout. Le reste de ce guide explique quoi regarder et comment ajuster.

---

## 1. Démarrer une session

```bash
source .venv/bin/activate
python -m scripts.paper_trade
```

Sortie d'en-tête attendue :

```
Starting paper trading bot...
  Exchange:       kraken
  Quote currency: EUR
  LLM provider:   openai-compat
  LLM model:      deepseek-v4.1-flash
  Cadence:        daily
  Daily budget:   $1.00
  Max drawdown:   5%
  Max leverage:   5x
```

Ensuite, le bot fait **un pas immédiat**, puis attend (scheduler adaptatif, 5–60 min).

**Arrêter proprement** : `Ctrl-C`.

---

## 2. Comprendre les lignes de log

Chaque ligne est un JSON. Les plus utiles :

```jsonc
// L'exchange est prêt (lecture publique, aucune clé)
{"exchange":"kraken","markets":1460,"event":"trading_exchange_ready",...}

// Les symboles EUR ont été retenus
{"symbols":["BTC/EUR","ETH/EUR","TRX/EUR","PAXG/EUR"],"event":"symbols_validated",...}

// Le contexte de marché a été collecté
{"symbols":4,"has_onchain":true,"has_sentiment":true,"has_macro":true,"event":"snapshot_collected",...}

// Coût de l'appel LLM
{"cost_usd":0.0018,"input_tokens":3939,"output_tokens":528,"daily_total":0.0018,"event":"llm_call_cost",...}

// La décision du LLM
{"actions":["hold"],"trend":"neutral","confidence":0.5,"next_check":60,"event":"llm_analysis_complete",...}

// La couche de risque a validé
{"action":"hold","symbol":"BTC/EUR:EUR","approved":true,"reason":"OK","event":"risk_validation",...}
```

**Messages normaux à ignorer** (paper sans clé, Kraken spot) :

```
"error":"kraken requires \"apiKey\" credential"   -> fetch_balance (normal, pas de clé)
"kraken fetchFundingRate () is not supported yet" -> pas de funding en spot
"macro_ticker_failed"                             -> source macro externe indisponible
```

Un run qui contient `symbols_validated`, `snapshot_collected` et
`llm_analysis_complete` **fonctionne**.

---

## 3. Voir les décisions enregistrées

Tout est écrit dans `data/trading_bot.db` (SQLite).

```bash
sqlite3 data/trading_bot.db \
  "SELECT id, action, symbol, confidence FROM decisions ORDER BY id DESC LIMIT 10;"
```

Exemple de sortie :

```
6|hold|BTC/EUR:EUR|0.5
5|hold|BTC/EUR:EUR|0.55
4|hold|BTC/EUR:EUR|0.5
```

Voir le raisonnement en clair d'une décision :

```bash
sqlite3 data/trading_bot.db \
  "SELECT reasoning FROM decisions ORDER BY id DESC LIMIT 1;"
```

Exemple :

```
No free equity ($0.00) means no valid entry is possible. Signals also conflict:
15m/1h momentum bullish but 4h MACD bearish, price stalling into resistance...
```

---

## 4. Surveiller les coûts

Le budget est un coupe-circuit : au-delà, le bot cesse d'appeler le LLM.

```bash
grep llm_call_cost logs/bot.log | tail -5        # 1 ligne par appel
```

Chaque ligne donne `cost_usd`, `input_tokens`, `output_tokens`, `daily_total`.
Pour un total :

```bash
grep llm_call_cost logs/bot.log | python3 -c \
'import sys,json;print(round(sum(json.loads(l[l.index("{"):])["cost_usd"] for l in sys.stdin),4),"USD")'
```

Ordre de grandeur observé : **~0,002 $ par appel**, soit ~0,02 $ pour 5 décisions.

Changer les plafonds : éditer `LLM_DAILY_BUDGET_USD` / `LLM_MONTHLY_BUDGET_USD` dans `.env`.

---

## 5. Ajuster le comportement

### Changer la devise / l'exchange
Dans `.env` :

```
EXCHANGE_NAME=kraken
EXCHANGE_QUOTE_CCY=EUR
EXCHANGE_MARKET_TYPE=spot
```

Le bot retombera sur les paires spot disponibles (ex. `BTC/EUR`) ; les symboles
absents de l'exchange sont simplement ignorés.

### Changer de LLM
Trois options dans `.env` :

```bash
# a) Modèle local llama.cpp (déjà actif, gratuit)
LLM_PROVIDER=openai-compat
LLM_BASE_URL=http://127.0.0.1:8081/v1
LLM_MODEL=qwen3.5-4b
LLM_EXTRA_HEADERS=

# b) DeepSeek direct (clé platform.deepseek.com)
LLM_PROVIDER=openai-compat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-...

# c) Passerelle OpenCode Go (config actuelle)
LLM_PROVIDER=openai-compat
LLM_BASE_URL=https://opencode.ai/inference/go/v1
LLM_MODEL=deepseek-v4.1-flash
LLM_EXTRA_HEADERS={"x-opencode-session":"ses_llmautotrader0001"}
```

### Changer la fréquence
Le scheduler est adaptatif (`min 5 min`, `max 60 min`) et suit l'intervalle que le
LLM demande (`next_check`). `REBALANCE_CADENCE` dans `.env` n'est qu'informatif.

---

## 6. Lancer les tests

```bash
source .venv/bin/activate
pytest -q            # rapide : ~624 passed
pytest -xvs          # détaillé, s'arrête au 1er échec
```

---

## 7. Backtest CPCV (mode d'emploi express)

```bash
source .venv/bin/activate
RUN=demo
CACHE=~/.cache/llm-auto-trader/ohlcv.db

# 1) Snapshots hebdomadaires
python scripts/prepare_snapshots.py \
  --window 2024-01-01:2025-12-31 \
  --universe BTC/USDT,ETH/USDT,PAXG/USDT \
  --cache-db "$CACHE" --run-id "$RUN"

# 2) Décisions LLM (headless = automatisé)
python scripts/dispatch_decisions.py \
  --run-id "$RUN" --runs-dir runs \
  --decisions-db "runs/$RUN/decisions.db" --mode headless --model opus

# 3) Simulation CPCV
python scripts/run_cpcv.py \
  --run-id "$RUN" --runs-dir runs \
  --cache-db "$CACHE" --decisions-db "runs/$RUN/decisions.db" \
  --cpcv-n 8 --cpcv-k 2 --purge 2 --embargo 2 --decisions-source cached
```

Résultats dans `runs/<run_id>/`.

---

## 8. Forcer plusieurs décisions en peu de temps (astuce debug)

Le scheduler n'appelle le LLM qu'une fois toutes les 5–60 min. Pour une prise en
main rapide, ce petit script (hors repo) enchaîne 5 cycles :

```bash
cat > /tmp/run5.py <<'PY'
import asyncio, os, sys
os.environ["BOT_MODE"]="paper"; sys.path.insert(0,".")
from config.settings import load_settings
from src.bot import TradingBot
async def main():
    bot = TradingBot(load_settings()); await bot.initialize()
    for _ in range(5):
        await bot._run_llm_analysis(0)
    print(bot.cost_tracker.get_summary())
    await bot.shutdown()
asyncio.run(main())
PY
python /tmp/run5.py
```

Exemple de sortie :

```
{'today_spend': 0.0187, 'calls_today': 10, 'daily_remaining': 0.9813, ...}
```

---

## 9. FAQ / problèmes courants

**« Toutes les décisions sont `hold` »**
→ Normal si le compte paper est vide : le LLM voit 0 € et refuse d'entrer
(*"No free equity"*). Pour de vraies décisions, il faudrait un solde paper simulé.

**« No valid symbols found »**
→ La devise de cotation ne correspond à aucune paire. Vérifier `EXCHANGE_QUOTE_CCY`
et que `EXCHANGE_NAME` liste bien cette devise.

**« 401 / MissingSessionID »**
→ En OpenCode Go, l'en-tête `x-opencode-session` manque dans `LLM_EXTRA_HEADERS`.

**« Ça coûte cher »**
→ Baisser `LLM_DAILY_BUDGET_USD`, choisir un modèle local (`:8081`), ou laisser le
scheduler espacer les appels.

**« Rien ne se passe après le démarrage »**
→ Le bot a fait son pas puis attend l'intervalle suivant. C'est normal ; `Ctrl-C`
pour quitter.

---

## 10. Rappels

- **Jamais** `BOT_MODE=live` sans décision réfléchie et explicite.
- Ne jamais committer `.env` ni une clé API.
- Fichiers à surveiller : `data/trading_bot.db` (décisions), `logs/bot.log` (traces).
