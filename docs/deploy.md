# Deploying STL Crime Tracker

This guide walks you through putting the tracker online for free, step by step, from a
Windows PC. You'll need about an hour the first time.

## How the pieces fit

```
GitHub Actions (every 10 min)          Render (web service, free)
  alembic upgrade head                   uvicorn api.main:app
  python -m jobs once  ──writes──►  Supabase Postgres  ◄──reads──  /, /timer, /health ...
        │                                                              ▲
        └── Discord alerts (optional)                UptimeRobot ──────┘ checks /health
```

- **Supabase** hosts the Postgres database (free tier).
- **GitHub Actions** runs the pipeline (fetch → classify → match) every 10 minutes. It
  also applies database migrations before every run. Optionally, **cron-job.org**
  triggers it on time through GitHub's API (step 3d); GitHub's own schedule is often late.
- **Render** runs the website and API.
- **UptimeRobot** checks `/health` every 5 minutes. `/health` answers **503** when no
  pipeline run has succeeded in 30 minutes, so you get an email if GitHub Actions stops.
  The checks also keep Render's free service from going to sleep.

All commands below are for **PowerShell**, run from the project folder:

```powershell
cd C:\Users\manis\projects\stl-crime-tracker
.venv\Scripts\activate
```

---

## 0. Before you start

1. Make sure the tests pass: `pytest`
2. Make sure your local database is up to date: `alembic upgrade head`
3. **Stop the local scheduler** if it's running (Ctrl+C in its window). From now on,
   GitHub Actions runs the pipeline. Running both would use your Gemini quota twice.
4. **Public or private repo?** Your repo page on GitHub shows "Public" or "Private"
   next to its name.
   - **Public**: GitHub Actions minutes are free and unlimited. Nothing to do.
   - **Private**: the free plan includes 2,000 Actions minutes a month, and every run
     bills at least 1 minute. A run every 10 minutes is about 4,300 runs a month, so a
     private repo runs out of minutes around day 10 and the pipeline stops. Make the repo
     public (Settings → General → Danger Zone → **Change visibility**). Your `.env`,
     databases, and logs are never committed (see step 7), and GitHub secrets are never
     visible to anyone, even on a public repo.

---

## 1. Supabase: create the database

1. Go to <https://supabase.com> and sign up (signing in with GitHub is easiest).
2. Click **New project**.
   - **Name**: `stl-crime-tracker`
   - **Database password**: click **Generate a password**, then **copy it into a password
     manager now**. You can't see it again later (you can only reset it).
     Tip: a password with only letters and digits avoids URL-escaping problems later.
   - **Region**: *East US (Ohio)*. It's closest to Render's Ohio region and to St. Louis.
   - Free plan. Click **Create new project** and wait a couple of minutes.
3. Get the connection string:
   - Click the **Connect** button at the top of the project page.
   - Choose **Session pooler** (not "Direct connection": that one is IPv6-only, and
     GitHub Actions and Render can't reach it; not "Transaction pooler" either).
   - Copy the **URI**. It looks like:
     ```
     postgresql://postgres.abcdefghijklmnop:[YOUR-PASSWORD]@aws-0-us-east-2.pooler.supabase.com:5432/postgres
     ```
   - Replace `[YOUR-PASSWORD]` (including the brackets) with your database password.
     If the password has symbols like `@ : / ? #`, they must be URL-escaped (`@` → `%40`,
     `#` → `%23`, ...), or just reset the password to letters and digits.
   - This full string is your **`DATABASE_URL`**. Treat it like a password.

> Supabase pauses free projects after a week with no activity. The pipeline connects
> every 10 minutes, so that won't happen while it's running.

---

## 2. Create the tables and copy your data

You'll point this PowerShell window at Supabase temporarily. A variable set in the
window wins over `.env`, and it disappears when you close the window.

1. Set `DATABASE_URL` for this window only. `Read-Host` keeps the password out of your
   PowerShell history:
   ```powershell
   $env:DATABASE_URL = Read-Host "Paste the Supabase Session pooler URI"
   ```
2. Create the tables in Supabase:
   ```powershell
   alembic upgrade head
   ```
   The last line should mention `alerts_sent` (the newest migration). If it hangs or
   says `Network is unreachable`, double-check you copied the **Session pooler** URI.
3. Copy your local data (`stl_crime.db`) into Supabase:
   ```powershell
   python scripts/copy_sqlite_to_postgres.py
   ```
   It prints a row count per table and `Done: ... rows copied, counts verified.`
   - It refuses to run if Supabase already has data, so it can't double-copy. If it
     failed halfway, nothing was written: fix the problem and run it again.
   - "source database is at revision ...": run `alembic upgrade head` on the local
     database first (in a **new** PowerShell window, where `DATABASE_URL` is still SQLite).
   - Starting fresh without your old data? Skip this step.
4. Clear the variable (or just close the window):
   ```powershell
   Remove-Item Env:DATABASE_URL
   ```

You can look at the tables in Supabase under **Table Editor**.

---

## 3. GitHub: secrets and the pipeline

### 3a. Add the secrets

On GitHub, open your repo → **Settings** → **Secrets and variables** → **Actions** →
**New repository secret**. Add each of these (name exactly as shown, value without
quotes):

| Name | Required? | Value |
|---|---|---|
| `DATABASE_URL` | **yes** | the Supabase Session pooler URI from step 1 |
| `GEMINI_API_KEY` | **yes** | your Gemini API key (same as in `.env`) |
| `GEMINI_MODEL` | recommended | same as in your `.env` |
| `GEMINI_FALLBACK_MODEL` | recommended | same as in your `.env` |
| `LLM_CHAIN` | optional | same as in your `.env` (e.g. `groq:openai/gpt-oss-120b,gemini:gemini-3.8-flash,gemini:gemini-3.5-flash-lite`); empty = the `GEMINI_*` models |
| `GROQ_API_KEY` | if `LLM_CHAIN` has a `groq:` entry | your Groq API key |
| `GROQ_OTPM` | if the Groq model has an output-token limit | e.g. `1000` for qwen (same as in your `.env`) |
| `ALERT_WEBHOOK_URL` | optional | your Discord webhook URL (alerts are only logged without it) |
| `ALERT_ON_NEW_INCIDENT` | optional | `1` to get a Discord message for every new confirmed incident |
| `GEMINI_RPM`, `GROQ_RPM`, `GROQ_TPM`, `GROQ_REASONING_EFFORT`, `LLM_PROVIDER`, `MATCH_LOCATION_THRESHOLD`, `ALERT_COOLDOWN_HOURS` | optional | only if you changed them in `.env` |

A secret you don't add is treated as empty and the default is used. Secrets are
never shown in logs (GitHub replaces them with `***`).

### 3b. Push the code

The workflows live in `.github/workflows/`. Scheduled runs only happen from the
repo's **default branch** (`master`), so push there:

```powershell
git push origin master
```

### 3c. First run

1. Open the repo's **Actions** tab. If GitHub asks, click **I understand my workflows,
   go ahead and enable them**.
2. You'll see two workflows:
   - **Tests** runs `pytest` on every push (SQLite plus a throwaway Postgres). It never
     touches your real database or Gemini.
   - **Pipeline** runs every 10 minutes.
3. Click **Pipeline** → **Run workflow** → **Run workflow** to start one now.
4. Click the run, then the **run** job, to watch the logs. A good run ends with a line
   like `Pipeline run 42 finished: success`. In Supabase's Table Editor,
   `pipeline_runs` gets a new row.

Good to know:
- GitHub starts scheduled runs **late** (often 5–15 minutes, sometimes skips one when
  busy). That's expected; the 30-minute health window allows for it.
- Runs never overlap: a run that's due while another is still going waits for it.
- **On a public repo, GitHub turns off scheduled workflows after 60 days with no commits.**
  You'll get an email from GitHub, and UptimeRobot will alert. Re-enable it under
  Actions → Pipeline → **Enable workflow**, or push any commit now and then.
- The free Gemini quota (about 20 requests/day per model) is shared by all runs; the
  classifier only calls Gemini when there are new crime-looking headlines, and it
  switches to the fallback model or stops cleanly when the quota is used up.

### 3d. Optional: on-time runs from cron-job.org

GitHub's own schedule is often late or skips runs (see above). cron-job.org (free) can
start the Pipeline workflow on time instead, through GitHub's `workflow_dispatch` API.
That's the same thing as clicking **Run workflow**. It needs a GitHub token, which you
make **only able to run Actions on this one repo**.

**1. Make a fine-grained token**

1. On GitHub, click your avatar → **Settings** → **Developer settings** →
   **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
2. Fill in:
   - **Token name**: `cron-job.org pipeline trigger`
   - **Expiration**: 1 year at most. Put a reminder in your calendar a week before it
     ends (see *Renewing the token* below).
   - **Resource owner**: your account.
   - **Repository access**: **Only select repositories** → `stl-crime-timer`.
   - **Permissions** → **Repository permissions** → **Actions**: **Read and write**.
     Leave everything else at **No access**. GitHub adds **Metadata: Read-only** by
     itself; that's required and harmless.
3. **Generate token** and copy it (starts with `github_pat_`). You can't see it again.
   Don't save it in a file in the project.

What this token can do: start, re-run, cancel and delete workflow runs, and turn
workflows on or off, on this repo only. It can't read or change your code, secrets,
or any other repo.

**2. Test it from PowerShell** (optional, but tells you the token works before
cron-job.org is involved). Paste your token into the first line, then run:

```powershell
$token = "github_pat_PASTE_HERE"
Invoke-WebRequest -UseBasicParsing -Method Post `
  -Uri "https://api.github.com/repos/manisharanthota/stl-crime-timer/actions/workflows/pipeline.yml/dispatches" `
  -Headers @{ Authorization = "Bearer $token"; Accept = "application/vnd.github+json"; "X-GitHub-Api-Version" = "2022-11-28" } `
  -ContentType "application/json" -Body '{"ref":"master"}'
```

`StatusCode : 204` (no content) means it worked: a new **Pipeline** run appears in the
Actions tab within a few seconds, marked as triggered by `workflow_dispatch`. Close the
PowerShell window afterwards so the token isn't left in it.

**3. Create the cron job**

1. Sign up at <https://cron-job.org> and go to **Cronjobs** → **Create cronjob**.
2. **Common** tab:
   - **Title**: `STL crime pipeline`
   - **URL**:
     `https://api.github.com/repos/manisharanthota/stl-crime-timer/actions/workflows/pipeline.yml/dispatches`
   - **Execution schedule**: every 10 minutes.
   - **Notify me when**: execution fails (and *after the job is disabled*).
3. **Advanced** tab:
   - **Request method**: `POST`
   - **Headers** (add one row each):

     | Key | Value |
     |---|---|
     | `Authorization` | `Bearer github_pat_...` (your token, with `Bearer ` in front) |
     | `Accept` | `application/vnd.github+json` |
     | `X-GitHub-Api-Version` | `2022-11-28` |
     | `Content-Type` | `application/json` |

   - **Request body**: `{"ref":"master"}`
4. **Create**, then open the job and use **Test run**. It should answer **204**, and a
   Pipeline run shows up in the Actions tab.

**Should the GitHub schedule stay on?** Yes, as a backup: if cron-job.org stops, the
GitHub schedule keeps the pipeline going. Two triggers don't double the work. Runs
never overlap (a run that's due while another is going waits, and GitHub keeps at most
one waiting). An extra run only fetches feeds; only new crime-looking headlines reach
the LLM.

**Renewing the token**: before it expires, open the token on GitHub → **Regenerate
token** (same permissions), then paste the new value into the job's `Authorization`
header on cron-job.org (keep `Bearer ` in front). If the token ever leaks, delete it
on GitHub's token page first, then make a new one.

---

## 4. Render: the website

1. Go to <https://render.com> and sign up **with GitHub**.
2. Make a long random admin token (you'll need it for the `/admin` endpoints):
   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   Save it in your password manager.
3. In Render: **New** → **Blueprint** → connect your GitHub account if asked → pick the
   `stl-crime-timer` repo. Render reads `render.yaml` and shows one web service,
   `stl-crime-tracker`, on the free plan.
4. Render asks for the values that `render.yaml` leaves blank:
   - `DATABASE_URL`: the same Supabase Session pooler URI
   - `ADMIN_TOKEN`: the token from step 2
5. Click **Apply** / **Deploy**. The first build takes a few minutes. Watch the
   **Logs** tab for `Uvicorn running on http://0.0.0.0:...`.

> **This project uses pip, not Poetry.** There's no Poetry config or `poetry.lock`,
> and you don't need to set `POETRY_VERSION` or anything else Poetry-related.
> The build log should show `python -m pip install ...`. If it shows `poetry install`
> instead, the service wasn't created from the Blueprint (with **New → Web Service**,
> Render ignores `render.yaml` and guesses Poetry from `pyproject.toml`). Fix it in the
> dashboard: the service → **Settings** → **Build & Deploy**:
> - **Build Command**: `python -m pip install --upgrade pip && python -m pip install .`
> - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
> - **Health Check Path** (under Settings → Health Checks): `/`
>
> Save, then **Manual Deploy** → **Deploy latest commit**. If you added a
> `POETRY_VERSION` environment variable while troubleshooting, delete it; it isn't used.
6. Open the URL Render gives you (like `https://stl-crime-tracker.onrender.com`). You
   should see the timer page. Also try `.../health`.

Good to know:
- Render redeploys automatically on every push to `master`. Migrations aren't run by
  Render: the next Pipeline run applies them (or run Pipeline manually after a push
  that adds a migration).
- Free services **sleep after 15 minutes without visitors** and take about a minute to
  wake up. UptimeRobot's checks (step 5) keep it awake. One always-on free service fits
  within Render's free monthly hours.
- Render's own health check uses `/`, not `/health`, on purpose: `/health` is 503
  whenever the pipeline is late, and that shouldn't make Render think the website is
  broken.
- To use the admin endpoints from PowerShell:
  ```powershell
  $h = @{ "X-Admin-Token" = (Read-Host "Admin token") }
  Invoke-RestMethod https://YOUR-APP.onrender.com/admin/review -Headers $h
  Invoke-RestMethod https://YOUR-APP.onrender.com/admin/incidents/12/confirm -Method Post -Headers $h
  ```

---

## 5. UptimeRobot: get told when something stops

1. Sign up at <https://uptimerobot.com> (free plan).
2. **New monitor**:
   - **Type**: HTTP(s)
   - **Friendly name**: `STL crime pipeline`
   - **URL**: `https://YOUR-APP.onrender.com/health`
   - **Interval**: 5 minutes
   - **Alert contacts**: your email (and/or the mobile app)
3. Save. Within a few minutes it should show **Up**.

What the alerts mean:

| `/health` | Meaning | What to do |
|---|---|---|
| 200 `"status": "ok"` | a pipeline run succeeded in the last 30 min | nothing |
| 503 `"status": "stale"` | no successful run in 30 min (or ever) | check the **Actions** tab: is Pipeline running? failing? disabled? |
| timeout / 5xx with no JSON | the website itself is down | check Render's **Logs** and **Events** |

A run whose status is `partial` (one step failed, e.g. a feed was down) doesn't count
as successful. If every run is partial for 30 minutes, `/health` goes stale. The
`last_run` field in the `/health` response says what the latest run did.

Right after you first deploy, `/health` is 503 until the first successful Pipeline
run, which is expected.

---

## 6. Changing a secret later

- **GitHub**: Settings → Secrets and variables → Actions → click the secret → **Update**.
- **Render**: the service → **Environment** → edit → **Save changes** (it redeploys).
- **Supabase password**: Project Settings → Database → **Reset database password**, then
  update `DATABASE_URL` in **both** GitHub and Render.
- **cron-job.org token** (step 3d): regenerate it on GitHub, then update the job's
  `Authorization` header.

If a secret ever leaks (pasted somewhere public, committed by accident): reset it at
the source first (new Gemini key, new Supabase password, delete and recreate the
Discord webhook), then update GitHub and Render.

---

## 7. What never goes into git

`.gitignore` keeps these out, and a test (`tests/test_repo_hygiene.py`) fails if
any of them is ever committed:

- `.env` (and `.env.*`, except the blank template `.env.example`)
- databases: `*.db`, `stl_crime.db.bak-*`, `*.before-*.db`, journals
- `logs/` and `*.log`

Before committing, `git status` should never list any of these. On GitHub Actions the
pipeline logs to the Actions log only (`LOG_TO_FILE=false`), and the webhook token is
redacted from log lines.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `alembic upgrade head` hangs or "Network is unreachable" | You used the Direct connection URI. Use **Session pooler**. |
| `password authentication failed` | Wrong password in the URI, or `[YOUR-PASSWORD]` brackets left in. Symbols in the password need URL-escaping. |
| `prepared statement ... does not exist` | You used the Transaction pooler (port 6543). Use the Session pooler (port 5432). |
| Render build log runs `poetry install` and fails | The service was made with New → Web Service, so `render.yaml` is ignored. Set the Build Command to `python -m pip install --upgrade pip && python -m pip install .` in the service's Settings (see the pip note in step 4). No `POETRY_VERSION` needed. |
| Render build: `No matching distribution` / wrong Python | Render reads `.python-version` (3.13). Remove any `PYTHON_VERSION` env var you added in the dashboard, or set it to a full version like `3.13.5`. |
| Pipeline log: `DATABASE_URL secret is not set` | Add the secret (name must match exactly) and re-run. |
| Pipeline never runs on schedule | Workflows must be on `master` (the default branch); check Actions isn't disabled. |
| cron-job.org: **401** | Token wrong, expired or revoked, or `Bearer ` missing in front of it. Regenerate it (step 3d). |
| cron-job.org: **403** "Resource not accessible by personal access token" | The token lacks **Actions: Read and write**, or `stl-crime-timer` isn't among its selected repositories. |
| cron-job.org: **404** | Typo in the URL (owner, repo, or `pipeline.yml`), or the token can't see the repo. GitHub answers 404 instead of 403 for repos a token can't access. |
| cron-job.org: **422** | Body isn't `{"ref":"master"}` (or `Content-Type` is missing), or the workflow is disabled: re-enable it under Actions → Pipeline. |
| Page loads but shows nothing | No confirmed incidents yet, or `DATABASE_URL` on Render points at an empty database. |
| Copy script: "target already has rows" | Supabase already has data (maybe a Pipeline run happened first). To start over, disable the Pipeline workflow (Actions → Pipeline → ⋯ → **Disable workflow**), then in Supabase → **SQL Editor** run `truncate sources, raw_items, classifications, incidents, incident_items, pipeline_runs, job_locks, alerts_sent restart identity cascade;` (**this deletes all tracker data in Supabase**), run step 2.3 again, and re-enable the workflow. |
| `/health` stuck at 503 though runs look fine | Runs are `partial`/`failed`. Open the latest run's log, or look at `pipeline_runs.*_error` in Supabase. |
