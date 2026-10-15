# Operations

How the assistant is monitored, and what to do when something goes wrong.

## Daily schedule (Asia/Taipei)

| Time | Job | Runs on | What it does |
|---|---|---|---|
| 10:30 | `local_worker` | Your computer (Windows Task Scheduler) | Finds new videos, skips Shorts, fetches captions |
| 11:00 | `daily_pipeline` | Modal | Finds new videos again (backup), summarizes with an LLM |
| 12:00 | `send_digest` | Modal | Sends the LINE digest and pings the heartbeat |
| Sunday 14:00 | `eval_drift` | Your computer (Windows Task Scheduler) | Re-runs the primary model on the evaluation videos and compares it with a reference run |

Each run is recorded in the `job_runs` table. Log lines carry `run=<id>` so one run can be followed end to end.

## Service goals

- **Delivery:** a digest arrives by 12:15 every day, even if it only explains what went wrong.
- **Quality:** at least half of a week's summaries are based on captions (⭐⭐ or ⭐⭐⭐).
- **Coverage:** every channel is read successfully at least once every 2 days.

## Alerts

| Layer | Catches | Where it shows up |
|---|---|---|
| Heartbeat ([Healthchecks.io](https://healthchecks.io) or compatible) | No digest today, for any reason: the schedule did not run, the job crashed, LINE rejected the push | Email |
| Modal failure notice | An error in a cloud job | Email from Modal |
| Digest notices | Degraded runs: worker did not run, captions blocked, LLM quota used up, plus problems that persist (see below) | Top of the LINE digest |

Persistent problems reported in the digest:

| Notice | Condition |
|---|---|
| A channel cannot be read | No job has read it for more than 2 days |
| Primary model unreliable | It failed in at least half of its analyses in the last 3 runs |
| Repeated caption errors | The last 2 worker runs had caption errors (videos without captions do not count) |
| Mostly topic guesses | More than half of the last 7 days' summaries (at least 4) had no captions |
| Answered by another model | In the last pipeline run, OpenRouter answered a model ID with a different model |
| Rule problems left in outputs | A summary from the last day still has a problem the post-processing rules remove (half-width gloss, wrapping quotes, too many key points, and so on) |
| Model drift | The latest completed weekly drift check found identity drift (another model answering) or quality drift (final-topic F1 or precision moved beyond its threshold) |
| Drift check overdue | No drift check completed in 8 days (computer off, check stopped by the quota, skipped because the reference is outdated, or failed); the warning says which |

Cloud feed failures alone are not reported: the cloud read is only a backup for the worker and fails routinely.

Drift warnings are derived from the drift check's job runs on every digest, not stored: a drift warning
disappears as soon as a later check is stable, and nothing is written twice. A provider change alone
("execution environment changed") is shown in the drift report's title but not alerted on: free models move
between providers routinely, and if it matters it shows up as quality drift.

"Answered by another model" and "rule problems" are invariants, not trends: one occurrence is enough. Latency, failure rates and the number of
topics per video are not alerted on, because with 1-5 videos a day no threshold could be justified; `yla metrics`
shows them as trends instead.

## Tools

| Command | Use |
|---|---|
| `uv run yla health` | Checks every service, the configuration and usage. Never calls the LLM. Exit code 1 on any failure |
| `uv run yla status --days 3` | Recent runs of each job with their key numbers and errors |
| `uv run yla eval drift` | The weekly drift check, run by hand; the result is in `eval/runs/drift-*/drift.md` |
| `uv run yla metrics --days 14` | LLM trends: results, latency and tokens per model, fallbacks, star mix, topics per video, rule fixes and problems per day |
| `uv run yla digest --dry-run` | Shows today's digest without sending it |
| `uv run modal app logs youtube-learning-assistant` | Cloud logs |
| `logs/worker.log` | Local worker output, one block per run |

Run `uv run yla health` once a week, and after any change to settings, keys or code.

## Runbook

### No digest today (heartbeat alert)

- **Likely causes:** the cloud job crashed before sending; the LINE token is invalid; the database is unreachable; the Modal app is not deployed or out of credits.
- **Check:** `uv run yla status --days 1`, then `uv run yla health`.
- **Decide:**
  - `send_digest` failed with `HTTP 401`: the LINE token is invalid.
  - `send_digest` failed with a database error: check the Neon project.
  - No `send_digest` run today: the schedule did not run. Check the app in the Modal dashboard.
- **Fix:** reissue the LINE token, update `.env` and the Modal secret (see Routine tasks), or redeploy.
- **Verify:** `uv run modal run src/yla/modal_app.py::send_digest`. It sends at most once per day; if it already went out, it reports that.

### "本機 worker 今天沒有執行"

- **Likely causes:** the computer was off or on battery (wake timers may be off on battery); the task is disabled; the virtual environment is missing.
- **Check:** the task's last run and result:

  ```powershell
  Get-ScheduledTaskInfo -TaskName "YouTube Learning Assistant - local worker"
  ```

  Also read the end of `logs/worker.log`.
- **Decide:** `LastTaskResult` 0 means it ran (check the log). A missing or old `LastRunTime` means it never started.
- **Fix:** run it now with `Start-ScheduledTask -TaskName "YouTube Learning Assistant - local worker"`, or re-register it with `scripts\windows\register-task.ps1`. If the log says `yla not found`, run `uv sync`.
- **Verify:** `uv run yla status --days 1` shows a succeeded `local_worker` run. Captions fetched within 3 days of a video's discovery upgrade its summary on the next pipeline run.

### Captions blocked by YouTube

- **Likely causes:** the network is blocked (cloud servers, some VPNs, corporate or mobile networks), or too many requests in a short time.
- **Check:** `uv run yla worker --check`. Exit code 2 means blocked.
- **Fix:** run the worker from a home network without a VPN. Wait a few hours if it was caused by repeated manual runs. Do not use proxies to get around blocks.
- **Verify:** `uv run yla worker --check` reports captions.

### A channel cannot be read for days

- **Likely causes:** a wrong `channel_id`, or the channel was removed or renamed. A short outage of YouTube's feeds is retried automatically.
- **Check:** open `https://www.youtube.com/feeds/videos.xml?channel_id=<id>` in a browser. A persistent 404 means the ID is wrong.
- **Fix:** correct `channel_id` in `config/settings.yaml`, then run `uv run yla sync-config`.
- **Verify:** `uv run yla discover` reads the channel.

### Primary model unreliable, or LLM quota used up

- **Likely causes:** the free model is overloaded, was removed, or is no longer free; the daily free request quota is spent.
- **Check:** `uv run yla health` (whether each model is still listed and free) and `uv run yla status` (`model_outcomes` in recent runs).
- **Fix:** pick another free long-context model with structured output from [OpenRouter's model list](https://openrouter.ai/models?max_price=0), set `llm.primary_model` or `llm.fallback_models` in `config/settings.yaml`, and redeploy. Deferred videos are analysed the next day.
- **Verify:** the next run's `model_outcomes` show `ok` for the primary model.

### Cloud pipeline failed (email from Modal)

- **Check:** `uv run yla status --days 1` shows the error. Read the full log with `uv run modal app logs youtube-learning-assistant`.
- **Fix:** fix the cause, redeploy, then rerun today's analysis (see Routine tasks). The digest at 12:00 still goes out and reports the failure.
- **Verify:** the rerun succeeds in `uv run yla status`.

### "要求 X 卻由 Y 回應" (answered by another model)

- **Cause:** OpenRouter served a model ID with a different model (rerouting, or the model was replaced). The
  summaries may change without any change on our side.
- **Check:** `uv run yla metrics --days 7`: the model's providers and "answered by another model".
- **Decide:** once, it may be a temporary reroute. If it persists, treat it as a model change: run the evaluation
  (`yla eval run`) on the current setting and compare with the last baseline before trusting the summaries.
- **Fix:** if the replacement is worse, choose another primary model (see "Primary model unreliable").

### "摘要仍有程式應修正的格式問題" (rule problems left in outputs)

- **Cause:** a rule in `normalize` or `tidy_text` stopped working, or the model's output changed shape so the
  rules no longer match it.
- **Check:** `uv run yla metrics --days 2` lists the videos and problems. Reproduce with
  `uv run yla analyze VIDEO_ID --dry-run --raw`.
- **Fix:** add the case to the rule tests, fix the rule, bump `RULES_VERSION`, and replay the evaluation runs
  (`yla eval report`) to check nothing else changed.

### "每週漂移檢查：主模型的表現和基準不同" (model drift)

- **Cause:** the model behind the configured ID changed (a new version, a reroute), so the same inputs now get
  different topics. The thresholds are computed on every check from the A/A pairs in `eval/drift.yaml`
  (for F1 and for precision, the largest 95% CI bound of the difference between two identical runs), so this
  is more than run-to-run noise has shown so far. They are provisional: add A/A pairs as more runs are made.
- **Scope:** the check compares outputs for the same videos. It cannot detect a shift in the inputs (new
  channels or new kinds of videos); for that, extend the evaluation dataset.
- **Check:** read `eval/runs/<run>/drift.md`, then compare in full with
  `uv run yla eval report <reference run> <drift run> --no-blind`.
- **Decide:** if quality dropped, evaluate another model (see "Primary model unreliable"); if it improved or
  only changed, run the blind review before trusting it.
- **After any change of primary model or prompt:** make a new full run, set it as `reference_run` in
  `eval/drift.yaml`, and replace the A/A pairs with runs of the new configuration; until then the check skips
  itself, and after 8 days the digest says the check is overdue and why.

### "每週漂移檢查已 N 天沒有完成" (drift check overdue)

- **Check:** `uv run yla status --days 10` and the end of `logs/drift.log`. The warning names the last result.
- **Fix:** computer off or asleep without wake timers: run it now with
  `Start-ScheduledTask -TaskName "YouTube Learning Assistant - drift check"`. Stopped by the quota: run
  `uv run yla eval drift` again after 08:00 the next day (it resumes). Skipped: set a new reference (above).

### Cloud settings differ from settings.yaml

- **Cause:** `config/settings.yaml` changed without redeploying, so the cloud still uses the old channels, topics or models.
- **Fix:** `uv run modal deploy src/yla/modal_app.py`.
- **Verify:** after the next pipeline run, `uv run yla health` shows the cloud settings matching.

### Runs stuck in "running"

- **Cause:** a run was stopped by its platform time limit or by hand before it could record the result.
- **Fix:** none needed. The next run of the same job marks it failed (`stale_run_timeout`).

### Repeated caption errors

- **Likely cause:** YouTube changed something and `youtube-transcript-api` needs an update.
- **Fix:** `uv lock --upgrade-package youtube-transcript-api`, `uv sync`, run the tests, then redeploy.
- **Verify:** `uv run yla worker --check` reports captions.

## Routine tasks

| Task | Command |
|---|---|
| Rerun today's analysis in the cloud | `uv run modal run src/yla/modal_app.py::daily_pipeline --skip-discover` |
| Preview today's digest from the cloud | `uv run modal run src/yla/modal_app.py::send_digest --dry-run` |
| Deploy code or settings changes | `uv run modal deploy src/yla/modal_app.py` |
| Apply database migrations | `uv run alembic upgrade head` (uses `DATABASE_URL`). Run it **before** deploying code that needs the new schema; new tables are unused by the old code, so this order is always safe |
| Update keys in the cloud after editing `.env` | `uv run modal secret create yla-secrets --from-dotenv .env --force` |
| Change channels or topics | Edit `config/settings.yaml`, then `uv run yla validate-config`, `uv run yla sync-config`, and deploy |
| Register the weekly drift check | `powershell -ExecutionPolicy Bypass -File scripts\windows\register-drift-task.ps1` |
| Upgrade dependencies | `uv lock --upgrade`, `uv sync`, `uv run pytest`, then deploy |
| Restore data | Use Neon's restore (branch from a point in time) in the Neon console; the free plan keeps a short history window |
