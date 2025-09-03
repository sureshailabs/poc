# Text-to-SQL — ClickHouse (Env Exec + Summary)

Single-file Streamlit app:
- Loads your **ClickHouse DDL** (from `DDL_PATH` or `orbit_japa.sql`)
- Generates a **single SELECT** (safe) using **Gemini/OpenAI**
- **Executes** it on ClickHouse **from env config** (no UI config)
- **Summarizes** the result in 2–4 lines

# can also run using docker desktop also
# or run following bash command in the terminal

# python version used 3.10.6
## Run
```bash
pip install -r requirements.txt

streamlit run app.py
```




## Questions

1) List all active users who joined in the last 30 days
2) What are the top categories by total mala count?
3) How many active users do we have?
4) Top 10 users by total sessions (lifetime)
5) List users with global leaderboard enabled
6) Top 10 temples by total mala (join sessions ↔ users)

Supports casual chit chat and generic question as well