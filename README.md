# Nifty 500 High-Probability CMF Breakout EOD Scanner

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. In Streamlit Community Cloud, create an app and select `app.py` as the main file.
3. Wait for dependencies to install, then open the app.
4. Run the scan after **3:45 PM IST** for completed daily candles.

## Nifty 500 universe handling
- The app fetches the official Nifty 500 constituent CSV from NSE/Nifty Indices.
- After a successful fetch, it saves the verified symbols to `data/nifty500_symbols.csv` and reuses that local copy if the live endpoint is temporarily unavailable.
- If no verified local copy exists and both endpoints fail, the app stops rather than silently scanning a small fallback list. Upload the official Nifty 500 CSV using the uploader in the app; it will be saved for later runs.
- Streamlit Cloud filesystems can be reset on redeploy/restart. If the saved CSV disappears, refresh the list or upload the official CSV again.

## Strategy rules preserved
- Trend: Close > EMA 50.
- CMF(20): base breakout > 0.05; A+ threshold defaults to >= 0.25.
- Resistance: highest high of the previous 20 sessions, excluding the current candle.
- Volume: current volume / previous 20-session average volume; base >= 2.0x and A+ default >= 2.3x.
- A+ breakout distance: +1.0% to +5.5% above prior 20-session resistance.
- A+ liquidity: daily turnover >= ₹10 Cr.
- Near-breakout watch: trend positive, CMF >= 0.20, volume >= 2.0x, and close from -2.5% to 0% versus resistance.
- Confirmed breakouts receive mechanical stop-loss and 1:2 / 1:3 targets. Near-breakout watch rows intentionally do not show trade levels.

## Technical corrections
- Batch downloads (40 symbols per request) to reduce oversized Yahoo Finance requests.
- Handles both common yfinance MultiIndex column layouts.
- Counts unavailable symbols and exposes them in the app instead of silently hiding all failures.
- Clamps the conviction-score components to 0..1 so the score remains within 0..100.
- Avoids calculating trade targets for near-breakout watch candidates.
- Telegram is optional and sends only when the user explicitly clicks the alert button.

## Important
This is a rule-based research screener, not a guarantee of profitable or high-probability trades. Yahoo Finance data can be delayed, incomplete, adjusted, or unavailable for individual symbols. Validate signals and execution prices independently. The app does not place orders.
