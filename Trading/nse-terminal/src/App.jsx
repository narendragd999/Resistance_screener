import { useState, useEffect, useRef, useCallback } from "react";

// ── Design tokens ────────────────────────────────────────────────────────────
const C = {
  bg:          "#06090D",
  bgPanel:     "#0B0F16",
  bgCard:      "#0F1520",
  border:      "#1A2535",
  borderBright:"#2A3F55",
  amber:       "#FFB300",
  amberDim:    "#7A5000",
  amberGlow:   "rgba(255,179,0,0.15)",
  green:       "#00D68F",
  greenDim:    "#003D28",
  red:         "#FF4555",
  redDim:      "#4D1018",
  blue:        "#4DA6FF",
  blueDim:     "#0D2440",
  purple:      "#A855F7",
  muted:       "#3D5468",
  mutedBright: "#5A7A90",
  text:        "#C0D4E8",
  textBright:  "#E4F2FF",
};

const MONO = "'Courier New', 'Lucida Console', monospace";
const UI   = "'Trebuchet MS', 'Arial Narrow', sans-serif";

// ── Formatters ───────────────────────────────────────────────────────────────
const pct  = (v) => (v >= 0 ? `+${v.toFixed(2)}%` : `${v.toFixed(2)}%`);
const fmt  = (v) => (v != null ? `₹${Number(v).toLocaleString("en-IN")}` : "—");
const clr  = (v) => (v >= 0 ? C.green : C.red);

// ── Shared styles ─────────────────────────────────────────────────────────────
const monoLabel = {
  fontFamily: MONO, fontSize: 10, color: C.muted,
  letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 6,
};

// ── Blinking cursor ───────────────────────────────────────────────────────────
function Cursor() {
  const [on, setOn] = useState(true);
  useEffect(() => { const t = setInterval(() => setOn(x => !x), 530); return () => clearInterval(t); }, []);
  return (
    <span style={{
      display: "inline-block", width: 9, height: "1em",
      background: on ? C.amber : "transparent",
      verticalAlign: "text-bottom", marginLeft: 2,
    }} />
  );
}

// ── Scanlines ─────────────────────────────────────────────────────────────────
function Scanlines() {
  return (
    <div style={{
      position: "fixed", inset: 0, pointerEvents: "none", zIndex: 9999,
      background: "repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,214,143,0.008) 2px,rgba(0,214,143,0.008) 4px)",
    }} />
  );
}

// ── Section header ────────────────────────────────────────────────────────────
function SectionHead({ label, color = C.amber, icon }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
      <div style={{
        width: 3, height: 18, background: color,
        boxShadow: `0 0 10px ${color}`, borderRadius: 2,
      }} />
      {icon && <span style={{ fontSize: 13 }}>{icon}</span>}
      <span style={{
        fontFamily: MONO, fontSize: 11, letterSpacing: 3,
        color, textTransform: "uppercase", fontWeight: 700,
      }}>{label}</span>
    </div>
  );
}

// ── Stat chip ─────────────────────────────────────────────────────────────────
function Stat({ label, value, color, sub }) {
  return (
    <div style={{
      background: C.bgCard, border: `1px solid ${C.border}`,
      borderRadius: 6, padding: "10px 14px", flex: "1 1 120px", minWidth: 100,
    }}>
      <div style={monoLabel}>{label}</div>
      <div style={{ fontFamily: MONO, fontSize: 15, fontWeight: 700, color: color || C.textBright }}>{value}</div>
      {sub && <div style={{ fontFamily: MONO, fontSize: 10, color: C.muted, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

// ── Price level row ───────────────────────────────────────────────────────────
function LevelRow({ label, price, pctVal, type }) {
  const colorMap = { sl: C.red, t1: "#FFD700", t2: C.green, t3: "#00BFFF", entry: C.amber };
  const c = colorMap[type] || C.text;
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "11px 16px", borderBottom: `1px solid ${C.border}`,
      background: type === "sl" ? "rgba(255,69,85,0.05)" : "transparent",
      transition: "background 0.2s",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: c, boxShadow: `0 0 6px ${c}` }} />
        <span style={{ fontFamily: MONO, fontSize: 12, color: C.text, letterSpacing: 1 }}>{label}</span>
      </div>
      <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
        {pctVal !== undefined && (
          <span style={{ fontFamily: MONO, fontSize: 11, color: type === "sl" ? C.red : C.green, opacity: 0.85 }}>
            {pctVal}
          </span>
        )}
        <span style={{ fontFamily: MONO, fontSize: 14, fontWeight: 700, color: c, minWidth: 90, textAlign: "right" }}>
          {price ? fmt(price) : "—"}
        </span>
      </div>
    </div>
  );
}

// ── Checklist item ────────────────────────────────────────────────────────────
function CheckItem({ label, status, note }) {
  const s = status === true
    ? { icon: "✓", color: C.green,  bg: "rgba(0,214,143,0.07)" }
    : status === false
    ? { icon: "✗", color: C.red,    bg: "rgba(255,69,85,0.07)"  }
    : { icon: "◎", color: C.amber,  bg: "rgba(255,179,0,0.05)"  };
  return (
    <div style={{
      display: "flex", alignItems: "flex-start", gap: 12,
      padding: "10px 14px", background: s.bg,
      borderRadius: 6, border: `1px solid ${C.border}`,
    }}>
      <span style={{ fontFamily: MONO, fontSize: 14, fontWeight: 900, color: s.color, lineHeight: 1.3, minWidth: 16, textAlign: "center" }}>
        {s.icon}
      </span>
      <div>
        <div style={{ fontFamily: MONO, fontSize: 12, color: C.text, marginBottom: note ? 3 : 0 }}>{label}</div>
        {note && <div style={{ fontFamily: MONO, fontSize: 10, color: C.muted }}>{note}</div>}
      </div>
    </div>
  );
}

// ── Factor pill ───────────────────────────────────────────────────────────────
function Factor({ text, positive }) {
  return (
    <div style={{
      display: "flex", alignItems: "flex-start", gap: 10, padding: "8px 12px",
      background: positive ? "rgba(0,214,143,0.05)" : "rgba(255,69,85,0.05)",
      borderLeft: `2px solid ${positive ? C.green : C.red}`,
      borderRadius: "0 4px 4px 0",
    }}>
      <span style={{ color: positive ? C.green : C.red, fontSize: 12, fontWeight: 900, marginTop: 1 }}>
        {positive ? "▲" : "▼"}
      </span>
      <span style={{ fontFamily: MONO, fontSize: 12, color: C.text, lineHeight: 1.55 }}>{text}</span>
    </div>
  );
}

// ── Sentiment badge ───────────────────────────────────────────────────────────
function SentimentBadge({ sentiment }) {
  const map = {
    BULLISH: { color: C.green, bg: "rgba(0,214,143,0.12)", label: "◆ BULLISH SETUP" },
    BEARISH: { color: C.red,   bg: "rgba(255,69,85,0.12)",  label: "◆ BEARISH CAUTION" },
    NEUTRAL: { color: C.amber, bg: "rgba(255,179,0,0.10)",  label: "◆ NEUTRAL / WATCH" },
  };
  const s = map[sentiment] || map.NEUTRAL;
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 8,
      padding: "6px 14px", background: s.bg,
      border: `1px solid ${s.color}`, borderRadius: 20,
      boxShadow: `0 0 14px ${s.color}25`,
    }}>
      <span style={{ fontFamily: MONO, fontSize: 11, fontWeight: 800, color: s.color, letterSpacing: 2 }}>
        {s.label}
      </span>
    </div>
  );
}

// ── Mini sparkline (purely decorative) ───────────────────────────────────────
function MiniChart({ positive }) {
  const pts = positive
    ? "0,30 10,28 20,25 30,20 40,22 50,15 60,10 70,12 80,8 90,5 100,2"
    : "0,5 10,8 20,10 30,15 40,12 50,18 60,22 70,20 80,25 90,28 100,30";
  const color = positive ? C.green : C.red;
  return (
    <svg width={100} height={32} style={{ opacity: 0.7 }}>
      <polyline fill="none" stroke={color} strokeWidth={1.5} points={pts} />
      <defs>
        <linearGradient id={`g${positive}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
    </svg>
  );
}

// ── Risk Calculator ───────────────────────────────────────────────────────────
function RiskCalc({ cmp, stopLoss }) {
  const [capital, setCapital]   = useState("100000");
  const [riskPct, setRiskPct]   = useState("2");

  const cap      = parseFloat(capital) || 0;
  const rPct     = parseFloat(riskPct) || 2;
  const sl       = parseFloat(stopLoss) || 0;
  const price    = parseFloat(cmp) || 1;
  const slPct    = sl ? Math.abs(((price - sl) / price) * 100) : 3.5;
  const riskAmt  = (cap * rPct) / 100;
  const lossPer  = price * (slPct / 100);
  const shares   = lossPer > 0 ? Math.floor(riskAmt / lossPer) : 0;
  const posSz    = shares * price;
  const profitAt5= posSz * 0.05;
  const rr       = lossPer > 0 ? ((price * 0.05) / lossPer).toFixed(1) : "—";

  const inputStyle = {
    width: "100%", background: C.bg, border: `1px solid ${C.border}`,
    borderRadius: 4, padding: "9px 10px", fontFamily: MONO, fontSize: 13,
    color: C.amber, outline: "none", boxSizing: "border-box",
    transition: "border-color 0.2s",
  };

  return (
    <div>
      <SectionHead label="Position Sizer" color={C.blue} icon="⚖" />
      <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
        <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
          {[
            { label: "Total Capital (₹)", val: capital, set: setCapital, type: "number" },
            { label: "Risk Per Trade (%)", val: riskPct, set: setRiskPct, type: "number" },
          ].map(({ label, val, set, type }) => (
            <div key={label} style={{ flex: 1, minWidth: 140 }}>
              <div style={monoLabel}>{label}</div>
              <input
                type={type} value={val}
                onChange={e => set(e.target.value)}
                style={inputStyle}
                onFocus={e => e.target.style.borderColor = C.amber}
                onBlur={e => e.target.style.borderColor = C.border}
              />
            </div>
          ))}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {[
            { label: "Max Risk Amount",  val: `₹${Math.round(riskAmt).toLocaleString("en-IN")}`,     c: C.red    },
            { label: "Shares to Buy",    val: `${shares.toLocaleString("en-IN")} shares`,             c: C.amber  },
            { label: "Position Size",    val: `₹${Math.round(posSz).toLocaleString("en-IN")}`,       c: C.blue   },
            { label: "5% Profit",        val: `+₹${Math.round(profitAt5).toLocaleString("en-IN")}`,  c: C.green  },
            { label: "SL Distance",      val: `${slPct.toFixed(2)}%`,                                 c: C.red    },
            { label: "Risk:Reward (5%)", val: `1 : ${rr}`,                                            c: C.purple },
          ].map(({ label, val, c }) => (
            <div key={label} style={{
              background: C.bg, border: `1px solid ${C.border}`,
              borderRadius: 6, padding: "10px 12px",
            }}>
              <div style={monoLabel}>{label}</div>
              <div style={{ fontFamily: MONO, fontSize: 14, fontWeight: 700, color: c }}>{val}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Loading terminal ──────────────────────────────────────────────────────────
function LoadingTerminal({ stock }) {
  const [dots, setDots]       = useState(0);
  const [lineIdx, setLineIdx] = useState(0);
  const lines = [
    `CONNECTING TO NSE DATA FEED...`,
    `FETCHING ${stock.toUpperCase()} LIVE PRICE DATA...`,
    `SCANNING TECHNICAL INDICATORS...`,
    `COMPUTING SUPPORT/RESISTANCE ZONES...`,
    `RUNNING CONFLUENCE ANALYSIS...`,
    `BUILDING STRATEGY MATRIX...`,
    `FINALISING TRADE PLAN...`,
  ];
  useEffect(() => {
    const t1 = setInterval(() => setDots(d => (d + 1) % 4), 300);
    const t2 = setInterval(() => setLineIdx(i => Math.min(i + 1, lines.length - 1)), 800);
    return () => { clearInterval(t1); clearInterval(t2); };
  }, []);

  return (
    <div style={{ padding: "48px 24px", display: "flex", flexDirection: "column", alignItems: "center", gap: 28 }}>
      <div style={{ position: "relative", width: 64, height: 64 }}>
        <div style={{
          position: "absolute", inset: 0,
          border: `2px solid ${C.border}`,
          borderTop: `2px solid ${C.amber}`,
          borderRadius: "50%",
          animation: "spin 1s linear infinite",
        }} />
        <div style={{
          position: "absolute", inset: 8,
          border: `2px solid ${C.border}`,
          borderBottom: `2px solid ${C.green}`,
          borderRadius: "50%",
          animation: "spin 1.5s linear infinite reverse",
        }} />
      </div>
      <div style={{
        background: C.bgCard, border: `1px solid ${C.border}`,
        borderRadius: 8, padding: "20px 28px", minWidth: 340, maxWidth: 500,
      }}>
        {lines.slice(0, lineIdx + 1).map((line, i) => (
          <div key={i} style={{
            fontFamily: MONO, fontSize: 11,
            color: i === lineIdx ? C.amber : C.muted,
            marginBottom: 7, display: "flex", alignItems: "center", gap: 10,
          }}>
            <span style={{ color: i < lineIdx ? C.green : C.amber, minWidth: 12 }}>
              {i < lineIdx ? "✓" : "›"}
            </span>
            {line}{i === lineIdx && ".".repeat(dots)}
          </div>
        ))}
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ── Global styles ─────────────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @keyframes spin        { to { transform: rotate(360deg); } }
  @keyframes fadeIn      { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
  @keyframes pulse       { 0%,100%{opacity:1} 50%{opacity:0.35} }
  @keyframes glow        { 0%,100%{box-shadow:0 0 8px ${C.amber}40} 50%{box-shadow:0 0 20px ${C.amber}80} }
  .result-section        { animation: fadeIn 0.4s ease both; }
  input:focus            { border-color: ${C.amber} !important; box-shadow: 0 0 0 2px ${C.amberDim} !important; }
  ::-webkit-scrollbar   { width: 6px; }
  ::-webkit-scrollbar-track  { background: ${C.bg}; }
  ::-webkit-scrollbar-thumb  { background: ${C.border}; border-radius: 3px; }
  .quick-btn:hover       { border-color: ${C.amberDim} !important; color: ${C.amber} !important; }
  .analyze-btn:hover:not(:disabled) { background: #FFC933 !important; transform: translateY(-1px); }
`;

// ── System prompt ─────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are an expert NSE equity trading analyst with 20 years of experience in Indian markets. 
Search the web for current, live market data for the given stock symbol on NSE (National Stock Exchange of India).

Return ONLY a single valid JSON object. No markdown, no explanation, no backticks, no preamble.

Schema (all price fields are plain numbers, no currency symbol):
{
  "symbol": "TICKER",
  "companyName": "Full Company Name",
  "sector": "Sector",
  "exchange": "NSE",
  "cmp": 1234.56,
  "change": 12.5,
  "changePct": 1.02,
  "dayHigh": 1240.00,
  "dayLow": 1220.00,
  "openPrice": 1225.00,
  "prevClose": 1222.00,
  "weekHigh52": 1500.00,
  "weekLow52": 900.00,
  "pe": 25.4,
  "pbv": 3.2,
  "beta": 1.2,
  "marketCap": "1.2L Cr",
  "dividendYield": 1.5,
  "roe": 18.5,
  "eps": 48.6,
  "debtToEquity": 0.3,
  "monthReturn": 5.2,
  "yearReturn": 22.1,
  "avgVolume20d": "2.1M",
  "analystTarget": 1400,
  "analystCount": 12,
  "analystBuy": 8,
  "analystHold": 3,
  "analystSell": 1,
  "sentiment": "BULLISH",
  "sentimentReason": "One-line reason",
  "technicalRating": "BUY",
  "bullishFactors": ["Factor 1", "Factor 2", "Factor 3"],
  "bearishFactors": ["Risk 1", "Risk 2"],
  "catalysts": ["Upcoming catalyst 1", "Upcoming catalyst 2"],
  "entryPullback": {
    "low": 1180,
    "high": 1200,
    "rationale": "Why this zone is strong support"
  },
  "entryBreakout": {
    "level": 1250,
    "volumeMultiple": 1.5,
    "rationale": "Why this breakout level matters"
  },
  "stopLoss": 1150,
  "stopLossPct": 3.5,
  "target1": { "price": 1270, "pct": 3.0, "label": "Conservative" },
  "target2": { "price": 1295, "pct": 5.0, "label": "Primary (5%)" },
  "target3": { "price": 1320, "pct": 7.0, "label": "Aspirational" },
  "confluenceChecklist": [
    { "item": "Price above 20 EMA on daily chart", "status": null, "note": "Check TradingView" },
    { "item": "RSI (14) between 50–70", "status": null, "note": "Momentum without overbought" },
    { "item": "Volume > 1.5× 20-day average on breakout", "status": null, "note": "Confirm institutional activity" },
    { "item": "Price above VWAP (intraday)", "status": null, "note": "Intraday strength" },
    { "item": "Nifty50 in uptrend (no breakdown)", "status": null, "note": "Macro alignment critical" },
    { "item": "No major earnings/event in next 5 days", "status": null, "note": "Avoid event risk" },
    { "item": "MACD bullish crossover or positive histogram", "status": null, "note": "Momentum confirmation" },
    { "item": "Sector index outperforming Nifty", "status": null, "note": "Sector tailwind" }
  ],
  "weeklyStrategy": "2–3 sentence tactical plan: when to enter, how to manage, when to exit.",
  "keyLevels": {
    "support1": 1190,
    "support2": 1160,
    "resistance1": 1260,
    "resistance2": 1300
  },
  "riskNote": "1–2 sentence honest caution about this specific trade.",
  "dataNote": "Note on data freshness and any caveats."
}

Rules:
- Use web search to find the LATEST available price for this NSE stock
- All price/number fields must be plain numbers (no ₹, no commas)
- sentiment: exactly "BULLISH", "BEARISH", or "NEUTRAL"
- technicalRating: "STRONG BUY", "BUY", "NEUTRAL", "SELL", or "STRONG SELL"
- status in confluenceChecklist: true=confirmed bullish, false=confirmed bearish, null=must verify
- Set realistic entry/SL/target levels relative to ACTUAL current price
- If symbol invalid: {"error": "Stock symbol not found on NSE. Please verify and try again."}`;

// ── MAIN APP ──────────────────────────────────────────────────────────────────
export default function TradingStrategyApp() {
  const [symbol,    setSymbol]   = useState("HDFCAMC");
  const [inputVal,  setInputVal] = useState("HDFCAMC");
  const [loading,   setLoading]  = useState(false);
  const [data,      setData]     = useState(null);
  const [error,     setError]    = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const inputRef = useRef();

  const QUICK = ["HDFCAMC","RELIANCE","INFY","TCS","ICICIBANK","SBIN","BAJFINANCE","HDFCBANK","WIPRO","ADANIENT","TATAMOTORS","NIFTY50"];

  const analyzeStock = useCallback(async (sym) => {
    const clean = sym.trim().toUpperCase();
    if (!clean) return;
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: SYSTEM_PROMPT,
          tools: [{ type: "web_search_20250305", name: "web_search" }],
          messages: [{
            role: "user",
            content: `Analyze NSE stock: ${clean}. Search for latest price, fundamentals, and technicals, then generate the complete trading strategy JSON.`,
          }],
        }),
      });

      const json = await res.json();

      if (!res.ok) {
        const msg = json?.error?.message || `API error ${res.status}`;
        throw new Error(msg);
      }

      // Extract all text blocks (tool use may produce several content blocks)
      const textBlocks = (json.content || []).filter(b => b.type === "text");
      let raw = textBlocks.map(b => b.text).join("");

      // Strip markdown fences
      raw = raw.replace(/```json\s*/gi, "").replace(/```\s*/g, "").trim();

      // Extract JSON object
      const start = raw.indexOf("{");
      const end   = raw.lastIndexOf("}");
      if (start === -1 || end === -1) throw new Error("Response did not contain valid JSON.");

      const parsed = JSON.parse(raw.slice(start, end + 1));

      if (parsed.error) {
        setError(parsed.error);
      } else {
        setData(parsed);
        setLastUpdated(new Date());
      }
    } catch (e) {
      setError(e.message || "Analysis failed. Please check the symbol and try again.");
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-load on mount
  useEffect(() => { analyzeStock("HDFCAMC"); }, []);

  const handleSearch = () => {
    const sym = inputVal.trim().toUpperCase();
    if (!sym || loading) return;
    setSymbol(sym);
    analyzeStock(sym);
  };

  const handleKey = (e) => { if (e.key === "Enter") handleSearch(); };

  const techColor = (r) => {
    if (!r) return C.muted;
    if (r.includes("STRONG BUY")) return C.green;
    if (r.includes("BUY")) return "#80FF80";
    if (r.includes("STRONG SELL")) return C.red;
    if (r.includes("SELL")) return "#FF8080";
    return C.amber;
  };

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: UI, position: "relative" }}>
      <style>{GLOBAL_CSS}</style>
      <Scanlines />

      {/* ── Top bar ── */}
      <div style={{
        background: C.bgPanel, borderBottom: `1px solid ${C.border}`,
        padding: "0 24px", display: "flex", alignItems: "center",
        justifyContent: "space-between", height: 56,
        position: "sticky", top: 0, zIndex: 100,
        backdropFilter: "blur(8px)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 34, height: 34, background: C.amber, borderRadius: 7,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 900, color: C.bg,
            boxShadow: `0 0 16px ${C.amberDim}`,
          }}>N</div>
          <div>
            <div style={{ fontFamily: MONO, fontSize: 13, fontWeight: 700, color: C.amber, letterSpacing: 2 }}>
              NSE STRATEGY TERMINAL
            </div>
            <div style={{ fontFamily: MONO, fontSize: 9, color: C.muted, letterSpacing: 1 }}>
              AI-POWERED · SWING TRADE ANALYZER · 5% WEEKLY TARGET
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {lastUpdated && (
            <span style={{ fontFamily: MONO, fontSize: 10, color: C.muted }}>
              UPD {lastUpdated.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}
            </span>
          )}
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontFamily: MONO, fontSize: 11, color: C.green }}>
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: C.green, boxShadow: `0 0 8px ${C.green}`,
              animation: "pulse 2s ease-in-out infinite",
            }} />
            LIVE
          </div>
        </div>
      </div>

      {/* ── Search bar ── */}
      <div style={{ background: C.bgPanel, borderBottom: `1px solid ${C.border}`, padding: "16px 24px" }}>
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
            <div style={{
              flex: 1, display: "flex", alignItems: "center",
              background: C.bg, border: `1px solid ${C.border}`,
              borderRadius: 6, padding: "0 14px", gap: 10,
            }}>
              <span style={{ fontFamily: MONO, fontSize: 14, color: C.amber, fontWeight: 700 }}>NSE:</span>
              <input
                ref={inputRef}
                value={inputVal}
                onChange={e => setInputVal(e.target.value.toUpperCase())}
                onKeyDown={handleKey}
                placeholder="SYMBOL  (e.g. RELIANCE, INFY, TCS)"
                style={{
                  flex: 1, background: "transparent", border: "none", outline: "none",
                  fontFamily: MONO, fontSize: 14, color: C.textBright,
                  letterSpacing: 1, padding: "13px 0",
                }}
              />
              {loading && (
                <div style={{
                  width: 16, height: 16,
                  border: `2px solid ${C.border}`,
                  borderTop: `2px solid ${C.amber}`,
                  borderRadius: "50%", animation: "spin 0.8s linear infinite",
                }} />
              )}
            </div>
            <button
              className="analyze-btn"
              onClick={handleSearch}
              disabled={loading}
              style={{
                background: loading ? C.bgCard : C.amber,
                color: loading ? C.muted : C.bg,
                border: "none", borderRadius: 6, padding: "0 26px",
                fontFamily: MONO, fontSize: 12, fontWeight: 800,
                letterSpacing: 2, cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.2s", whiteSpace: "nowrap",
              }}
            >
              ANALYZE ▶
            </button>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
            <span style={{ fontFamily: MONO, fontSize: 10, color: C.muted, letterSpacing: 1 }}>QUICK:</span>
            {QUICK.map(s => (
              <button
                key={s}
                className="quick-btn"
                onClick={() => { setInputVal(s); setSymbol(s); analyzeStock(s); }}
                style={{
                  background: symbol === s && data ? "rgba(255,179,0,0.10)" : C.bgCard,
                  border: `1px solid ${symbol === s && data ? C.amber : C.border}`,
                  borderRadius: 4, padding: "4px 10px",
                  fontFamily: MONO, fontSize: 10,
                  color: symbol === s && data ? C.amber : C.mutedBright,
                  cursor: "pointer", letterSpacing: 1, transition: "all 0.15s",
                }}
              >{s}</button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Main content ── */}
      <div style={{ maxWidth: 900, margin: "0 auto", padding: "28px 24px 56px" }}>

        {loading && <LoadingTerminal stock={symbol} />}

        {error && !loading && (
          <div style={{
            background: "rgba(255,69,85,0.07)", border: `1px solid ${C.red}`,
            borderRadius: 8, padding: "20px 24px",
            display: "flex", alignItems: "flex-start", gap: 16, marginTop: 16,
          }}>
            <span style={{ fontSize: 22, color: C.red, marginTop: 2 }}>⚠</span>
            <div>
              <div style={{ fontFamily: MONO, fontSize: 13, color: C.red, fontWeight: 700, marginBottom: 5 }}>
                ANALYSIS FAILED
              </div>
              <div style={{ fontFamily: MONO, fontSize: 12, color: C.text, lineHeight: 1.6 }}>{error}</div>
              <div style={{ fontFamily: MONO, fontSize: 10, color: C.muted, marginTop: 8 }}>
                Tip: Try NSE symbols like RELIANCE, INFY, TCS, ICICIBANK, SBIN
              </div>
            </div>
          </div>
        )}

        {data && !loading && (
          <div style={{ display: "flex", flexDirection: "column", gap: 30 }}>

            {/* ── Header ── */}
            <div className="result-section" style={{ animationDelay: "0ms" }}>
              <div style={{
                display: "flex", alignItems: "flex-start",
                justifyContent: "space-between", flexWrap: "wrap",
                gap: 16, paddingBottom: 22, borderBottom: `1px solid ${C.border}`,
              }}>
                <div>
                  <div style={{ fontFamily: MONO, fontSize: 11, color: C.muted, letterSpacing: 2, marginBottom: 4 }}>
                    {data.exchange || "NSE"}:{data.symbol} · {data.sector}
                  </div>
                  <div style={{ fontFamily: MONO, fontSize: 22, fontWeight: 900, color: C.textBright, letterSpacing: 1, marginBottom: 10 }}>
                    {data.companyName}
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                    <SentimentBadge sentiment={data.sentiment} />
                    {data.technicalRating && (
                      <div style={{
                        display: "inline-flex", alignItems: "center", gap: 6,
                        padding: "5px 12px", border: `1px solid ${techColor(data.technicalRating)}`,
                        borderRadius: 20, background: `${techColor(data.technicalRating)}15`,
                      }}>
                        <span style={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, color: techColor(data.technicalRating), letterSpacing: 1.5 }}>
                          ⚡ {data.technicalRating}
                        </span>
                      </div>
                    )}
                  </div>
                  {data.sentimentReason && (
                    <div style={{ fontFamily: MONO, fontSize: 11, color: C.muted, marginTop: 9, maxWidth: 420, lineHeight: 1.6 }}>
                      {data.sentimentReason}
                    </div>
                  )}
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{
                    fontFamily: MONO, fontSize: 36, fontWeight: 900, color: C.amber,
                    lineHeight: 1, textShadow: `0 0 24px ${C.amberDim}`,
                    animation: "glow 3s ease-in-out infinite",
                  }}>{fmt(data.cmp)}</div>
                  {data.change != null && (
                    <div style={{ fontFamily: MONO, fontSize: 13, color: clr(data.change || 0), marginTop: 5 }}>
                      {data.change >= 0 ? "▲" : "▼"} {Math.abs(data.change).toFixed(2)} ({data.changePct?.toFixed(2)}%)
                    </div>
                  )}
                  {data.prevClose && (
                    <div style={{ fontFamily: MONO, fontSize: 11, color: C.muted, marginTop: 3 }}>
                      Prev Close: {fmt(data.prevClose)}
                    </div>
                  )}
                  <div style={{ fontFamily: MONO, fontSize: 11, color: C.muted, marginTop: 3 }}>
                    Mkt Cap: {data.marketCap}
                  </div>
                </div>
              </div>
            </div>

            {/* ── Market Snapshot ── */}
            <div className="result-section" style={{ animationDelay: "60ms" }}>
              <SectionHead label="Market Snapshot" icon="📊" />
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                <Stat label="Open"      value={fmt(data.openPrice)} />
                <Stat label="Day H / L" value={`${fmt(data.dayHigh)} / ${fmt(data.dayLow)}`} />
                <Stat label="52W High"  value={fmt(data.weekHigh52)}  color={C.green} />
                <Stat label="52W Low"   value={fmt(data.weekLow52)}   color={C.red} />
                <Stat label="P/E"       value={data.pe    ?? "—"} />
                <Stat label="P/BV"      value={data.pbv   ?? "—"} />
                <Stat label="EPS (TTM)" value={data.eps   ? `₹${data.eps}` : "—"} />
                <Stat label="Beta"      value={data.beta  ?? "—"} color={data.beta > 1.3 ? C.amber : C.text} />
                <Stat label="ROE"       value={data.roe   ? `${data.roe}%` : "—"} color={C.green} />
                <Stat label="D/E Ratio" value={data.debtToEquity ?? "—"} color={data.debtToEquity > 1 ? C.red : C.text} />
                <Stat label="Div Yield" value={data.dividendYield ? `${data.dividendYield}%` : "—"} />
                <Stat label="Avg Vol (20D)" value={data.avgVolume20d ?? "—"} />
                <Stat label="1M Return" value={data.monthReturn != null ? pct(data.monthReturn) : "—"} color={clr(data.monthReturn || 0)} />
                <Stat label="1Y Return" value={data.yearReturn  != null ? pct(data.yearReturn)  : "—"} color={clr(data.yearReturn  || 0)} />
                {data.analystTarget && (
                  <Stat
                    label={`Analyst Target (${data.analystCount || "—"})`}
                    value={fmt(data.analystTarget)}
                    color={C.blue}
                    sub={data.analystBuy != null ? `B:${data.analystBuy} H:${data.analystHold} S:${data.analystSell}` : undefined}
                  />
                )}
              </div>
            </div>

            {/* ── Key Levels ── */}
            {data.keyLevels && (
              <div className="result-section" style={{ animationDelay: "100ms" }}>
                <SectionHead label="Key Levels" color={C.blue} icon="🎯" />
                <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                  <Stat label="Support 1"    value={fmt(data.keyLevels.support1)}    color={C.green} />
                  <Stat label="Support 2"    value={fmt(data.keyLevels.support2)}    color={C.green} />
                  <Stat label="Resistance 1" value={fmt(data.keyLevels.resistance1)} color={C.red} />
                  <Stat label="Resistance 2" value={fmt(data.keyLevels.resistance2)} color={C.red} />
                </div>
              </div>
            )}

            {/* ── Bullish / Bearish / Catalysts ── */}
            <div className="result-section" style={{ animationDelay: "120ms" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                <div>
                  <SectionHead label="Bullish Factors" color={C.green} icon="📈" />
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {(data.bullishFactors || []).map((f, i) => <Factor key={i} text={f} positive={true} />)}
                  </div>
                </div>
                <div>
                  <SectionHead label="Risk / Caution" color={C.red} icon="⚠️" />
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {(data.bearishFactors || []).map((f, i) => <Factor key={i} text={f} positive={false} />)}
                  </div>
                </div>
              </div>
              {data.catalysts?.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <SectionHead label="Upcoming Catalysts" color={C.purple} icon="⚡" />
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {data.catalysts.map((c, i) => (
                      <div key={i} style={{
                        padding: "6px 12px", background: "rgba(168,85,247,0.08)",
                        border: `1px solid rgba(168,85,247,0.3)`, borderRadius: 20,
                        fontFamily: MONO, fontSize: 11, color: C.purple,
                      }}>{c}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* ── Entry Zones ── */}
            <div className="result-section" style={{ animationDelay: "160ms" }}>
              <SectionHead label="Entry Zones" color={C.amber} icon="🎯" />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                <div style={{
                  background: C.bgCard, border: `1px solid ${C.amberDim}`,
                  borderRadius: 8, padding: 18,
                }}>
                  <div style={{ fontFamily: MONO, fontSize: 10, letterSpacing: 2, color: C.amber, marginBottom: 8, textTransform: "uppercase" }}>
                    ◈ Zone 1 — Pullback Buy
                  </div>
                  <div style={{ fontFamily: MONO, fontSize: 17, fontWeight: 700, color: C.textBright, marginBottom: 8 }}>
                    {fmt(data.entryPullback?.low)} – {fmt(data.entryPullback?.high)}
                  </div>
                  <div style={{ fontFamily: MONO, fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
                    {data.entryPullback?.rationale}
                  </div>
                </div>
                <div style={{
                  background: C.bgCard, border: `1px solid ${C.greenDim}`,
                  borderRadius: 8, padding: 18,
                }}>
                  <div style={{ fontFamily: MONO, fontSize: 10, letterSpacing: 2, color: C.green, marginBottom: 8, textTransform: "uppercase" }}>
                    ◈ Zone 2 — Breakout Buy
                  </div>
                  <div style={{ fontFamily: MONO, fontSize: 17, fontWeight: 700, color: C.textBright, marginBottom: 5 }}>
                    Above {fmt(data.entryBreakout?.level)}
                  </div>
                  <div style={{ fontFamily: MONO, fontSize: 11, color: C.amber, marginBottom: 8 }}>
                    Vol: {data.entryBreakout?.volumeMultiple}× avg required
                  </div>
                  <div style={{ fontFamily: MONO, fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
                    {data.entryBreakout?.rationale}
                  </div>
                </div>
              </div>
            </div>

            {/* ── Trade Levels ── */}
            <div className="result-section" style={{ animationDelay: "200ms" }}>
              <SectionHead label="Trade Levels" color={C.amber} icon="📐" />
              <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 8, overflow: "hidden" }}>
                <LevelRow label="🛑 STOP LOSS"                          type="sl"   price={data.stopLoss}      pctVal={`-${data.stopLossPct}%`} />
                <LevelRow label={`✦ T1 — ${data.target1?.label || ""}`} type="t1"   price={data.target1?.price} pctVal={`+${data.target1?.pct}%`} />
                <LevelRow label={`✦ T2 — ${data.target2?.label || ""}`} type="t2"   price={data.target2?.price} pctVal={`+${data.target2?.pct}%`} />
                <LevelRow label={`✦ T3 — ${data.target3?.label || ""}`} type="t3"   price={data.target3?.price} pctVal={`+${data.target3?.pct}%`} />
              </div>
              <div style={{
                marginTop: 10, padding: "10px 14px",
                background: C.bgCard, border: `1px solid ${C.border}`,
                borderRadius: 6, display: "flex", gap: 24, flexWrap: "wrap",
              }}>
                {[
                  { label: "Max Risk",    val: `${data.stopLossPct}%`,               c: C.red   },
                  { label: "Max Reward",  val: `${data.target3?.pct}%`,              c: C.green },
                  { label: "R:R (T2)",    val: `1:${(data.target2?.pct / data.stopLossPct).toFixed(1)}`, c: C.blue },
                ].map(({ label, val, c }) => (
                  <div key={label} style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <span style={{ fontFamily: MONO, fontSize: 10, color: C.muted, letterSpacing: 1 }}>{label}:</span>
                    <span style={{ fontFamily: MONO, fontSize: 13, fontWeight: 700, color: c }}>{val}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Confluence Checklist ── */}
            <div className="result-section" style={{ animationDelay: "240ms" }}>
              <SectionHead label="Confluence Checklist" color={C.green} icon="✅" />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {(data.confluenceChecklist || []).map((item, i) => (
                  <CheckItem key={i} label={item.item} status={item.status} note={item.note} />
                ))}
              </div>
              <div style={{ marginTop: 10, fontFamily: MONO, fontSize: 10, color: C.muted, display: "flex", gap: 18, flexWrap: "wrap" }}>
                <span style={{ color: C.green }}>✓ Confirmed Bullish</span>
                <span style={{ color: C.red }}>✗ Confirmed Bearish</span>
                <span style={{ color: C.amber }}>◎ Verify on Chart</span>
              </div>
            </div>

            {/* ── Weekly Trade Plan ── */}
            <div className="result-section" style={{ animationDelay: "280ms" }}>
              <SectionHead label="Weekly Trade Plan" color={C.amber} icon="📅" />
              <div style={{
                background: C.bgCard, border: `1px solid ${C.border}`,
                borderLeft: `3px solid ${C.amber}`, borderRadius: "0 8px 8px 0",
                padding: "18px 22px", fontFamily: MONO, fontSize: 13,
                color: C.text, lineHeight: 1.85,
              }}>
                {data.weeklyStrategy}
              </div>
            </div>

            {/* ── Position Sizer ── */}
            <div className="result-section" style={{ animationDelay: "320ms" }}>
              <RiskCalc cmp={data.cmp} stopLoss={data.stopLoss} />
            </div>

            {/* ── Risk Notice ── */}
            <div className="result-section" style={{ animationDelay: "360ms" }}>
              <div style={{
                background: "rgba(255,69,85,0.06)", border: `1px solid ${C.redDim}`,
                borderRadius: 8, padding: "16px 20px",
                display: "flex", gap: 14, alignItems: "flex-start",
              }}>
                <span style={{ fontSize: 20, color: C.red, marginTop: 2 }}>⚠</span>
                <div>
                  <div style={{ fontFamily: MONO, fontSize: 10, color: C.red, letterSpacing: 2, marginBottom: 6, textTransform: "uppercase" }}>
                    Risk Notice
                  </div>
                  <div style={{ fontFamily: MONO, fontSize: 12, color: C.text, lineHeight: 1.7 }}>
                    {data.riskNote}
                  </div>
                  {data.dataNote && (
                    <div style={{ fontFamily: MONO, fontSize: 10, color: C.muted, marginTop: 8, lineHeight: 1.6 }}>
                      ⓘ {data.dataNote}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* ── Disclaimer ── */}
            <div style={{
              fontFamily: MONO, fontSize: 10, color: C.muted,
              textAlign: "center", lineHeight: 1.7,
              borderTop: `1px solid ${C.border}`, paddingTop: 18,
            }}>
              This analysis is AI-generated for <strong style={{ color: C.amber }}>educational purposes only</strong>.
              Not SEBI-registered investment advice. Always conduct your own due diligence.
              <br />Past performance does not guarantee future results. Equity trading involves substantial risk.
            </div>
          </div>
        )}

        {/* ── Empty state ── */}
        {!data && !loading && !error && (
          <div style={{ textAlign: "center", padding: "70px 24px", fontFamily: MONO, color: C.muted }}>
            <div style={{ fontSize: 44, marginBottom: 18 }}>📈</div>
            <div style={{ fontSize: 14, letterSpacing: 2, marginBottom: 8 }}>ENTER AN NSE SYMBOL TO BEGIN ANALYSIS</div>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>
              Examples: RELIANCE · INFY · TCS · HDFCAMC · BAJFINANCE
            </div>
            <Cursor />
          </div>
        )}
      </div>
    </div>
  );
}