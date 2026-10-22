export default function LoadingSpinner() {
  return (
    <div style={{ textAlign: "center", padding: "40px 0", color: "#64748b" }}>
      <div
        style={{
          display: "inline-block",
          width: "32px",
          height: "32px",
          border: "3px solid #e2e8f0",
          borderTopColor: "#2563eb",
          borderRadius: "50%",
          animation: "spin 0.8s linear infinite",
        }}
      />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <p style={{ marginTop: "12px", fontSize: "14px" }}>Searching…</p>
    </div>
  );
}
