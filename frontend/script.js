async function summarize() {
  const url = document.getElementById("url").value.trim();
  const status = document.getElementById("status");
  const result = document.getElementById("result");
  status.innerText = "⏳ Processing...";
  result.innerText = "";

  const res = await fetch("https://your-backend.onrender.com/summarize", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ url }),
  });

  if (!res.ok) {
    status.innerText = "❌ Failed to summarize.";
    return;
  }

  const data = await res.json();
  status.innerText = "✅ Done!";
  result.innerText = data.summary;
}
