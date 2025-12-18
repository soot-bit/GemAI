document.getElementById("predictionForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const loader = document.getElementById("loader");
    const resultCard = document.getElementById("result");

    // Show loader and hide previous result
    loader.style.display = "block";
    resultCard.style.display = "none";

    const data = {
        carat: parseFloat(document.getElementById("carat").value),
        cut: parseInt(document.getElementById("cut").value),
        color: parseInt(document.getElementById("color").value),
        clarity: parseInt(document.getElementById("clarity").value),
        depth: parseFloat(document.getElementById("depth").value),
        table: parseFloat(document.getElementById("table").value),
        x: parseFloat(document.getElementById("x").value),
        y: parseFloat(document.getElementById("y").value),
        z: parseFloat(document.getElementById("z").value)
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        // Hide loader
        loader.style.display = "none";

        if (!response.ok) {
            const error = await response.text();
            resultCard.innerHTML = `<p style="color: red;">Error: ${error}</p>`;
            resultCard.style.display = "block";
            return;
        }

        const result = await response.json();

        function formatCurrency(value, currency) {
            return value.toLocaleString(undefined, {
                style: "currency",
                currency: currency,
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
        }

        resultCard.innerHTML = `
            <h2>💎 Predicted Price</h2>
            <p><strong>USD:</strong> ${formatCurrency(result.price_usd, "USD")}</p>
            <p><strong>BWP:</strong> P ${result.price_bwp.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            })}</p>
        `;
        resultCard.style.display = "block";

    } catch (err) {
        loader.style.display = "none";
        resultCard.innerHTML = `<p style="color: red;">Error: ${err.message}</p>`;
        resultCard.style.display = "block";
    }
});
